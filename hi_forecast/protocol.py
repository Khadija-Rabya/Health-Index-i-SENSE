"""
PROTOCOLE D'EVALUATION GELE — prediction de `health_index` a t+18 (~3h).

Ce fichier est la source de verite unique de l'evaluation. Une fois valide il
n'est PLUS modifie : toute iteration d'optimisation change le modele ou le
pipeline de features, jamais la definition du decoupage, de la cible ou des
metriques.

------------------------------------------------------------------------------
DECISIONS FIGEES (Phase 1)
------------------------------------------------------------------------------
1. CIBLE      : health_index[t+18] (~3h), valeur continue dans [0, 1].
2. ACCURACY   : % de predictions du jeu de test dont |y_pred - y_vrai| <= 0.01
                (TOLERANCE absolue, ~22% de l'ecart-type de la cible).
                R2 est rapporte systematiquement a cote.
3. TEST GELE  : les 20% de lignes les plus RECENTES de chaque machine, definies
                par un horodatage de coupure par machine. Utilise UNE SEULE FOIS
                par iteration pour le score final, jamais pour la selection.
4. CV         : 5 blocs temporels expansifs (PurgedTimeSeriesSplit) sur les 80%
                restants, avec purge + embargo de 3h de chaque cote de la
                frontiere de validation (= l'horizon de prediction), pour qu'
                aucune ligne d'entrainement n'ait sa cible dans le bloc de
                validation. C'est l'analogue temporel du 5-fold stratifie.
5. SEED       : 42 partout.

------------------------------------------------------------------------------
FUITES CORRIGEES ICI (par rapport au pipeline d'origine)
------------------------------------------------------------------------------
F1. LABEL  : PCA / IsolationForest / seuils d'alarme etaient ajustes sur TOUT le
             dataset, periode de test comprise. Ils sont ici reajustes sur la
             SEULE fenetre d'entrainement, puis appliques (transform) au test.
             -> colonnes *_lf ("leak-free").
F2. vi_proxy : np.polyfit ajuste sur toutes les lignes de chaque machine.
             Reajuste ici sur la fenetre d'entrainement uniquement, puis ses
             derivees (lag/diff/slope/ewma/zscore) sont recalculees.
F3. *_zscore : moyenne/ecart-type calcules sur tout le dataset.
             Recalcules ici sur la fenetre d'entrainement uniquement.
F4. HI_regles : la reference de Kinematic/Dynamic Viscosity etait la mediane de
             TOUT le dataset. Recalculee sur la fenetre d'entrainement.
F5. IMPUTATION : SimpleImputer etait ajuste avant le decoupage. Il fait
             desormais partie du Pipeline sklearn, donc ajuste par pli de CV.
F6. SELECTION : le meilleur modele etait choisi au score de test. Toute
             selection passe desormais par la CV uniquement.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    explained_variance_score, max_error, mean_absolute_error,
    median_absolute_error, mean_squared_error, r2_score,
)
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------- constantes
SEED = 42
HORIZON = 18                     # pas de ~10 min -> ~3h
HORIZON_TD = pd.Timedelta(hours=3)
TOLERANCE = 0.01                 # bande d'accuracy, en unites de health_index
TEST_FRACTION = 0.20
N_SPLITS = 5

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(ROOT, "isense_oil_data_health_index.csv")
CACHE_FILE = os.path.join(ROOT, "hi_forecast", "cache", "prepared.parquet")

ASSET_COLS = {"Motosoufflante A": "asset_Motosoufflante A",
              "Motosoufflante B": "asset_Motosoufflante B"}
FLAG_COLS = ["flag_high_contamination", "flag_high_water",
             "flag_high_temperature", "flag_high_vibration"]

PCA_VARS = [
    "ISO 4_filled", "ISO 6", "ISO 14",
    "Oil H2O ppm_filled", "Oil H2O Saturation_filled",
    "Viscosity at 40°C_filled", "Kinematic Viscosity_filled",
    "viscosity_grade_gap", "vi_proxy",
    "density_15C", "density_deviation_pct",
    "Oil System Vibration_filled", "Oil Temperature_filled",
]

# Colonnes dont la version d'origine est ajustee sur tout le dataset : elles
# sont recalculees "train-only" et les versions d'origine sont retirees des
# predicteurs (cf. F1-F4).
ZSCORE_BASE = [
    "ISO 4_filled", "ISO 6", "ISO 14", "Viscosity at 40°C_filled",
    "Dynamic Viscosity_filled", "Kinematic Viscosity_filled",
    "Oil H2O ppm_filled", "Oil Temperature_filled", "DC_filled",
    "Oil Conductivity_nSm_filled", "Oil System Vibration_filled",
    "density_15C", "vi_proxy",
]

# Identifiants + doublons categoriels + toutes les colonnes construites AVEC
# des statistiques globales (remplacees par leur version _lf).
EXCLUDE_COLS = {
    "created_at", "session_id",
    "vibration_confidence", "sensor_gap_source",
    # versions fuitees du label et de ses composants
    "health_index", "health_state", "HI_pca_t2", "HI_pca_spe", "HI_isoforest",
    "health_state_pca", "health_state_isoforest",
    "HI_regles", "health_state_regles",
    # remplacees par des versions train-only
    "vi_proxy", "vi_proxy_lag1", "vi_proxy_lag2", "vi_proxy_lag3",
    "vi_proxy_diff1", "vi_proxy_slope_3h", "vi_proxy_ewma",
    "dist_Kinematic Viscosity", "dist_Dynamic Viscosity",
}
EXCLUDE_COLS |= {f"{c}_zscore" for c in ZSCORE_BASE}

TARGET_LF = "health_index_lf"
STATE_LF = "health_state_lf"

SURVEILLANCE_PCTL, ALARME_PCTL = 0.95, 0.99
EXPLAINED_VARIANCE_TARGET = 0.90
SEVERITY_LABEL = {0: "Normal", 1: "Surveillance", 2: "Alarme"}

# Reference TD46 / regle relative du document de cadrage OCP, pour F4
_VISC_REF_PCT = {"Kinematic Viscosity": ("Kinematic Viscosity_filled", 5.0, 10.0),
                 "Dynamic Viscosity": ("Dynamic Viscosity_filled", 5.0, 10.0)}
_N_RULE_VARS = 12                # 12 variables a poids egaux dans HI_regles

# Grades ISO VG nominaux (identiques a feature_engineering.py) — necessaires pour
# recalculer viscosity_grade_gap apres masquage de la viscosite.
ISO_VG_GRADES = [22, 32, 46, 68, 100, 150]


# ============================================================ decoupage gele
def frozen_cutoffs(df: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """Horodatage de coupure par machine : les TEST_FRACTION dernieres lignes
    (chronologiquement) de chaque machine constituent le test gele.

    Defini sur le dataframe COMPLET (pas par horizon) pour que la frontiere
    soit identique quel que soit l'horizon etudie."""
    cut = {}
    for asset, col in ASSET_COLS.items():
        times = df.loc[df[col] == 1, "created_at"].sort_values()
        cut[asset] = times.iloc[int(len(times) * (1 - TEST_FRACTION))]
    return cut


def train_mask_from_cutoffs(df: pd.DataFrame, cutoffs) -> pd.Series:
    m = pd.Series(False, index=df.index)
    for asset, col in ASSET_COLS.items():
        m |= (df[col] == 1) & (df["created_at"] < cutoffs[asset])
    return m


# ================================================ reconstruction "leak-free"
VISCOSITY_COLS = ["Viscosity at 40°C_filled", "Kinematic Viscosity_filled",
                  "Dynamic Viscosity_filled"]


def _mask_off_viscosity(df, train_mask):
    """EXPERIENCE DECISIVE — traite la viscosite comme MANQUANTE sur toutes les
    lignes a l'arret (state_OFF == 1), puis l'impute.

    Justification (cf. Note_Viscosite_iSENSE) : sur la Motosoufflante B a l'arret
    le viscosimetre renvoie ~12 cSt pour une ISO VG 46 (~46 cSt), et les trois
    colonnes de viscosite sont touchees simultanement. Le signe est en outre
    physiquement inverse : a l'arret l'huile est plus FROIDE donc devrait etre
    plus VISQUEUSE — c'est ce que fait la Motosoufflante A (49,2 cSt a l'arret
    contre 35,0 en marche) et l'inverse exact de ce que fait B (8,2 contre 32,2).

    Imputation : mediane des lignes EN MARCHE de la meme machine, calculee sur la
    seule fenetre d'entrainement. Hypothese physique : a l'arret l'huile conserve
    la viscosite qu'elle avait en fonctionnement.

    Toutes les colonnes derivees de la viscosite sont recalculees ensuite, sans
    quoi l'artefact resterait present via les lags/EWMA/pentes.
    """
    off = (df["state_OFF"] == 1).to_numpy()
    for col in VISCOSITY_COLS:
        df.loc[off, col] = np.nan
        for asset, acol in ASSET_COLS.items():
            m = (df[acol] == 1).to_numpy()
            ref = df.loc[m & train_mask.to_numpy() & ~off, col].median()
            fill = m & df[col].isna().to_numpy()
            df.loc[fill, col] = ref

    # --- indices metier derives de la viscosite ---
    v40, kin = df["Viscosity at 40°C_filled"], df["Kinematic Viscosity_filled"]
    df["viscosity_grade_gap"] = v40.map(
        lambda x: np.nan if pd.isna(x) else (x - min(ISO_VG_GRADES, key=lambda g: abs(g - x)))
        / min(ISO_VG_GRADES, key=lambda g: abs(g - x)))
    df["viscosity_temp_ratio"] = v40 / kin
    df["temp_viscosity_interaction"] = df["Oil Temperature_filled"] * v40

    # --- lags / diff / EWMA / pentes des colonnes de viscosite ---
    for col in VISCOSITY_COLS:
        g = df.groupby("session_id", sort=False)[col]
        df[f"{col}_diff1"] = g.diff()
        for lag in (1, 2, 3):
            df[f"{col}_lag{lag}"] = g.shift(lag)
        df[f"{col}_ewma"] = g.transform(lambda s: s.ewm(span=18, min_periods=3).mean())
        df[f"{col}_slope_3h"] = (
            df.groupby("session_id", sort=False)[col]
            .rolling(window=18, min_periods=6).apply(_slope, raw=False)
            .reset_index(level=0, drop=True))
    return df


def _rebuild_vi_proxy(df, train_mask):
    """F2 : vi_proxy = residu de viscosity_temp_ratio ~ temperature, regression
    ajustee sur la fenetre d'entrainement uniquement, par machine.

    Les coefficients sont renvoyes pour pouvoir etre embarques dans l'artefact
    final : sans eux, un deploiement ne peut pas recalculer la feature."""
    df["vi_proxy_lf"] = np.nan
    coefs = {}
    for asset, col in ASSET_COLS.items():
        m = df[col] == 1
        fit = df.loc[m & train_mask, ["viscosity_temp_ratio", "Oil Temperature_filled"]].dropna()
        if len(fit) < 3:
            continue
        coef = np.polyfit(fit["Oil Temperature_filled"], fit["viscosity_temp_ratio"], 1)
        coefs[asset] = [float(c) for c in coef]
        pred = np.polyval(coef, df.loc[m, "Oil Temperature_filled"])
        df.loc[m, "vi_proxy_lf"] = df.loc[m, "viscosity_temp_ratio"] - pred
    return df, coefs


def _slope(y):
    y = y.dropna()
    if len(y) < 2:
        return np.nan
    return np.polyfit(np.arange(len(y)), y.to_numpy(), 1)[0]


def _rebuild_vi_proxy_derivatives(df):
    """Recalcule lag/diff/slope/ewma de vi_proxy_lf, causalement (dans la session)."""
    g = df.groupby("session_id", sort=False)["vi_proxy_lf"]
    df["vi_proxy_lf_diff1"] = g.diff()
    for lag in (1, 2, 3):
        df[f"vi_proxy_lf_lag{lag}"] = g.shift(lag)
    df["vi_proxy_lf_ewma"] = g.transform(lambda s: s.ewm(span=18, min_periods=3).mean())
    df["vi_proxy_lf_slope_3h"] = (
        df.groupby("session_id", sort=False)["vi_proxy_lf"]
        .rolling(window=18, min_periods=6).apply(_slope, raw=False)
        .reset_index(level=0, drop=True)
    )
    return df


def _rebuild_zscores(df, train_mask):
    """F3 : z-scores avec moyenne/ecart-type de la fenetre d'entrainement."""
    stats = {}
    for base in ZSCORE_BASE:
        src = "vi_proxy_lf" if base == "vi_proxy" else base
        out = f"{base}_zscore_lf"
        df[out] = np.nan
        for asset, col in ASSET_COLS.items():
            m = df[col] == 1
            ref = df.loc[m & train_mask, src]
            mu, sd = ref.mean(), ref.std()
            sd = sd if (sd and np.isfinite(sd) and sd > 0) else 1.0
            df.loc[m, out] = (df.loc[m, src] - mu) / sd
            stats[(asset, base)] = (float(mu), float(sd))
    return df, stats


def _rebuild_hi_regles(df, train_mask):
    """F4 : distances Kinematic/Dynamic Viscosity avec une reference (mediane)
    calculee sur la fenetre d'entrainement, puis HI_regles recompose."""
    refs = {}
    for var, (col_src, normal_pct, surv_pct) in _VISC_REF_PCT.items():
        out = f"dist_{var}_lf"
        df[out] = np.nan
        for asset, col in ASSET_COLS.items():
            m = df[col] == 1
            ref = df.loc[m & train_mask, col_src].median()
            refs[(asset, var)] = float(ref)
            dev = (df.loc[m, col_src] - ref).abs() / ref * 100
            df.loc[m, out] = ((dev - normal_pct) / (surv_pct - normal_pct)).clip(0, 1)

    dist_cols = [c for c in df.columns
                 if c.startswith("dist_") and not c.endswith("_lf")
                 and c not in ("dist_Kinematic Viscosity", "dist_Dynamic Viscosity")]
    dist_cols += ["dist_Kinematic Viscosity_lf", "dist_Dynamic Viscosity_lf"]
    assert len(dist_cols) == _N_RULE_VARS, f"attendu {_N_RULE_VARS} distances, obtenu {len(dist_cols)}"
    df["HI_regles_lf"] = (1 - df[dist_cols].sum(axis=1) / _N_RULE_VARS).clip(0, 1)
    return df, refs


def _rebuild_label(df, train_mask):
    """F1 : PCA (T2/SPE) + IsolationForest + seuils d'alarme ajustes sur les
    lignes SAINES de la fenetre d'entrainement uniquement, puis appliques a
    toutes les lignes. C'est exactement ce qu'un deploiement temps reel peut
    faire : le label du futur n'entre jamais dans sa propre definition."""
    vars_lf = [("vi_proxy_lf" if v == "vi_proxy" else v) for v in PCA_VARS]
    for c in ("HI_pca_t2_lf", "HI_pca_spe_lf", "HI_isoforest_lf", TARGET_LF):
        df[c] = np.nan
    df[STATE_LF] = "Normal"

    fitted = {}
    for asset, col in ASSET_COLS.items():
        m = (df[col] == 1).to_numpy()
        healthy_train = m & train_mask.to_numpy() & (~df[FLAG_COLS].any(axis=1)).to_numpy()
        healthy = df.loc[healthy_train, vars_lf].dropna()

        scaler = StandardScaler().fit(healthy)
        hs = scaler.transform(healthy)
        cum = np.cumsum(PCA().fit(hs).explained_variance_ratio_)
        k = max(2, min(int(np.searchsorted(cum, EXPLAINED_VARIANCE_TARGET) + 1), len(vars_lf) - 1))
        pca = PCA(n_components=k).fit(hs)

        iso_scaler = StandardScaler().fit(healthy)
        iso = IsolationForest(n_estimators=300, contamination="auto",
                              random_state=SEED, n_jobs=-1).fit(iso_scaler.transform(healthy))

        def score(frame):
            Xs = scaler.transform(frame[vars_lf])
            sc = pca.transform(Xs)
            t2 = np.sum(sc ** 2 / pca.explained_variance_, axis=1)
            spe = np.sum((Xs - pca.inverse_transform(sc)) ** 2, axis=1)
            return t2, spe, -iso.score_samples(iso_scaler.transform(frame[vars_lf]))

        t2_h, spe_h, iso_h = score(healthy)
        t2_s, t2_a = np.quantile(t2_h, [SURVEILLANCE_PCTL, ALARME_PCTL])
        spe_s, spe_a = np.quantile(spe_h, [SURVEILLANCE_PCTL, ALARME_PCTL])
        iso_s, iso_a = np.quantile(iso_h, [SURVEILLANCE_PCTL, ALARME_PCTL])

        t2, spe, isc = score(df.loc[m])
        df.loc[m, "HI_pca_t2_lf"], df.loc[m, "HI_pca_spe_lf"], df.loc[m, "HI_isoforest_lf"] = t2, spe, isc
        severity = np.maximum(np.maximum(t2 / t2_a, spe / spe_a), isc / iso_a)
        df.loc[m, TARGET_LF] = 1.0 / (1.0 + severity)

        st_pca = np.select([t2 > t2_a, spe > spe_a, t2 > t2_s, spe > spe_s], [2, 2, 1, 1], default=0)
        st_iso = np.select([isc > iso_a, isc > iso_s], [2, 1], default=0)
        df.loc[m, STATE_LF] = [SEVERITY_LABEL[s] for s in np.maximum(st_pca, st_iso)]

        fitted[asset] = dict(scaler=scaler, pca=pca, k=k, iso_scaler=iso_scaler, iso=iso,
                             t2_surv=float(t2_s), t2_alarm=float(t2_a),
                             spe_surv=float(spe_s), spe_alarm=float(spe_a),
                             iso_surv=float(iso_s), iso_alarm=float(iso_a),
                             n_healthy_train=int(len(healthy)))
    return df, fitted


def prepare(use_cache: bool = True, verbose: bool = True,
            mask_off_viscosity: bool = False):
    """Charge le dataset, applique les corrections F1-F4 et renvoie
    (df, train_mask, cutoffs, meta_fit).

    mask_off_viscosity : si True, la viscosite est traitee comme manquante sur
    toutes les lignes a l'arret puis imputee (experience decisive, cf.
    _mask_off_viscosity). Le decoupage gele est INCHANGE dans les deux cas :
    memes coupures, memes lignes de test — seule l'etiquette change."""
    df = pd.read_csv(INPUT_FILE, parse_dates=["created_at"], low_memory=False)
    df = df.sort_values(["session_id", "created_at"]).reset_index(drop=True)
    for c in df.columns:                       # bool -> int, homogeneite
        if df[c].dtype == bool:
            df[c] = df[c].astype(np.int8)

    cutoffs = frozen_cutoffs(df)
    train_mask = train_mask_from_cutoffs(df, cutoffs)
    if verbose:
        for a, c in cutoffs.items():
            n_tr = int((train_mask & (df[ASSET_COLS[a]] == 1)).sum())
            n_te = int(((~train_mask) & (df[ASSET_COLS[a]] == 1)).sum())
            print(f"  [split] {a}: cutoff={c}  train={n_tr}  test={n_te}")

    if mask_off_viscosity:
        n_off = int((df["state_OFF"] == 1).sum())
        df = _mask_off_viscosity(df, train_mask)
        df = df.copy()
        if verbose:
            print(f"  [masquage] viscosite traitee comme manquante sur {n_off} lignes "
                  f"a l'arret, puis imputee (mediane ON de la machine, train uniquement)")

    df, vi_coefs = _rebuild_vi_proxy(df, train_mask)
    df = _rebuild_vi_proxy_derivatives(df)
    df, zstats = _rebuild_zscores(df, train_mask)
    df, refs = _rebuild_hi_regles(df, train_mask)
    df = df.copy()                              # defragmente apres les inserts
    df, fitted = _rebuild_label(df, train_mask)
    df = df.copy()

    meta = dict(cutoffs=cutoffs, zscore_stats=zstats, visc_refs=refs,
                vi_proxy_coefs=vi_coefs, label_fit=fitted)
    return df, train_mask, cutoffs, meta


# ===================================================== construction X / y
def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def build_xy(df: pd.DataFrame, horizon: int = HORIZON):
    """X = features a t ; y = health_index_lf a t+horizon (dans la meme session).

    `health_index_lf` a t RESTE un predicteur (c'est l'ancre de persistance) —
    t precede strictement t+horizon, ce n'est pas une fuite."""
    d = df.sort_values(["session_id", "created_at"]).reset_index(drop=True)
    y = d.groupby("session_id", sort=False)[TARGET_LF].shift(-horizon)
    valid = y.notna().to_numpy()

    feats = feature_columns(d)
    X = d.loc[valid, feats].reset_index(drop=True).astype(np.float64)
    y = y[valid].reset_index(drop=True)
    meta = d.loc[valid, ["created_at", "session_id", TARGET_LF] + list(ASSET_COLS.values())]
    meta = meta.reset_index(drop=True).rename(columns={TARGET_LF: "hi_now"})
    return X, y, meta


def split_frozen(X, y, meta, cutoffs):
    """Applique le decoupage gele (defini une seule fois, sur les horodatages)."""
    is_train = pd.Series(False, index=meta.index)
    for asset, col in ASSET_COLS.items():
        is_train |= (meta[col] == 1) & (meta["created_at"] < cutoffs[asset])
    tr, te = np.flatnonzero(is_train), np.flatnonzero(~is_train)
    return (X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True),
            y.iloc[tr].reset_index(drop=True), y.iloc[te].reset_index(drop=True),
            meta.iloc[tr].reset_index(drop=True), meta.iloc[te].reset_index(drop=True))


# ================================================================ CV temporelle
class PurgedTimeSeriesSplit:
    """5 blocs temporels expansifs avec purge + embargo.

    Fold i : train = blocs [0..i], validation = bloc i+1. Toute ligne
    d'entrainement dont la cible (a t+3h) tomberait dans le bloc de validation
    est retiree (purge), plus un embargo de 3h supplementaire — sans quoi les
    dernieres lignes du train partagent leur cible avec le debut de la
    validation, ce qui gonfle artificiellement le score de CV.
    """

    def __init__(self, n_splits=N_SPLITS, horizon_td=HORIZON_TD, embargo_td=HORIZON_TD):
        self.n_splits, self.horizon_td, self.embargo_td = n_splits, horizon_td, embargo_td

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None, times=None):
        if times is None:
            raise ValueError("PurgedTimeSeriesSplit requiert `times` (created_at).")
        times = pd.Series(pd.to_datetime(np.asarray(times)))
        order = np.argsort(times.to_numpy(), kind="mergesort")
        blocks = np.array_split(order, self.n_splits + 1)
        for i in range(self.n_splits):
            val = blocks[i + 1]
            train_pool = np.concatenate(blocks[: i + 1])
            val_start = times.iloc[val].min()
            keep = times.iloc[train_pool].to_numpy() < (val_start - self.horizon_td - self.embargo_td)
            yield train_pool[keep], val


# ==================================================================== metriques
def regression_metrics(y_true, y_pred, n_features=None, hi_now=None, prefix=""):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    err = y_true - y_pred
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    m = {
        f"{prefix}n": n,
        f"{prefix}acc_tol": float(np.mean(np.abs(err) <= TOLERANCE)),
        f"{prefix}r2": float(r2),
        f"{prefix}rmse": rmse,
        f"{prefix}mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}medae": float(median_absolute_error(y_true, y_pred)),
        f"{prefix}mape": float(np.mean(np.abs(err / y_true)) * 100),
        f"{prefix}explained_var": float(explained_variance_score(y_true, y_pred)),
        f"{prefix}max_error": float(max_error(y_true, y_pred)),
        f"{prefix}bias": float(np.mean(y_pred - y_true)),
    }
    if n_features and n - n_features - 1 > 0:
        m[f"{prefix}adj_r2"] = float(1 - (1 - r2) * (n - 1) / (n - n_features - 1))
    else:
        m[f"{prefix}adj_r2"] = np.nan
    if hi_now is not None:                       # skill vs persistance
        hi_now = np.asarray(hi_now, dtype=float)
        mse_p = mean_squared_error(y_true, hi_now)
        m[f"{prefix}skill_vs_persist"] = float(1 - mean_squared_error(y_true, y_pred) / mse_p) if mse_p > 0 else np.nan
        m[f"{prefix}r2_persist"] = float(r2_score(y_true, hi_now))
        m[f"{prefix}acc_tol_persist"] = float(np.mean(np.abs(y_true - hi_now) <= TOLERANCE))
    return m


def data_fingerprint(X, y):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.nan_to_num(X.to_numpy(dtype=np.float64))).tobytes())
    h.update(np.ascontiguousarray(y.to_numpy(dtype=np.float64)).tobytes())
    return h.hexdigest()[:16]


if __name__ == "__main__":
    df, train_mask, cutoffs, meta_fit = prepare()
    X, y, meta = build_xy(df)
    Xtr, Xte, ytr, yte, mtr, mte = split_frozen(X, y, meta, cutoffs)
    print(f"\n  features         : {X.shape[1]}")
    print(f"  train / test     : {len(Xtr)} / {len(Xte)}  ({len(Xte)/len(X):.1%} test)")
    print(f"  y train mean/std : {ytr.mean():.4f} / {ytr.std():.4f}")
    print(f"  y test  mean/std : {yte.mean():.4f} / {yte.std():.4f}")
    print(f"  fingerprint      : {data_fingerprint(X, y)}")
    cv = PurgedTimeSeriesSplit()
    for i, (tr, va) in enumerate(cv.split(Xtr, times=mtr["created_at"]), 1):
        print(f"  fold {i}: train={len(tr):>6}  val={len(va):>6}  "
              f"val_window={str(mtr['created_at'].iloc[va].min())[:16]} -> "
              f"{str(mtr['created_at'].iloc[va].max())[:16]}")
