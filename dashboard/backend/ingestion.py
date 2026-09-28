"""Chemin d'INFERENCE INCREMENTALE — de la ligne qui arrive a l'indice.

PROBLEME QUE CE MODULE RESOUT
-----------------------------------------------------------------------------
Jusqu'ici, le service appelait `protocol.prepare()`, qui relit les 42 141 lignes
du jeu complet et REAJUSTE tout : moyennes et ecarts-types des z-scores,
coefficients du vi_proxy, references de viscosite, ACP et Isolation Forest de
l'etiquette, ancrages de l'indice systeme. C'est le bon comportement pour
entrainer ; c'est le mauvais pour servir, et pour deux raisons :

  1. COUT — recalculer sept mois d'historique pour obtenir la valeur d'une seule
     ligne qui vient d'arriver.
  2. CORRECTION — des parametres reajustes derivent avec les donnees qu'ils
     jugent. Deux lignes identiques recevraient deux indices differents selon le
     lot dans lequel elles arrivent, et l'indice cesserait d'etre comparable
     dans le temps. C'est le SKEW ENTRAINEMENT / SERVICE decrit au point dur
     n° 1 de ARCHITECTURE_PIPELINE.md.

CE QUE FAIT CE MODULE
  - `geler_pretraitement()` : execute la chaine d'ajustement UNE FOIS et fige
    tous les parametres dans un artefact unique.
  - `FluxIncremental` : tient un TAMPON des dernieres lignes par machine, et
    n'applique les transformations figees qu'a ce tampon. Jamais au jeu complet.

POURQUOI UN TAMPON ET NON UNE LIGNE SEULE
Certaines variables ont besoin de passe : lags 1 a 3, difference premiere, pente
sur 18 pas (~3 h), EWMA de portee 18. Une ligne isolee ne suffit pas. Le tampon
par defaut fait 600 lignes, soit la meme amorce que le mode rejeu — assez pour
que l'EWMA converge a mieux que 1e-9 pres de sa valeur hors ligne (la memoire
d'une EWMA de portee 18 est epuisee bien avant 600 pas).

LIMITE CONNUE — l'entree attendue est une ligne DEJA NETTOYEE ET IMPUTEE, avec
les colonnes `*_filled`. Les modeles d'imputation (RandomForest de la vibration,
comblement du trou capteur) ne sont pas encore persistes en artefact : c'est
l'etage qui manque pour partir du JSON brut de l'API. Le reste de la chaine, du
calcul des variables jusqu'a la prevision, est couvert ici.
"""
from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
HI = os.path.join(ROOT, "hi_forecast")
for p in (ROOT, HI, "C:\\pylib"):
    if p not in sys.path:
        sys.path.insert(0, p)

import feature_engineering as fe                                   # noqa: E402
import health_index_baseline as hib                                # noqa: E402
from protocol import (ALARME_PCTL, ASSET_COLS, PCA_VARS,           # noqa: E402
                      SEVERITY_LABEL, SURVEILLANCE_PCTL, TARGET_LF, STATE_LF,
                      ZSCORE_BASE, _VISC_REF_PCT, _N_RULE_VARS,
                      _rebuild_vi_proxy_derivatives)
from systeme_hi import appliquer_systeme_hi                        # noqa: E402

ARTEFACT = os.path.join(HI, "artifacts", "pretraitement.joblib")
CLEANED = os.path.join(ROOT, "isense_oil_data_cleaned.csv")
TAMPON_DEFAUT = 600


# ═══════════════════════════════════════════════════ 1. GEL DES PARAMETRES
def geler_pretraitement(chemin: str = ARTEFACT) -> dict:
    """Ajuste la chaine UNE FOIS et fige tous ses parametres dans un artefact.

    A relancer uniquement lors d'un reentrainement — jamais au fil de l'eau."""
    from protocol import prepare
    from systeme_hi import build_systeme_hi

    df, train_mask, cutoffs, meta = prepare(verbose=False, mask_off_viscosity=False)
    df, calibrage = build_systeme_hi(df, train_mask)

    art = {
        "version": 1,
        "genere_le": pd.Timestamp.now().isoformat(),
        "cutoffs": cutoffs,
        "zscore_stats": meta["zscore_stats"],
        "visc_refs": meta["visc_refs"],
        "vi_proxy_coefs": meta["vi_proxy_coefs"],
        "label_fit": meta["label_fit"],
        "systeme_calibrage": calibrage,
        # Ordre et presence des colonnes apres la chaine complete : sert a
        # realigner un tampon ou une modalite categorielle serait absente.
        "colonnes": list(df.columns),
        "n_lignes_ajustement": int(len(df)),
        # Dernier etage manquant du chemin temps reel : l'imputation.
        "imputeur_vibration": _geler_imputeur_vibration(cutoffs),
        "medianes_capteurs": _geler_medianes_capteurs(cutoffs),
    }
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    joblib.dump(art, chemin, compress=3)
    return art


def _geler_imputeur_vibration(cutoffs: dict) -> dict | None:
    """Fige l'imputeur de vibration : mediane des predicteurs + Random Forest.

    Regle metier (fill_vibration.py) : a l'ARRET la vibration vaut 0 — decision
    du specialiste, le systeme ne vibre pas ; EN MARCHE une valeur manquante est
    PREDITE. Le hors-ligne ajuste cette foret sur tout le lot ; en service elle
    doit etre figee, sinon l'imputation derive avec les donnees qu'elle comble."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    import fill_vibration as fv

    if not os.path.exists(CLEANED):
        return None
    df = pd.read_csv(CLEANED, parse_dates=["created_at"])
    # ajustement sur la FENETRE D'ENTRAINEMENT uniquement
    tr = pd.Series(False, index=df.index)
    for asset, coupe in cutoffs.items():
        tr |= (df["asset_name"] == asset) & (df["created_at"] < coupe)
    ok = tr & (df["machine_state"] == "ON") & df[fv.TARGET].notna()
    if ok.sum() < 100:
        return None

    X = df[fv.RF_PREDICTORS].copy()
    X["hour_sin"] = np.sin(2 * np.pi * df["created_at"].dt.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * df["created_at"].dt.hour / 24)
    X = pd.get_dummies(X.join(df[["asset_name"]]), columns=["asset_name"])
    imputer = SimpleImputer(strategy="median").fit(X.loc[tr])
    Xi = pd.DataFrame(imputer.transform(X), columns=X.columns, index=X.index)

    rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(Xi.loc[ok], df.loc[ok, fv.TARGET])
    return {"imputeur": imputer, "foret": rf, "colonnes": list(X.columns),
            "predicteurs": list(fv.RF_PREDICTORS), "n_ajustement": int(ok.sum())}


def _temps(serie: pd.Series) -> pd.Series:
    """Horodatages naifs en UTC. Tolere texte, datetime naif ou avec fuseau —
    un instant avec fuseau ne se compare pas a un instant sans, et le jeu local
    est naif."""
    if not pd.api.types.is_datetime64_any_dtype(serie):
        serie = pd.to_datetime(serie, format="ISO8601", utc=True)
    if getattr(serie.dt, "tz", None) is not None:
        serie = serie.dt.tz_localize(None)
    return serie


CAPTEURS_FILLED = ["DC", "Density", "Dynamic Viscosity", "Kinematic Viscosity",
                   "Oil H2O Saturation", "Oil H2O ppm", "Oil Temperature",
                   "Viscosity at 40°C", "Oil Conductivity", "Oil Conductivity_nSm",
                   "ISO 4"]


def _geler_medianes_capteurs(cutoffs: dict) -> dict:
    """Mediane par machine et par capteur, sur la fenetre d'entrainement.

    Dernier recours du comblement : quand une valeur manque ET qu'aucune valeur
    anterieure n'est disponible dans le tampon (demarrage a froid), on retombe
    sur cette mediane figee plutot que de laisser un NaN qui ferait echouer
    l'inference."""
    if not os.path.exists(CLEANED):
        return {}
    df = pd.read_csv(CLEANED, parse_dates=["created_at"])
    med = {}
    for asset, coupe in cutoffs.items():
        m = (df["asset_name"] == asset) & (df["created_at"] < coupe)
        for c in CAPTEURS_FILLED:
            if c in df.columns:
                v = df.loc[m, c].median()
                if np.isfinite(v):
                    med[(asset, c)] = float(v)
    return med


def appliquer_imputation_capteurs(df: pd.DataFrame, art: dict) -> pd.DataFrame:
    """Comble les capteurs : valeur mesuree, sinon report de la derniere connue
    dans le tampon, sinon mediane FIGEE.

    C'est la version en ligne du trou capteur synchrone de 994 lignes traite
    hors ligne : DC, ISO 4, H2O, temperature et viscosite manquent sur LES MEMES
    lignes, signe d'une coupure d'acquisition et non d'une panne de capteur
    isolee. Le report de la derniere valeur est le comportement correct pour une
    grandeur physique qui ne saute pas."""
    med = art.get("medianes_capteurs", {})
    manquait = pd.Series(False, index=df.index)
    for c in CAPTEURS_FILLED:
        if c not in df.columns:
            continue
        cible = f"{c}_filled"
        df[cible] = df[c]
        manquait |= df[c].isna()
        for asset in df.get("asset_name", pd.Series(dtype=object)).unique():
            m = (df["asset_name"] == asset).to_numpy()
            if not m.any():
                continue
            serie = df.loc[m, cible].ffill()
            defaut = med.get((asset, c))
            if defaut is not None:
                serie = serie.fillna(defaut)
            df.loc[m, cible] = serie
    df["sensor_gap_source"] = np.where(manquait, "comblé", "mesuré")
    return df


def appliquer_imputation_vibration(df: pd.DataFrame, art: dict) -> pd.DataFrame:
    """Remplit `Oil System Vibration` avec l'imputeur FIGE, et pose les colonnes
    de tracabilite que le reste de la chaine attend."""
    import fill_vibration as fv
    cible = fv.TARGET
    if cible not in df.columns:
        return df
    vib = art.get("imputeur_vibration")
    connu = df[cible].notna()
    off = df["machine_state"] == "OFF"
    a_predire = (~connu) & (df["machine_state"] == "ON")

    df[f"{cible}_filled"] = df[cible]
    df.loc[(~connu) & off, f"{cible}_filled"] = 0.0
    if vib is not None and a_predire.any():
        X = df[vib["predicteurs"]].copy()
        X["hour_sin"] = np.sin(2 * np.pi * df["created_at"].dt.hour / 24)
        X["hour_cos"] = np.cos(2 * np.pi * df["created_at"].dt.hour / 24)
        X = pd.get_dummies(X.join(df[["asset_name"]]), columns=["asset_name"])
        X = X.reindex(columns=vib["colonnes"], fill_value=0)
        Xi = pd.DataFrame(vib["imputeur"].transform(X), columns=X.columns, index=X.index)
        df.loc[a_predire, f"{cible}_filled"] = vib["foret"].predict(Xi.loc[a_predire])

    df["vibration_source"] = "mesuré"
    df.loc[(~connu) & off, "vibration_source"] = "imputé_zero_off"
    df.loc[a_predire, "vibration_source"] = "imputé_rf_on"
    df["vibration_confidence"] = df["vibration_source"]
    df["vibration_available"] = connu.astype(int)
    # Distance a la mesure reelle la plus proche : plus elle est grande, moins
    # l'imputation est fiable. Le tableau de bord s'en sert comme score de
    # confiance.
    dist = np.full(len(df), 0.0)
    depuis = np.inf
    for i, k in enumerate(connu.to_numpy()):
        depuis = 0.0 if k else depuis + 1
        dist[i] = depuis
    df["vibration_gap_distance"] = np.where(np.isfinite(dist), dist, 999.0)
    return df


def charger_pretraitement(chemin: str = ARTEFACT) -> dict:
    if not os.path.exists(chemin):
        raise FileNotFoundError(
            f"Artefact de pretraitement absent : {chemin}\n"
            "Le produire avec :  python -c \"import ingestion; ingestion.geler_pretraitement()\"")
    return joblib.load(chemin)


# ═══════════════════════════════════════ 2. APPLICATION A PARAMETRES FIGES
def _appliquer_vi_proxy(df: pd.DataFrame, coefs: dict) -> pd.DataFrame:
    """Residu de viscosity_temp_ratio ~ temperature, coefficients FIGES."""
    df["vi_proxy_lf"] = np.nan
    for asset, col in ASSET_COLS.items():
        coef = coefs.get(asset)
        if coef is None:
            continue
        m = (df[col] == 1).to_numpy()
        if not m.any():
            continue
        pred = np.polyval(coef, df.loc[m, "Oil Temperature_filled"])
        df.loc[m, "vi_proxy_lf"] = df.loc[m, "viscosity_temp_ratio"] - pred
    return df


def _appliquer_zscores(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """z-scores avec les mu et sigma FIGES de la fenetre d'entrainement."""
    for base in ZSCORE_BASE:
        src = "vi_proxy_lf" if base == "vi_proxy" else base
        out = f"{base}_zscore_lf"
        df[out] = np.nan
        if src not in df:
            continue
        for asset, col in ASSET_COLS.items():
            p = stats.get((asset, base))
            if p is None:
                continue
            mu, sd = p
            m = (df[col] == 1).to_numpy()
            if m.any():
                df.loc[m, out] = (df.loc[m, src] - mu) / (sd if sd else 1.0)
    return df


def _appliquer_hi_regles(df: pd.DataFrame, refs: dict) -> pd.DataFrame:
    """Distances aux references de viscosite FIGEES, puis HI_regles recompose."""
    for var, (col_src, normal_pct, surv_pct) in _VISC_REF_PCT.items():
        out = f"dist_{var}_lf"
        df[out] = np.nan
        for asset, col in ASSET_COLS.items():
            ref = refs.get((asset, var))
            if ref is None or not np.isfinite(ref) or ref == 0:
                continue
            m = (df[col] == 1).to_numpy()
            if not m.any():
                continue
            dev = (df.loc[m, col_src] - ref).abs() / ref * 100
            df.loc[m, out] = ((dev - normal_pct) / (surv_pct - normal_pct)).clip(0, 1)

    dist_cols = [c for c in df.columns
                 if c.startswith("dist_") and not c.endswith("_lf")
                 and c not in ("dist_Kinematic Viscosity", "dist_Dynamic Viscosity")]
    dist_cols += ["dist_Kinematic Viscosity_lf", "dist_Dynamic Viscosity_lf"]
    if len(dist_cols) != _N_RULE_VARS:
        raise ValueError(f"attendu {_N_RULE_VARS} distances, obtenu {len(dist_cols)} — "
                         "le tampon ne porte pas les memes colonnes que l'ajustement")
    df["HI_regles_lf"] = (1 - df[dist_cols].sum(axis=1) / _N_RULE_VARS).clip(0, 1)
    return df


def _appliquer_label(df: pd.DataFrame, fitted: dict) -> pd.DataFrame:
    """ACP (T2/SPE) + Isolation Forest FIGEES, puis etiquette et severite.

    Reproduit exactement protocol._rebuild_label, mais en n'utilisant que les
    objets deja ajustes et les seuils deja calcules."""
    vars_lf = [("vi_proxy_lf" if v == "vi_proxy" else v) for v in PCA_VARS]
    for c in ("HI_pca_t2_lf", "HI_pca_spe_lf", "HI_isoforest_lf", TARGET_LF):
        df[c] = np.nan
    df[STATE_LF] = "Normal"

    for asset, col in ASSET_COLS.items():
        f = fitted.get(asset)
        if f is None:
            continue
        m = (df[col] == 1).to_numpy()
        if not m.any():
            continue
        frame = df.loc[m, vars_lf]
        # Une ligne fraiche peut porter des NaN : au demarrage d'une session,
        # les variables temporelles (lags, pente 3 h, EWMA) ne sont pas encore
        # constituees. L'ACP refuse les NaN — hors ligne le cas n'existe pas,
        # le jeu est complet. On substitue la MOYENNE DE LA POPULATION SAINE,
        # deja portee par le scaler ajuste : c'est la valeur neutre, celle qui
        # n'apporte aucune information et ne cree donc pas d'anomalie
        # artificielle. L'indice reste calculable, simplement moins informe
        # tant que la fenetre n'est pas pleine.
        neutre = pd.Series(f["scaler"].mean_, index=vars_lf)
        frame = frame.fillna(neutre)
        Xs = f["scaler"].transform(frame)
        sc = f["pca"].transform(Xs)
        t2 = np.sum(sc ** 2 / f["pca"].explained_variance_, axis=1)
        spe = np.sum((Xs - f["pca"].inverse_transform(sc)) ** 2, axis=1)
        isc = -f["iso"].score_samples(f["iso_scaler"].transform(frame))

        df.loc[m, "HI_pca_t2_lf"] = t2
        df.loc[m, "HI_pca_spe_lf"] = spe
        df.loc[m, "HI_isoforest_lf"] = isc
        severity = np.maximum(np.maximum(t2 / f["t2_alarm"], spe / f["spe_alarm"]),
                              isc / f["iso_alarm"])
        df.loc[m, TARGET_LF] = 1.0 / (1.0 + severity)

        st_pca = np.select([t2 > f["t2_alarm"], spe > f["spe_alarm"],
                            t2 > f["t2_surv"], spe > f["spe_surv"]],
                           [2, 2, 1, 1], default=0)
        st_iso = np.select([isc > f["iso_alarm"], isc > f["iso_surv"]], [2, 1], default=0)
        df.loc[m, STATE_LF] = [SEVERITY_LABEL[s] for s in np.maximum(st_pca, st_iso)]
    return df


def _corriger_session(df: pd.DataFrame, sessions: dict) -> pd.DataFrame:
    """Retablit les deux variables de session que la troncature du tampon fausse.

    `time_in_session_h` et `measure_index_in_session` sont calcules par
    l'ingenierie hors ligne par rapport a la PREMIERE ligne de la session. Dans
    un tampon glissant, cette ligne a souvent ete evincee : le hors-ligne dirait
    « 40e mesure, 6,6 h de session », le tampon dirait « 1re mesure, 0 h ».
    On reinjecte donc l'etat conserve — horodatage de debut et nombre de lignes
    deja sorties du tampon. C'est le `SessionState` de ARCHITECTURE_PIPELINE.md.
    """
    if not sessions:
        return df
    for sid, etat in sessions.items():
        m = (df["session_id"] == sid).to_numpy()
        if not m.any():
            continue
        df.loc[m, "time_in_session_h"] = (
            (df.loc[m, "created_at"] - etat["debut"]).dt.total_seconds() / 3600.0)
        # rang dans la session = lignes deja evincees + rang dans le tampon
        rang = np.arange(int(m.sum()))
        df.loc[m, "measure_index_in_session"] = etat.get("evincees", 0) + rang
    return df


def traiter_fenetre(brut: pd.DataFrame, art: dict, sessions: dict | None = None) -> pd.DataFrame:
    """Chaine complete appliquee a UNE FENETRE de lignes deja nettoyees/imputees.

    Ingenierie de variables -> transformations figees -> etiquette -> indice
    systeme. Aucun parametre n'est reestime ici."""
    df = brut.sort_values(["session_id", "created_at"]).reset_index(drop=True).copy()

    # --- 1. ingenierie de variables (fonctions pures du hors-ligne) ---
    for etape in (fe.add_calendar_features, fe.add_session_features,
                  fe.add_physicochemical_indices, fe.add_trend_features,
                  fe.add_lag_features, fe.add_ewma_features, fe.add_domain_indices,
                  fe.add_zscore_per_machine, fe.add_threshold_flags,
                  fe.add_vibration_trust_features, fe.add_sensor_gap_trust_feature,
                  fe.add_categorical_encoding):
        df = etape(df)
    df = _corriger_session(df, sessions or {})

    # --- 1bis. distances aux seuils officiels OCP ---
    # Douze variables a poids egaux. Dix reposent sur des seuils ABSOLUS et ne
    # dependent donc pas du lot : on peut les calculer sur le tampon sans biais.
    # Les deux distances de viscosite, elles, se mesurent a une mediane de
    # reference — un parametre ajuste, donc fige, recalcule juste apres par
    # _appliquer_hi_regles qui ecrase ces deux colonnes par leurs versions _lf.
    for var in hib.OFFICIAL_THRESHOLDS:
        df[f"dist_{var}"] = np.nan
    for asset, col in ASSET_COLS.items():
        m = (df[col] == 1).to_numpy()
        if not m.any():
            continue
        d = hib.compute_distances(df.loc[m])
        for var in hib.OFFICIAL_THRESHOLDS:
            df.loc[m, f"dist_{var}"] = d[var].to_numpy()

    # Une modalite absente du tampon (machine muette, etat jamais atteint) ferait
    # disparaitre sa colonne indicatrice. On realigne sur les colonnes de
    # l'ajustement, les manquantes a zero.
    for c in art["colonnes"]:
        if c not in df.columns and (c.startswith("asset_") or c.startswith("state_")
                                    or c.startswith("vibsource_")):
            df[c] = 0
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(np.int8)

    # --- 2. transformations a parametres figes, dans l'ordre du hors-ligne ---
    df = _appliquer_vi_proxy(df, art["vi_proxy_coefs"])
    df = _rebuild_vi_proxy_derivatives(df)          # causal, rien a ajuster
    df = _appliquer_zscores(df, art["zscore_stats"])
    df = _appliquer_hi_regles(df, art["visc_refs"])
    df = df.copy()
    df = _appliquer_label(df, art["label_fit"])
    df = appliquer_systeme_hi(df, art["systeme_calibrage"])

    # --- 3. alias attendus par la couche de service ---
    df["health_index"] = df[TARGET_LF]
    df["health_state"] = df[STATE_LF]
    # Deux indicateurs INDEPENDANTS, plus d'agregat : voir systeme_hi.py.
    for canal in ("pression", "vibration"):
        df[f"indice_{canal}"] = df[f"indice_{canal}_lf"]
    return df


# ═══════════════════════════════════════════════════ 3. FLUX INCREMENTAL
class FluxIncremental:
    """Tampon par machine + application des transformations figees.

    Interface volontairement identique a celle de `ReplayFeed` : `window()`
    renvoie « les lignes traitees dont on dispose maintenant ». Tout l'etage aval
    (predict_latest, history_series, l'API, le tableau de bord) est inchange.
    """

    def __init__(self, art: dict | None = None, taille_tampon: int = TAMPON_DEFAUT):
        self.art = art if art is not None else charger_pretraitement()
        self.taille = int(taille_tampon)
        self._tampons: dict[str, pd.DataFrame] = {}
        self._sessions: dict = {}          # etat par session : debut, total, evincees
        self._traite: pd.DataFrame | None = None

    def _maj_sessions(self, lignes: pd.DataFrame) -> None:
        """Met a jour, pour chaque session vue, son horodatage de debut et le
        nombre total de lignes rencontrees."""
        for sid, part in lignes.groupby("session_id"):
            e = self._sessions.setdefault(sid, {"debut": part["created_at"].min(),
                                                "total": 0, "evincees": 0})
            e["debut"] = min(e["debut"], part["created_at"].min())
            e["total"] += int(len(part))

    def _maj_evincees(self) -> None:
        """Lignes de chaque session sorties du tampon = total vu - presentes."""
        presentes: dict = {}
        for t in self._tampons.values():
            if t is None or not len(t):
                continue
            for sid, n in t["session_id"].value_counts().items():
                presentes[sid] = presentes.get(sid, 0) + int(n)
        for sid, e in self._sessions.items():
            e["evincees"] = max(0, e["total"] - presentes.get(sid, 0))

    def amorcer(self, historique: pd.DataFrame) -> None:
        """Remplit les tampons a froid depuis l'historique — l'equivalent du
        `warmup_rows` du rejeu. Sans amorce, les 18 premieres lignes d'une
        session sortent avec des variables temporelles incompletes.

        L'historique COMPLET sert a etablir l'etat de session (vrai debut, vrai
        compteur) ; seules ses dernieres lignes entrent dans le tampon."""
        h = historique.copy()
        if "created_at" in h:
            h["created_at"] = _temps(h["created_at"])
        self._maj_sessions(h)
        for asset, col in ASSET_COLS.items():
            if col in historique.columns:
                sub = historique[historique[col] == 1]
            else:
                sub = historique[historique.get("asset_name", "") == asset]
            if len(sub):
                self._tampons[asset] = sub.sort_values("created_at").tail(self.taille).copy()
        self._maj_evincees()
        # Traiter tout de suite : sans cela window() reste vide jusqu'a la
        # premiere ligne ingeree. Si l'API n'a rien de neuf au demarrage — cas
        # courant entre deux mesures — le tableau de bord n'afficherait rien.
        morceaux = [t for t in self._tampons.values() if t is not None and len(t)]
        if morceaux:
            self._traite = traiter_fenetre(pd.concat(morceaux, ignore_index=True),
                                           self.art, self._sessions)

    def ingerer(self, lignes: pd.DataFrame) -> pd.DataFrame:
        """Ajoute des lignes brutes (deja nettoyees/imputees) et renvoie la
        fenetre TRAITEE. Le calcul ne porte que sur les tampons, jamais sur
        l'historique complet."""
        if "created_at" in lignes:
            lignes = lignes.copy()
            lignes["created_at"] = _temps(lignes["created_at"])

        self._maj_sessions(lignes)
        for asset, col in ASSET_COLS.items():
            part = (lignes[lignes[col] == 1] if col in lignes.columns
                    else lignes[lignes.get("asset_name", "") == asset])
            if not len(part):
                continue
            base = self._tampons.get(asset)
            fusion = part if base is None else pd.concat([base, part], ignore_index=True)
            fusion = (fusion.drop_duplicates(subset=["created_at"], keep="last")
                            .sort_values("created_at").tail(self.taille))
            self._tampons[asset] = fusion.reset_index(drop=True)

        self._maj_evincees()
        morceaux = [t for t in self._tampons.values() if t is not None and len(t)]
        if not morceaux:
            self._traite = None
            return pd.DataFrame()
        fenetre = pd.concat(morceaux, ignore_index=True)
        self._traite = traiter_fenetre(fenetre, self.art, self._sessions)
        return self._traite

    def window(self, n_rows: int = 4000) -> pd.DataFrame:
        """Interface commune avec ReplayFeed : la fenetre traitee courante."""
        if self._traite is None:
            return pd.DataFrame()
        return self._traite.sort_values("created_at").tail(n_rows)

    def status(self) -> dict:
        return {
            "mode": "flux incremental",
            "taille_tampon": self.taille,
            "lignes_en_tampon": {a: int(len(t)) for a, t in self._tampons.items()},
            "artefact_genere_le": self.art.get("genere_le"),
            "sessions_suivies": len(self._sessions),
        }
