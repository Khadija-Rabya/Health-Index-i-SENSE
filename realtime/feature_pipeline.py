"""
Version temps reel du pipeline batch (clean_isense_data.py -> fill_vibration.py ->
fill_sensor_gap.py -> feature_engineering.py -> health_index_baseline/pca/isolation_forest.py),
appliquee a une petite fenetre de donnees fraichement recuperees (realtime/api_client.py)
plutot qu'a tout l'historique. Reutilise directement les fonctions batch existantes partout
ou leur comportement ne depend pas de la taille du dataframe recu ; seules deux etapes sont
reimplementees en version "figee" (parametres appris a l'entrainement, pas recalcules a la
volee) - cf. realtime/artifacts.py :

  - le z-score par machine (feature_engineering.add_zscore_per_machine recalcule
    moyenne/ecart-type sur le dataframe recu - correct sur tout l'historique, faux sur un
    buffer de quelques dizaines de lignes) ;
  - le residu vi_proxy (meme probleme : une regression re-ajustee sur un petit buffer est
    instable, alors que vi_proxy sert ensuite de variable PCA).

Le remplissage du trou capteur synchrone (fill_sensor_gap.py) est simplifie en un simple
forward-fill par machine (avec repli sur la mediane historique geleee, fill_medians.json) :
le Random Forest dedie a ISO 4 dans le pipeline batch ciblait un episode de coupure
historique precis (994 lignes), pas un pattern recurrent attendu en temps reel.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
sys.path.insert(0, PROJECT_DIR)

import clean_isense_data as cid  # noqa: E402
import feature_engineering as fe  # noqa: E402
import fill_sensor_gap as fsg  # noqa: E402
import health_index_baseline as hib  # noqa: E402
import health_index_isolation_forest as hiso  # noqa: E402
import health_index_pca as hpca  # noqa: E402

ASSET_COLS = {"Motosoufflante A": "asset_Motosoufflante A", "Motosoufflante B": "asset_Motosoufflante B"}
ASSET_KEY = {"Motosoufflante A": "A", "Motosoufflante B": "B"}


def load_artifacts():
    """Charge une fois tous les artefacts produits par realtime/artifacts.py."""
    artifacts = {
        "health_index": {
            asset: joblib.load(os.path.join(ARTIFACTS_DIR, f"health_index_model_{key}.joblib"))
            for asset, key in ASSET_KEY.items()
        },
        "vibration_imputer": joblib.load(os.path.join(ARTIFACTS_DIR, "vibration_imputer.joblib")),
    }
    with open(os.path.join(ARTIFACTS_DIR, "zscore_stats.json"), encoding="utf-8") as f:
        artifacts["zscore_stats"] = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "vi_proxy_coefs.json"), encoding="utf-8") as f:
        artifacts["vi_proxy_coefs"] = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "fill_medians.json"), encoding="utf-8") as f:
        artifacts["fill_medians"] = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "regles_thresholds.json"), encoding="utf-8") as f:
        artifacts["regles_thresholds"] = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "manifest.json"), encoding="utf-8") as f:
        artifacts["manifest"] = json.load(f)
    return artifacts


def fill_vibration_live(df, bundle):
    """OFF & NaN -> 0 (regle physique) ; ON & NaN -> predit par le Random Forest sauvegarde
    (fill_vibration.py, reentraine par realtime/artifacts.py). Meme structure de features
    que le batch (RF_PREDICTORS + hour_sin/cos + one-hot asset_name), realignee sur les
    colonnes d'entrainement avant transformation par l'imputer sauvegarde."""
    target = "Oil System Vibration"
    off_missing = (df["machine_state"] == "OFF") & df[target].isna()
    on_missing = (df["machine_state"] == "ON") & df[target].isna()

    df[f"{target}_filled"] = df[target]
    df.loc[off_missing, f"{target}_filled"] = 0.0

    if on_missing.any():
        X = df.loc[on_missing, bundle["predictors"]].copy()
        X["hour_sin"] = np.sin(2 * np.pi * df.loc[on_missing, "created_at"].dt.hour / 24)
        X["hour_cos"] = np.cos(2 * np.pi * df.loc[on_missing, "created_at"].dt.hour / 24)
        X = pd.get_dummies(X.join(df.loc[on_missing, ["asset_name"]]), columns=["asset_name"])
        X = X.reindex(columns=bundle["columns"], fill_value=0)
        X_imp = bundle["imputer"].transform(X)
        df.loc[on_missing, f"{target}_filled"] = bundle["model"].predict(X_imp)

    df["vibration_source"] = "mesuré"
    df.loc[off_missing, "vibration_source"] = "imputé_zero_off"
    df.loc[on_missing, "vibration_source"] = "imputé_rf_on"
    df["vibration_confidence"] = df["vibration_source"]
    return df


def fill_sensor_gaps_live(df, fill_medians):
    """Forward-fill par machine (simplification assumee du Random Forest ISO 4 du batch,
    cf. docstring du module) ; repli sur la mediane historique gelee si aucune valeur
    anterieure n'est disponible dans la fenetre live (ex. tout debut de fenetre)."""
    cols = fsg.FFILL_TARGETS + [fsg.RF_TARGET]
    for col in cols:
        filled = df.groupby("asset_name")[col].ffill()
        medians = df["asset_name"].map(lambda a: fill_medians.get(a, {}).get(col, np.nan))
        df[f"{col}_filled"] = filled.fillna(medians)
    df["sensor_gap_source"] = np.where(df[cols].isna().any(axis=1), "comblé", "mesuré")
    return df


def add_zscore_per_machine_live(df, zscore_stats):
    """Variante figee de feature_engineering.add_zscore_per_machine : utilise la
    moyenne/ecart-type appris a l'entrainement (realtime/artifacts.py) au lieu de les
    recalculer sur le petit buffer live."""
    for col in fe.HEALTH_VARS:
        mean = df["asset_name"].map(lambda a: zscore_stats.get(a, {}).get(col, {}).get("mean"))
        std = df["asset_name"].map(lambda a: zscore_stats.get(a, {}).get(col, {}).get("std"))
        df[f"{col}_zscore"] = (df[col] - mean) / std
    return df


def add_physicochemical_indices_live(df, vi_proxy_coefs):
    """Identique a feature_engineering.add_physicochemical_indices, sauf vi_proxy qui
    utilise la regression figee (pente/ordonnee) apprise a l'entrainement au lieu d'un
    polyfit recalcule sur le petit buffer live (cf. docstring du module)."""
    df["density_15C"] = df[fe.DENSITY_COL] / (1 - fe.ALPHA_THERMAL_EXPANSION * (df[fe.TEMP_COL] - 15))
    df["density_deviation_pct"] = (df["density_15C"] - fe.DENSITY_BASELINE) / fe.DENSITY_BASELINE * 100
    df["viscosity_temp_ratio"] = df[fe.VISC40_COL] / df[fe.KINVISC_COL]

    df["vi_proxy"] = np.nan
    for asset, coef in vi_proxy_coefs.items():
        mask = df["asset_name"] == asset
        pred = coef["slope"] * df.loc[mask, fe.TEMP_COL] + coef["intercept"]
        df.loc[mask, "vi_proxy"] = df.loc[mask, "viscosity_temp_ratio"] - pred
    return df


_SEVERITY_LABEL = {0: "Normal", 1: "Surveillance", 2: "Alarme"}


def compute_health_index_live(df, hi_bundles, regles_thresholds):
    """Reproduit health_index_comparison.build_final_health_index (health_index/health_state
    finaux, PCA T2/SPE + Isolation Forest geles) ET les 3 etats intermediaires
    (health_state_regles, health_state_pca, health_state_isoforest) - ces 3 colonnes sont
    des predicteurs categoriels one-hot-encodes par
    health_index_prediction_data.build_feature_columns (STATE_COLS), donc necessaires meme
    si seul health_index final est affiche au tableau de bord."""
    df["HI_regles"] = np.nan
    df["HI_pca_t2"] = np.nan
    df["HI_pca_spe"] = np.nan
    df["HI_isoforest"] = np.nan
    df["health_index"] = np.nan
    df["health_state"] = "Normal"
    df["health_state_regles"] = "Normal"
    df["health_state_pca"] = "Normal"
    df["health_state_isoforest"] = "Normal"

    for asset, col in ASSET_COLS.items():
        mask = df[col] == 1
        if not mask.any():
            continue
        sub = df.loc[mask]
        b = hi_bundles[asset]

        t2, spe = hpca.compute_t2_spe(sub, b["scaler_pca"], b["pca"])
        iso_score = hiso.compute_isoforest_score(sub, b["scaler_iso"], b["iso"])
        df.loc[mask, "HI_pca_t2"] = t2
        df.loc[mask, "HI_pca_spe"] = spe
        df.loc[mask, "HI_isoforest"] = iso_score

        severity_pca = np.maximum(t2 / b["t2_alarm"], spe / b["spe_alarm"])
        severity_iso = iso_score / b["iso_alarm"]
        combined_severity = np.maximum(severity_pca, severity_iso)
        df.loc[mask, "health_index"] = 1 / (1 + combined_severity)

        state = np.select(
            [t2 > b["t2_alarm"], spe > b["spe_alarm"], iso_score > b["iso_alarm"],
             t2 > b["t2_surv"], spe > b["spe_surv"], iso_score > b["iso_surv"]],
            [2, 2, 2, 1, 1, 1], default=0,
        )
        df.loc[mask, "health_state"] = [_SEVERITY_LABEL[s] for s in state]

        state_pca = np.select([t2 > b["t2_alarm"], spe > b["spe_alarm"], t2 > b["t2_surv"], spe > b["spe_surv"]],
                               [2, 2, 1, 1], default=0)
        df.loc[mask, "health_state_pca"] = [_SEVERITY_LABEL[s] for s in state_pca]

        state_iso = np.select([iso_score > b["iso_alarm"], iso_score > b["iso_surv"]], [2, 1], default=0)
        df.loc[mask, "health_state_isoforest"] = [_SEVERITY_LABEL[s] for s in state_iso]

        # Regles metier (health_index_baseline.py) - seuils Surveillance/Alarme geles a
        # l'entrainement (realtime/artifacts.py::build_regles_thresholds), distances
        # recalculees sur la fenetre live (health_index_baseline.compute_distances).
        distances = hib.compute_distances(sub)
        hi_regles = (1 - hib.WEIGHT * distances.sum(axis=1)).clip(0, 1)
        df.loc[mask, "HI_regles"] = hi_regles
        rt = regles_thresholds[asset]
        state_regles = np.select([hi_regles <= rt["alarm"], hi_regles <= rt["surv"]], [2, 1], default=0)
        df.loc[mask, "health_state_regles"] = [_SEVERITY_LABEL[s] for s in state_regles]

    return df


def build_live_snapshot(raw_wide_df, artifacts):
    """Pipeline complet, sur la fenetre live recuperee par realtime/api_client.py.
    Retourne le dataframe entierement enrichi (une ligne par mesure) - l'appelant prend
    la derniere ligne par machine comme etat "maintenant"."""
    df, _report = cid.clean(raw_wide_df.copy())

    df = fill_vibration_live(df, artifacts["vibration_imputer"])
    df = fill_sensor_gaps_live(df, artifacts["fill_medians"])

    df = fe.add_calendar_features(df)
    df = fe.add_session_features(df)
    df = add_physicochemical_indices_live(df, artifacts["vi_proxy_coefs"])
    df = fe.add_trend_features(df)
    df = fe.add_lag_features(df)
    df = fe.add_ewma_features(df)
    df = fe.add_domain_indices(df)
    df = add_zscore_per_machine_live(df, artifacts["zscore_stats"])
    df = fe.add_threshold_flags(df)
    df = fe.add_vibration_trust_features(df)
    df = fe.add_sensor_gap_trust_feature(df)

    df["asset_name_label"] = df["asset_name"]  # add_categorical_encoding consomme asset_name (one-hot)
    df = fe.add_categorical_encoding(df)

    df = compute_health_index_live(df, artifacts["health_index"], artifacts["regles_thresholds"])
    return df


if __name__ == "__main__":
    from api_client import fetch_recent_wide

    wide, fetched_at = fetch_recent_wide()
    artifacts = load_artifacts()
    snapshot = build_live_snapshot(wide, artifacts)
    latest = snapshot.sort_values("created_at").groupby("asset_name_label").tail(1)
    print(latest[["asset_name_label", "created_at", "health_index", "health_state"]])
