"""
Entraine et persiste (joblib/json) les artefacts necessaires a l'inference temps reel
du health_index, aujourd'hui recalcules a la volee et jetes par les scripts batch du
projet (health_index_pca.py, health_index_isolation_forest.py,
health_index_prediction_baseline.py, fill_vibration.py). A executer une fois (ou pour
rafraichir les artefacts si les CSV historiques sont regeneres) AVANT de lancer le
dashboard (realtime/dashboard.py).

Modeles retenus pour la prediction t+n (cf. rapport_prediction_health_index_comparaison_finale.md,
decision utilisateur : seuls Random Forest / Persistance, jamais XGBoost/autoencodeur en
production) : le gagnant reel par (horizon, machine) est relu directement depuis
rapport_prediction_health_index_baseline_resultats.csv plutot que code en dur, pour
rester automatiquement coherent avec le rapport si celui-ci est regenere.

Sortie : realtime/artifacts/
  - health_index_model_{A,B}.joblib   (PCA + Isolation Forest + seuils, par machine)
  - zscore_stats.json                 (moyenne/ecart-type geles par machine x HEALTH_VAR)
  - vi_proxy_coefs.json               (pente/ordonnee geles de la regression vi_proxy~temperature, par machine)
  - regles_thresholds.json            (seuils Surveillance/Alarme de HI_regles, par machine)
  - fill_medians.json                 (mediane historique par machine, filet de secours du forward-fill live)
  - vibration_imputer.joblib          (Random Forest OFF->0/ON->RF, reentrainee)
  - rf_{horizon}_{A,B}.joblib         (un par combinaison ou Random Forest gagne)
  - manifest.json                     (modele a utiliser + metriques historiques, par horizon x machine)
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
sys.path.insert(0, PROJECT_DIR)

import feature_engineering as fe  # noqa: E402
import fill_sensor_gap as fsg  # noqa: E402
import fill_vibration as fv  # noqa: E402
import health_index_baseline as hib  # noqa: E402
import health_index_isolation_forest as hiso  # noqa: E402
import health_index_pca as hpca  # noqa: E402
import health_index_prediction_data as hpd  # noqa: E402

ASSET_COLS = {"Motosoufflante A": "asset_Motosoufflante A", "Motosoufflante B": "asset_Motosoufflante B"}
ASSET_KEY = {"Motosoufflante A": "A", "Motosoufflante B": "B"}
FLAG_COLS = ["flag_high_contamination", "flag_high_water", "flag_high_temperature", "flag_high_vibration"]
SURVEILLANCE_PCTL = 0.95
ALARME_PCTL = 0.99
RANDOM_STATE = 42

# Les seuls modeles autorises en production (decision utilisateur) - XGBoost et les 3
# variantes autoencodeur ne gagnent jamais dans le rapport final, on ne les entraine pas ici.
ALLOWED_MODELS = {"Persistance (référence)": "persistence", "Random Forest": "random_forest"}


def build_health_index_artifacts(df):
    """PCA (T2/SPE) + Isolation Forest + seuils, par machine - meme logique que
    health_index_pca.fit_pca_for_asset / health_index_isolation_forest.fit_isoforest_for_asset,
    persistee au lieu d'etre jetee."""
    bundles = {}
    for asset, col in ASSET_COLS.items():
        mask = df[col] == 1
        sub = df.loc[mask]
        healthy_mask = ~sub[FLAG_COLS].any(axis=1)

        scaler_pca, pca, k, cum_var = hpca.fit_pca_for_asset(sub, healthy_mask)
        t2_healthy, spe_healthy = hpca.compute_t2_spe(sub.loc[healthy_mask], scaler_pca, pca)
        t2_surv, t2_alarm = np.quantile(t2_healthy, [SURVEILLANCE_PCTL, ALARME_PCTL])
        spe_surv, spe_alarm = np.quantile(spe_healthy, [SURVEILLANCE_PCTL, ALARME_PCTL])

        scaler_iso, iso = hiso.fit_isoforest_for_asset(sub, healthy_mask)
        iso_healthy = hiso.compute_isoforest_score(sub.loc[healthy_mask], scaler_iso, iso)
        iso_surv, iso_alarm = np.quantile(iso_healthy, [SURVEILLANCE_PCTL, ALARME_PCTL])

        # Seuils sur health_index LUI-MEME (pas T2/SPE/IsoForest) - necessaires en temps reel
        # pour classer une valeur PREDITE a t+n, pour laquelle on n'a pas de mesures futures
        # permettant de recalculer T2/SPE/IsoForest. Meme formule de combinaison que
        # health_index_comparison.build_final_health_index, appliquee au sous-ensemble sain ;
        # meme convention de percentile que health_index_baseline.py (health_index bas = pire,
        # donc quantile 1-0.95/1-0.99 cote bas).
        severity_pca_healthy = np.maximum(t2_healthy / t2_alarm, spe_healthy / spe_alarm)
        severity_iso_healthy = iso_healthy / iso_alarm
        hi_healthy = 1 / (1 + np.maximum(severity_pca_healthy, severity_iso_healthy))
        hi_surv, hi_alarm = np.quantile(hi_healthy, [1 - SURVEILLANCE_PCTL, 1 - ALARME_PCTL])

        bundle = {
            "scaler_pca": scaler_pca, "pca": pca, "t2_surv": t2_surv, "t2_alarm": t2_alarm,
            "spe_surv": spe_surv, "spe_alarm": spe_alarm,
            "scaler_iso": scaler_iso, "iso": iso, "iso_surv": iso_surv, "iso_alarm": iso_alarm,
            "hi_surv": float(hi_surv), "hi_alarm": float(hi_alarm),
        }
        bundles[asset] = bundle
        path = os.path.join(ARTIFACTS_DIR, f"health_index_model_{ASSET_KEY[asset]}.joblib")
        joblib.dump(bundle, path)
        print(f"[health_index] {asset} -> {path} (k={k} composantes, "
              f"var. expliquee={cum_var[k-1]:.1%}, {int(healthy_mask.sum())} lignes saines, "
              f"seuils HI surveillance={hi_surv:.3f}/alarme={hi_alarm:.3f})")
    return bundles


def build_zscore_stats(df):
    """Moyenne/ecart-type geles par machine x HEALTH_VAR (feature_engineering.add_zscore_per_machine
    les recalcule a la volee sur tout le dataframe recu - correct en batch, faux sur un
    petit buffer temps reel, d'ou la necessite de les figer ici)."""
    stats = {}
    for asset, col in ASSET_COLS.items():
        mask = df[col] == 1
        sub = df.loc[mask]
        stats[asset] = {
            var: {"mean": float(sub[var].mean()), "std": float(sub[var].std())}
            for var in fe.HEALTH_VARS
        }
    path = os.path.join(ARTIFACTS_DIR, "zscore_stats.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[zscore] -> {path} ({len(fe.HEALTH_VARS)} variables x {len(stats)} machines)")
    return stats


def build_fill_medians(df_cleaned):
    """Mediane historique par machine, pour les variables du trou capteur synchrone
    (fill_sensor_gap.FFILL_TARGETS + RF_TARGET) - sert de filet de secours en temps reel
    quand le forward-fill (fenetre glissante recente) n'a aucune valeur anterieure a
    propager (ex. tout debut de fenetre, machine qui redemarre). Equivalent temps reel du
    `global_median` utilise par fill_sensor_gap.forward_fill_column en batch, mais calcule
    par machine plutot que globalement (les deux machines ont des profils physico-chimiques
    differents, cf. rapport_data_quality.md)."""
    cols = fsg.FFILL_TARGETS + [fsg.RF_TARGET]
    medians = {}
    for asset in df_cleaned["asset_name"].unique():
        sub = df_cleaned[df_cleaned["asset_name"] == asset]
        medians[asset] = {col: float(sub[col].median()) for col in cols}
    path = os.path.join(ARTIFACTS_DIR, "fill_medians.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(medians, f, indent=2, ensure_ascii=False)
    print(f"[fill_medians] -> {path} ({len(cols)} variables x {len(medians)} machines)")
    return medians


def build_physicochem_coefs(df_health_index):
    """Coefficients geles (pente, ordonnee) de la regression vi_proxy ~ Oil Temperature_filled,
    par machine (feature_engineering.add_physicochemical_indices la re-ajuste a chaque appel
    sur tout le dataframe recu - correct en batch avec des milliers de lignes, instable sur un
    petit buffer temps reel puisque vi_proxy est ensuite utilisee comme HEALTH_VAR ET comme
    variable PCA)."""
    coefs = {}
    for asset, col in ASSET_COLS.items():
        mask = df_health_index[col] == 1
        sub = df_health_index.loc[mask, ["viscosity_temp_ratio", fe.TEMP_COL]].dropna()
        slope, intercept = np.polyfit(sub[fe.TEMP_COL], sub["viscosity_temp_ratio"], 1)
        coefs[asset] = {"slope": float(slope), "intercept": float(intercept)}
    path = os.path.join(ARTIFACTS_DIR, "vi_proxy_coefs.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(coefs, f, indent=2, ensure_ascii=False)
    print(f"[vi_proxy_coefs] -> {path} {coefs}")
    return coefs


def build_regles_thresholds(df_health_index):
    """Seuils Surveillance/Alarme de HI_regles, par machine - meme convention que
    health_index_baseline.py::main() (percentiles 1-0.95/1-0.99 sur TOUTES les lignes de la
    machine, pas seulement le sous-ensemble sain). health_state_regles est l'un des 4
    predicteurs categoriels one-hot-encodes par health_index_prediction_data.build_feature_columns
    (STATE_COLS) - necessaire pour que l'inference live reproduise exactement le meme espace
    de features que celui utilise a l'entrainement des modeles Random Forest."""
    thresholds = {}
    for asset, col in ASSET_COLS.items():
        hi = df_health_index.loc[df_health_index[col] == 1, "HI_regles"]
        surv, alarm = hi.quantile(1 - SURVEILLANCE_PCTL), hi.quantile(1 - ALARME_PCTL)
        thresholds[asset] = {"surv": float(surv), "alarm": float(alarm)}
    path = os.path.join(ARTIFACTS_DIR, "regles_thresholds.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2, ensure_ascii=False)
    print(f"[regles_thresholds] -> {path} {thresholds}")
    return thresholds


def build_vibration_imputer(df_cleaned):
    """Reentraine le Random Forest de fill_vibration.py (etat ON, vibration manquante) sur
    tout l'historique nettoye, et persiste modele + imputer + ordre des colonnes - la regle
    OFF -> 0 reste appliquee directement au moment de l'inference (pas besoin de modele)."""
    on_mask = df_cleaned["machine_state"] == "ON"
    known_mask = df_cleaned[fv.TARGET].notna()

    X = df_cleaned[fv.RF_PREDICTORS].copy()
    X["hour_sin"] = np.sin(2 * np.pi * df_cleaned["created_at"].dt.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * df_cleaned["created_at"].dt.hour / 24)
    X = pd.get_dummies(X.join(df_cleaned[["asset_name"]]), columns=["asset_name"])

    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    train_idx = df_cleaned.index[on_mask & known_mask]
    rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_imp.loc[train_idx], df_cleaned.loc[train_idx, fv.TARGET])

    bundle = {"model": rf, "imputer": imputer, "predictors": fv.RF_PREDICTORS, "columns": list(X.columns)}
    path = os.path.join(ARTIFACTS_DIR, "vibration_imputer.joblib")
    joblib.dump(bundle, path)
    print(f"[vibration_imputer] -> {path} (entraine sur {len(train_idx)} lignes ON connues)")
    return bundle


def build_prediction_manifest_and_models(df_health_index):
    """Relit le classement reel Persistance vs Random Forest (rapport_prediction_health_index_baseline_resultats.csv)
    pour decider, par horizon x machine, quel modele est effectivement le plus precis - puis
    n'entraine (et ne persiste) un Random Forest QUE pour les combinaisons ou il gagne
    reellement. Ailleurs, la persistance est appliquee directement a l'inference (aucun
    artefact necessaire)."""
    results_path = os.path.join(PROJECT_DIR, "rapport_prediction_health_index_baseline_resultats.csv")
    results = pd.read_csv(results_path)
    results = results[results["model"].isin(ALLOWED_MODELS)]

    manifest = {}
    for horizon in hpd.HORIZONS:
        for asset in ASSET_COLS:
            sub = results[(results.horizon == horizon) & (results.asset == asset)]
            if sub.empty:
                continue
            best = sub.loc[sub["r2"].idxmax()]
            manifest[f"{horizon}|{asset}"] = {
                "horizon_steps": int(horizon),
                "horizon_label": hpd.HORIZON_LABELS[horizon],
                "asset": asset,
                "model": ALLOWED_MODELS[best["model"]],
                "mae": float(best["mae"]),
                "rmse": float(best["rmse"]),
                "r2": float(best["r2"]),
            }

    rf_needed = [(v["horizon_steps"], v["asset"]) for v in manifest.values() if v["model"] == "random_forest"]

    print(f"[manifest] {len(manifest)} combinaisons horizon x machine ; "
          f"Random Forest retenu pour {rf_needed}, persistance ailleurs.")

    for horizon, asset in rf_needed:
        X, y, meta = hpd.build_dataset_for_horizon(df_health_index, horizon)
        imputer = SimpleImputer(strategy="median")
        X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

        asset_mask = (meta[ASSET_COLS[asset]] == 1).values
        X_asset = X_imp.loc[asset_mask]
        y_asset = y.loc[asset_mask]
        delta = y_asset - X_asset["health_index"]

        model = RandomForestRegressor(n_estimators=300, max_depth=14, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X_asset, delta)

        bundle = {"model": model, "imputer": imputer, "columns": list(X.columns),
                  "horizon": horizon, "asset": asset}
        fname = f"rf_{horizon}_{ASSET_KEY[asset]}.joblib"
        path = os.path.join(ARTIFACTS_DIR, fname)
        joblib.dump(bundle, path)
        manifest[f"{horizon}|{asset}"]["artifact"] = fname
        print(f"[rf model] {hpd.HORIZON_LABELS[horizon]} / {asset} -> {path} "
              f"({len(X_asset)} lignes d'entrainement, toutes disponibles)")

    manifest_path = os.path.join(ARTIFACTS_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[manifest] -> {manifest_path}")
    return manifest


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    df_health_index = pd.read_csv(
        os.path.join(PROJECT_DIR, "isense_oil_data_health_index.csv"), parse_dates=["created_at"]
    )
    df_health_index = df_health_index.sort_values(["session_id", "created_at"]).reset_index(drop=True)

    df_cleaned = pd.read_csv(
        os.path.join(PROJECT_DIR, "isense_oil_data_cleaned.csv"), parse_dates=["created_at"]
    )
    df_cleaned = df_cleaned.sort_values(["asset_name", "created_at"]).reset_index(drop=True)

    build_health_index_artifacts(df_health_index)
    build_zscore_stats(df_health_index)
    build_physicochem_coefs(df_health_index)
    build_regles_thresholds(df_health_index)
    build_fill_medians(df_cleaned)
    build_vibration_imputer(df_cleaned)
    build_prediction_manifest_and_models(df_health_index)

    print("\nTermine. Artefacts dans", ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
