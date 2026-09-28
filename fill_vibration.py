"""
Remplissage HYBRIDE de `Oil System Vibration`, base sur `machine_state`
(derive de Oil Pressure, cf. data_quality_isense.py / clean_isense_data.py) :

  - machine_state == OFF & NaN -> remplace par 0. Decision du specialiste
    metier : la machine ne tourne pas, donc pas de vibration mecanique du
    systeme.
  - machine_state == ON  & NaN -> PREDIT par Random Forest. La machine tourne
    et vibre reellement ; un remplacement par 0 serait physiquement faux.
    Le choix de Random Forest s'appuie sur une comparaison chiffree de 8
    methodes (impute_vibration_on_state.py) : interpolation temporelle,
    Random Forest, XGBoost, KNN, moyenne/mediane globale et
    moyenne/mediane conditionnelle par machine. Random Forest l'emporte
    (RMSE=0.302, R²=0.438 sur 1525 points masques en etat ON) devant
    XGBoost (RMSE=0.307) et l'interpolation (RMSE=0.331) ; les 4 methodes
    statistiques simples sont nettement plus faibles (R² negatif ou proche
    de 0). Voir rapport_imputation_vibration_on_state.md et
    explication_remplissage_vibration.md pour le detail complet.

Historique : l'approche precedente (RF sur trous courts, abstention sur
trous longs) avait ete ecartee sur la base d'une analyse
(plan_refonte_pipeline.md, Partie 1.1) qui s'appuyait a tort sur
`machine_state` pour juger DIRECTEMENT de l'etat de vibration - un
raisonnement indirect qui melangeait deux mesures physiques distinctes.
L'approche hybride actuelle n'a pas ce defaut : machine_state sert
uniquement a separer OFF (remplacement physique par 0) de ON (prediction
statistique), jamais a estimer la valeur de vibration elle-meme.

Entree : isense_oil_data_cleaned.csv
Sortie : isense_oil_data_vibration_filled.csv / .xlsx + rapport_remplissage_vibration.md
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

INPUT_FILE = "isense_oil_data_cleaned.csv"
TARGET = "Oil System Vibration"

RF_PREDICTORS = [
    "DC", "Density", "Dynamic Viscosity", "Kinematic Viscosity",
    "ISO 4", "ISO 6", "ISO 14", "Oil Conductivity",
    "Oil H2O Saturation", "Oil H2O ppm", "Oil Pressure", "Oil Temperature",
    "Viscosity at 40°C",
]
RANDOM_STATE = 42


def build_feature_matrix(df):
    X = df[RF_PREDICTORS].copy()
    X["hour_sin"] = np.sin(2 * np.pi * df["created_at"].dt.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * df["created_at"].dt.hour / 24)
    X = pd.get_dummies(X.join(df[["asset_name"]]), columns=["asset_name"])
    imputer = SimpleImputer(strategy="median")
    return pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)


def predict_on_state_vibration(df):
    """Entraine un Random Forest sur toutes les lignes ON a vibration connue,
    predit sur toutes les lignes ON a vibration NaN. Retourne une Series
    alignee sur df.index, NaN en dehors du perimetre ON&NaN."""
    on_mask = df["machine_state"] == "ON"
    known_mask = df[TARGET].notna()
    train_idx = df.index[on_mask & known_mask]
    predict_idx = df.index[on_mask & ~known_mask]

    predictions = pd.Series(index=df.index, dtype=float)
    if len(predict_idx) == 0:
        return predictions

    X_full = build_feature_matrix(df)
    rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_full.loc[train_idx], df.loc[train_idx, TARGET])
    predictions.loc[predict_idx] = rf.predict(X_full.loc[predict_idx])
    return predictions


def compute_distance_to_known(df):
    """Distance (en nb. de mesures) au point mesure connu le plus proche, par
    machine - conservee a titre informatif/feature uniquement ; ne conditionne
    plus le remplissage (qui est desormais un simple remplacement par 0)."""
    distances = pd.Series(index=df.index, dtype=float)
    for asset in df["asset_name"].unique():
        sub = df[df["asset_name"] == asset].sort_values("created_at")
        known = sub[TARGET].notna().values
        n = len(known)
        forward = np.full(n, np.inf)
        last_known = -np.inf
        for i in range(n):
            if known[i]:
                last_known = i
            forward[i] = i - last_known
        backward = np.full(n, np.inf)
        next_known = np.inf
        for i in range(n - 1, -1, -1):
            if known[i]:
                next_known = i
            backward[i] = next_known - i
        dist = np.minimum(forward, backward)
        distances.loc[sub.index] = dist
    return distances


def main():
    df = pd.read_csv(INPUT_FILE, parse_dates=["created_at"])
    df = df.sort_values(["asset_name", "created_at"]).reset_index(drop=True)

    report = ["# Rapport — Remplissage hybride de `Oil System Vibration` (OFF→0 / ON→Random Forest)\n"]

    n_total = len(df)
    known_mask = df[TARGET].notna()
    n_missing_before = int((~known_mask).sum())
    off_missing_mask = (df["machine_state"] == "OFF") & (~known_mask)
    on_missing_mask = (df["machine_state"] == "ON") & (~known_mask)
    n_off_missing = int(off_missing_mask.sum())
    n_on_missing = int(on_missing_mask.sum())

    report.append(f"- Lignes totales : {n_total}")
    report.append(f"- Valeurs manquantes avant remplissage : {n_missing_before} "
                   f"({n_missing_before / n_total:.1%})")
    report.append(f"  - dont machine OFF : {n_off_missing} -> remplacées par 0")
    report.append(f"  - dont machine ON : {n_on_missing} -> prédites par Random Forest\n")
    report.append(
        "- **Méthode retenue** : approche hybride fondée sur `machine_state` (dérivé de "
        "`Oil Pressure`). À l'arrêt (OFF), le système ne vibre pas — remplacement direct par 0, "
        "décision du spécialiste métier. En marche (ON), la machine vibre réellement — un "
        "remplacement par 0 serait physiquement faux, la valeur est donc **prédite** par un "
        "modèle Random Forest, retenu après comparaison chiffrée de 8 méthodes "
        "(`impute_vibration_on_state.py` / `rapport_imputation_vibration_on_state.md`) : "
        "RF RMSE=0.302, R²=0.438, devant XGBoost (RMSE=0.307), KNN (RMSE=0.325), interpolation "
        "temporelle (RMSE=0.331), et loin devant les méthodes statistiques simples "
        "(moyenne/médiane globale ou conditionnelle, R² ≤ 0).\n"
    )

    df[f"{TARGET}_filled"] = df[TARGET].copy()
    on_predictions = predict_on_state_vibration(df)
    df.loc[on_missing_mask, f"{TARGET}_filled"] = on_predictions.loc[on_missing_mask]
    df.loc[off_missing_mask, f"{TARGET}_filled"] = 0.0

    df["vibration_source"] = "mesuré"
    df.loc[off_missing_mask, "vibration_source"] = "imputé_zero_off"
    df.loc[on_missing_mask, "vibration_source"] = "imputé_rf_on"
    # Conservee pour compatibilite avec feature_engineering.py (add_vibration_trust_features) ;
    # avec cette methode, confidence et source portent la meme information.
    df["vibration_confidence"] = df["vibration_source"]

    distance = compute_distance_to_known(df)
    df["vibration_gap_distance"] = distance.values  # informatif uniquement

    n_missing_after = int(df[f"{TARGET}_filled"].isna().sum())
    report.append(f"- Valeurs remplacées par 0 (OFF) : {n_off_missing}")
    report.append(f"- Valeurs prédites par Random Forest (ON) : {n_on_missing}")
    report.append(f"- Valeurs manquantes après remplissage : {n_missing_after}\n")

    report.append("## Colonnes ajoutées\n")
    report.append(f"- `{TARGET}_filled` : valeurs mesurées + 0 (OFF) + prédictions RF (ON) — colonne complète, sans NaN")
    report.append("- `vibration_source` : `mesuré`, `imputé_zero_off` ou `imputé_rf_on`")
    report.append("- `vibration_confidence` : identique à `vibration_source` (conservée pour compatibilité avec feature_engineering.py)")
    report.append("- `vibration_gap_distance` : distance (nb. de mesures) au point mesuré connu le plus proche (0 si mesuré) — informative uniquement, ne conditionne plus le remplissage")

    df.to_csv("isense_oil_data_vibration_filled.csv", index=False, encoding="utf-8-sig")
    df.to_excel("isense_oil_data_vibration_filled.xlsx", index=False, engine="openpyxl")

    with open("rapport_remplissage_vibration.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    print("\n".join(report))
    print("\nFichiers générés :")
    print("- isense_oil_data_vibration_filled.csv")
    print("- isense_oil_data_vibration_filled.xlsx")
    print("- rapport_remplissage_vibration.md")


if __name__ == "__main__":
    main()
