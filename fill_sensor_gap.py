"""
Remplissage du trou synchrone de 994 mesures affectant desormais 11 variables
sante simultanement, en appliquant la methode hybride retenue suite a la
comparaison de 7 methodes (voir explication_imputation_capteurs.md /
impute_sensor_gap.py) :

  - Forward-fill (au niveau de la machine) pour les 10 variables ou cette
    methode simple etait quasi optimale (R2 ~0.75-1.0) :
    DC, Density, Dynamic Viscosity, Kinematic Viscosity, Oil H2O Saturation,
    Oil H2O ppm, Oil Temperature, Viscosity at 40°C, Oil Conductivity,
    Oil Conductivity_nSm
  - Random Forest pour ISO 4, seule variable ou forward-fill/interpolation
    echouaient (R2 ~0) : le modele s'appuie sur ISO 6, ISO 14,
    Oil Conductivity_nSm_filled, Oil Pressure et Oil System Vibration_filled,
    restes disponibles pendant ce trou.

Density et Kinematic Viscosity ont ete ajoutees suite a la Phase 0 (audit
physico-chimique, plan_refonte_pipeline.md) : elles partagent EXACTEMENT le
meme episode de 994 lignes manquantes que les 7 variables deja traitees (meme
masque de NaN verifie), donc le meme traitement forward-fill s'applique
naturellement. Oil Conductivity / Oil Conductivity_nSm ajoutees le 2026-09-01
suite a la decouverte d'une sentinelle -0.09999 non detectee par
clean_isense_data.py, touchant le meme evenement de coupure.

Entree : isense_oil_data_vibration_filled.csv
Sortie : isense_oil_data_sensors_filled.csv / .xlsx + rapport_remplissage_capteurs.md
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

INPUT_FILE = "isense_oil_data_vibration_filled.csv"

FFILL_TARGETS = [
    "DC", "Density", "Dynamic Viscosity", "Kinematic Viscosity", "Oil H2O Saturation",
    "Oil H2O ppm", "Oil Temperature", "Viscosity at 40°C",
    "Oil Conductivity", "Oil Conductivity_nSm",
]
# Oil Conductivity/Oil Conductivity_nSm ajoutees (2026-09-01) : la sentinelle -0.09999
# de cette variable, non standard, n'etait pas detectee par clean_isense_data.py -
# corrige, elle partage EXACTEMENT le meme masque de 994 lignes manquantes que les
# 8 variables deja traitees (meme evenement de coupure synchrone).
RF_TARGET = "ISO 4"
RF_PREDICTORS = ["ISO 6", "ISO 14", "Oil Conductivity_nSm_filled", "Oil Pressure", "Oil System Vibration_filled"]

RANDOM_STATE = 42


def forward_fill_column(df, col):
    filled = df.groupby("asset_name")[col].ffill()
    global_median = df[col].median()
    return filled.fillna(global_median)


def build_rf_predictor_matrix(df):
    X = df[RF_PREDICTORS].copy()
    X["hour_sin"] = np.sin(2 * np.pi * df["created_at"].dt.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * df["created_at"].dt.hour / 24)
    X = pd.get_dummies(X.join(df[["asset_name", "machine_state"]]), columns=["asset_name", "machine_state"])
    imputer = SimpleImputer(strategy="median")
    return pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)


def main():
    df = pd.read_csv(INPUT_FILE, parse_dates=["created_at"])
    df = df.sort_values(["asset_name", "created_at"]).reset_index(drop=True)

    report = ["# Rapport — Remplissage du trou capteur synchrone (forward-fill + Random Forest)\n"]

    # Flag partage : ces 9 variables sont NaN sur (quasiment) les memes lignes
    gap_mask = df[FFILL_TARGETS + [RF_TARGET]].isna().any(axis=1)
    df["sensor_gap_source"] = np.where(gap_mask, "comblé", "mesuré")
    report.append(f"- Lignes concernées par le trou synchrone (≥1 des 11 variables NaN) : "
                   f"{gap_mask.sum()} ({gap_mask.sum() / len(df):.2%})\n")

    report.append(f"## Forward-fill ({len(FFILL_TARGETS)} variables)\n")
    report.append("| Variable | NaN avant | NaN après |")
    report.append("|---|---|---|")
    for col in FFILL_TARGETS:
        n_before = df[col].isna().sum()
        df[f"{col}_filled"] = forward_fill_column(df, col)
        n_after = df[f"{col}_filled"].isna().sum()
        report.append(f"| {col} | {n_before} | {n_after} |")

    report.append("\n## Random Forest (ISO 4)\n")
    X = build_rf_predictor_matrix(df)
    known_mask = df[RF_TARGET].notna()
    n_before = (~known_mask).sum()

    model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X.loc[known_mask], df.loc[known_mask, RF_TARGET])
    predicted = model.predict(X.loc[~known_mask])

    df[f"{RF_TARGET}_filled"] = df[RF_TARGET]
    df.loc[~known_mask, f"{RF_TARGET}_filled"] = predicted
    n_after = df[f"{RF_TARGET}_filled"].isna().sum()
    report.append(f"| Variable | NaN avant | NaN après |")
    report.append(f"|---|---|---|")
    report.append(f"| {RF_TARGET} | {n_before} | {n_after} |")

    total_missing_before = df[FFILL_TARGETS + [RF_TARGET]].isna().sum().sum()
    total_missing_after = df[[f"{c}_filled" for c in FFILL_TARGETS] + [f"{RF_TARGET}_filled"]].isna().sum().sum()
    report.append(f"\n- Total NaN (11 variables) avant : {total_missing_before}")
    report.append(f"- Total NaN (11 variables) après : {total_missing_after}")

    report.append("\n## Colonnes ajoutées\n")
    for col in FFILL_TARGETS + [RF_TARGET]:
        report.append(f"- `{col}_filled`")
    report.append("- `sensor_gap_source` : `mesuré` ou `comblé` (partagé par les 11 variables, "
                   "puisqu'elles étaient manquantes sur les mêmes lignes)")

    df.to_csv("isense_oil_data_sensors_filled.csv", index=False, encoding="utf-8-sig")
    df.to_excel("isense_oil_data_sensors_filled.xlsx", index=False, engine="openpyxl")

    with open("rapport_remplissage_capteurs.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    print("\n".join(report))
    print("\nFichiers générés :")
    print("- isense_oil_data_sensors_filled.csv")
    print("- isense_oil_data_sensors_filled.xlsx")
    print("- rapport_remplissage_capteurs.md")


if __name__ == "__main__":
    main()
