"""
Feature engineering pour le dataset i-SENSE (qualite huile de lubrification).

Entree : isense_oil_data_sensors_filled.csv (sortie de fill_sensor_gap.py, qui
         complete DC, Dynamic Viscosity, ISO 4, Oil H2O Saturation, Oil H2O ppm,
         Oil Temperature et Viscosity at 40°C via forward-fill/Random Forest -
         voir explication_remplissage_capteurs.md - en plus de
         Oil System Vibration_filled deja complete par fill_vibration.py)
Sortie : isense_oil_data_features.csv / .xlsx + rapport (Markdown)

Voir explication_feature_engineering.md pour le detail et la justification
de chaque famille de features avant d'executer ce script.
"""

import numpy as np
import pandas as pd

INPUT_FILE = "isense_oil_data_sensors_filled.csv"

# Colonnes completees (suffixe _filled) utilisees a la place des versions brutes
# encore trouees, cf. explication_remplissage_capteurs.md et
# explication_remplissage_vibration.md.
DC_COL = "DC_filled"
DENSITY_COL = "Density_filled"
DYNVISC_COL = "Dynamic Viscosity_filled"
KINVISC_COL = "Kinematic Viscosity_filled"
ISO4_COL = "ISO 4_filled"
H2OPPM_COL = "Oil H2O ppm_filled"
TEMP_COL = "Oil Temperature_filled"
VISC40_COL = "Viscosity at 40°C_filled"
VIBRATION_COL = "Oil System Vibration_filled"

ALPHA_THERMAL_EXPANSION = 0.00075  # ASTM D1298, /degC
DENSITY_BASELINE = 869.0  # kg/m3, ligne de base identifiee en Phase 0 (data_quality_isense.py)

# Variables "sante" jugees les plus pertinentes pour le suivi de degradation
# (tendance, lags, EWMA) - on evite de le faire sur TOUTES les colonnes pour
# limiter le nombre de features et le temps de calcul.
# Kinematic Viscosity_filled, density_15C et vi_proxy REINTEGREES suite a la
# Phase 0 (audit physico-chimique) : leur correlation avec Viscosity at 40C /
# Oil Temperature est une loi physique (mu=nu*rho, ASTM D1298), pas une
# redondance a eliminer - cf. plan_refonte_pipeline.md. density_15C et
# vi_proxy sont calcules par add_physicochemical_indices() avant d'etre
# utilisees ici.
# ISO 6 et ISO 14 restent en version brute : jamais de NaN sur ces colonnes
# (cf. explication_remplissage_capteurs.md, capteurs restes actifs). Oil Conductivity
# utilise desormais sa version nettoyee/convertie (nS/m, cf. clean_isense_data.py) :
# la colonne brute avait en realite une sentinelle non standard (-0.09999) restee
# non detectee jusqu'au 2026-09-01, cf. explication_remplissage_capteurs.md.
CONDUCTIVITY_COL = "Oil Conductivity_nSm_filled"
HEALTH_VARS = [
    ISO4_COL, "ISO 6", "ISO 14",
    VISC40_COL, DYNVISC_COL, KINVISC_COL,
    H2OPPM_COL, TEMP_COL,
    DC_COL, CONDUCTIVITY_COL,
    VIBRATION_COL,
    "density_15C", "vi_proxy",
]

VIBRATION_CONFIDENCE_ORDER = {
    "mesuré": 0, "imputé_zero_off": 1, "imputé_rf_on": 2,
}  # cf. fill_vibration.py - hybride OFF->0 (physique) / ON->Random Forest (predit)

LAGS = [1, 2, 3]
EWMA_SPAN = 18  # ~3h, plus reactif qu'un rolling mean classique

# Grades ISO VG nominaux (reference tribologie) pour l'ecart de viscosite
ISO_VG_GRADES = [22, 32, 46, 68, 100, 150]


def load_data():
    df = pd.read_csv(INPUT_FILE, parse_dates=["created_at"])
    df = df.sort_values(["asset_name", "session_id", "created_at"]).reset_index(drop=True)
    return df


def add_calendar_features(df):
    df["hour"] = df["created_at"].dt.hour
    df["day_of_week"] = df["created_at"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def add_session_features(df):
    g = df.groupby("session_id")
    session_start = g["created_at"].transform("min")
    df["time_in_session_h"] = (df["created_at"] - session_start).dt.total_seconds() / 3600
    df["measure_index_in_session"] = g.cumcount()
    return df


def add_physicochemical_indices(df):
    """Phase 4 : transforme Density_filled et Kinematic Viscosity_filled (reintegrees
    en Phase 1) en indices metier normalises, plutot que de les utiliser brutes.
    Doit s'executer AVANT add_trend_features/add_lag_features/add_ewma_features car
    density_15C et vi_proxy font partie de HEALTH_VARS."""
    # Densite normalisee a 15C (ASTM D1298) : isole la composition de l'huile de
    # l'effet temperature. Verifiee quasi-constante en Phase 0 (ecart-type 0.17 kg/m3).
    df["density_15C"] = df[DENSITY_COL] / (1 - ALPHA_THERMAL_EXPANSION * (df[TEMP_COL] - 15))
    df["density_deviation_pct"] = (df["density_15C"] - DENSITY_BASELINE) / DENSITY_BASELINE * 100

    # Ratio de sensibilite thermique de la viscosite
    df["viscosity_temp_ratio"] = df[VISC40_COL] / df[KINVISC_COL]

    # vi_proxy : residu du ratio apres retrait de l'effet temperature (regression
    # lineaire par machine, cf. Phase 0 - la correlation ratio~T differe entre A et B,
    # r=0.989 vs r=0.792, donc une regression par machine est necessaire)
    df["vi_proxy"] = np.nan
    for asset in df["asset_name"].unique():
        mask = df["asset_name"] == asset
        sub = df.loc[mask, ["viscosity_temp_ratio", TEMP_COL]].dropna()
        if len(sub) > 2:
            coef = np.polyfit(sub[TEMP_COL], sub["viscosity_temp_ratio"], 1)
            pred = np.polyval(coef, df.loc[mask, TEMP_COL])
            df.loc[mask, "vi_proxy"] = df.loc[mask, "viscosity_temp_ratio"] - pred

    return df


def _slope(y):
    y = y.dropna()
    if len(y) < 2:
        return np.nan
    x = np.arange(len(y))
    return np.polyfit(x, y.values, 1)[0]


def add_trend_features(df, window=18):
    grouped = df.groupby("session_id")[HEALTH_VARS]
    for col in HEALTH_VARS:
        df[f"{col}_diff1"] = df.groupby("session_id")[col].diff()
        df[f"{col}_slope_3h"] = (
            df.groupby("session_id")[col]
            .rolling(window=window, min_periods=6)
            .apply(_slope, raw=False)
            .reset_index(level=0, drop=True)
        )
    return df


def add_lag_features(df):
    for col in HEALTH_VARS:
        for lag in LAGS:
            df[f"{col}_lag{lag}"] = df.groupby("session_id")[col].shift(lag)
    return df


def add_ewma_features(df):
    for col in HEALTH_VARS:
        df[f"{col}_ewma"] = df.groupby("session_id")[col].transform(
            lambda s: s.ewm(span=EWMA_SPAN, min_periods=3).mean()
        )
    return df


def add_domain_indices(df):
    # Indice de contamination global (moyenne des 3 codes ISO 4406)
    df["contamination_index"] = df[[ISO4_COL, "ISO 6", "ISO 14"]].mean(axis=1)

    # Ecart relatif au grade ISO VG le plus proche (viscosite hors norme = usure/dilution)
    def nearest_grade_gap(v):
        if pd.isna(v):
            return np.nan
        nearest = min(ISO_VG_GRADES, key=lambda g: abs(g - v))
        return (v - nearest) / nearest

    df["viscosity_grade_gap"] = df[VISC40_COL].apply(nearest_grade_gap)

    # Interaction temperature x viscosite (relation physique connue : viscosite chute si T augmente)
    df["temp_viscosity_interaction"] = df[TEMP_COL] * df[VISC40_COL]

    return df


def add_zscore_per_machine(df):
    for col in HEALTH_VARS:
        df[f"{col}_zscore"] = df.groupby("asset_name")[col].transform(
            lambda s: (s - s.mean()) / s.std()
        )
    return df


def add_threshold_flags(df):
    # Seuils indicatifs (a ajuster selon les specifications reelles de l'huile utilisee)
    df["flag_high_contamination"] = df["contamination_index"] > 20
    df["flag_high_water"] = df[H2OPPM_COL] > 100
    df["flag_high_temperature"] = df[TEMP_COL] > 55
    df["flag_high_vibration"] = df[VIBRATION_COL] > 1.5
    return df


def add_vibration_trust_features(df):
    # Score numerique ordinal (0=mesure, 1=impute par 0 en OFF, 2=predit par RF en ON) :
    # permet a un modele de distinguer mesures reelles, remplacement physique et
    # prediction statistique sans filtrer les lignes.
    df["vibration_confidence_score"] = df["vibration_confidence"].map(VIBRATION_CONFIDENCE_ORDER)
    df = pd.get_dummies(df, columns=["vibration_source"], prefix="vibsource")
    return df


def add_sensor_gap_trust_feature(df):
    # DC_filled/ISO 4_filled/etc. melangent mesures et valeurs comblees (forward-fill/RF) sur
    # les memes 994 lignes (trou synchrone) - ce flag laisse un modele en aval distinguer les deux.
    df["sensor_gap_filled"] = (df["sensor_gap_source"] == "comblé").astype(int)
    return df


def add_categorical_encoding(df):
    df = pd.get_dummies(df, columns=["asset_name", "machine_state"], prefix=["asset", "state"])
    return df


def main():
    df = load_data()
    n_cols_start = df.shape[1]

    df = add_calendar_features(df)
    df = add_session_features(df)
    df = add_physicochemical_indices(df)
    df = add_trend_features(df)
    df = add_lag_features(df)
    df = add_ewma_features(df)
    df = add_domain_indices(df)
    df = add_zscore_per_machine(df)
    df = add_threshold_flags(df)
    df = add_vibration_trust_features(df)
    df = add_sensor_gap_trust_feature(df)
    df = add_categorical_encoding(df)

    n_cols_end = df.shape[1]
    print(f"Colonnes : {n_cols_start} -> {n_cols_end} ({n_cols_end - n_cols_start} features ajoutees)")

    df.to_csv("isense_oil_data_features.csv", index=False, encoding="utf-8-sig")
    df.to_excel("isense_oil_data_features.xlsx", index=False, engine="openpyxl")

    print("Fichiers generes :")
    print("- isense_oil_data_features.csv")
    print("- isense_oil_data_features.xlsx")


if __name__ == "__main__":
    main()
