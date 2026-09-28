"""
Phase 3 du plan de construction du Health Index (plan_construction_health_index.md) :
Isolation Forest par machine, en verification/complement de la PCA (Phase 2) -
robuste aux distributions non gaussiennes (ex. Oil H2O ppm, asymetrique, deja
observe dans l'EDA post-imputation), la ou la PCA suppose implicitement une
structure lineaire/gaussienne pour ses seuils bases sur les valeurs propres.

Meme perimetre que la Phase 2 : memes variables (PCA_VARS), meme sous-ensemble
"sain" approxime (aucun flag_high_* actif), entrainement separe par machine.

Entree : isense_oil_data_health_index.csv (sortie de health_index_pca.py)
Sortie : colonne HI_isoforest, health_state_isoforest
         + eda_output/health_index/03_isoforest_timeline.png
         + rapport_health_index_isoforest.md
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

INPUT_FILE = "isense_oil_data_health_index.csv"
OUTPUT_DIR = "eda_output/health_index"

ASSET_COLS = {"Motosoufflante A": "asset_Motosoufflante A", "Motosoufflante B": "asset_Motosoufflante B"}

PCA_VARS = [
    "ISO 4_filled", "ISO 6", "ISO 14",
    "Oil H2O ppm_filled", "Oil H2O Saturation_filled",
    "Viscosity at 40°C_filled", "Kinematic Viscosity_filled", "viscosity_grade_gap", "vi_proxy",
    "density_15C", "density_deviation_pct",
    "Oil System Vibration_filled",
    "Oil Temperature_filled",
]

FLAG_COLS = ["flag_high_contamination", "flag_high_water", "flag_high_temperature", "flag_high_vibration"]

RANDOM_STATE = 42
SURVEILLANCE_PCTL = 0.95
ALARME_PCTL = 0.99


def fit_isoforest_for_asset(sub, healthy_mask):
    """Ajuste scaler + IsolationForest sur le sous-ensemble sain d'une machine.
    Exposee separement pour etre reutilisee par health_index_comparison.py
    (test d'injection de defauts synthetiques)."""
    scaler = StandardScaler().fit(sub.loc[healthy_mask, PCA_VARS])
    X_healthy = scaler.transform(sub.loc[healthy_mask, PCA_VARS])
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(X_healthy)
    return scaler, iso


def compute_isoforest_score(sub, scaler, iso):
    X = scaler.transform(sub[PCA_VARS])
    return -iso.score_samples(X)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_FILE, parse_dates=["created_at"])

    df["HI_isoforest"] = np.nan

    report = ["# Rapport — Health Index Phase 3 (Isolation Forest par machine)\n"]
    report.append(f"- Variables utilisées : identiques à la Phase 2 ({len(PCA_VARS)} variables)")
    report.append("- Score = `-score_samples` (plus haut = plus anormal, même convention que T²/SPE)\n")
    report.append("| Machine | Lignes saines (entraînement) | Seuil Surveillance | Seuil Alarme |")
    report.append("|---|---|---|---|")

    thresholds = {}
    for asset, col in ASSET_COLS.items():
        mask = df[col] == 1
        sub = df.loc[mask]
        healthy_mask = ~sub[FLAG_COLS].any(axis=1)

        scaler, iso = fit_isoforest_for_asset(sub, healthy_mask)
        score_all = compute_isoforest_score(sub, scaler, iso)
        score_healthy = compute_isoforest_score(sub.loc[healthy_mask], scaler, iso)
        df.loc[mask, "HI_isoforest"] = score_all

        surv, alarm = np.quantile(score_healthy, [SURVEILLANCE_PCTL, ALARME_PCTL])
        thresholds[asset] = (surv, alarm)

        report.append(f"| {asset} | {int(healthy_mask.sum())}/{len(sub)} ({healthy_mask.mean():.1%}) | {surv:.3f} | {alarm:.3f} |")

    df["health_state_isoforest"] = "Normal"
    for asset, col in ASSET_COLS.items():
        mask = df[col] == 1
        surv, alarm = thresholds[asset]
        df.loc[mask & (df["HI_isoforest"] > surv), "health_state_isoforest"] = "Surveillance"
        df.loc[mask & (df["HI_isoforest"] > alarm), "health_state_isoforest"] = "Alarme"

    report.append("\n## Répartition finale des états (Isolation Forest)\n")
    report.append("| État | Lignes |")
    report.append("|---|---|")
    for state, count in df["health_state_isoforest"].value_counts().items():
        report.append(f"| {state} | {count} |")

    report.append("\n## Cohérence avec la PCA (Phase 2)\n")
    cross = pd.crosstab(df["health_state_pca"], df["health_state_isoforest"])
    report.append("| PCA \\ IsoForest | " + " | ".join(cross.columns) + " |")
    report.append("|---|" + "---|" * len(cross.columns))
    for idx, row in cross.iterrows():
        report.append(f"| {idx} | " + " | ".join(str(v) for v in row.values) + " |")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    for ax, (asset, col) in zip(axes, ASSET_COLS.items()):
        mask = df[col] == 1
        sub = df.loc[mask].sort_values("created_at")
        surv, alarm = thresholds[asset]
        ax.plot(sub["created_at"], sub["HI_isoforest"], linewidth=0.5, color="#2b6cb0")
        ax.axhline(surv, color="#dd6b20", linestyle="--", linewidth=1, label="Seuil Surveillance")
        ax.axhline(alarm, color="#e53e3e", linestyle="--", linewidth=1, label="Seuil Alarme")
        ax.set_title(f"Isolation Forest — {asset}")
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/03_isoforest_timeline.png", dpi=120)
    plt.close(fig)

    df.to_csv("isense_oil_data_health_index.csv", index=False, encoding="utf-8-sig")

    with open("rapport_health_index_isoforest.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    print("\n".join(report))
    print(f"\nFichiers générés : isense_oil_data_health_index.csv (mis à jour), rapport_health_index_isoforest.md, {OUTPUT_DIR}/03_isoforest_timeline.png")


if __name__ == "__main__":
    main()
