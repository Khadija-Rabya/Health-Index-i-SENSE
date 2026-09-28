"""
EDA post-imputation : reanalyse le dataset i-SENSE APRES le remplissage de
Oil System Vibration (fill_vibration.py) et du trou capteur synchrone
(fill_sensor_gap.py), pour verifier l'effet du remplissage et mettre a jour
les statistiques/corrélations/distributions de l'etape EDA initiale
(eda_isense.py, faite AVANT toute imputation).

Entree : isense_oil_data_sensors_filled.csv
Sortie : eda_output/post_imputation/*.png + rapport_eda_post_imputation.md

Ne fait AUCUNE nouvelle imputation ni feature engineering - purement analytique.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT_FILE = "isense_oil_data_sensors_filled.csv"
OUTPUT_DIR = "eda_output/post_imputation"

# Variables imputees : colonne brute (avec NaN d'origine) -> colonne completee
# Density et Kinematic Viscosity ajoutees suite a la Phase 0 (reintegrees dans le
# trou capteur synchrone, cf. plan_refonte_pipeline.md)
IMPUTED_VARS = {
    "DC": "DC_filled",
    "Density": "Density_filled",
    "Dynamic Viscosity": "Dynamic Viscosity_filled",
    "Kinematic Viscosity": "Kinematic Viscosity_filled",
    "ISO 4": "ISO 4_filled",
    "Oil H2O Saturation": "Oil H2O Saturation_filled",
    "Oil H2O ppm": "Oil H2O ppm_filled",
    "Oil Temperature": "Oil Temperature_filled",
    "Viscosity at 40°C": "Viscosity at 40°C_filled",
    "Oil System Vibration": "Oil System Vibration_filled",
}

# Variables "sante" post-imputation, pour la matrice de correlation et les stats
HEALTH_VARS_FILLED = [
    "DC_filled", "Dynamic Viscosity_filled", "Kinematic Viscosity_filled",
    "ISO 4_filled", "ISO 6", "ISO 14",
    "Oil Conductivity", "Oil H2O Saturation_filled", "Oil H2O ppm_filled",
    "Oil Pressure", "Oil Temperature_filled", "Viscosity at 40°C_filled",
    "Oil System Vibration_filled",
]

ASSETS = ["Motosoufflante A", "Motosoufflante B"]
COLORS = {"Motosoufflante A": "#2b6cb0", "Motosoufflante B": "#c53030"}


def load_data():
    return pd.read_csv(INPUT_FILE, parse_dates=["created_at"])


def report_line(report, text):
    print(text)
    report.append(text)


def plot_missing_before_after(df, report):
    before = {raw: df[raw].isna().mean() * 100 for raw in IMPUTED_VARS}
    after = {raw: df[filled].isna().mean() * 100 for raw, filled in IMPUTED_VARS.items()}

    labels = list(IMPUTED_VARS.keys())
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width / 2, [before[l] for l in labels], width, label="Avant remplissage", color="#c53030")
    ax.bar(x + width / 2, [after[l] for l in labels], width, label="Après remplissage", color="#2f855a")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("% de valeurs manquantes")
    ax.set_title("Taux de NaN avant vs après remplissage (vibration + trou capteur)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/01_missing_before_after.png", dpi=120)
    plt.close(fig)

    report_line(report, "\n### Valeurs manquantes avant / après\n")
    report_line(report, "| Variable | % NaN avant | % NaN après |")
    report_line(report, "|---|---|---|")
    for l in labels:
        report_line(report, f"| {l} | {before[l]:.2f}% | {after[l]:.2f}% |")


def plot_distribution_overlay(df, report):
    fig, axes = plt.subplots(4, 3, figsize=(15, 17))
    for ax, (raw, filled) in zip(axes.flat, IMPUTED_VARS.items()):
        ax.hist(df[raw].dropna(), bins=40, alpha=0.5, label="Mesuré (avant)", color="#4a5568", density=True)
        ax.hist(df[filled].dropna(), bins=40, alpha=0.5, label="Complet (après)", color="#2f855a", density=True)
        ax.set_title(raw, fontsize=9)
        ax.legend(fontsize=6)
    for ax in axes.flat[len(IMPUTED_VARS):]:
        ax.axis("off")
    fig.suptitle("Distribution avant (valeurs mesurées seules) vs après remplissage (densité normalisée)")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/02_distribution_overlay.png", dpi=120)
    plt.close(fig)
    report_line(report, "\n### Distributions avant / après\nVoir `02_distribution_overlay.png`.")


def plot_correlation_heatmap(df, report):
    corr = df[HEALTH_VARS_FILLED].corr()
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(HEALTH_VARS_FILLED)))
    ax.set_yticks(range(len(HEALTH_VARS_FILLED)))
    ax.set_xticklabels(HEALTH_VARS_FILLED, rotation=90, fontsize=7)
    ax.set_yticklabels(HEALTH_VARS_FILLED, fontsize=7)
    for i in range(len(HEALTH_VARS_FILLED)):
        for j in range(len(HEALTH_VARS_FILLED)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=5.5)
    fig.colorbar(im, ax=ax, label="Corrélation de Pearson")
    ax.set_title("Matrice de corrélation — dataset complété (vibration incluse)")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/03_correlation_heatmap_filled.png", dpi=120)
    plt.close(fig)

    report_line(report, "\n### Corrélations fortes post-imputation (|r| > 0.6)\n")
    strong = []
    for i in range(len(HEALTH_VARS_FILLED)):
        for j in range(i + 1, len(HEALTH_VARS_FILLED)):
            r = corr.iloc[i, j]
            if abs(r) > 0.6:
                strong.append((HEALTH_VARS_FILLED[i], HEALTH_VARS_FILLED[j], r))
    strong.sort(key=lambda t: -abs(t[2]))
    for a, b, r in strong:
        report_line(report, f"- {a} / {b} : r = {r:.3f}")


def plot_gap_zoom(df, report):
    """Zoome sur le vrai trou capteur (juillet 2026, Machine A) pour visualiser
    avant/apres remplissage sur la meme fenetre temporelle."""
    sub = df[(df["asset_name"] == "Motosoufflante A") &
              (df["created_at"] >= "2026-06-25") & (df["created_at"] <= "2026-07-28")].sort_values("created_at")

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(sub["created_at"], sub["DC"], ".", color="#c53030", markersize=3, label="DC (brut, NaN pendant le trou)")
    axes[0].set_title("Avant remplissage — DC brut (trou visible)")
    axes[0].legend()

    axes[1].plot(sub["created_at"], sub["DC_filled"], ".", color="#2f855a", markersize=3, label="DC_filled")
    axes[1].set_title("Après remplissage — DC_filled (trou comblé par forward-fill)")
    axes[1].legend()

    fig.suptitle("Zoom sur le trou capteur synchrone — Motosoufflante A (juillet 2026)")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/04_gap_zoom_sensor.png", dpi=120)
    plt.close(fig)
    report_line(report, "\n### Zoom trou capteur (avant/après)\nVoir `04_gap_zoom_sensor.png`.")


def descriptive_stats_table(df, report):
    report_line(report, "\n### Statistiques descriptives post-imputation (colonnes complétées)\n")
    report_line(report, "| Variable | Valides A | Min A | Max A | Moy. A | Valides B | Min B | Max B | Moy. B |")
    report_line(report, "|---|---|---|---|---|---|---|---|---|")
    for col in HEALTH_VARS_FILLED:
        cells = []
        for asset in ASSETS:
            s = df.loc[df["asset_name"] == asset, col].dropna()
            cells += [len(s), f"{s.min():.2f}", f"{s.max():.2f}", f"{s.mean():.2f}"]
        report_line(
            report,
            f"| {col} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} "
            f"| {cells[4]} | {cells[5]} | {cells[6]} | {cells[7]} |",
        )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()

    report = ["# Rapport EDA post-imputation — dataset i-SENSE\n"]
    report_line(report, f"- Lignes analysées : {len(df)}")
    report_line(report, f"- Dataset : {INPUT_FILE} (vibration + trou capteur remplis)")

    plot_missing_before_after(df, report)
    plot_distribution_overlay(df, report)
    plot_correlation_heatmap(df, report)
    plot_gap_zoom(df, report)
    descriptive_stats_table(df, report)

    with open("rapport_eda_post_imputation.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    print(f"\nFichiers générés :\n- {OUTPUT_DIR}/ (4 graphiques)\n- rapport_eda_post_imputation.md")


if __name__ == "__main__":
    main()
