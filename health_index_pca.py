"""
Phase 2 du plan de construction du Health Index (plan_construction_health_index.md) :
PCA par machine (statistiques T^2 de Hotelling et SPE/Q), entrainee sur un
sous-ensemble "sain" approxime (aucun flag_high_* actif), faute de periode de
reference officiellement confirmee par l'equipe i-SENSE (hypothese documentee,
cf. explication_health_index.md section 1 et plan_construction_health_index.md
section 0).

T^2 mesure l'ecart DANS le sous-espace principal retenu (regimes de variation
normaux, connus) ; SPE/Q mesure le residu HORS de ce sous-espace (ce que la PCA
lineaire ne peut pas expliquer). Cf. explication_health_index_biblio_comparaison.md
section 4 pour le lien avec l'erreur de reconstruction d'un autoencodeur (SPE est
l'equivalent lineaire exact de epsilon_REC).

DC/DC_filled exclue (anomalie de calibration non resolue sur Motosoufflante B,
cf. rapport_data_quality.md).

Entree : isense_oil_data_health_index.csv (sortie de health_index_baseline.py)
Sortie : colonnes HI_pca_t2, HI_pca_spe, health_state_pca
         + eda_output/health_index/02_pca_*.png
         + rapport_health_index_pca.md
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

INPUT_FILE = "isense_oil_data_health_index.csv"
OUTPUT_DIR = "eda_output/health_index"

ASSET_COLS = {"Motosoufflante A": "asset_Motosoufflante A", "Motosoufflante B": "asset_Motosoufflante B"}

# Variables "sante" retenues (explication_health_index.md, section 5) - DC exclue
PCA_VARS = [
    "ISO 4_filled", "ISO 6", "ISO 14",
    "Oil H2O ppm_filled", "Oil H2O Saturation_filled",
    "Viscosity at 40°C_filled", "Kinematic Viscosity_filled", "viscosity_grade_gap", "vi_proxy",
    "density_15C", "density_deviation_pct",
    "Oil System Vibration_filled",
    "Oil Temperature_filled",
]

FLAG_COLS = ["flag_high_contamination", "flag_high_water", "flag_high_temperature", "flag_high_vibration"]

EXPLAINED_VARIANCE_TARGET = 0.90
SURVEILLANCE_PCTL = 0.95
ALARME_PCTL = 0.99


def fit_pca_for_asset(sub, healthy_mask):
    healthy = sub.loc[healthy_mask, PCA_VARS]
    scaler = StandardScaler().fit(healthy)
    healthy_scaled = scaler.transform(healthy)

    pca_full = PCA().fit(healthy_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    k = int(np.searchsorted(cum_var, EXPLAINED_VARIANCE_TARGET) + 1)
    k = max(2, min(k, len(PCA_VARS) - 1))

    pca = PCA(n_components=k).fit(healthy_scaled)
    return scaler, pca, k, cum_var


def compute_t2_spe(sub, scaler, pca):
    X_scaled = scaler.transform(sub[PCA_VARS])
    scores = pca.transform(X_scaled)
    t2 = np.sum((scores ** 2) / pca.explained_variance_, axis=1)
    X_reconstructed = pca.inverse_transform(scores)
    spe = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)
    return t2, spe


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_FILE, parse_dates=["created_at"])

    n_missing = df[PCA_VARS].isna().sum().sum()
    assert n_missing == 0, f"NaN résiduels dans les variables PCA : {n_missing}"

    df["HI_pca_t2"] = np.nan
    df["HI_pca_spe"] = np.nan

    report = ["# Rapport — Health Index Phase 2 (PCA par machine, T² / SPE)\n"]
    report.append(f"- Variables utilisées ({len(PCA_VARS)}) : {', '.join(PCA_VARS)}")
    report.append(
        "- **Sous-ensemble sain (hypothèse documentée)** : lignes sans aucun `flag_high_*` actif. "
        "À confirmer avec l'équipe i-SENSE (cf. `plan_construction_health_index.md`, section 0).\n"
    )
    report.append("| Machine | Lignes saines (entraînement) | Composantes retenues (k) | Variance expliquée | Seuil Surveillance T² | Seuil Alarme T² | Seuil Surveillance SPE | Seuil Alarme SPE |")
    report.append("|---|---|---|---|---|---|---|---|")

    thresholds = {}
    for asset, col in ASSET_COLS.items():
        mask = df[col] == 1
        sub = df.loc[mask]
        healthy_mask = ~sub[FLAG_COLS].any(axis=1)

        scaler, pca, k, cum_var = fit_pca_for_asset(sub, healthy_mask)
        t2, spe = compute_t2_spe(sub, scaler, pca)
        df.loc[mask, "HI_pca_t2"] = t2
        df.loc[mask, "HI_pca_spe"] = spe

        t2_healthy, spe_healthy = compute_t2_spe(sub.loc[healthy_mask], scaler, pca)
        t2_surv, t2_alarm = np.quantile(t2_healthy, [SURVEILLANCE_PCTL, ALARME_PCTL])
        spe_surv, spe_alarm = np.quantile(spe_healthy, [SURVEILLANCE_PCTL, ALARME_PCTL])
        thresholds[asset] = (t2_surv, t2_alarm, spe_surv, spe_alarm)

        report.append(
            f"| {asset} | {int(healthy_mask.sum())}/{len(sub)} ({healthy_mask.mean():.1%}) | {k} | "
            f"{cum_var[k-1]:.1%} | {t2_surv:.2f} | {t2_alarm:.2f} | {spe_surv:.2f} | {spe_alarm:.2f} |"
        )

    df["health_state_pca"] = "Normal"
    for asset, col in ASSET_COLS.items():
        mask = df[col] == 1
        t2_surv, t2_alarm, spe_surv, spe_alarm = thresholds[asset]
        is_surv = mask & ((df["HI_pca_t2"] > t2_surv) | (df["HI_pca_spe"] > spe_surv))
        is_alarm = mask & ((df["HI_pca_t2"] > t2_alarm) | (df["HI_pca_spe"] > spe_alarm))
        df.loc[is_surv, "health_state_pca"] = "Surveillance"
        df.loc[is_alarm, "health_state_pca"] = "Alarme"

    report.append("\n## Répartition finale des états (PCA)\n")
    report.append("| État | Lignes |")
    report.append("|---|---|")
    for state, count in df["health_state_pca"].value_counts().items():
        report.append(f"| {state} | {count} |")

    # --- Comparaison avec la Phase 1 (regles metier) ---
    report.append("\n## Cohérence avec la Phase 1 (règles métier)\n")
    cross = pd.crosstab(df["health_state_regles"], df["health_state_pca"])
    report.append("| Règles \\ PCA | " + " | ".join(cross.columns) + " |")
    report.append("|---|" + "---|" * len(cross.columns))
    for idx, row in cross.iterrows():
        report.append(f"| {idx} | " + " | ".join(str(v) for v in row.values) + " |")

    # --- Graphique : T2 et SPE dans le temps, par machine ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    for i, (asset, col) in enumerate(ASSET_COLS.items()):
        mask = df[col] == 1
        sub = df.loc[mask].sort_values("created_at")
        t2_surv, t2_alarm, spe_surv, spe_alarm = thresholds[asset]

        axes[i, 0].plot(sub["created_at"], sub["HI_pca_t2"], linewidth=0.5, color="#2b6cb0")
        axes[i, 0].axhline(t2_surv, color="#dd6b20", linestyle="--", linewidth=1)
        axes[i, 0].axhline(t2_alarm, color="#e53e3e", linestyle="--", linewidth=1)
        axes[i, 0].set_title(f"T² — {asset}")

        axes[i, 1].plot(sub["created_at"], sub["HI_pca_spe"], linewidth=0.5, color="#2b6cb0")
        axes[i, 1].axhline(spe_surv, color="#dd6b20", linestyle="--", linewidth=1)
        axes[i, 1].axhline(spe_alarm, color="#e53e3e", linestyle="--", linewidth=1)
        axes[i, 1].set_title(f"SPE — {asset}")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/02_pca_timeline.png", dpi=120)
    plt.close(fig)

    df.to_csv("isense_oil_data_health_index.csv", index=False, encoding="utf-8-sig")

    with open("rapport_health_index_pca.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    print("\n".join(report))
    print(f"\nFichiers générés : isense_oil_data_health_index.csv (mis à jour), rapport_health_index_pca.md, {OUTPUT_DIR}/02_pca_timeline.png")


if __name__ == "__main__":
    main()
