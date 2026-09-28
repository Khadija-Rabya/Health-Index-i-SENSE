"""
Analyse exploratoire (EDA) du dataset i-SENSE nettoye.

Etape distincte du nettoyage (clean_isense_data.py) et du feature engineering
(feature_engineering.py) : ce script n'ajoute et ne modifie AUCUNE colonne,
il se contente d'explorer et de visualiser les donnees deja propres.

Entree : isense_oil_data_cleaned.csv
Sortie : dossier eda_output/ (graphiques PNG) + rapport_eda.md (chiffres cles)

Voir explication_eda.md pour le detail de chaque analyse et sa justification.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT_FILE = "isense_oil_data_cleaned.csv"
OUTPUT_DIR = "eda_output"

HEALTH_VARS = [
    "DC", "Density", "Dynamic Viscosity", "Kinematic Viscosity",
    "ISO 4", "ISO 6", "ISO 14",
    "Oil Conductivity", "Oil H2O Saturation", "Oil H2O ppm",
    "Oil Pressure", "Oil Temperature", "Viscosity at 40°C",
]  # Kinematic Viscosity reintegree suite a la Phase 0 (audit physico-chimique) :
# sa correlation avec Viscosity at 40C est une loi physique, pas une redondance
# a eliminer - cf. plan_refonte_pipeline.md.

ALPHA_THERMAL_EXPANSION = 0.00075  # ASTM D1298, /degC

ASSETS = ["Motosoufflante A", "Motosoufflante B"]
COLORS = {"Motosoufflante A": "#2b6cb0", "Motosoufflante B": "#c53030"}

# Unites, pour reproduire le tableau de l'audit PDF (section 1.5)
UNITS = {
    "DC": "-",
    "Density": "kg/m³",
    "Dynamic Viscosity": "cP",
    "ISO 4": "-",
    "ISO 6": "-",
    "ISO 14": "-",
    "Oil Conductivity": "nS/m",
    "Oil H2O Saturation": "%",
    "Oil H2O ppm": "ppm",
    "Oil Pressure": "Bar",
    "Oil Temperature": "°C",
    "Viscosity at 40°C": "cSt",
    "Kinematic Viscosity": "cSt",
}


def load_data():
    df = pd.read_csv(INPUT_FILE, parse_dates=["created_at"])
    return df


def report_line(report, text):
    print(text)
    report.append(text)


def plot_missing_data(df, report):
    missing = df[HEALTH_VARS + ["Oil System Vibration"]].isna().mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    missing.plot(kind="barh", ax=ax, color="#718096")
    ax.set_xlabel("% de valeurs manquantes")
    ax.set_title("Taux de valeurs manquantes par variable (post-nettoyage)")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/01_missing_data.png", dpi=120)
    plt.close(fig)
    report_line(report, "\n### Valeurs manquantes")
    report_line(report, "Voir `eda_output/01_missing_data.png`.")
    for var, pct in missing.items():
        report_line(report, f"- {var} : {pct:.2f} %")


def plot_distributions(df, report):
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    for ax, var in zip(axes.flat, HEALTH_VARS):
        for asset in ASSETS:
            data = df.loc[df["asset_name"] == asset, var].dropna()
            ax.hist(data, bins=40, alpha=0.5, label=asset, color=COLORS[asset])
        ax.set_title(var)
        ax.legend(fontsize=7)
    fig.suptitle("Distributions par variable et par machine")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/02_distributions.png", dpi=120)
    plt.close(fig)
    report_line(report, "\n### Distributions")
    report_line(report, "Voir `eda_output/02_distributions.png`.")


def plot_boxplots(df, report):
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    for ax, var in zip(axes.flat, HEALTH_VARS):
        data = [df.loc[df["asset_name"] == a, var].dropna() for a in ASSETS]
        ax.boxplot(data, tick_labels=["A", "B"])
        ax.set_title(var)
    fig.suptitle("Box plots comparatifs Machine A vs B")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/03_boxplots.png", dpi=120)
    plt.close(fig)
    report_line(report, "\n### Box plots comparatifs")
    report_line(report, "Voir `eda_output/03_boxplots.png`.")


def plot_correlation_heatmap(df, report):
    corr = df[HEALTH_VARS].corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(HEALTH_VARS)))
    ax.set_yticks(range(len(HEALTH_VARS)))
    ax.set_xticklabels(HEALTH_VARS, rotation=90, fontsize=8)
    ax.set_yticklabels(HEALTH_VARS, fontsize=8)
    for i in range(len(HEALTH_VARS)):
        for j in range(len(HEALTH_VARS)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="Corrélation de Pearson")
    ax.set_title("Matrice de corrélation des variables santé")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/04_correlation_heatmap.png", dpi=120)
    plt.close(fig)

    report_line(report, "\n### Corrélations fortes (|r| > 0.6, hors diagonale)")
    strong = []
    for i in range(len(HEALTH_VARS)):
        for j in range(i + 1, len(HEALTH_VARS)):
            r = corr.iloc[i, j]
            if abs(r) > 0.6:
                strong.append((HEALTH_VARS[i], HEALTH_VARS[j], r))
    strong.sort(key=lambda t: -abs(t[2]))
    for a, b, r in strong:
        report_line(report, f"- {a} / {b} : r = {r:.3f}")
    if not strong:
        report_line(report, "- Aucune paire avec |r| > 0.6.")


def plot_timeseries_overview(df, report):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    for ax, var in zip(axes, ["Viscosity at 40°C", "Oil Temperature", "contamination_index"]):
        if var == "contamination_index":
            df["_contam_tmp"] = df[["ISO 4", "ISO 6", "ISO 14"]].mean(axis=1)
            var_series = "_contam_tmp"
            title = "Indice de contamination (moyenne ISO 4/6/14)"
        else:
            var_series = var
            title = var
        for asset in ASSETS:
            sub = df[df["asset_name"] == asset].sort_values("created_at")
            ax.plot(sub["created_at"], sub[var_series], label=asset, color=COLORS[asset], linewidth=0.6)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/05_timeseries_overview.png", dpi=120)
    plt.close(fig)
    if "_contam_tmp" in df.columns:
        df.drop(columns=["_contam_tmp"], inplace=True)
    report_line(report, "\n### Vue temporelle globale")
    report_line(report, "Voir `eda_output/05_timeseries_overview.png`.")


def plot_machine_state(df, report):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, asset in zip(axes, ASSETS):
        counts = df.loc[df["asset_name"] == asset, "machine_state"].value_counts()
        ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", colors=["#2f855a", "#a0aec0"])
        ax.set_title(asset)
    fig.suptitle("Répartition du temps ON vs OFF/veille")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/06_machine_state.png", dpi=120)
    plt.close(fig)
    report_line(report, "\n### État machine (ON/OFF)")
    report_line(report, "Voir `eda_output/06_machine_state.png`.")


def plot_session_durations(df, report):
    session_stats = df.groupby("session_id")["created_at"].agg(["min", "max", "count"])
    session_stats["duration_h"] = (session_stats["max"] - session_stats["min"]).dt.total_seconds() / 3600

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(session_stats["duration_h"], bins=40, color="#6b46c1")
    ax.set_xlabel("Durée de session (heures)")
    ax.set_ylabel("Nombre de sessions")
    ax.set_title(f"Distribution de la durée des {len(session_stats)} sessions continues")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/07_session_durations.png", dpi=120)
    plt.close(fig)

    report_line(report, "\n### Sessions continues")
    report_line(report, f"- Nombre total de sessions : {len(session_stats)}")
    report_line(report, f"- Durée médiane : {session_stats['duration_h'].median():.1f} h")
    report_line(report, f"- Durée max : {session_stats['duration_h'].max():.1f} h")
    report_line(report, f"- Sessions < 1h (probables artefacts / redémarrages courts) : "
                          f"{(session_stats['duration_h'] < 1).sum()}")


def plot_conditional_correlation(df, report):
    """Phase 3 : verifie si la correlation DC / Oil Pressure (r=0.94 dans l'EDA globale)
    est un vrai lien physico-chimique ou un artefact confondu par l'etat ON/OFF de la
    machine (cf. explication_etat_systeme_vs_huile.md)."""
    report_line(report, "\n### Corrélation conditionnelle DC / Oil Pressure (par état machine)")
    pairs_results = {}
    s_all = df[["DC", "Oil Pressure", "machine_state"]].dropna()
    r_global = s_all["DC"].corr(s_all["Oil Pressure"])
    pairs_results["Globale"] = r_global
    for state in ["ON", "OFF"]:
        sub = s_all[s_all["machine_state"] == state]
        pairs_results[f"{state} uniquement"] = sub["DC"].corr(sub["Oil Pressure"]) if len(sub) > 2 else float("nan")

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = list(pairs_results.keys())
    values = [pairs_results[k] for k in labels]
    colors = ["#718096", "#2f855a", "#c53030"]
    ax.bar(labels, values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("Corrélation de Pearson DC / Oil Pressure")
    ax.set_title("Corrélation DC/Oil Pressure : globale vs conditionnée sur machine_state")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/08_conditional_correlation.png", dpi=120)
    plt.close(fig)

    report_line(report, "Voir `eda_output/08_conditional_correlation.png`.")
    for label, r in pairs_results.items():
        report_line(report, f"- {label} : r = {r:.3f}")
    if abs(pairs_results["ON uniquement"]) < abs(r_global) * 0.6:
        report_line(
            report,
            "- **Constat : la corrélation s'effondre largement une fois conditionnée sur "
            "l'état ON** — elle était en grande partie confondue par l'état machine (ON/OFF), "
            "pas un vrai lien physico-chimique direct entre DC et la pression.",
        )
    else:
        report_line(
            report,
            "- **Constat : la corrélation persiste même conditionnée sur l'état ON** — ce "
            "n'est donc pas un simple artefact de confusion par l'état machine.",
        )


def plot_physicochemical_indices(df, report):
    """Phase 3 : suivi temporel des indices derives valides par la Phase 0
    (density_15C, proxy d'indice de viscosite) - calcules ici sur le dataset
    nettoye (avant imputation), donc encore troues sur les 994 lignes du trou
    capteur synchrone."""
    report_line(report, "\n### Indices physico-chimiques dérivés (density_15C, proxy VI)")
    s = df[["asset_name", "created_at", "Density", "Oil Temperature",
            "Kinematic Viscosity", "Viscosity at 40°C"]].copy()
    s["density_15C"] = s["Density"] / (1 - ALPHA_THERMAL_EXPANSION * (s["Oil Temperature"] - 15))
    s["vi_proxy"] = s["Viscosity at 40°C"] / s["Kinematic Viscosity"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    for asset in ASSETS:
        sub = s[s["asset_name"] == asset].sort_values("created_at")
        axes[0].plot(sub["created_at"], sub["density_15C"], ".", markersize=1.5,
                      color=COLORS[asset], label=asset, alpha=0.5)
        axes[1].plot(sub["created_at"], sub["vi_proxy"], ".", markersize=1.5,
                      color=COLORS[asset], label=asset, alpha=0.5)
    axes[0].set_title("density_15C — densité normalisée à 15°C (ASTM D1298)")
    axes[0].set_ylabel("kg/m³")
    axes[0].legend(fontsize=8)
    axes[1].set_title("vi_proxy — proxy d'indice de viscosité (Viscosity@40°C / Kinematic Viscosity)")
    axes[1].set_ylabel("ratio")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/09_physicochemical_indices.png", dpi=120)
    plt.close(fig)

    report_line(report, "Voir `eda_output/09_physicochemical_indices.png`.")
    for asset in ASSETS:
        sub = s[s["asset_name"] == asset]
        d = sub["density_15C"].dropna()
        v = sub["vi_proxy"].dropna()
        report_line(
            report,
            f"- {asset} : density_15C moy={d.mean():.2f} kg/m³ (écart-type {d.std():.3f}) ; "
            f"vi_proxy moy={v.mean():.3f} (écart-type {v.std():.3f})",
        )


def descriptive_stats_table(df, report):
    report_line(report, "\n### Statistiques descriptives par machine (post-nettoyage)")
    report_line(
        report,
        "| Variable | Unité | Valides A | Min A | Max A | Moy. A | Valides B | Min B | Max B | Moy. B |",
    )
    report_line(
        report,
        "|---|---|---|---|---|---|---|---|---|---|",
    )
    for var in HEALTH_VARS:
        cells = []
        for asset in ASSETS:
            s = df.loc[df["asset_name"] == asset, var].dropna()
            cells += [f"{len(s)}", f"{s.min():.2f}", f"{s.max():.2f}", f"{s.mean():.2f}"]
        report_line(
            report,
            f"| {var} | {UNITS[var]} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} "
            f"| {cells[4]} | {cells[5]} | {cells[6]} | {cells[7]} |",
        )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()

    report = ["# Rapport EDA — dataset i-SENSE nettoyé\n"]
    report_line(report, f"- Lignes analysées : {len(df)}")
    report_line(report, f"- Période : {df['created_at'].min()} → {df['created_at'].max()}")

    descriptive_stats_table(df, report)
    plot_missing_data(df, report)
    plot_distributions(df, report)
    plot_boxplots(df, report)
    plot_correlation_heatmap(df, report)
    plot_timeseries_overview(df, report)
    plot_machine_state(df, report)
    plot_session_durations(df, report)
    plot_conditional_correlation(df, report)
    plot_physicochemical_indices(df, report)

    with open("rapport_eda.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    print("\nFichiers generes :")
    print(f"- {OUTPUT_DIR}/ (9 graphiques PNG)")
    print("- rapport_eda.md")


if __name__ == "__main__":
    main()
