"""
Comparaison finale des 5 modeles ENTRAINES de prediction de health_index a t+n
(2 baselines : Random Forest, XGBoost ; 3 a base d'autoencodeur : A, B, C), sur
les 3 horizons communs (10 min, 3h, 24h - les seuls testes par les 3 modeles a
base d'autoencodeur, cf. health_index_prediction_autoencoder.py).

La reference de persistance (calculee dans les rapports baseline/autoencodeur
comme garde-fou methodologique - cf. explication_prediction_health_index.md,
section 3, decouverte qui a motive la formulation en delta) est deliberement
EXCLUE de cette comparaison finale : seuls les modeles reellement entraines
sont compares entre eux ici.

Entree : rapport_prediction_health_index_baseline_resultats.csv
         rapport_prediction_health_index_autoencoder_resultats.csv
Sortie : rapport_prediction_health_index_comparaison_finale.md
         + eda_output/health_index_prediction/03_comparaison_finale.png
         + eda_output/health_index_prediction/04_meilleur_modele.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = "eda_output/health_index_prediction"
COMMON_HORIZONS = [1, 18, 144]
HORIZON_LABELS = {1: "10 min", 18: "3h", 144: "24h"}
ASSETS = ["Motosoufflante A", "Motosoufflante B"]

MODEL_ORDER = [
    "Random Forest", "XGBoost",
    "A. Autoencodeur+RF (latent)", "B. LSTM Encoder-Decoder", "C. Débruiteur + fine-tuning",
]
COLORS = {
    "Random Forest": "#2b6cb0", "XGBoost": "#dd6b20",
    "A. Autoencodeur+RF (latent)": "#38a169", "B. LSTM Encoder-Decoder": "#805ad5",
    "C. Débruiteur + fine-tuning": "#e53e3e",
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    baseline = pd.read_csv("rapport_prediction_health_index_baseline_resultats.csv")
    autoenc = pd.read_csv("rapport_prediction_health_index_autoencoder_resultats.csv")

    all_results = pd.concat([baseline, autoenc], ignore_index=True)
    all_results = all_results[all_results["horizon"].isin(COMMON_HORIZONS)].copy()
    all_results = all_results[all_results["model"] != "Persistance (référence)"].copy()
    all_results = all_results.drop_duplicates(subset=["horizon", "asset", "model"])

    report = [
        "# Comparaison finale — 5 modèles de prédiction de `health_index` à t+n\n",
        "Comparaison des 5 modèles entraînés (Random Forest, XGBoost, 3 architectures "
        "autoencodeur) sur les 3 horizons communs (10 min, 3h, 24h). La référence de persistance "
        "n'est volontairement pas incluse ici — voir `explication_prediction_health_index.md` "
        "pour la découverte méthodologique qui a motivé la formulation en delta. Détail complet "
        "des 6 horizons pour les 2 baselines dans `rapport_prediction_health_index_baseline.md`.\n",
    ]

    # --- Tableau complet ---
    report.append("| Horizon | Machine | Modèle | MAE | RMSE | R² |")
    report.append("|---|---|---|---|---|---|")
    for h in COMMON_HORIZONS:
        for asset in ASSETS:
            sub = all_results[(all_results.horizon == h) & (all_results.asset == asset)]
            sub = sub.set_index("model").reindex(MODEL_ORDER).reset_index()
            best_r2 = sub["r2"].max()
            for _, row in sub.iterrows():
                marker = " 🏆" if row["r2"] == best_r2 else ""
                report.append(
                    f"| {HORIZON_LABELS[h]} | {asset} | {row['model']}{marker} | {row['mae']:.4f} | "
                    f"{row['rmse']:.4f} | {row['r2']:.4f} |"
                )

    # --- Meilleur modele par (horizon, machine) ---
    report.append("\n## Meilleur modèle par horizon et par machine\n")
    report.append("| Horizon | Machine | Meilleur modèle | R² |")
    report.append("|---|---|---|---|")
    best_counts = {}
    for h in COMMON_HORIZONS:
        for asset in ASSETS:
            sub = all_results[(all_results.horizon == h) & (all_results.asset == asset)]
            best_row = sub.loc[sub["r2"].idxmax()]
            report.append(f"| {HORIZON_LABELS[h]} | {asset} | {best_row['model']} | {best_row['r2']:.4f} |")
            best_counts[best_row["model"]] = best_counts.get(best_row["model"], 0) + 1

    # --- Classement global (moyenne des rangs de R2, plus robuste que la moyenne brute de R2
    #     vu les valeurs tres negatives a certains horizons qui ecraseraient la moyenne) ---
    report.append("\n## Classement global (rang moyen de R², sur les 6 combinaisons horizon×machine)\n")
    rank_table = []
    for (h, asset), group in all_results.groupby(["horizon", "asset"]):
        ranked = group.sort_values("r2", ascending=False).reset_index(drop=True)
        for rank, row in ranked.iterrows():
            rank_table.append((row["model"], rank + 1))
    rank_df = pd.DataFrame(rank_table, columns=["model", "rank"])
    mean_ranks = rank_df.groupby("model")["rank"].mean().reindex(MODEL_ORDER).sort_values()

    report.append("| Rang | Modèle | Rang moyen | Nb. fois meilleur (/6) |")
    report.append("|---|---|---|---|")
    for i, (model, rank) in enumerate(mean_ranks.items(), 1):
        report.append(f"| {i} | {model} | {rank:.2f} | {best_counts.get(model, 0)} |")

    winner = mean_ranks.index[0]
    report.append(f"\n**Meilleur choix global : {winner}** (rang moyen le plus bas = le plus "
                   f"souvent proche du sommet du classement, {best_counts.get(winner, 0)}/6 fois "
                   "meilleur modèle sur les combinaisons horizon×machine testées).")

    # --- Graphique 1 : R2 par horizon/machine, tous modeles ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, asset in zip(axes, ASSETS):
        sub = all_results[all_results.asset == asset]
        x = np.arange(len(COMMON_HORIZONS))
        width = 0.13
        for i, model in enumerate(MODEL_ORDER):
            vals = [sub[(sub.horizon == h) & (sub.model == model)]["r2"].values[0]
                    if len(sub[(sub.horizon == h) & (sub.model == model)]) else np.nan
                    for h in COMMON_HORIZONS]
            ax.bar(x + (i - len(MODEL_ORDER) / 2) * width, vals, width, label=model, color=COLORS[model])
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([HORIZON_LABELS[h] for h in COMMON_HORIZONS])
        ax.set_title(asset)
        ax.set_xlabel("Horizon")
    axes[0].set_ylabel("R² (test, découpage temporel)")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Comparaison des 5 modèles + persistance — R² par horizon et par machine")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/03_comparaison_finale.png", dpi=120)
    plt.close(fig)

    # --- Graphique 2 : rang moyen (classement global) ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors_sorted = [COLORS[m] for m in mean_ranks.index]
    ax.barh(mean_ranks.index[::-1], mean_ranks.values[::-1], color=colors_sorted[::-1])
    ax.set_xlabel("Rang moyen (1 = toujours le meilleur)")
    ax.set_title("Classement global des 6 méthodes (rang moyen sur 6 combinaisons horizon×machine)")
    for i, v in enumerate(mean_ranks.values[::-1]):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/04_meilleur_modele.png", dpi=120)
    plt.close(fig)

    with open("rapport_prediction_health_index_comparaison_finale.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    print("\n".join(report))
    print(f"\nFichiers générés : rapport_prediction_health_index_comparaison_finale.md, "
          f"{OUTPUT_DIR}/03_comparaison_finale.png, {OUTPUT_DIR}/04_meilleur_modele.png")


if __name__ == "__main__":
    main()
