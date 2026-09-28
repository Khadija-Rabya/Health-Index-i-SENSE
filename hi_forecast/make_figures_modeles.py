"""Figures dediees a la comparaison des 14 familles de modeles (figures 10 a 16)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, RED, GREEN, PURPLE, GREY = ("#2b6cb0", "#dd6b20", "#e53e3e",
                                          "#38a169", "#805ad5", "#718096")

lb = pd.read_csv(os.path.join(HERE, "leaderboard_familles.csv"))
PERS_ACC, PERS_R2 = 0.8164207825529185, 0.11423529583335357

# famille -> categorie (pour la couleur)
CAT = {
    "Linear (OLS)": "Linéaire", "Ridge": "Linéaire", "Lasso": "Linéaire",
    "ElasticNet": "Linéaire",
    "Random Forest": "Arbres (bagging)", "Extra Trees": "Arbres (bagging)",
    "Gradient Boosting": "Arbres (boosting)", "XGBoost": "Arbres (boosting)",
    "LightGBM": "Arbres (boosting)", "CatBoost": "Arbres (boosting)",
    "SVM (RBF, Nystroem)": "Noyau / distance", "SVM (RBF exact)": "Noyau / distance",
    "KNN (k=25)": "Noyau / distance", "MLP (128,64)": "Réseau de neurones",
}
CATCOL = {"Linéaire": BLUE, "Arbres (bagging)": GREEN, "Arbres (boosting)": ORANGE,
          "Noyau / distance": PURPLE, "Réseau de neurones": RED}
lb["cat"] = lb["name"].map(CAT)
lb["col"] = lb["cat"].map(CATCOL)


def legend_cats(ax, loc="lower right"):
    ax.legend(handles=[Line2D([0], [0], marker="s", ls="", mfc=c, mec=c, ms=8, label=k)
                       for k, c in CATCOL.items()], fontsize=7.5, loc=loc)


# ---------------------------------------- 10 : R2 et accuracy de test, classes
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
d = lb.sort_values("test_r2")
y = np.arange(len(d))
ax[0].barh(y, d["test_r2"], color=d["col"])
ax[0].axvline(PERS_R2, color="black", ls="--", lw=1.6, label=f"persistance = {PERS_R2:.3f}")
ax[0].axvline(0, color=GREY, lw=1)
for i, v in enumerate(d["test_r2"]):
    ax[0].text(v, i, f" {v:.3f}" if v > 0 else f"{v:.3f} ", va="center",
               ha="left" if v > 0 else "right", fontsize=7.5)
ax[0].set_yticks(y); ax[0].set_yticklabels(d["name"], fontsize=8)
ax[0].set_xlabel("R² sur le test gelé"); ax[0].set_title("Qualité de reconstruction (R²)")
ax[0].legend(fontsize=8, loc="lower right")

d2 = lb.sort_values("test_acc_tol")
ax[1].barh(y, d2["test_acc_tol"], color=d2["col"])
ax[1].axvline(PERS_ACC, color="black", ls="--", lw=1.6, label=f"persistance = {PERS_ACC:.3f}")
for i, v in enumerate(d2["test_acc_tol"]):
    ax[1].text(v, i, f" {v:.3f}", va="center", fontsize=7.5)
ax[1].set_yticks(y); ax[1].set_yticklabels(d2["name"], fontsize=8)
ax[1].set_xlabel("accuracy @ ±0,01 sur le test gelé")
ax[1].set_title("Précision dans la bande de tolérance")
legend_cats(ax[1])
fig.suptitle("Classement des 14 familles sur le test gelé — deux métriques, deux classements",
             fontsize=13)
fig.tight_layout(); fig.savefig(f"{FIG}/10_familles_classement.png", dpi=130); plt.close(fig)

# ---------------------------------------- 11 : CV contre test (desaccord)
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ax[0].scatter(lb["cv_r2_mean"], lb["test_r2"], s=90, c=lb["col"], edgecolors="white", zorder=3)
for _, r in lb.iterrows():
    ax[0].annotate(r["name"], (r["cv_r2_mean"], r["test_r2"]), fontsize=7,
                   xytext=(4, 4), textcoords="offset points")
ax[0].axhline(PERS_R2, color=GREY, ls="--", lw=1)
ax[0].axvline(0, color=GREY, ls="--", lw=1)
ax[0].set_xscale("symlog", linthresh=1)
ax[0].set_xlabel("R² moyen en validation croisée (échelle symlog)")
ax[0].set_ylabel("R² sur le test gelé")
ax[0].set_title("CV contre test : aucune corrélation utile\n(seul Extra Trees a un R² CV positif)")

d3 = lb.sort_values("cv_acc_tol_mean")
y = np.arange(len(d3))
ax[1].barh(y, d3["cv_acc_tol_mean"], xerr=d3["cv_acc_tol_std"], color=d3["col"],
           error_kw=dict(lw=1.1, ecolor="#444", capsize=3))
ax[1].axvline(0.7912, color="black", ls="--", lw=1.6, label="persistance CV = 0,791")
ax[1].set_yticks(y); ax[1].set_yticklabels(d3["name"], fontsize=8)
ax[1].set_xlabel("accuracy @ ±0,01 en CV (moyenne ± écart-type)")
ax[1].set_title("Instabilité en CV : les barres d'erreur\nrecouvrent presque tout le classement")
ax[1].legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(f"{FIG}/11_cv_vs_test_familles.png", dpi=130); plt.close(fig)

# ---------------------------------------- 12 : carte thermique multi-metriques
metrics = ["test_r2", "test_acc_tol", "test_rmse", "test_mae", "test_medae",
           "test_mape", "test_explained_var", "test_max_error", "test_skill_vs_persist"]
labels = ["R²", "acc ±0,01", "RMSE", "MAE", "MedAE", "MAPE", "var. expl.",
          "err. max", "skill"]
lower_better = {"test_rmse", "test_mae", "test_medae", "test_mape", "test_max_error"}
d4 = lb.sort_values("test_r2", ascending=False).reset_index(drop=True)
Z = np.zeros((len(d4), len(metrics)))
for j, m in enumerate(metrics):
    v = d4[m].to_numpy(dtype=float)
    r = pd.Series(v).rank(ascending=(m in lower_better), pct=True).to_numpy()
    Z[:, j] = r
fig, ax = plt.subplots(figsize=(11, 7.5))
im = ax.imshow(Z, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=9)
ax.set_yticks(range(len(d4))); ax.set_yticklabels(d4["name"], fontsize=8.5)
for i in range(len(d4)):
    for j, m in enumerate(metrics):
        val = d4[m].iloc[i]
        txt = f"{val:.3f}" if abs(val) < 100 else f"{val:.0f}"
        ax.text(j, i, txt, ha="center", va="center", fontsize=6.8,
                color="black" if 0.25 < Z[i, j] < 0.85 else "white")
ax.set_title("Toutes les métriques de test, par famille\n"
             "(couleur = rang relatif, vert = meilleur ; tri par R²)", fontsize=12)
fig.colorbar(im, ax=ax, shrink=0.7, label="rang relatif")
fig.tight_layout(); fig.savefig(f"{FIG}/12_heatmap_metriques.png", dpi=130); plt.close(fig)

# ---------------------------------------- 13 : cout de calcul contre performance
fig, ax = plt.subplots(figsize=(10, 6.5))
ax.scatter(lb["seconds"], lb["test_r2"], s=110, c=lb["col"], edgecolors="white", zorder=3)
for _, r in lb.iterrows():
    ax.annotate(r["name"], (r["seconds"], r["test_r2"]), fontsize=7.5,
                xytext=(5, 4), textcoords="offset points")
ax.axhline(PERS_R2, color="black", ls="--", lw=1.4, label="persistance")
ax.set_xscale("log")
ax.set_xlabel("temps d'entraînement des 5 plis + test (s, échelle log)")
ax.set_ylabel("R² sur le test gelé")
ax.set_title("Coût de calcul contre performance — le temps passé n'achète rien ici")
legend_cats(ax, loc="lower left")
fig.tight_layout(); fig.savefig(f"{FIG}/13_cout_vs_performance.png", dpi=130); plt.close(fig)

# ---------------------------------------- 14 : courbes de retrecissement alpha
sw = pd.read_csv(os.path.join(HERE, "sweep_alpha.csv"), index_col=0)
fig, ax = plt.subplots(figsize=(10.5, 6))
cols = [BLUE, ORANGE, GREEN, PURPLE, RED]
for c, col in zip(sw.columns, cols):
    ax.plot(sw.index, sw[c], marker="o", ms=5, lw=1.8, color=col, label=c)
ax.axhline(0.7912, color="black", ls="--", lw=1.6, label="persistance (α = 0)")
ax.set_xlabel("facteur de rétrécissement α  —  prédiction = hi[t] + α · δ̂")
ax.set_ylabel("accuracy @ ±0,01 en validation croisée")
ax.set_title("Sans porte, corriger la persistance DÉGRADE toujours la précision\n"
             "(α optimal = 0 pour toutes les familles sauf une)")
ax.legend(fontsize=8.5)
ax.annotate("α = 0 : ne rien corriger", xy=(0.0, 0.7912), xytext=(0.35, 0.76),
            arrowprops=dict(arrowstyle="->", color=GREY), fontsize=9, color=GREY)
fig.tight_layout(); fig.savefig(f"{FIG}/14_courbes_alpha.png", dpi=130); plt.close(fig)

# ---------------------------------------- 15 : familles contre modele a porte
fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.6))
best4 = lb.sort_values("test_r2", ascending=False).head(4)
names = list(best4["name"]) + ["Modèle à porte", "Persistance"]
cv = list(best4["cv_acc_tol_mean"]) + [0.8094, 0.7912]
cvsd = list(best4["cv_acc_tol_std"]) + [0.0370, 0.0391]
te = list(best4["test_acc_tol"]) + [0.8259, PERS_ACC]
cols6 = list(best4["col"]) + ["#1a365d", GREY]
x = np.arange(len(names))
ax[0].bar(x, cv, yerr=cvsd, color=cols6, capsize=4,
          error_kw=dict(lw=1.2, ecolor="#333"))
for i, (v, s) in enumerate(zip(cv, cvsd)):
    ax[0].text(i, v + s + 0.012, f"{v:.3f}\n±{s:.3f}", ha="center", fontsize=7.5)
ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
ax[0].set_ylabel("accuracy @ ±0,01 en CV"); ax[0].set_ylim(0, 1.02)
ax[0].set_title("Validation croisée — le critère de sélection")
ax[1].bar(x, te, color=cols6)
for i, v in enumerate(te):
    ax[1].text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
ax[1].set_ylabel("accuracy @ ±0,01 sur le test"); ax[1].set_ylim(0, 1.02)
ax[1].set_title("Test gelé — mesuré une seule fois")
fig.suptitle("Les 4 meilleures familles contre l'architecture retenue", fontsize=12.5)
fig.tight_layout(); fig.savefig(f"{FIG}/15_familles_vs_porte.png", dpi=130); plt.close(fig)

# ---------------------------------------- 16 : performance par machine
fig, ax = plt.subplots(figsize=(11, 6))
d5 = lb.sort_values("test_r2", ascending=False)
x = np.arange(len(d5)); w = 0.38
ax.bar(x - w / 2, d5["test_acc_A"], w, color=BLUE, label="Motosoufflante A")
ax.bar(x + w / 2, d5["test_acc_B"], w, color=ORANGE, label="Motosoufflante B")
ax.axhline(0.9258, color=BLUE, ls="--", lw=1.2, label="persistance A = 0,926")
ax.axhline(0.7264, color=ORANGE, ls="--", lw=1.2, label="persistance B = 0,726")
ax.set_xticks(x); ax.set_xticklabels(d5["name"], rotation=32, ha="right", fontsize=8)
ax.set_ylabel("accuracy @ ±0,01 sur le test")
ax.set_title("Aucune famille ne bat la persistance sur la Motosoufflante A\n"
             "(les barres bleues restent sous la ligne bleue)")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/16_familles_par_machine.png", dpi=130); plt.close(fig)

print("Figures generees :")
for f in ["10_familles_classement", "11_cv_vs_test_familles", "12_heatmap_metriques",
          "13_cout_vs_performance", "14_courbes_alpha", "15_familles_vs_porte",
          "16_familles_par_machine"]:
    print(f"  figures/{f}.png")
