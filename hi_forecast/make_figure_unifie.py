"""Figure 21 — tableau unifie : skill et IC de tous les modeles."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
BLUE, ORANGE, RED, GREEN, PURPLE, GREY = ("#2b6cb0", "#dd6b20", "#e53e3e",
                                          "#38a169", "#805ad5", "#718096")
CAT = {"reference": GREY, "classique": BLUE, "autoencodeur": PURPLE}

d = pd.read_csv(os.path.join(HERE, "tableau_unifie_actuelle.csv"))
d = d.sort_values("test_skill")

fig, ax = plt.subplots(1, 2, figsize=(16, 7))

# --- panneau 1 : skill de test avec IC 95 %
y = np.arange(len(d))
for i, (_, r) in enumerate(d.iterrows()):
    lo, hi = r["ic_bas"], r["ic_haut"]
    c = GREEN if r["significatif"] else RED
    if np.isfinite(lo) and np.isfinite(hi) and (lo != 0 or hi != 0):
        ax[0].plot([lo, hi], [i, i], lw=4, color=c, alpha=0.5, solid_capstyle="butt", zorder=1)
    ax[0].plot([r["test_skill"]], [i], "o", ms=8, mfc=CAT[r["categorie"]],
               mec="white", mew=1.3, zorder=3)
ax[0].axvline(0, color="black", lw=1.4)
ax[0].set_yticks(y); ax[0].set_yticklabels(d["famille"], fontsize=8)
ax[0].set_xlabel("skill vs persistance (test gelé) — barre = IC 95 % bootstrap par blocs")
ax[0].set_title("Skill et significativité\nvert = IC au-dessus de zéro, rouge = croise zéro",
                fontsize=11)
ax[0].legend(handles=[plt.Line2D([0], [0], marker="o", ls="", mfc=c, mec="white", ms=9, label=k)
                      for k, c in CAT.items()], fontsize=8, loc="lower right")

# --- panneau 2 : CV acc contre CV skill (les deux criteres de selection)
for _, r in d.iterrows():
    ax[1].scatter(r["cv_acc"], r["cv_skill"], s=110, c=CAT[r["categorie"]],
                  edgecolors="white", zorder=3)
    ax[1].annotate(r["famille"][:26], (r["cv_acc"], r["cv_skill"]), fontsize=7,
                   xytext=(5, 3), textcoords="offset points")
ax[1].axhline(0, color=GREY, ls="--", lw=1)
ax[1].axvline(0.7912, color=GREY, ls="--", lw=1)
ax[1].text(0.7915, ax[1].get_ylim()[0], " persistance", fontsize=8, color=GREY, rotation=90,
           va="bottom")
ax[1].set_xlabel("accuracy @ ±0,01 en CV  (le critère verrouillé)")
ax[1].set_ylabel("skill en CV  (le critère utile pour l'alerte)")
ax[1].set_title("Les deux critères de sélection ne désignent pas le même modèle", fontsize=11)

fig.suptitle("Tableau unifié — 18 modèles, protocole strictement identique, étiquette actuelle",
             fontsize=13)
fig.tight_layout(); fig.savefig(f"{FIG}/21_tableau_unifie.png", dpi=130); plt.close(fig)
print("figures/21_tableau_unifie.png")
