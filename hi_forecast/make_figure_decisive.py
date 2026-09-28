"""Figure 09 — experience decisive : avant / apres masquage de la viscosite."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
BLUE, ORANGE, RED, GREEN, GREY = "#2b6cb0", "#dd6b20", "#e53e3e", "#38a169", "#718096"

with open(os.path.join(HERE, "verif_decisive.json"), encoding="utf-8") as f:
    D = json.load(f)
av, ap = D["decisive"]["avant"], D["decisive"]["apres"]
keys = ["global", "Motosoufflante A", "Motosoufflante B"]
labs = ["Global", "Motosoufflante A", "Motosoufflante B"]

fig, ax = plt.subplots(1, 3, figsize=(16, 5))

# --- panneau 1 et 2 : IC avant / apres
for j, (metric, ci, title) in enumerate(
        [("delta_acc", "ci_acc", "Écart d'accuracy (modèle − persistance)"),
         ("skill", "ci_skill", "Skill score vs persistance")]):
    y = np.arange(len(keys))
    for i, k in enumerate(keys):
        for src, off, col, name in [(av, +0.16, RED, "avant"), (ap, -0.16, GREEN, "après")]:
            lo, hi = src[k][ci]
            p = src[k][metric]
            crosses = lo <= 0 <= hi
            c = col if not crosses else GREY
            ax[j].plot([lo, hi], [i + off, i + off], lw=4, color=c, alpha=0.55,
                       solid_capstyle="butt", zorder=1)
            ax[j].plot([lo, lo], [i + off - .07, i + off + .07], lw=2, color=c)
            ax[j].plot([hi, hi], [i + off - .07, i + off + .07], lw=2, color=c)
            ax[j].plot([p], [i + off], "o", ms=8, mfc=col, mec="white", mew=1.3, zorder=3)
            ax[j].text(hi, i + off, f"  {name}", va="center", fontsize=7.5, color=col)
    ax[j].axvline(0, color="black", lw=1.3)
    ax[j].set_yticks(y); ax[j].set_yticklabels(labs, fontsize=9)
    ax[j].set_title(title + "\nIC 95 % — gris = croise zéro", fontsize=10)

# --- panneau 3 : n_eff
y = np.arange(len(keys)); w = 0.36
n_av = [av[k]["n_eff"] for k in keys]
n_ap = [ap[k]["n_eff"] for k in keys]
ax[2].barh(y + w / 2, n_av, w, color=RED, label="avant")
ax[2].barh(y - w / 2, n_ap, w, color=GREEN, label="après")
for i, (a, b) in enumerate(zip(n_av, n_ap)):
    ax[2].text(a, i + w / 2, f" {a:.0f}", va="center", fontsize=8, color=RED)
    ax[2].text(b, i - w / 2, f" {b:.0f}", va="center", fontsize=8, color=GREEN)
ax[2].set_yticks(y); ax[2].set_yticklabels(labs, fontsize=9)
ax[2].set_xscale("log")
ax[2].set_xlabel("taille d'échantillon effective (échelle log)")
ax[2].set_title("n_eff : l'alignement des signes\nrestaure la puissance statistique", fontsize=10)
ax[2].legend(fontsize=8, loc="lower right")

fig.suptitle("Expérience décisive — viscosité masquée sur les lignes à l'arrêt", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "09_experience_decisive.png"), dpi=130)
plt.close(fig)
print("figures/09_experience_decisive.png")
