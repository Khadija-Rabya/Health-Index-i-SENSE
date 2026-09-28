"""Figures des autoencodeurs (17-20)."""
import json
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

A = json.load(open(os.path.join(HERE, "ae_roleA_resultats.json"), encoding="utf-8"))
dA = pd.DataFrame([{k: v for k, v in r.items() if k != "scopes"} for r in A])
B = pd.read_csv(os.path.join(HERE, "ae_roleB_resultats.csv"))
feas = pd.read_csv(os.path.join(HERE, "ae_faisabilite_fenetres.csv"))

# ------------------------------------------------ 17 : faisabilite des fenetres
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
f = feas[feas["W"] > 1]
x = np.arange(len(f)); w = 0.38
ax[0].bar(x - w / 2, f["sessions_ok"], w, color=BLUE, label="sessions utilisables")
ax[0].bar(x + w / 2, f["echantillons"] / 400, w, color=ORANGE,
          label="échantillons / 400")
for i, (s, e) in enumerate(zip(f["sessions_ok"], f["echantillons"])):
    ax[0].text(i - w / 2, s, f"{s}", ha="center", va="bottom", fontsize=8)
    ax[0].text(i + w / 2, e / 400, f"{e/1000:.1f}k", ha="center", va="bottom", fontsize=8)
ax[0].set_xticks(x); ax[0].set_xticklabels([f"W={int(v)}" for v in f["W"]])
ax[0].set_title("Fenêtres utilisables sans franchir de session\n(352 sessions, 42 141 lignes)")
ax[0].legend(fontsize=8)
ax[1].plot(f["W"], f["pct_lignes"] * 100, marker="o", lw=2, color=BLUE,
           label="% de lignes conservées")
ax[1].plot(f["W"], f["pct_sessions"] * 100, marker="s", lw=2, color=ORANGE,
           label="% de sessions utilisables")
ax[1].set_xlabel("longueur de fenêtre W"); ax[1].set_ylabel("%")
ax[1].set_ylim(0, 100); ax[1].legend(fontsize=8)
ax[1].set_title("Coût du fenêtrage\n(médiane de session = 3 lignes)")
fig.tight_layout(); fig.savefig(f"{FIG}/17_ae_faisabilite.png", dpi=130); plt.close(fig)

# ------------------------------------------------ 18 : role A, skill et criteres
fig, ax = plt.subplots(1, 2, figsize=(15.5, 6.5))
for j, (lab, title) in enumerate([("etiquette actuelle", "Étiquette actuelle"),
                                  ("etiquette viscosite masquee",
                                   "Étiquette viscosité masquée")]):
    g = dA[dA["etiquette"] == lab].sort_values("test_skill")
    y = np.arange(len(g))
    cols = [GREEN if p else (ORANGE if a else RED)
            for p, a in zip(g["passe"], g["critere_a"])]
    ax[j].barh(y, g["test_skill"], color=cols)
    ax[j].axvline(0, color="black", lw=1.2)
    for i, (v, p) in enumerate(zip(g["test_skill"], g["passe"])):
        ax[j].text(v, i, f" {v:+.3f}" + ("  ✓" if p else ""), va="center", fontsize=7.5)
    ax[j].set_yticks(y); ax[j].set_yticklabels(g["nom"], fontsize=7.5)
    ax[j].set_xlabel("skill vs persistance (test gelé)")
    ax[j].set_title(title, fontsize=11)
ax[0].legend(handles=[plt.Line2D([0], [0], marker="s", ls="", mfc=c, mec=c, ms=8, label=l)
                      for c, l in [(GREEN, "passe (a) et (b)"),
                                   (ORANGE, "passe (a) seulement"),
                                   (RED, "échoue (a)")]], fontsize=8, loc="lower right")
fig.suptitle("Rôle A — prévision : skill de chaque architecture d'autoencodeur", fontsize=13)
fig.tight_layout(); fig.savefig(f"{FIG}/18_ae_roleA_skill.png", dpi=130); plt.close(fig)

# ------------------------------------------------ 19 : IC par perimetre
best = max(A, key=lambda r: (r["passe"], r["test_skill"]))
gate = {"global": (0.0941, [-0.0045, 0.1662]), "A": (-0.1191, [-0.3331, 0.4949]),
        "B": (0.1017, [0.0844, 0.2007])}
keys = ["global", "A", "B", "ON", "OFF"]
fig, ax = plt.subplots(figsize=(11, 5.5))
y = np.arange(len(keys))
for i, k in enumerate(keys):
    s = best["scopes"].get(k)
    if s:
        lo, hi = s["ci_skill"]
        c = GREEN if lo > 0 else RED
        ax.plot([lo, hi], [i + 0.16] * 2, lw=5, color=c, alpha=0.6, solid_capstyle="butt")
        ax.plot([s["skill"]], [i + 0.16], "o", ms=9, mfc=BLUE, mec="white", mew=1.4, zorder=3)
        ax.text(hi, i + 0.16, f"  AE ({s['n_blocks']} blocs)", va="center", fontsize=8, color=c)
    if k in gate:
        sk, (lo, hi) = gate[k]
        c = GREEN if lo > 0 else RED
        ax.plot([lo, hi], [i - 0.16] * 2, lw=5, color=c, alpha=0.3, solid_capstyle="butt")
        ax.plot([sk], [i - 0.16], "s", ms=8, mfc=GREY, mec="white", mew=1.2, zorder=3)
        ax.text(hi, i - 0.16, "  porte LightGBM", va="center", fontsize=8, color=GREY)
ax.axvline(0, color="black", lw=1.4)
ax.set_yticks(y); ax.set_yticklabels(keys)
ax.set_xlabel("skill vs persistance — IC 95 % (bootstrap par blocs)")
ax.set_title(f"Critère (b) par périmètre — {best['nom']}\n"
             f"({best['etiquette']}) contre le modèle à porte précédent", fontsize=11)
fig.tight_layout(); fig.savefig(f"{FIG}/19_ae_criteres.png", dpi=130); plt.close(fig)

# ------------------------------------------------ 20 : role B
fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.5))
for j, base in enumerate(["etiquette actuelle", "viscosite masquee"]):
    g = B[B["base"] == base].copy()
    if g.empty:
        continue
    g["court"] = g["tag"].str.replace(r" \(.*\)", "", regex=True).str.slice(0, 40)
    y = np.arange(len(g))
    ax[j].barh(y - 0.2, g["sd_label"], 0.4, color=PURPLE, label="écart-type de l'étiquette")
    ax[j].barh(y + 0.2, g["test_acc"], 0.4, color=BLUE, label="accuracy de test")
    ax[j].barh(y + 0.2, g["test_acc_persist"], 0.4, color="none", edgecolor=RED, lw=1.4,
               label="accuracy de la persistance")
    ax[j].set_yticks(y); ax[j].set_yticklabels(g["court"], fontsize=7)
    ax[j].set_title(f"Base : {base}", fontsize=10)
    ax[j].legend(fontsize=7, loc="lower right")
fig.suptitle("Rôle B — étiquette reconstruite par autoencodeur : plus rugueuse, "
             "donc moins prévisible\n(les accuracies ne sont PAS comparables au rôle A)",
             fontsize=12)
fig.tight_layout(); fig.savefig(f"{FIG}/20_ae_roleB.png", dpi=130); plt.close(fig)

print("Figures generees :")
for f_ in ["17_ae_faisabilite", "18_ae_roleA_skill", "19_ae_criteres", "20_ae_roleB"]:
    print(f"  figures/{f_}.png")
