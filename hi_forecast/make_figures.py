"""Figures de verification (taches 1-3) et preuve viscosite (note i-SENSE)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, RED, GREEN, GREY = "#2b6cb0", "#dd6b20", "#e53e3e", "#38a169", "#718096"

with open(os.path.join(HERE, "verif_resultats.json"), encoding="utf-8") as f:
    V = json.load(f)

# ---------------------------------------------- 04 : CV par pli, RF vs porte
cv = pd.read_csv(os.path.join(HERE, "verif_A_cv_comparee.csv"))
fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
x = np.arange(len(cv)); w = 0.27
ax[0].bar(x - w, cv["rf_acc"], w, label="RF d'origine", color=RED)
ax[0].bar(x, cv["gate_acc"], w, label="Modele final a porte", color=BLUE)
ax[0].bar(x + w, cv["persist_acc"], w, label="Persistance", color=GREY)
ax[0].axhline(0.80, color=GREEN, ls=":", lw=1.5, label="objectif 80%")
ax[0].set_xticks(x); ax[0].set_xticklabels([f"pli {i}" for i in cv["pli"]])
ax[0].set_ylabel("accuracy @ ±0.01"); ax[0].set_ylim(0, 1)
ax[0].set_title("Validation croisee — accuracy par pli (memes plis)")
ax[0].legend(fontsize=8, loc="lower right")

lab = ["RF d'origine", "Modele a porte"]
cvm = [cv["rf_acc"].mean(), cv["gate_acc"].mean()]
cvs = [cv["rf_acc"].std(), cv["gate_acc"].std()]
tst = [0.8310, 0.8259]
xx = np.arange(2)
ax[1].bar(xx - 0.2, cvm, 0.4, yerr=cvs, capsize=5, color=BLUE, label="CV (selection)")
ax[1].bar(xx + 0.2, tst, 0.4, color=ORANGE, label="Test gele")
ax[1].axhline(0.7912, color=GREY, ls="--", lw=1.2, label="persistance CV")
for i, (c, t) in enumerate(zip(cvm, tst)):
    ax[1].text(i - 0.2, c + 0.02, f"{c:.3f}", ha="center", fontsize=9)
    ax[1].text(i + 0.2, t + 0.02, f"{t:.3f}", ha="center", fontsize=9)
ax[1].set_xticks(xx); ax[1].set_xticklabels(lab)
ax[1].set_ylabel("accuracy @ ±0.01"); ax[1].set_ylim(0, 1.05)
ax[1].set_title("CV vs test : le test seul inverse le classement")
ax[1].legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(f"{FIG}/04_cv_vs_test_rf_porte.png", dpi=130); plt.close(fig)

# ---------------------------------------------- 05 : intervalles de confiance
B = V["B"]["bootstrap"]
labels = ["Global", "Motosoufflante A", "Motosoufflante B"]
keys = ["global", "Motosoufflante A", "Motosoufflante B"]
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
for j, (metric, title) in enumerate([("acc", "Ecart d'accuracy (modele - persistance)"),
                                     ("skill", "Skill score vs persistance")]):
    pts = [B[k]["delta_acc" if metric == "acc" else "skill"] for k in keys]
    los = [B[k]["ci_acc" if metric == "acc" else "ci_skill"][0] for k in keys]
    his = [B[k]["ci_acc" if metric == "acc" else "ci_skill"][1] for k in keys]
    y = np.arange(len(keys))
    cols = [RED if lo <= 0 <= hi else GREEN for lo, hi in zip(los, his)]
    for i, (p, lo, hi, c) in enumerate(zip(pts, los, his, cols)):
        ax[j].plot([lo, hi], [i, i], lw=4, color=c, alpha=0.55, zorder=1,
                   solid_capstyle="butt")
        ax[j].plot([lo, lo], [i - .12, i + .12], lw=2, color=c, zorder=1)
        ax[j].plot([hi, hi], [i - .12, i + .12], lw=2, color=c, zorder=1)
        ax[j].plot([p], [i], "o", ms=9, mfc=BLUE, mec="white", mew=1.4, zorder=3)
        ax[j].text(p, i + 0.22, f"{p:+.4f}", ha="center", fontsize=8.5, color=BLUE)
    ax[j].axvline(0, color="black", lw=1.2)
    ax[j].set_yticks(y); ax[j].set_yticklabels(
        [f"{l}\n(n_eff={B[k]['n_eff']:.0f}, {B[k]['n_sessions']} sessions)"
         for l, k in zip(labels, keys)], fontsize=8)
    ax[j].set_title(title + "\nIC 95% — bootstrap par blocs (sessions)", fontsize=10)
    ax[j].set_xlabel("rouge = l'intervalle croise zero")
fig.tight_layout(); fig.savefig(f"{FIG}/05_intervalles_confiance.png", dpi=130); plt.close(fig)

# ---------------------------------------------- 06 : arbitrage des metriques
C = pd.read_csv(os.path.join(HERE, "verif_C_arbitrage.csv"))
C = C[C["configuration"] != "Persistance"].reset_index(drop=True)
pers = pd.read_csv(os.path.join(HERE, "verif_C_arbitrage.csv"))
pers = pers[pers["configuration"] == "Persistance"].iloc[0]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
cols3 = [BLUE, ORANGE, GREEN]
for j, (col, name, ref) in enumerate([("acc", "accuracy @ ±0.01", pers["acc"]),
                                      ("r2", "R²", pers["r2"]),
                                      ("skill", "skill vs persistance", 0.0)]):
    ax[j].bar(C["configuration"], C[col], color=cols3)
    ax[j].axhline(ref, color=GREY, ls="--", lw=1.4, label=f"persistance = {ref:.4f}")
    for i, v in enumerate(C[col]):
        ax[j].text(i, v, f" {v:.4f}", ha="center", va="bottom", fontsize=9)
    ax[j].set_title(name); ax[j].tick_params(axis="x", labelrotation=18, labelsize=7.5)
    ax[j].legend(fontsize=7.5)
fig.suptitle("Le choix de la metrique decide du gagnant — memes lignes de test, meme protocole",
             fontsize=12)
fig.tight_layout(); fig.savefig(f"{FIG}/06_arbitrage_metriques.png", dpi=130); plt.close(fig)

# ---------------------------------------------- 07 : preuve viscosite
raw = pd.read_csv(os.path.join(os.path.dirname(HERE), "isense_oil_data_health_index.csv"),
                  parse_dates=["created_at"], low_memory=False)
A_COL, B_COL = "asset_Motosoufflante A", "asset_Motosoufflante B"
v = raw["Viscosity at 40°C_filled"]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))

groups = [("A / ON", (raw[A_COL] == 1) & (raw["state_OFF"] == 0), GREEN),
          ("A / OFF", (raw[A_COL] == 1) & (raw["state_OFF"] == 1), "#68d391"),
          ("B / ON", (raw[B_COL] == 1) & (raw["state_OFF"] == 0), ORANGE),
          ("B / OFF", (raw[B_COL] == 1) & (raw["state_OFF"] == 1), RED)]
for name, m, c in groups:
    ax[0].hist(v[m].dropna(), bins=70, alpha=0.65, label=f"{name} (n={int(m.sum())})", color=c)
ax[0].axvline(46.0, color="black", ls="--", lw=1.6, label="reference TD46 = 46,0 cSt")
ax[0].axvspan(0, 20, color=RED, alpha=0.08)
ax[0].set_xlabel("Viscosite a 40 °C (cSt)"); ax[0].set_ylabel("Effectif")
ax[0].set_title("Distribution par machine et par etat")
ax[0].legend(fontsize=7.5)

ev = pd.DataFrame(V["E"]["par_groupe"])
ev["lab"] = ev["machine"].str[-1] + " / " + ev["etat"]
ax[1].bar(ev["lab"], ev["pct"] * 100, color=[GREEN, "#68d391", ORANGE, RED])
for i, (p, n) in enumerate(zip(ev["pct"], ev["n_basse"])):
    ax[1].text(i, p * 100, f" {p:.1%}\n(n={n})", ha="center", va="bottom", fontsize=8.5)
ax[1].set_ylabel("% de lignes avec viscosite < 20 cSt"); ax[1].set_ylim(0, 100)
ax[1].set_title("Part des valeurs physiquement impossibles")

sub = raw[raw[B_COL] == 1].sort_values("created_at")
ax[2].scatter(sub["created_at"], sub["Viscosity at 40°C_filled"], s=2, alpha=0.3,
              c=np.where(sub["state_OFF"] == 1, RED, ORANGE))
ax[2].axhline(46.0, color="black", ls="--", lw=1.4)
ax[2].axhspan(0, 20, color=RED, alpha=0.08)
ax[2].set_title("Motosoufflante B dans le temps\n(rouge = OFF, orange = ON)")
ax[2].set_ylabel("Viscosite a 40 °C (cSt)")
ax[2].tick_params(axis="x", labelrotation=25, labelsize=7)
fig.suptitle("Viscosite hors plage physique — Motosoufflante B a l'arret", fontsize=13)
fig.tight_layout(); fig.savefig(f"{FIG}/07_viscosite.png", dpi=130); plt.close(fig)

# ---------------------------------------------- 08 : schema du pipeline
fig, ax = plt.subplots(figsize=(15, 6.5))
ax.axis("off")
boxes = [
    (0.02, 0.62, 0.17, 0.26, "ENTREE\nisense_oil_data_\nhealth_index.csv\n42 141 x 171", "#e2e8f0"),
    (0.22, 0.62, 0.17, 0.26, "1. CORRECTION\nDES FUITES\nF1 label, F2 vi_proxy,\nF3 z-scores, F4 regles", "#fed7d7"),
    (0.42, 0.62, 0.17, 0.26, "2. PROTOCOLE GELE\ntest = 20% recents\nCV = 5 blocs purges\nembargo 3 h", "#c6f6d5"),
    (0.62, 0.62, 0.17, 0.26, "3. CIBLE\nhealth_index[t+18]\ndelta = y[t+18] - y[t]\ntolerance ±0,01", "#bee3f8"),
    (0.82, 0.62, 0.16, 0.26, "4. 33 ITERATIONS\n14 familles,\n10 leviers,\nselection en CV", "#fefcbf"),
    (0.22, 0.20, 0.24, 0.28, "REGRESSEUR\nLightGBM Huber\n400 arbres, 20 features\n-> delta predit", "#bee3f8"),
    (0.52, 0.20, 0.24, 0.28, "CLASSIFIEUR (PORTE)\nLightGBM\nP(|delta| > 0,01)\nAUC test 0,706", "#fbb6ce"),
]
for x0, y0, w, h, txt, col in boxes:
    ax.add_patch(plt.Rectangle((x0, y0), w, h, fc=col, ec="#2d3748", lw=1.3, zorder=1))
    ax.text(x0 + w / 2, y0 + h / 2, txt, ha="center", va="center", fontsize=8.5, zorder=2)
for x0 in (0.19, 0.39, 0.59, 0.79):
    ax.annotate("", xy=(x0 + 0.03, 0.75), xytext=(x0, 0.75),
                arrowprops=dict(arrowstyle="->", lw=1.6, color="#2d3748"))
ax.annotate("", xy=(0.40, 0.48), xytext=(0.88, 0.62),
            arrowprops=dict(arrowstyle="->", lw=1.6, color="#2d3748"))
ax.add_patch(plt.Rectangle((0.28, 0.02), 0.44, 0.12, fc="#9ae6b4", ec="#2d3748", lw=1.6))
ax.text(0.50, 0.08, "SORTIE :  health_index[t] + α·δ̂ · 1[P > 0,35]      α = 1,0\n"
                    "CV 0,8094 ± 0,0413      Test 0,8259      skill +0,094",
        ha="center", va="center", fontsize=10, weight="bold")
ax.annotate("", xy=(0.44, 0.14), xytext=(0.34, 0.20),
            arrowprops=dict(arrowstyle="->", lw=1.6, color="#2d3748"))
ax.annotate("", xy=(0.56, 0.14), xytext=(0.64, 0.20),
            arrowprops=dict(arrowstyle="->", lw=1.6, color="#2d3748"))
ax.set_xlim(0, 1); ax.set_ylim(0, 0.95)
ax.set_title("Chaine de traitement — de la donnee brute au modele final", fontsize=13, pad=6)
fig.tight_layout(); fig.savefig(f"{FIG}/08_pipeline.png", dpi=130); plt.close(fig)

print("Figures generees :")
for f in ["04_cv_vs_test_rf_porte.png", "05_intervalles_confiance.png",
          "06_arbitrage_metriques.png", "07_viscosite.png", "08_pipeline.png"]:
    print(f"  figures/{f}")
