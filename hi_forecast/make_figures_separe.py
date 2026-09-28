"""Visualisations de l'entrainement separe par machine (figures 22 a 26)."""
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
CV = "croisee (blocs expansifs)"
WF = "glissante (walk-forward)"
G = "G* porte réglée"

d = pd.read_csv(os.path.join(HERE, "separe_par_machine_complet.csv"))
MACH = sorted(d["machine"].unique())

# ------------------------------- 22 : classement par machine (validation croisee)
fig, ax = plt.subplots(1, 2, figsize=(16, 6.5))
for j, mach in enumerate(MACH):
    s = d[(d.machine == mach) & (d.validation == CV) & (d.config == G)].sort_values("val_acc")
    y = np.arange(len(s))
    cols = [GREEN if v else RED for v in s["significatif"]]
    ax[j].barh(y - 0.2, s["val_acc"], 0.4, xerr=s["val_sd"], color=BLUE,
               error_kw=dict(lw=1, ecolor="#444", capsize=3), label="validation croisée")
    ax[j].barh(y + 0.2, s["acc_tol"], 0.4, color=cols, label="test gelé (vert = significatif)")
    ax[j].axvline(s["acc_persist"].iloc[0], color="black", ls="--", lw=1.6,
                  label=f"persistance = {s['acc_persist'].iloc[0]:.3f}")
    ax[j].set_yticks(y); ax[j].set_yticklabels(s["famille"], fontsize=9)
    ax[j].set_xlabel("accuracy @ ±0,01"); ax[j].set_title(mach, fontsize=12)
    ax[j].legend(fontsize=8, loc="lower right")
    ax[j].set_xlim(0, 1.02)
fig.suptitle("Modèles entraînés SÉPARÉMENT par machine — validation croisée contre test gelé",
             fontsize=13)
fig.tight_layout(); fig.savefig(f"{FIG}/22_separe_classement.png", dpi=130); plt.close(fig)

# ------------------------------- 23 : croisee contre glissante
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
for j, mach in enumerate(MACH):
    a = d[(d.machine == mach) & (d.validation == CV) & (d.config == G)].set_index("famille")
    b = d[(d.machine == mach) & (d.validation == WF) & (d.config == G)].set_index("famille")
    fams = [f for f in a.index if f in b.index]
    x = np.arange(len(fams)); w = 0.38
    ax[j].bar(x - w / 2, a.loc[fams, "val_acc"], w, yerr=a.loc[fams, "val_sd"],
              color=BLUE, capsize=3, label="croisée (blocs expansifs)")
    ax[j].bar(x + w / 2, b.loc[fams, "val_acc"], w, yerr=b.loc[fams, "val_sd"],
              color=ORANGE, capsize=3, label="glissante (walk-forward)")
    ax[j].set_xticks(x); ax[j].set_xticklabels(fams, rotation=32, ha="right", fontsize=8)
    ax[j].set_ylabel("accuracy de validation"); ax[j].set_title(mach, fontsize=11)
    ax[j].legend(fontsize=8)
    ax[j].set_ylim(0, 1.0)
fig.suptitle("Deux schémas de validation — la glissante entraîne sur un historique borné,\n"
             "donc elle est plus sévère et plus proche du régime de production", fontsize=12)
fig.tight_layout(); fig.savefig(f"{FIG}/23_separe_validations.png", dpi=130); plt.close(fig)

# ------------------------------- 24 : configurations G0 / G1 / G*
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
CFG = ["G0 persistance", "G1 sans porte", "G* porte réglée"]
CC = {"G0 persistance": GREY, "G1 sans porte": RED, "G* porte réglée": GREEN}
for j, mach in enumerate(MACH):
    s = d[(d.machine == mach) & (d.validation == CV)]
    fams = sorted(s["famille"].unique())
    x = np.arange(len(fams)); w = 0.27
    for k, cfgn in enumerate(CFG):
        v = [s[(s.famille == f) & (s.config == cfgn)]["val_acc"].mean() for f in fams]
        ax[j].bar(x + (k - 1) * w, v, w, color=CC[cfgn], label=cfgn)
    ax[j].set_xticks(x); ax[j].set_xticklabels(fams, rotation=32, ha="right", fontsize=8)
    ax[j].set_ylabel("accuracy de validation croisée"); ax[j].set_title(mach, fontsize=11)
    ax[j].legend(fontsize=8); ax[j].set_ylim(0, 1.0)
fig.suptitle("Trois configurations de décision — la porte réglée domine partout,\n"
             "la correction systématique (G1) dégrade la persistance", fontsize=12)
fig.tight_layout(); fig.savefig(f"{FIG}/24_separe_configs.png", dpi=130); plt.close(fig)

# ------------------------------- 25 : carte thermique multi-metriques par machine
MET = ["acc_tol", "r2", "rmse", "mae", "medae", "mape", "var_expl", "err_max",
       "skill", "etat_acc", "etat_bal", "etat_mcc", "alerte_rappel", "alerte_prec", "alerte_f1"]
LAB = ["acc±0,01", "R²", "RMSE", "MAE", "MedAE", "MAPE", "var.expl", "err.max",
       "skill", "état acc", "état bal", "MCC", "rappel al.", "préc. al.", "F1 al."]
LOW = {"rmse", "mae", "medae", "mape", "err_max"}
fig, ax = plt.subplots(1, 2, figsize=(17, 6))
for j, mach in enumerate(MACH):
    s = d[(d.machine == mach) & (d.validation == CV) & (d.config == G)].sort_values(
        "skill", ascending=False).reset_index(drop=True)
    Z = np.zeros((len(s), len(MET)))
    for k, m in enumerate(MET):
        Z[:, k] = pd.Series(s[m]).rank(ascending=(m in LOW), pct=True).to_numpy()
    im = ax[j].imshow(Z, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax[j].set_xticks(range(len(LAB))); ax[j].set_xticklabels(LAB, rotation=42, ha="right", fontsize=7.5)
    ax[j].set_yticks(range(len(s))); ax[j].set_yticklabels(s["famille"], fontsize=8)
    for r_ in range(len(s)):
        for c_ in range(len(MET)):
            v = s[MET[c_]].iloc[r_]
            ax[j].text(c_, r_, f"{v:.3f}" if abs(v) < 100 else f"{v:.0f}", ha="center",
                       va="center", fontsize=5.6,
                       color="black" if 0.25 < Z[r_, c_] < 0.85 else "white")
    ax[j].set_title(mach, fontsize=11)
fig.suptitle("Toutes les métriques par machine (couleur = rang relatif, vert = meilleur)",
             fontsize=12)
fig.tight_layout(); fig.savefig(f"{FIG}/25_separe_heatmap.png", dpi=130); plt.close(fig)

# ------------------------------- 26 : skill et IC par machine
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
for j, mach in enumerate(MACH):
    s = d[(d.machine == mach) & (d.validation == CV) & (d.config == G)].sort_values("skill")
    y = np.arange(len(s))
    for i, (_, r) in enumerate(s.iterrows()):
        c = GREEN if r["significatif"] else RED
        ax[j].plot([r["ic_bas"], r["ic_haut"]], [i, i], lw=4, color=c, alpha=0.5,
                   solid_capstyle="butt")
        ax[j].plot([r["skill"]], [i], "o", ms=8, mfc=BLUE, mec="white", mew=1.3, zorder=3)
    ax[j].axvline(0, color="black", lw=1.4)
    ax[j].set_yticks(y); ax[j].set_yticklabels(s["famille"], fontsize=9)
    ax[j].set_xlabel("skill vs persistance — IC 95 % bootstrap par sessions")
    ax[j].set_title(f"{mach}  ({int(s['n_blocs'].iloc[0])} blocs)", fontsize=11)
fig.suptitle("Significativité par machine — vert : l'intervalle exclut zéro", fontsize=12)
fig.tight_layout(); fig.savefig(f"{FIG}/26_separe_significativite.png", dpi=130); plt.close(fig)

print("Figures generees :")
for f in ["22_separe_classement", "23_separe_validations", "24_separe_configs",
          "25_separe_heatmap", "26_separe_significativite"]:
    print(f"  figures/{f}.png")
