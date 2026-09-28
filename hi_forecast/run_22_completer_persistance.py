"""Complete la ligne PERSISTANCE du tableau unifie : metriques de regression
manquantes, metriques de classification d'etat, et R2 ajuste pour tous.

Sans cette ligne, les colonnes de classification classent les modeles entre eux
sans dire si l'un d'eux bat « ne rien faire » — ce qui est precisement la question.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, "C:\\pylib")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             f1_score, matthews_corrcoef, precision_score, recall_score)

from protocol import regression_metrics
from run_16_ae_roleA import load

CSV = os.path.join(HERE, "tableau_unifie_actuelle.csv")
d = pd.read_csv(CSV)
data = load(False)
yte, nte, ntr = data["yte"], data["nte"], data["ntr"]

# --- regression complete de la persistance ---
m = regression_metrics(yte, nte, n_features=1, hi_now=nte)

# --- classification de l'etat, MEMES seuils que pour les modeles ---
thr_s = float(np.quantile(ntr, 0.05))
thr_a = float(np.quantile(ntr, 0.01))
to_state = lambda v: np.where(v <= thr_a, "Alarme",
                              np.where(v <= thr_s, "Surveillance", "Normal"))
LB = ["Normal", "Surveillance", "Alarme"]
s_true, s_pers = to_state(yte), to_state(nte)
deg_t, deg_p = (s_true != "Normal").astype(int), (s_pers != "Normal").astype(int)

vals = dict(
    adj_r2=m["adj_r2"], mae=m["mae"], medae=m["medae"], mape=m["mape"],
    var_expliquee=m["explained_var"], erreur_max=m["max_error"], biais=m["bias"],
    etat_accuracy=float(accuracy_score(s_true, s_pers)),
    etat_bal_accuracy=float(balanced_accuracy_score(s_true, s_pers)),
    etat_f1_macro=float(f1_score(s_true, s_pers, average="macro", labels=LB, zero_division=0)),
    etat_mcc=float(matthews_corrcoef(deg_t, deg_p)),
    etat_kappa=float(cohen_kappa_score(s_true, s_pers)),
    alerte_rappel=float(recall_score(deg_t, deg_p, zero_division=0)),
    alerte_precision=float(precision_score(deg_t, deg_p, zero_division=0)),
    alerte_f1=float(f1_score(deg_t, deg_p, zero_division=0)))

i = d.index[d["famille"] == "Persistance (reference)"][0]
for k, v in vals.items():
    d.loc[i, k] = v

# --- R2 ajuste pour tous les modeles : 1-(1-R2)(n-1)/(n-p-1), p = 20 ou 21 ---
n = int(d.loc[i, "n_test"])
for j, r in d.iterrows():
    if j == i:
        continue
    p = 21 if r["categorie"] == "autoencodeur" else 20   # top-20 (+ latent/erreur pour les AE)
    p = 37 if r["categorie"] == "autoencodeur" else 20
    d.loc[j, "adj_r2"] = 1 - (1 - r["test_r2"]) * (n - 1) / (n - p - 1)

d.to_csv(CSV, index=False, encoding="utf-8-sig")

print("=" * 96)
print("LIGNE PERSISTANCE COMPLETEE")
print("=" * 96)
print(f"  regression  : R2 {m['r2']:.4f}  R2aj {m['adj_r2']:.4f}  RMSE {m['rmse']:.4f}  "
      f"MAE {m['mae']:.4f}  MedAE {m['medae']:.4f}  MAPE {m['mape']:.3f}")
print(f"  etat        : accuracy {vals['etat_accuracy']:.4f}  "
      f"bal_acc {vals['etat_bal_accuracy']:.4f}  F1 macro {vals['etat_f1_macro']:.4f}  "
      f"MCC {vals['etat_mcc']:.4f}  kappa {vals['etat_kappa']:.4f}")
print(f"  alerte      : rappel {vals['alerte_rappel']:.4f}  "
      f"precision {vals['alerte_precision']:.4f}  F1 {vals['alerte_f1']:.4f}")

print("\n" + "=" * 96)
print("QUI BAT REELLEMENT LA PERSISTANCE, SUR CHAQUE METRIQUE ?")
print("=" * 96)
mods = d[d["categorie"] != "reference"]
for col, sens, lab in [("etat_accuracy", "max", "accuracy de l'etat"),
                       ("etat_bal_accuracy", "max", "accuracy equilibree"),
                       ("etat_f1_macro", "max", "F1 macro"),
                       ("etat_mcc", "max", "MCC"),
                       ("etat_kappa", "max", "kappa"),
                       ("alerte_rappel", "max", "rappel d'alerte"),
                       ("alerte_precision", "max", "precision d'alerte"),
                       ("alerte_f1", "max", "F1 d'alerte")]:
    ref = vals[col]
    better = mods[mods[col] > ref] if sens == "max" else mods[mods[col] < ref]
    best = mods.loc[mods[col].idxmax()] if sens == "max" else mods.loc[mods[col].idxmin()]
    flag = "OUI" if len(better) else "AUCUN"
    print(f"  {lab:<22} persistance={ref:.4f}  meilleur modele={best[col]:.4f} "
          f"({best['famille'][:34]})  -> {len(better):>2} modeles la battent [{flag}]")
print(f"\n  -> {CSV}")
