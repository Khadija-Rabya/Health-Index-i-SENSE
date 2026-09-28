"""Le modele retenu est-il UTILISABLE en exploitation ?

Trois questions concretes, au-dela du R2 :
 1. Que vaut reellement une erreur de 0,0085 sur cet indice ?
 2. Le modele predit-il correctement l'ETAT (Normal / Surveillance / Alarme) a t+3h,
    qui est la seule chose sur laquelle une equipe de maintenance agit ?
 3. Ou est le plafond : que reste-t-il de previsible dans l'ecart ?
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, "C:\\pylib")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, mean_squared_error)

from ae_models import DenseAE
from protocol import ASSET_COLS, HORIZON, SEED, TOLERANCE, regression_metrics
from run_16_ae_roleA import (ALPHAS, PROBS, TOP20, dense_builder, evaluate, load)

MASKED = os.environ.get("AE_MASKED", "1") == "1"
d = load(MASKED)
lab = "viscosite masquee" if MASKED else "etiquette actuelle"

print("=" * 100)
print(f"MODELE RETENU — AE-1 dense debruiteur (latent 16, bruit 0,3) — {lab}")
print("=" * 100)
res = evaluate("AE-1 dense debruiteur (latent 16, bruit 0,3)", d,
               dense_builder(lambda: DenseAE(latent=16, hidden=96, noise=0.3)), verbose=False)

# reconstruire les predictions du test pour l'analyse operationnelle
from ae_models import DenseAE as _AE
Ca = np.nan_to_num(d["Ctr"].to_numpy(), nan=0.0)
Cb = np.nan_to_num(d["Cte"].to_numpy(), nan=0.0)
ae = _AE(latent=16, hidden=96, noise=0.3).fit(Ca)
Fa = np.hstack([d["Ttr"].to_numpy(), ae.encode(Ca), ae.reconstruction_error(Ca)[:, None]])
Fb = np.hstack([d["Tte"].to_numpy(), ae.encode(Cb), ae.reconstruction_error(Cb)[:, None]])
from run_16_ae_roleA import make_clf, make_reg
r = make_reg(); r.fit(Fa, d["ytr"] - d["ntr"])
c = make_clf(); c.fit(Fa, (np.abs(d["ytr"] - d["ntr"]) > TOLERANCE).astype(int))
dte, pte = r.predict(Fb), c.predict_proba(Fb)[:, 1]
yhat = np.clip(d["nte"] + res["alpha"] * dte * (pte > res["seuil"]), 0, 1)
y, n = d["yte"], d["nte"]

print("\n" + "=" * 100)
print("1. CE QUE VAUT L'ERREUR, EN UNITES METIER")
print("=" * 100)
m = regression_metrics(y, yhat, hi_now=n)
mp = regression_metrics(y, n, hi_now=n)
print(f"  etendue reelle de l'indice sur le test : {y.min():.4f} a {y.max():.4f} "
      f"(ecart-type {y.std():.4f})")
print(f"  {'':<24}{'MODELE':>12}{'PERSISTANCE':>14}{'gain':>10}")
for k, lab2 in [("rmse", "RMSE"), ("mae", "MAE"), ("medae", "erreur mediane")]:
    print(f"  {lab2:<24}{m[k]:>12.5f}{mp[k]:>14.5f}"
          f"{(1 - m[k] / mp[k]) * 100:>9.1f}%")
print(f"  {'part |erreur| <= 0,01':<24}{m['acc_tol']:>12.1%}{mp['acc_tol']:>14.1%}")
print(f"\n  R² = {m['r2']:.4f}   MAIS le R² se mesure contre la MOYENNE de la cible,")
print(f"  reference sans interet pour une serie aussi autocorrelee. La reference utile")
print(f"  est la persistance : skill = {m['skill_vs_persist']:+.4f}")
print(f"  -> le modele retire {m['skill_vs_persist']*100:.1f} % de l'erreur quadratique")
print(f"     que ferait « rien ne change en 3 h ».")

print("\n" + "=" * 100)
print("2. PREDICTION DE L'ETAT A t+3h — ce sur quoi une equipe agit reellement")
print("=" * 100)
# seuils d'etat : centiles des lignes saines d'entrainement (meme convention que le projet)
sev_tr = 1.0 / d["ntr"] - 1.0
thr_surv = float(np.quantile(d["ntr"], 0.05))
thr_alarm = float(np.quantile(d["ntr"], 0.01))
print(f"  seuils appliques a la VALEUR de l'indice (centiles 5 % / 1 % du train) : "
      f"Surveillance <= {thr_surv:.4f}, Alarme <= {thr_alarm:.4f}")


def to_state(v):
    return np.where(v <= thr_alarm, "Alarme",
                    np.where(v <= thr_surv, "Surveillance", "Normal"))


s_true, s_mod, s_per = to_state(y), to_state(yhat), to_state(n)
labels = ["Normal", "Surveillance", "Alarme"]
print(f"\n  repartition reelle a t+3h : " +
      ", ".join(f"{l} {np.mean(s_true == l):.1%}" for l in labels))
for nm, s in [("MODELE", s_mod), ("PERSISTANCE", s_per)]:
    print(f"\n  --- {nm} ---")
    print(f"    accuracy         {accuracy_score(s_true, s):.4f}")
    print(f"    accuracy equilibree {balanced_accuracy_score(s_true, s):.4f}")
    print(f"    F1 macro         {f1_score(s_true, s, average='macro', labels=labels):.4f}")
    cm = confusion_matrix(s_true, s, labels=labels)
    print(f"    matrice de confusion (lignes = vrai, colonnes = predit) :")
    print("      " + "".join(f"{l:>14}" for l in labels))
    for i, l in enumerate(labels):
        print(f"      {l:<14}" + "".join(f"{v:>14}" for v in cm[i]))
    # detection des degradations
    deg_true = s_true != "Normal"
    deg_pred = s != "Normal"
    tp = int((deg_true & deg_pred).sum()); fn = int((deg_true & ~deg_pred).sum())
    fp = int((~deg_true & deg_pred).sum())
    rec = tp / max(tp + fn, 1); prec = tp / max(tp + fp, 1)
    print(f"    detection « non Normal » : rappel {rec:.1%}, precision {prec:.1%} "
          f"(VP={tp}, FN={fn}, FP={fp})")

print("\n" + "=" * 100)
print("3. OU EST LE PLAFOND ?")
print("=" * 100)
delta = y - n
print(f"  ecart reel a 3 h : ecart-type {delta.std():.5f}, mediane |ecart| {np.median(np.abs(delta)):.5f}")
print(f"  part des lignes ou |ecart| <= 0,01 (donc rien a predire) : "
      f"{np.mean(np.abs(delta) <= TOLERANCE):.1%}")
big = np.abs(delta) > TOLERANCE
print(f"  sur les {int(big.sum())} lignes qui bougent vraiment :")
print(f"    ecart-type de l'ecart          {delta[big].std():.5f}")
print(f"    R² du modele sur ce sous-ensemble {regression_metrics(y[big], yhat[big])['r2']:.4f}")
print(f"    skill sur ce sous-ensemble        "
      f"{regression_metrics(y[big], yhat[big], hi_now=n[big])['skill_vs_persist']:+.4f}")
corr = np.corrcoef(delta, yhat - n)[0, 1]
print(f"  correlation entre ecart reel et ecart predit : {corr:.4f}  "
      f"(part de variance de l'ecart expliquee : {corr**2:.1%})")

out = dict(label=lab, alpha=res["alpha"], seuil=res["seuil"],
           rmse=float(m["rmse"]), rmse_persist=float(mp["rmse"]),
           mae=float(m["mae"]), mae_persist=float(mp["mae"]),
           r2=float(m["r2"]), skill=float(m["skill_vs_persist"]),
           acc_tol=float(m["acc_tol"]), acc_tol_persist=float(mp["acc_tol"]),
           etat_acc_modele=float(accuracy_score(s_true, s_mod)),
           etat_acc_persist=float(accuracy_score(s_true, s_per)),
           etat_bal_modele=float(balanced_accuracy_score(s_true, s_mod)),
           etat_bal_persist=float(balanced_accuracy_score(s_true, s_per)),
           part_sans_mouvement=float(np.mean(np.abs(delta) <= TOLERANCE)),
           corr_delta=float(corr), var_delta_expliquee=float(corr ** 2))
with open(os.path.join(HERE, f"utilisabilite_{'masquee' if MASKED else 'actuelle'}.json"),
          "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\n  -> hi_forecast/utilisabilite_*.json")
