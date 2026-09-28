"""PHASE 3 — iterations 6 a 9 : les deux leviers dictes par la metrique.

Constat des iterations 1-5 : tout modele appris perd contre la persistance en CV
(0.50 vs 0.79). La cible `delta = hi[t+18] - hi[t]` est concentree autour de 0 ;
une perte quadratique fait courir le modele apres les gros deltas rares et lui
fait rater les nombreux petits, ce que la bande de tolerance ±0.01 sanctionne
lourdement.

Deux corrections :
  (A) RETRECISSEMENT  : pred = hi_now + alpha * delta_predit, alpha choisi en CV.
  (B) PERTE ROBUSTE   : entrainer sur l'erreur absolue (mediane conditionnelle,
                        proche de 0) plutot que quadratique (moyenne).
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import lightgbm as lgb

from harness import evaluate, get_data, show_log
from protocol import (N_SPLITS, SEED, TOLERANCE, PurgedTimeSeriesSplit,
                      regression_metrics)
from transforms import (ShrunkRegressor, chain, drop_constant_and_dupes,
                        prune_correlated)

CLEAN = chain(drop_constant_and_dupes, prune_correlated(0.995))
d = get_data()
Xtr, ytr, mtr = d["Xtr"], d["ytr"], d["mtr"]

ALPHAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]

CANDIDATES = {
    "RF (mse, sqrt)": lambda: RandomForestRegressor(
        n_estimators=300, max_depth=14, max_features="sqrt", random_state=SEED, n_jobs=-1),
    "LGBM l2": lambda: lgb.LGBMRegressor(
        objective="l2", n_estimators=400, learning_rate=0.05, num_leaves=31,
        random_state=SEED, n_jobs=-1, verbose=-1),
    "LGBM l1 (MAE)": lambda: lgb.LGBMRegressor(
        objective="l1", n_estimators=400, learning_rate=0.05, num_leaves=31,
        random_state=SEED, n_jobs=-1, verbose=-1),
    "LGBM huber": lambda: lgb.LGBMRegressor(
        objective="huber", alpha=0.005, n_estimators=400, learning_rate=0.05,
        num_leaves=31, random_state=SEED, n_jobs=-1, verbose=-1),
    "HistGB absolute_error": lambda: HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=400, learning_rate=0.05, random_state=SEED),
}

print("=" * 112)
print("BALAYAGE : perte d'entrainement x facteur de retrecissement alpha — SELECTION EN CV UNIQUEMENT")
print("=" * 112)
print(f"  reference persistance (= alpha 0) : CV acc = 0.7912")
print()

cv = PurgedTimeSeriesSplit(n_splits=N_SPLITS)
folds = list(cv.split(Xtr, times=mtr["created_at"]))
results = {}

for cname, cfn in CANDIDATES.items():
    t0 = time.time()
    oof = []                       # (y_true, hi_now, delta_pred) par pli
    for itr, iva in folds:
        Xa, Xb = CLEAN(Xtr.iloc[itr], Xtr.iloc[iva])
        na, nb = mtr["hi_now"].iloc[itr], mtr["hi_now"].iloc[iva]
        ya, yb = ytr.iloc[itr], ytr.iloc[iva]
        p = Pipeline([("imp", SimpleImputer(strategy="median")), ("m", cfn())])
        p.fit(Xa, ya - na)
        oof.append((yb.to_numpy(), nb.to_numpy(), p.predict(Xb)))

    accs = {}
    for a in ALPHAS:
        per_fold = [np.mean(np.abs(yb - (nb + a * dp)) <= TOLERANCE) for yb, nb, dp in oof]
        accs[a] = (float(np.mean(per_fold)), float(np.std(per_fold)))
    results[cname] = accs
    best_a = max(accs, key=lambda k: accs[k][0])
    print(f"  {cname:<24} [{time.time()-t0:>5.0f}s]  " +
          "  ".join(f"a={a}:{accs[a][0]:.3f}" for a in ALPHAS))
    print(f"  {'':<24}          -> meilleur alpha = {best_a}  "
          f"CV acc = {accs[best_a][0]:.4f} ± {accs[best_a][1]:.4f}")

best_pair = max(((c, a) for c in results for a in ALPHAS),
                key=lambda ca: results[ca[0]][ca[1]][0])
print(f"\n  >>> MEILLEURE COMBINAISON EN CV : {best_pair[0]}  alpha={best_pair[1]}  "
      f"acc={results[best_pair[0]][best_pair[1]][0]:.4f}")

pd.DataFrame({c: {a: v[0] for a, v in r.items()} for c, r in results.items()}).to_csv(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_alpha.csv"))

# ------------------------------------------------- iterations journalisees
print("\n" + "=" * 112)
print("EVALUATION SUR LE TEST GELE DES CONFIGURATIONS RETENUES EN CV")
print("=" * 112)


def shrunk(model_fn, alpha):
    return lambda: Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("m", ShrunkRegressor(model_fn(), alpha=alpha))])


it = 6
for cname in ["LGBM l1 (MAE)", "LGBM huber", "HistGB absolute_error", "RF (mse, sqrt)"]:
    a = max(ALPHAS, key=lambda k: results[cname][k][0])
    evaluate(it, f"{cname} + shrink a={a}",
             f"Perte robuste + retrecissement alpha={a} choisi en CV "
             f"(balayage {ALPHAS})",
             shrunk(CANDIDATES[cname], a), transform=CLEAN, delta=True,
             notes=f"CV acc du balayage = {results[cname][a][0]:.4f}")
    it += 1

print("\n" + "=" * 118)
print("JOURNAL (trie par CV acc)")
print("=" * 118)
show_log(sort_by="cv_acc_tol_mean")
