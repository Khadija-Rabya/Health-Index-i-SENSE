"""PHASE 3 — iteration 27 : mise a l'echelle (levier 5).

Reprise de l'iteration 27 avec Ridge au lieu de Lasso comme regresseur-sonde :
meme sensibilite a l'echelle (modele lineaire regularise) pour ~35x moins de
temps de calcul. L'iteration 26 a montre que le choix d'imputation ne deplace
l'accuracy en CV que de 0.0019 (0.7908 a 0.7927) — on garde donc la mediane,
nettement moins couteuse que le KNN pour un ecart dans le bruit.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, PowerTransformer, QuantileTransformer,
                                   RobustScaler, StandardScaler)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import lightgbm as lgb

from gated import compute_oof, fit_and_test, get_folds, make_row, summarise, sweep_gate
from harness import append_log, show_log
from protocol import SEED
from transforms import chain, drop_constant_and_dupes, prune_correlated

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = chain(drop_constant_and_dupes, prune_correlated(0.995))
d, folds = get_folds()

SCALERS = {
    "aucune":      None,
    "standard":    StandardScaler,
    "robuste":     RobustScaler,
    "min-max":     MinMaxScaler,
    "quantile":    lambda: QuantileTransformer(output_distribution="normal", random_state=SEED),
    "yeo-johnson": lambda: PowerTransformer(method="yeo-johnson", standardize=True),
}


def clf_fn():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                              num_leaves=63, min_child_samples=40,
                                              subsample=0.8, subsample_freq=1,
                                              colsample_bytree=0.7, random_state=SEED,
                                              n_jobs=-1, verbose=-1))])


def reg_fn(scaler):
    steps = [("imp", SimpleImputer(strategy="median"))]
    if scaler is not None:
        steps.append(("sc", scaler()))
    steps.append(("m", Ridge(alpha=10.0, random_state=SEED)))
    return lambda: Pipeline(steps)


print("=" * 118)
print("ITERATION 27 — mise a l'echelle (regresseur-sonde Ridge, porte LGBM)")
print("=" * 118)
best, best_score, rows = None, -1, []
for sname, sfn in SCALERS.items():
    t0 = time.time()
    try:
        oof = compute_oof(reg_fn(sfn), clf_fn, CLEAN, d, folds)
        grid, b = sweep_gate(oof)
        rows.append(dict(echelle=sname, alpha=b["alpha"], seuil=b["seuil"],
                         cv_acc=b["cv_acc"], cv_std=b["cv_std"],
                         secondes=round(time.time() - t0, 1)))
        print(f"  {sname:<14} CV acc={b['cv_acc']:.4f}±{b['cv_std']:.4f}"
              f"  (alpha={b['alpha']}, seuil={b['seuil']})  [{time.time()-t0:>5.0f}s]")
        if b["cv_acc"] > best_score:
            best, best_score, best_cfg, best_oof, best_fn = sname, b["cv_acc"], b, oof, sfn
    except Exception as e:
        print(f"  {sname:<14} ECHEC : {type(e).__name__}: {e}")

pd.DataFrame(rows).to_csv(os.path.join(HERE, "sweep_echelle.csv"), index=False,
                          encoding="utf-8-sig")
print(f"\n  >>> retenu en CV : {best}  (acc={best_score:.4f})")
print("  Note : le modele finalement retenu (LGBM, a base d'arbres) est invariant "
      "par transformation monotone des features — ce levier ne joue que pour les "
      "familles lineaires / a noyau du classement.")

res = fit_and_test(reg_fn(best_fn), clf_fn, CLEAN, best_cfg["alpha"], best_cfg["seuil"], d)
row = make_row(27, f"Porte + echelle {best}",
               f"Comparaison de 6 mises a l'echelle (regresseur-sonde Ridge) ; "
               f"retenue : {best}", best_oof, best_cfg["alpha"], best_cfg["seuil"], res, d)
append_log(row); summarise(row, f"it27 {best}")

print("\n" + "=" * 118)
show_log(sort_by="cv_acc_tol_mean")
