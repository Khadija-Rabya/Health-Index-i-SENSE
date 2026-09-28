"""PHASE 2 — Baseline : le modele d'origine, re-evalue sous le protocole gele.

Produit trois chiffres a ne pas confondre :
  (a) "tel que publie"  : le RF d'origine, protocole d'origine (fuites incluses)
  (b) "persistance"     : y_pred = health_index[t], protocole gele
  (c) "baseline it00"   : le MEME RF d'origine, protocole gele sans fuites
L'ecart (a) - (c) mesure ce que les fuites ajoutaient artificiellement.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from harness import LOG_CSV, append_log, evaluate, get_data, persistence_reference, show_log
from protocol import ASSET_COLS, SEED, TOLERANCE

if os.path.exists(LOG_CSV):
    os.remove(LOG_CSV)

print("=" * 100)
print("(a) MODELE D'ORIGINE, PROTOCOLE D'ORIGINE (fuites incluses) — pour memoire, non comparable")
print("=" * 100)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from health_index_prediction_data import (  # noqa: E402
    build_dataset_for_horizon, load_data, temporal_train_test_split,
)
from sklearn.metrics import r2_score  # noqa: E402

df_old = load_data()
X_old, y_old, meta_old = build_dataset_for_horizon(df_old, 18)
imp = SimpleImputer(strategy="median")                      # <- ajuste AVANT le split (la fuite F5)
X_old_i = pd.DataFrame(imp.fit_transform(X_old), columns=X_old.columns, index=X_old.index)
for asset, col in ASSET_COLS.items():
    Xa, Xb, ya, yb = temporal_train_test_split(X_old_i, y_old, meta_old, col)
    m = RandomForestRegressor(n_estimators=300, max_depth=14, random_state=SEED, n_jobs=-1)
    m.fit(Xa, ya - Xa["health_index"])
    yp = Xb["health_index"] + m.predict(Xb)
    print(f"    {asset}: R² = {r2_score(yb, yp):.4f}   acc@±{TOLERANCE} = "
          f"{np.mean(np.abs(yb - yp) <= TOLERANCE):.4f}   (n={len(yb)})")

print("\n" + "=" * 100)
print("(b) PERSISTANCE — protocole gele, sans fuites")
print("=" * 100)
ref = persistence_reference()
append_log(ref)
print(f"    CV   acc@±{TOLERANCE} = {ref['cv_acc_tol_mean']:.4f} ± {ref['cv_acc_tol_std']:.4f}"
      f"   R² = {ref['cv_r2_mean']:.4f}")
print(f"    TEST acc@±{TOLERANCE} = {ref['test_acc_tol']:.4f}   R² = {ref['test_r2']:.4f}"
      f"   RMSE = {ref['test_rmse']:.5f}")
print(f"    per-machine acc: A={ref['test_acc_A']:.4f}  B={ref['test_acc_B']:.4f}")

print("\n" + "=" * 100)
print("(c) BASELINE it00 — RF d'origine (300 arbres, profondeur 14), protocole gele")
print("=" * 100)


def make_rf_original():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),          # ajuste PAR PLI desormais
        ("rf", RandomForestRegressor(n_estimators=300, max_depth=14,
                                     random_state=SEED, n_jobs=-1)),
    ])


evaluate(0, "RF d'origine (300, depth14)",
         "Baseline : hyperparametres d'origine, formulation delta, imputation mediane "
         "dans le pipeline, features corrigees des fuites F1-F4",
         make_rf_original, delta=True,
         notes="reference que toute iteration doit battre en CV")

print("\n" + "=" * 100)
print("JOURNAL")
print("=" * 100)
show_log()
