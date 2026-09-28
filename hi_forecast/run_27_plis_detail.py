"""Detail des 5 plis PAR MACHINE, pour le rapport : tailles, fenetres de dates,
et accuracy pli par pli du modele retenu (Lasso) contre un modele instable.
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

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from protocol import ASSET_COLS, N_SPLITS, SEED, TOLERANCE, PurgedTimeSeriesSplit
from run_16_ae_roleA import ALPHAS, PROBS, load, make_clf
from run_26_separe_complet import WalkForwardSplit

d = load(False)
mtr = d["mtr"]
MACH = {a: np.flatnonzero((mtr[c] == 1).to_numpy()) for a, c in ASSET_COLS.items()}


def lasso():
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("m", Lasso(alpha=1e-4, max_iter=5000, random_state=SEED))])


def rf():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", RandomForestRegressor(n_estimators=300, max_depth=14,
                                                 random_state=SEED, n_jobs=-1))])


OUT = {}
for asset, tr in MACH.items():
    print("=" * 96)
    print(f"{asset} — {len(tr)} lignes d'entrainement")
    print("=" * 96)
    times = mtr["created_at"].iloc[tr]
    folds = list(PurgedTimeSeriesSplit(n_splits=N_SPLITS).split(
        np.zeros((len(tr), 1)), times=times))

    struct, acc_lasso, acc_rf = [], [], []
    for k, (a, b) in enumerate(folds, 1):
        ia, ib = tr[a], tr[b]
        t0 = mtr["created_at"].iloc[ib].min()
        t1 = mtr["created_at"].iloc[ib].max()
        Fa, Fb = d["Ttr"].to_numpy()[ia], d["Ttr"].to_numpy()[ib]
        ya, na = d["ytr"][ia], d["ntr"][ia]
        yb, nb = d["ytr"][ib], d["ntr"][ib]
        c = make_clf(); c.fit(Fa, (np.abs(ya - na) > TOLERANCE).astype(int))
        p = c.predict_proba(Fb)[:, 1]
        row = dict(pli=k, n_train=int(len(ia)), n_val=int(len(ib)),
                   debut=t0.strftime("%d/%m/%Y"), fin=t1.strftime("%d/%m/%Y"))
        for nom, mk, acc in [("lasso", lasso, acc_lasso), ("rf", rf, acc_rf)]:
            m = mk(); m.fit(Fa, ya - na)
            dd = m.predict(Fb)
            best = max(((al, pt, np.mean(np.abs(yb - (nb + al * dd * (p > pt))) <= TOLERANCE))
                        for al in ALPHAS for pt in PROBS), key=lambda x: x[2])
            acc.append(float(best[2])); row[f"acc_{nom}"] = float(best[2])
        struct.append(row)
        print(f"  pli {k} : train {len(ia):>6}  validation {len(ib):>5}  "
              f"{row['debut']} -> {row['fin']}  |  Lasso {row['acc_lasso']:.4f}  "
              f"RF {row['acc_rf']:.4f}")

    print(f"\n  Lasso : moyenne {np.mean(acc_lasso):.4f}  ecart-type {np.std(acc_lasso):.4f}")
    print(f"  RF    : moyenne {np.mean(acc_rf):.4f}  ecart-type {np.std(acc_rf):.4f}")
    OUT[asset] = dict(plis=struct,
                      lasso_moy=float(np.mean(acc_lasso)), lasso_sd=float(np.std(acc_lasso)),
                      rf_moy=float(np.mean(acc_rf)), rf_sd=float(np.std(acc_rf)),
                      n_train=int(len(tr)))

with open(os.path.join(HERE, "plis_detail.json"), "w", encoding="utf-8") as f:
    json.dump(OUT, f, indent=2, ensure_ascii=False)
print(f"\n  -> hi_forecast/plis_detail.json")
