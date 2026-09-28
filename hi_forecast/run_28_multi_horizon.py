"""Modeles MULTI-HORIZONS par machine, pour le tableau de bord.

Le tableau de bord doit afficher quatre horizons : 10 min (t+1), 20 min (t+2),
3 h (t+18) et 24 h (t+144). Seul t+18 etait entraine. On entraine ici les quatre,
avec la
configuration retenue (Lasso + porte), separement pour chaque machine, sous le
MEME protocole gele : CV 5 blocs purges + embargo proportionnel a l'horizon,
selection de (alpha, seuil) en CV, test score une seule fois.

L'embargo suit l'horizon : 10 min pour t+1, 20 min pour t+2, 3 h pour t+18,
24 h pour t+144.
Sans cela, la purge serait insuffisante aux horizons longs.
"""
import json
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "C:\\pylib")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from protocol import (ASSET_COLS, N_SPLITS, SEED, TOLERANCE, PurgedTimeSeriesSplit,
                      build_xy, feature_columns, prepare, regression_metrics,
                      split_frozen)
from run_16_ae_roleA import ALPHAS, PROBS, TOP20, make_clf

HORIZONS = {1: "10 min", 2: "20 min", 18: "3 h", 144: "24 h"}
ART = os.path.join(HERE, "artifacts")


def lasso():
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("m", Lasso(alpha=1e-4, max_iter=5000, random_state=SEED))])


print("=" * 96)
print("MODELES MULTI-HORIZONS PAR MACHINE — Lasso + porte")
print("=" * 96)
df, _, cutoffs, _ = prepare(verbose=False, mask_off_viscosity=False)
BUNDLE, METRICS = {}, []

for h, label in HORIZONS.items():
    X, y, meta = build_xy(df, h)
    Xtr, Xte, ytr, yte, mtr, mte = split_frozen(X, y, meta, cutoffs)
    cols = [c for c in TOP20 if c in Xtr.columns]
    Ttr, Tte = Xtr[cols].to_numpy(), Xte[cols].to_numpy()
    ntr, nte = mtr["hi_now"].to_numpy(), mte["hi_now"].to_numpy()
    ytr_a, yte_a = ytr.to_numpy(), yte.to_numpy()
    emb = pd.Timedelta(minutes=10 * h)

    print(f"\n### Horizon t+{h} ({label})  —  train {len(Xtr)}, test {len(Xte)}, "
          f"embargo {emb}")
    for asset, col in ASSET_COLS.items():
        tr = np.flatnonzero((mtr[col] == 1).to_numpy())
        te = np.flatnonzero((mte[col] == 1).to_numpy())
        if len(tr) < 500 or len(te) < 50:
            print(f"  {asset} : trop peu de donnees — ECARTE")
            continue
        t0 = time.time()

        cv = PurgedTimeSeriesSplit(n_splits=N_SPLITS, horizon_td=emb, embargo_td=emb)
        oof = []
        for a, b in cv.split(np.zeros((len(tr), 1)), times=mtr["created_at"].iloc[tr]):
            ia, ib = tr[a], tr[b]
            if len(ia) < 100:
                continue
            r = lasso(); r.fit(Ttr[ia], ytr_a[ia] - ntr[ia])
            c = make_clf(); c.fit(Ttr[ia], (np.abs(ytr_a[ia] - ntr[ia]) > TOLERANCE).astype(int))
            oof.append(dict(y=ytr_a[ib], n=ntr[ib], d=r.predict(Ttr[ib]),
                            p=c.predict_proba(Ttr[ib])[:, 1]))
        best = None
        for al in ALPHAS:
            for pt in PROBS:
                accs = [np.mean(np.abs(o["y"] - (o["n"] + al * o["d"] * (o["p"] > pt))) <= TOLERANCE)
                        for o in oof]
                s = float(np.mean(accs))
                if best is None or s > best["cv_acc"]:
                    best = dict(alpha=al, seuil=pt, cv_acc=s, cv_sd=float(np.std(accs)))

        r = lasso(); r.fit(Ttr[tr], ytr_a[tr] - ntr[tr])
        c = make_clf(); c.fit(Ttr[tr], (np.abs(ytr_a[tr] - ntr[tr]) > TOLERANCE).astype(int))
        dte = r.predict(Tte[te]); pte = c.predict_proba(Tte[te])[:, 1]
        yhat = np.clip(nte[te] + best["alpha"] * dte * (pte > best["seuil"]), 0, 1)
        m = regression_metrics(yte_a[te], yhat, n_features=len(cols), hi_now=nte[te])

        BUNDLE[f"{h}|{asset}"] = dict(horizon=h, horizon_label=label, machine=asset,
                                      regresseur=r, porte=c, alpha=best["alpha"],
                                      seuil=best["seuil"], colonnes=cols,
                                      n_train=int(len(tr)))
        METRICS.append(dict(horizon=h, horizon_label=label, machine=asset,
                            cv_acc=best["cv_acc"], cv_sd=best["cv_sd"],
                            alpha=best["alpha"], seuil=best["seuil"],
                            test_acc=m["acc_tol"], acc_persist=m["acc_tol_persist"],
                            r2=m["r2"], rmse=m["rmse"], mae=m["mae"],
                            skill=m["skill_vs_persist"], n_test=int(len(te))))
        print(f"  {asset:<20} CV {best['cv_acc']:.4f}±{best['cv_sd']:.4f}  "
              f"test acc {m['acc_tol']:.4f} (pers {m['acc_tol_persist']:.4f})  "
              f"R² {m['r2']:.4f}  skill {m['skill_vs_persist']:+.4f}  [{time.time()-t0:.0f}s]")

p = os.path.join(ART, "modeles_multi_horizon.joblib")
joblib.dump(BUNDLE, p, compress=3)
pd.DataFrame(METRICS).to_csv(os.path.join(HERE, "multi_horizon_metriques.csv"),
                             index=False, encoding="utf-8-sig")
with open(os.path.join(ART, "multi_horizon_metriques.json"), "w", encoding="utf-8") as f:
    json.dump(METRICS, f, indent=2, ensure_ascii=False, default=float)
print(f"\n  artefact : {p}  ({os.path.getsize(p)/1e6:.2f} Mo)  — {len(BUNDLE)} modeles")
print(f"  -> hi_forecast/multi_horizon_metriques.csv")
