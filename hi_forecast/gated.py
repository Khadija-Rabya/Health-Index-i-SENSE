"""Architecture a porte, factorisee — reutilisee par les iterations suivantes.

    prediction = health_index[t] + alpha * delta_predit * 1[P(mouvement) > seuil]

Deux modeles : un regresseur de delta et un classifieur "le health index va-t-il
bouger de plus que la tolerance ?". La porte protege les ~79% de lignes ou la
persistance est deja juste.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from harness import get_data
from protocol import (ASSET_COLS, N_SPLITS, TOLERANCE, PurgedTimeSeriesSplit,
                      regression_metrics)

ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.0]
PROBS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]


def get_folds():
    d = get_data()
    cv = PurgedTimeSeriesSplit(n_splits=N_SPLITS)
    return d, list(cv.split(d["Xtr"], times=d["mtr"]["created_at"]))


def compute_oof(reg_fn, clf_fn, transform, d=None, folds=None):
    """Predictions hors-pli du regresseur et du classifieur, par pli."""
    if d is None or folds is None:
        d, folds = get_folds()
    Xtr, ytr, mtr = d["Xtr"], d["ytr"], d["mtr"]
    out = []
    for itr, iva in folds:
        Xa, Xb = transform(Xtr.iloc[itr], Xtr.iloc[iva])
        na = mtr["hi_now"].iloc[itr].to_numpy()
        nb = mtr["hi_now"].iloc[iva].to_numpy()
        ya = ytr.iloc[itr].to_numpy()
        yb = ytr.iloc[iva].to_numpy()
        r = reg_fn(); r.fit(Xa, ya - na)
        c = clf_fn(); c.fit(Xa, (np.abs(ya - na) > TOLERANCE).astype(int))
        out.append(dict(y=yb, n=nb, d=r.predict(Xb), p=c.predict_proba(Xb)[:, 1],
                        lab=(np.abs(yb - nb) > TOLERANCE).astype(int)))
    return out


def sweep_gate(oof, alphas=ALPHAS, probs=PROBS):
    """Balaye (alpha, seuil) sur les predictions hors-pli. Renvoie la grille
    triee et la meilleure combinaison, choisies EN CV uniquement."""
    rows = []
    for a in alphas:
        for p in probs:
            per_fold = [np.mean(np.abs(o["y"] - (o["n"] + a * o["d"] * (o["p"] > p))) <= TOLERANCE)
                        for o in oof]
            rows.append(dict(alpha=a, seuil=p, cv_acc=float(np.mean(per_fold)),
                             cv_std=float(np.std(per_fold))))
    g = pd.DataFrame(rows).sort_values("cv_acc", ascending=False).reset_index(drop=True)
    return g, g.iloc[0]


def cv_metrics(oof, alpha, prob):
    per_fold = [regression_metrics(o["y"], o["n"] + alpha * o["d"] * (o["p"] > prob), hi_now=o["n"])
                for o in oof]
    return pd.DataFrame(per_fold)


def fit_and_test(reg_fn, clf_fn, transform, alpha, prob, d=None):
    """Ajuste sur l'integralite du train, score UNE fois le test gele."""
    if d is None:
        d, _ = get_folds()
    Xtr, ytr, mtr = d["Xtr"], d["ytr"], d["mtr"]
    Xte, yte, mte = d["Xte"], d["yte"], d["mte"]
    Xa, Xb = transform(Xtr, Xte)
    na, nb = mtr["hi_now"].to_numpy(), mte["hi_now"].to_numpy()
    r = reg_fn(); r.fit(Xa, ytr.to_numpy() - na)
    c = clf_fn(); c.fit(Xa, (np.abs(ytr.to_numpy() - na) > TOLERANCE).astype(int))
    dte, pte = r.predict(Xb), c.predict_proba(Xb)[:, 1]
    gate = pte > prob
    yhat = np.clip(nb + alpha * dte * gate, 0.0, 1.0)
    return dict(yhat=yhat, delta=dte, prob=pte, gate=gate, reg=r, clf=c,
                n_features=int(Xa.shape[1]),
                metrics=regression_metrics(yte, yhat, Xa.shape[1], hi_now=nb))


def make_row(iteration, name, change, oof, alpha, prob, res, d=None, notes=""):
    if d is None:
        d, _ = get_folds()
    yte, mte = d["yte"], d["mte"]
    cvdf = cv_metrics(oof, alpha, prob)
    t = res["metrics"]
    row = {"iteration": iteration, "name": name, "change": change,
           "n_features": res["n_features"], "delta_form": True,
           "cv_acc_tol_mean": cvdf["acc_tol"].mean(), "cv_acc_tol_std": cvdf["acc_tol"].std(),
           "cv_r2_mean": cvdf["r2"].mean(), "cv_r2_std": cvdf["r2"].std(),
           "cv_rmse_mean": cvdf["rmse"].mean(), "cv_mae_mean": cvdf["mae"].mean(),
           "cv_skill_mean": cvdf["skill_vs_persist"].mean(),
           **{f"test_{k}": v for k, v in t.items()},
           "gap_acc_tol": t["acc_tol"] - cvdf["acc_tol"].mean(),
           "gap_r2": t["r2"] - cvdf["r2"].mean(), "seconds": 0.0,
           "notes": notes or f"alpha={alpha} seuil={prob} corrigees={res['gate'].mean():.1%}"}
    yhat = res["yhat"]
    for asset, col in ASSET_COLS.items():
        m = (mte[col] == 1).to_numpy()
        row[f"test_acc_{asset[-1]}"] = float(np.mean(np.abs(yte[m] - yhat[m]) <= TOLERANCE))
        row[f"test_r2_{asset[-1]}"] = float(regression_metrics(yte[m], yhat[m])["r2"])
    return row


def summarise(row, label=""):
    print(f"  {label:<38} CV acc={row['cv_acc_tol_mean']:.4f}±{row['cv_acc_tol_std']:.4f}"
          f"  TEST acc={row['test_acc_tol']:.4f}  R²={row['test_r2']:.4f}"
          f"  skill={row['test_skill_vs_persist']:+.4f}  gap={row['gap_acc_tol']:+.4f}")
