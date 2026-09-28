"""Diagnostic avant optimisation : d'ou vient l'instabilite enorme de la CV
(acc 0.52 ± 0.30, R² -3.91 ± 5.69) alors que le test gele donne 0.83 / R² 0.28 ?"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from harness import get_data
from protocol import (ASSET_COLS, N_SPLITS, SEED, TOLERANCE, PurgedTimeSeriesSplit,
                      regression_metrics)

d = get_data()
Xtr, ytr, mtr, Xte, yte, mte = d["Xtr"], d["ytr"], d["mtr"], d["Xte"], d["yte"], d["mte"]

print("=" * 105)
print("D1 — comportement pli par pli (RF d'origine, 150 arbres pour la vitesse)")
print("=" * 105)
cv = PurgedTimeSeriesSplit(n_splits=N_SPLITS)
print(f"{'fold':<6}{'n_tr':>7}{'n_va':>7}{'%A_tr':>7}{'%A_va':>7}"
      f"{'y_tr_mu':>9}{'y_va_mu':>9}{'y_va_sd':>9}{'acc':>8}{'R2':>10}{'R2_pers':>9}{'skill':>9}")
for k, (itr, iva) in enumerate(cv.split(Xtr, times=mtr["created_at"]), 1):
    Xa, Xb, ya, yb = Xtr.iloc[itr], Xtr.iloc[iva], ytr.iloc[itr], ytr.iloc[iva]
    na, nb = mtr["hi_now"].iloc[itr], mtr["hi_now"].iloc[iva]
    p = Pipeline([("imp", SimpleImputer(strategy="median")),
                  ("rf", RandomForestRegressor(n_estimators=150, max_depth=14,
                                               random_state=SEED, n_jobs=-1))])
    p.fit(Xa, ya - na)
    yhat = nb + p.predict(Xb)
    m = regression_metrics(yb, yhat, Xa.shape[1], hi_now=nb)
    pa_tr = mtr[ASSET_COLS["Motosoufflante A"]].iloc[itr].mean()
    pa_va = mtr[ASSET_COLS["Motosoufflante A"]].iloc[iva].mean()
    print(f"{k:<6}{len(itr):>7}{len(iva):>7}{pa_tr:>7.0%}{pa_va:>7.0%}"
          f"{ya.mean():>9.4f}{yb.mean():>9.4f}{yb.std():>9.4f}"
          f"{m['acc_tol']:>8.4f}{m['r2']:>10.4f}{m['r2_persist']:>9.4f}{m['skill_vs_persist']:>9.4f}")

print("\n" + "=" * 105)
print("D2 — repartition des machines dans le temps (explique-t-elle les plis ?)")
print("=" * 105)
tmp = mtr.copy()
tmp["month"] = tmp["created_at"].dt.to_period("M")
g = tmp.groupby("month").agg(n=("hi_now", "size"),
                             pct_A=(ASSET_COLS["Motosoufflante A"], "mean"),
                             hi_mu=("hi_now", "mean"))
g["y_mu"] = ytr.groupby(tmp["month"]).mean()
g["y_sd"] = ytr.groupby(tmp["month"]).std()
print(g.to_string(float_format=lambda v: f"{v:.4f}"))
print("\n  TEST :")
t2 = mte.copy(); t2["month"] = t2["created_at"].dt.to_period("M")
g2 = t2.groupby("month").agg(n=("hi_now", "size"), pct_A=(ASSET_COLS["Motosoufflante A"], "mean"))
g2["y_mu"] = yte.groupby(t2["month"]).mean(); g2["y_sd"] = yte.groupby(t2["month"]).std()
print(g2.to_string(float_format=lambda v: f"{v:.4f}"))

print("\n" + "=" * 105)
print("D3 — qualite des donnees : valeurs impossibles / extremes (avant toute suppression)")
print("=" * 105)
base = ["Oil Temperature_filled", "Oil H2O ppm_filled", "Oil H2O Saturation_filled",
        "Viscosity at 40°C_filled", "Kinematic Viscosity_filled", "Dynamic Viscosity_filled",
        "Density_filled", "ISO 4_filled", "ISO 6", "ISO 14", "DC_filled",
        "Oil Conductivity_nSm_filled", "Oil System Vibration_filled", "Oil Pressure"]
rows = []
for c in base:
    if c not in Xtr.columns:
        continue
    s = Xtr[c]
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    rows.append(dict(var=c, min=s.min(), p1=s.quantile(.01), med=s.median(),
                     p99=s.quantile(.99), max=s.max(), n_neg=int((s < 0).sum()),
                     n_out=int(((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).sum()),
                     pct_out=float(((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).mean())))
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

print("\n" + "=" * 105)
print("D4 — colonnes constantes / quasi-constantes / dupliquees dans le TRAIN")
print("=" * 105)
nun = Xtr.nunique()
const = list(nun[nun <= 1].index)
near = [c for c in Xtr.columns if c not in const and Xtr[c].value_counts(normalize=True).iloc[0] > 0.999]
print(f"  constantes ({len(const)}) : {const}")
print(f"  quasi-constantes >99.9% ({len(near)}) : {near}")
num = Xtr.drop(columns=const, errors="ignore")
corr = num.fillna(num.median()).corr().abs()
np.fill_diagonal(corr.values, 0)
pairs = (corr.stack().loc[lambda s: s > 0.999].reset_index()
         .rename(columns={"level_0": "a", "level_1": "b", 0: "r"}))
pairs = pairs[pairs["a"] < pairs["b"]]
print(f"  paires |r| > 0.999 ({len(pairs)}) :")
print(pairs.head(30).to_string(index=False, float_format=lambda v: f"{v:.5f}"))

print("\n" + "=" * 105)
print("D5 — distribution de la cible : train vs test (derive de regime ?)")
print("=" * 105)
print(f"  y_train : mean={ytr.mean():.4f} std={ytr.std():.4f} "
      f"q=[{ytr.quantile(.01):.4f}, {ytr.quantile(.5):.4f}, {ytr.quantile(.99):.4f}]")
print(f"  y_test  : mean={yte.mean():.4f} std={yte.std():.4f} "
      f"q=[{yte.quantile(.01):.4f}, {yte.quantile(.5):.4f}, {yte.quantile(.99):.4f}]")
for asset, col in ASSET_COLS.items():
    a, b = ytr[(mtr[col] == 1).to_numpy()], yte[(mte[col] == 1).to_numpy()]
    print(f"  {asset}: train mu={a.mean():.4f} sd={a.std():.4f} | test mu={b.mean():.4f} sd={b.std():.4f}")
delta_tr = (ytr - mtr["hi_now"]); delta_te = (yte - mte["hi_now"])
print(f"  delta train: mu={delta_tr.mean():+.5f} sd={delta_tr.std():.5f} "
      f"| test: mu={delta_te.mean():+.5f} sd={delta_te.std():.5f}")
print(f"  part de |delta| <= {TOLERANCE} : train={np.mean(np.abs(delta_tr) <= TOLERANCE):.4f} "
      f"test={np.mean(np.abs(delta_te) <= TOLERANCE):.4f}")
