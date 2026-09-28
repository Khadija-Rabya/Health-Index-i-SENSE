"""Suite du diagnostic : redondance de features, valeurs physiquement suspectes,
et test de l'hypothese principale (le melange des 2 machines casse la CV)."""
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

print("=" * 100)
print("D4 — colonnes constantes et paires redondantes (|r| > 0.999) dans le TRAIN")
print("=" * 100)
nun = Xtr.nunique()
const = list(nun[nun <= 1].index)
print(f"  constantes ({len(const)}) : {const}")
num = Xtr.drop(columns=const, errors="ignore")
c = num.fillna(num.median()).corr().abs().to_numpy(copy=True)
np.fill_diagonal(c, 0.0)
cols = list(num.columns)
iu = np.triu_indices_from(c, k=1)
hits = [(cols[i], cols[j], c[i, j]) for i, j in zip(*iu) if c[i, j] > 0.999]
print(f"  paires |r| > 0.999 : {len(hits)}")
for a, b, r in hits[:40]:
    print(f"    {r:.5f}  {a}  <->  {b}")

print("\n" + "=" * 100)
print("D5 — valeurs physiquement suspectes : viscosite tres basse pour une ISO VG 46")
print("=" * 100)
v40 = Xtr["Viscosity at 40°C_filled"]
low = v40 < 20
print(f"  lignes avec Viscosity@40C < 20 cSt : {int(low.sum())} ({low.mean():.2%})")
if low.sum():
    print(f"    part en etat OFF        : {Xtr.loc[low, 'state_OFF'].mean():.1%} "
          f"(vs {Xtr['state_OFF'].mean():.1%} globalement)")
    print(f"    part machine A          : {mtr.loc[low.to_numpy(), ASSET_COLS['Motosoufflante A']].mean():.1%}")
    print(f"    Oil Pressure mediane    : {Xtr.loc[low, 'Oil Pressure'].median():.3f} "
          f"(vs {Xtr['Oil Pressure'].median():.3f})")
    print(f"    temperature mediane     : {Xtr.loc[low, 'Oil Temperature_filled'].median():.2f} "
          f"(vs {Xtr['Oil Temperature_filled'].median():.2f})")
    print(f"    |delta| median sur ces lignes : "
          f"{(ytr - mtr['hi_now'])[low.to_numpy()].abs().median():.5f} "
          f"(vs {(ytr - mtr['hi_now']).abs().median():.5f})")

print("\n  Etat machine (ON/OFF) et difficulte de prediction :")
for st, lab in [(1, "OFF"), (0, "ON ")]:
    m = (Xtr["state_OFF"] == st).to_numpy()
    dd = (ytr - mtr["hi_now"])[m]
    print(f"    {lab} : n={m.sum():>6} ({m.mean():>5.1%})  |delta| median={dd.abs().median():.5f}  "
          f"sd(delta)={dd.std():.5f}  part |delta|<=tol={np.mean(np.abs(dd) <= TOLERANCE):.1%}")

print("\n" + "=" * 100)
print("D6 — HYPOTHESE PRINCIPALE : un modele POOLE vs un modele PAR MACHINE")
print("=" * 100)


def rf():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("rf", RandomForestRegressor(n_estimators=150, max_depth=14,
                                                  random_state=SEED, n_jobs=-1))])


for asset, col in ASSET_COLS.items():
    sel_tr = (mtr[col] == 1).to_numpy()
    Xa_all, ya_all, ma_all = Xtr[sel_tr], ytr[sel_tr], mtr[sel_tr].reset_index(drop=True)
    Xa_all = Xa_all.reset_index(drop=True); ya_all = ya_all.reset_index(drop=True)
    cv = PurgedTimeSeriesSplit(n_splits=N_SPLITS)
    accs, r2s, sk = [], [], []
    for itr, iva in cv.split(Xa_all, times=ma_all["created_at"]):
        if len(itr) < 100:
            accs.append(np.nan); r2s.append(np.nan); sk.append(np.nan); continue
        na, nb = ma_all["hi_now"].iloc[itr], ma_all["hi_now"].iloc[iva]
        p = rf(); p.fit(Xa_all.iloc[itr], ya_all.iloc[itr] - na)
        yhat = nb + p.predict(Xa_all.iloc[iva])
        m = regression_metrics(ya_all.iloc[iva], yhat, hi_now=nb)
        accs.append(m["acc_tol"]); r2s.append(m["r2"]); sk.append(m["skill_vs_persist"])
    print(f"  {asset} — CV PAR MACHINE :")
    print(f"    acc   par pli : {['%.3f' % a for a in accs]}  -> {np.nanmean(accs):.4f} ± {np.nanstd(accs):.4f}")
    print(f"    R²    par pli : {['%.3f' % a for a in r2s]}  -> {np.nanmean(r2s):.4f}")
    print(f"    skill par pli : {['%.3f' % a for a in sk]}  -> {np.nanmean(sk):.4f}")

print("\n" + "=" * 100)
print("D7 — combien de lignes du TEST viennent de sessions vues en TRAIN ?")
print("=" * 100)
s_tr, s_te = set(mtr["session_id"]), set(mte["session_id"])
both = s_tr & s_te
n_rows = int(mte["session_id"].isin(both).sum())
print(f"  sessions communes train/test : {len(both)}  ->  {n_rows} lignes de test ({n_rows/len(mte):.2%})")
print(f"  (le decoupage etant purement temporel, ces lignes sont posterieures a la coupure ;")
print(f"   elles restent legitimes mais sont les plus correlees au train)")
