"""PHASE 3 — iterations 11 a 14 : correction SELECTIVE.

Diagnostic des iterations 6-9 : le alpha optimal en CV vaut 0 pour toutes les
familles, c'est-a-dire que corriger la persistance degrade la bande ±0.01. La
raison est mecanique : la persistance est deja juste sur ~79% des lignes
(|delta| reel <= 0.01) ; y ajouter une correction bruitee casse ces lignes-la
plus vite qu'elle ne rattrape les 21% restantes.

Consequence : il ne faut corriger QUE les lignes ou une vraie variation est
attendue. Trois mecanismes de porte sont testes, tous regles en CV :

  P1  porte sur l'amplitude : corriger si |delta_predit| > tau
  P2  porte par classifieur : corriger si P(|delta| > tolerance) > seuil
  P3  porte par regime      : corriger seulement en marche (ON), ou l'huile
                              circule et le health index bouge reellement

On mesure aussi l'AUC du classifieur : si l'on ne sait pas distinguer les lignes
qui vont bouger, aucune porte ne peut fonctionner, et c'est en soi le resultat.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import lightgbm as lgb

from harness import append_log, get_data, show_log
from protocol import (ASSET_COLS, N_SPLITS, SEED, TOLERANCE, PurgedTimeSeriesSplit,
                      regression_metrics)
from transforms import chain, drop_constant_and_dupes, prune_correlated

CLEAN = chain(drop_constant_and_dupes, prune_correlated(0.995))
d = get_data()
Xtr, ytr, mtr, Xte, yte, mte = d["Xtr"], d["ytr"], d["mtr"], d["Xte"], d["yte"], d["mte"]

TAUS = [0.0, 0.002, 0.004, 0.006, 0.008, 0.010, 0.015, 0.020, 0.030, 0.050]
PROBS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
ALPHAS = [0.25, 0.5, 0.75, 1.0]


def reg():
    return lgb.LGBMRegressor(objective="huber", alpha=0.005, n_estimators=500,
                             learning_rate=0.05, num_leaves=63, min_child_samples=40,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                             random_state=SEED, n_jobs=-1, verbose=-1)


def clf():
    return lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
                              min_child_samples=40, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.7, random_state=SEED, n_jobs=-1, verbose=-1)


print("=" * 118)
print("CONSTRUCTION DES PREDICTIONS HORS-PLI (regresseur + classifieur)")
print("=" * 118)
cv = PurgedTimeSeriesSplit(n_splits=N_SPLITS)
folds = list(cv.split(Xtr, times=mtr["created_at"]))
oof, aucs = [], []
t0 = time.time()
for k, (itr, iva) in enumerate(folds, 1):
    Xa, Xb = CLEAN(Xtr.iloc[itr], Xtr.iloc[iva])
    na, nb = mtr["hi_now"].iloc[itr].to_numpy(), mtr["hi_now"].iloc[iva].to_numpy()
    ya, yb = ytr.iloc[itr].to_numpy(), ytr.iloc[iva].to_numpy()
    off_b = Xb["state_OFF"].to_numpy() if "state_OFF" in Xb else np.zeros(len(Xb))

    pr = Pipeline([("imp", SimpleImputer(strategy="median")), ("m", reg())])
    pr.fit(Xa, ya - na)
    dhat = pr.predict(Xb)

    lab_a = (np.abs(ya - na) > TOLERANCE).astype(int)
    pc = Pipeline([("imp", SimpleImputer(strategy="median")), ("m", clf())])
    pc.fit(Xa, lab_a)
    phat = pc.predict_proba(Xb)[:, 1]

    lab_b = (np.abs(yb - nb) > TOLERANCE).astype(int)
    auc = roc_auc_score(lab_b, phat) if len(np.unique(lab_b)) > 1 else np.nan
    ap = average_precision_score(lab_b, phat) if len(np.unique(lab_b)) > 1 else np.nan
    aucs.append((auc, ap, lab_b.mean()))
    oof.append(dict(y=yb, n=nb, d=dhat, p=phat, off=off_b))
    print(f"  pli {k}: n_val={len(yb):>5}  part |delta|>tol = {lab_b.mean():.1%}  "
          f"AUC = {auc:.4f}  PR-AUC = {ap:.4f}")
print(f"  AUC moyenne = {np.nanmean([a for a, _, _ in aucs]):.4f}   [{time.time()-t0:.0f}s]")


def cv_acc(fn):
    return float(np.mean([np.mean(np.abs(o["y"] - fn(o)) <= TOLERANCE) for o in oof])), \
           float(np.std([np.mean(np.abs(o["y"] - fn(o)) <= TOLERANCE) for o in oof]))


base_acc, base_sd = cv_acc(lambda o: o["n"])
print(f"\n  persistance (reference CV) : {base_acc:.4f} ± {base_sd:.4f}")

print("\n" + "=" * 118)
print("P1 — porte sur l'amplitude : corriger si |delta_predit| > tau")
print("=" * 118)
best = {}
for a in ALPHAS:
    line = []
    for tau in TAUS:
        acc, _ = cv_acc(lambda o, t=tau, al=a: o["n"] + al * o["d"] * (np.abs(o["d"]) > t))
        line.append(acc); best[("P1", a, tau)] = acc
    print(f"  alpha={a:<5} " + "  ".join(f"t={t}:{v:.4f}" for t, v in zip(TAUS, line)))

print("\n" + "=" * 118)
print("P2 — porte par classifieur : corriger si P(|delta|>tol) > seuil")
print("=" * 118)
for a in ALPHAS:
    line = []
    for pth in PROBS:
        acc, _ = cv_acc(lambda o, p=pth, al=a: o["n"] + al * o["d"] * (o["p"] > p))
        line.append(acc); best[("P2", a, pth)] = acc
    print(f"  alpha={a:<5} " + "  ".join(f"p={p}:{v:.4f}" for p, v in zip(PROBS, line)))

print("\n" + "=" * 118)
print("P3 — porte par regime : corriger seulement en marche (state_OFF == 0)")
print("=" * 118)
for a in ALPHAS:
    line = []
    for tau in TAUS[:6]:
        acc, _ = cv_acc(lambda o, t=tau, al=a: o["n"] + al * o["d"] * ((np.abs(o["d"]) > t) & (o["off"] == 0)))
        line.append(acc); best[("P3", a, tau)] = acc
    print(f"  alpha={a:<5} " + "  ".join(f"t={t}:{v:.4f}" for t, v in zip(TAUS[:6], line)))

bk = max(best, key=best.get)
print(f"\n  >>> MEILLEURE PORTE EN CV : {bk[0]}  alpha={bk[1]}  seuil={bk[2]}  acc={best[bk]:.4f}"
      f"   (persistance = {base_acc:.4f}, gain = {best[bk]-base_acc:+.4f})")

pd.DataFrame([{"porte": k[0], "alpha": k[1], "seuil": k[2], "cv_acc": v}
              for k, v in best.items()]).to_csv(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_portes.csv"), index=False)

# ------------------------------------------------------------------ test gele
print("\n" + "=" * 118)
print("EVALUATION SUR LE TEST GELE — configuration de porte retenue en CV")
print("=" * 118)
Xa, Xb = CLEAN(Xtr, Xte)
na = mtr["hi_now"].to_numpy(); nb = mte["hi_now"].to_numpy()
pr = Pipeline([("imp", SimpleImputer(strategy="median")), ("m", reg())])
pr.fit(Xa, ytr.to_numpy() - na)
dte = pr.predict(Xb)
pc = Pipeline([("imp", SimpleImputer(strategy="median")), ("m", clf())])
pc.fit(Xa, (np.abs(ytr.to_numpy() - na) > TOLERANCE).astype(int))
pte = pc.predict_proba(Xb)[:, 1]
off_te = Xb["state_OFF"].to_numpy() if "state_OFF" in Xb else np.zeros(len(Xb))

lab_te = (np.abs(yte.to_numpy() - nb) > TOLERANCE).astype(int)
print(f"  AUC test du classifieur de mouvement : {roc_auc_score(lab_te, pte):.4f}"
      f"   PR-AUC : {average_precision_score(lab_te, pte):.4f}"
      f"   (part reelle |delta|>tol = {lab_te.mean():.1%})")

kind, alpha, thr = bk
if kind == "P1":
    gate = (np.abs(dte) > thr)
elif kind == "P2":
    gate = (pte > thr)
else:
    gate = (np.abs(dte) > thr) & (off_te == 0)
yhat = np.clip(nb + alpha * dte * gate, 0, 1)

test = regression_metrics(yte, yhat, Xa.shape[1], hi_now=nb)
cvsel = [regression_metrics(o["y"],
                            o["n"] + alpha * o["d"] * ((np.abs(o["d"]) > thr) if kind == "P1"
                                                       else (o["p"] > thr) if kind == "P2"
                                                       else ((np.abs(o["d"]) > thr) & (o["off"] == 0))),
                            hi_now=o["n"]) for o in oof]
cvdf = pd.DataFrame(cvsel)

row = {"iteration": 24, "name": f"Correction selective {kind} (a={alpha}, seuil={thr})",
       "change": f"Porte {kind} : correction appliquee seulement si le critere est franchi ; "
                 f"alpha={alpha}, seuil={thr}, regles en CV",
       "n_features": int(Xa.shape[1]), "delta_form": True,
       "cv_acc_tol_mean": cvdf["acc_tol"].mean(), "cv_acc_tol_std": cvdf["acc_tol"].std(),
       "cv_r2_mean": cvdf["r2"].mean(), "cv_r2_std": cvdf["r2"].std(),
       "cv_rmse_mean": cvdf["rmse"].mean(), "cv_mae_mean": cvdf["mae"].mean(),
       "cv_skill_mean": cvdf["skill_vs_persist"].mean(),
       **{f"test_{k}": v for k, v in test.items()},
       "gap_acc_tol": test["acc_tol"] - cvdf["acc_tol"].mean(),
       "gap_r2": test["r2"] - cvdf["r2"].mean(),
       "seconds": 0.0, "notes": f"AUC test classifieur = {roc_auc_score(lab_te, pte):.4f} ; "
                                f"part de lignes corrigees = {gate.mean():.1%}"}
for asset, col in ASSET_COLS.items():
    m = (mte[col] == 1).to_numpy()
    row[f"test_acc_{asset[-1]}"] = float(np.mean(np.abs(yte[m] - yhat[m]) <= TOLERANCE))
    row[f"test_r2_{asset[-1]}"] = float(regression_metrics(yte[m], yhat[m])["r2"])
append_log(row)

print(f"\n  CV   acc = {row['cv_acc_tol_mean']:.4f} ± {row['cv_acc_tol_std']:.4f}   R² = {row['cv_r2_mean']:.4f}")
print(f"  TEST acc = {row['test_acc_tol']:.4f}   R² = {row['test_r2']:.4f}   "
      f"RMSE = {row['test_rmse']:.5f}   skill = {row['test_skill_vs_persist']:+.4f}")
print(f"  lignes effectivement corrigees : {gate.mean():.1%}")
print(f"  per-machine acc : A={row['test_acc_A']:.4f}  B={row['test_acc_B']:.4f}")

print("\n" + "=" * 118)
show_log(sort_by="cv_acc_tol_mean")
