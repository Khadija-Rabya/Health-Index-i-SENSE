"""PHASE 3 — iterations 25 a 28 : reglage de l'architecture a porte.

La porte P2 (corriger seulement si le classifieur juge un mouvement probable)
est la premiere configuration a battre la persistance EN CV. On l'optimise ici
sur ses quatre degres de liberte, tous regles en CV :

  - le REGRESSEUR de delta        (6 familles)
  - le CLASSIFIEUR de mouvement   (brut, Platt/sigmoide, isotonique)
  - le facteur de retrecissement  alpha
  - le seuil de probabilite       de la porte

Les predictions hors-pli sont calculees UNE fois par modele, puis la grille de
portes est balayee dessus : le cout est celui des modeles, pas de la grille.

La calibration du classifieur est un levier a part entiere ici : on seuille sa
probabilite, donc une probabilite mal calibree deplace la porte. Score de Brier
et courbe de fiabilite sont rapportes avant/apres.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb

from harness import append_log, get_data, show_log
from protocol import (ASSET_COLS, N_SPLITS, SEED, TOLERANCE, PurgedTimeSeriesSplit,
                      regression_metrics)
from transforms import chain, drop_constant_and_dupes, prune_correlated

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = chain(drop_constant_and_dupes, prune_correlated(0.995))
d = get_data()
Xtr, ytr, mtr, Xte, yte, mte = d["Xtr"], d["ytr"], d["mtr"], d["Xte"], d["yte"], d["mte"]

ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.0]
PROBS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]


def sc(*steps):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler())] + list(steps))


def ns(*steps):
    return Pipeline([("imp", SimpleImputer(strategy="median"))] + list(steps))


REGRESSORS = {
    "Lasso":       lambda: sc(("m", Lasso(alpha=1e-4, max_iter=5000, random_state=SEED))),
    "ElasticNet":  lambda: sc(("m", ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=5000, random_state=SEED))),
    "Ridge":       lambda: sc(("m", Ridge(alpha=10.0, random_state=SEED))),
    "RF sqrt":     lambda: ns(("m", RandomForestRegressor(n_estimators=300, max_depth=14, max_features="sqrt", random_state=SEED, n_jobs=-1))),
    "ExtraTrees":  lambda: ns(("m", ExtraTreesRegressor(n_estimators=300, max_depth=14, max_features="sqrt", random_state=SEED, n_jobs=-1))),
    "LGBM huber":  lambda: ns(("m", lgb.LGBMRegressor(objective="huber", alpha=0.005, n_estimators=500, learning_rate=0.05, num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, random_state=SEED, n_jobs=-1, verbose=-1))),
}


def base_clf():
    return lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
                              min_child_samples=40, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.7, random_state=SEED, n_jobs=-1, verbose=-1)


CLASSIFIERS = {
    "brut":        lambda: ns(("m", base_clf())),
    "Platt":       lambda: ns(("m", CalibratedClassifierCV(base_clf(), method="sigmoid", cv=3))),
    "isotonique":  lambda: ns(("m", CalibratedClassifierCV(base_clf(), method="isotonic", cv=3))),
}

cv = PurgedTimeSeriesSplit(n_splits=N_SPLITS)
folds = list(cv.split(Xtr, times=mtr["created_at"]))

# --------------------------------------------------------- predictions hors-pli
print("=" * 118)
print("PREDICTIONS HORS-PLI")
print("=" * 118)
oof_common = [dict(y=ytr.iloc[iva].to_numpy(), n=mtr["hi_now"].iloc[iva].to_numpy())
              for _, iva in folds]
for o in oof_common:
    o["lab"] = (np.abs(o["y"] - o["n"]) > TOLERANCE).astype(int)

reg_oof = {}
for rname, rfn in REGRESSORS.items():
    t0 = time.time(); preds = []
    for itr, iva in folds:
        Xa, Xb = CLEAN(Xtr.iloc[itr], Xtr.iloc[iva])
        p = rfn(); p.fit(Xa, ytr.iloc[itr].to_numpy() - mtr["hi_now"].iloc[itr].to_numpy())
        preds.append(p.predict(Xb))
    reg_oof[rname] = preds
    print(f"  regresseur {rname:<12} [{time.time()-t0:>5.0f}s]")

clf_oof, clf_quality = {}, {}
for cname, cfn in CLASSIFIERS.items():
    t0 = time.time(); probs = []
    for itr, iva in folds:
        Xa, Xb = CLEAN(Xtr.iloc[itr], Xtr.iloc[iva])
        lab = (np.abs(ytr.iloc[itr].to_numpy() - mtr["hi_now"].iloc[itr].to_numpy()) > TOLERANCE).astype(int)
        p = cfn(); p.fit(Xa, lab)
        probs.append(p.predict_proba(Xb)[:, 1])
    clf_oof[cname] = probs
    yl = np.concatenate([o["lab"] for o in oof_common]); pl = np.concatenate(probs)
    clf_quality[cname] = dict(auc=roc_auc_score(yl, pl), ap=average_precision_score(yl, pl),
                              brier=brier_score_loss(yl, pl),
                              logloss=log_loss(yl, np.clip(pl, 1e-6, 1 - 1e-6)))
    q = clf_quality[cname]
    print(f"  classifieur {cname:<11} [{time.time()-t0:>5.0f}s]  AUC={q['auc']:.4f}  "
          f"PR-AUC={q['ap']:.4f}  Brier={q['brier']:.5f}  logloss={q['logloss']:.5f}")

# --------------------------------------------------------- courbes de fiabilite
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="parfaitement calibre")
for cname in CLASSIFIERS:
    yl = np.concatenate([o["lab"] for o in oof_common])
    pl = np.concatenate(clf_oof[cname])
    fr, mp = calibration_curve(yl, pl, n_bins=15, strategy="quantile")
    ax[0].plot(mp, fr, marker="o", ms=4, lw=1.4,
               label=f"{cname} (Brier={clf_quality[cname]['brier']:.4f})")
    ax[1].hist(pl, bins=40, histtype="step", lw=1.4, label=cname)
ax[0].set_xlabel("Probabilite predite"); ax[0].set_ylabel("Frequence observee")
ax[0].set_title("Courbe de fiabilite — classifieur de mouvement (hors-pli)")
ax[0].legend(fontsize=8)
ax[1].set_xlabel("Probabilite predite"); ax[1].set_ylabel("Effectif")
ax[1].set_title("Distribution des probabilites"); ax[1].legend(fontsize=8)
fig.tight_layout()
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
fig.savefig(os.path.join(HERE, "figures", "02_calibration.png"), dpi=120); plt.close(fig)

# --------------------------------------------------------- balayage de la porte
print("\n" + "=" * 118)
print("BALAYAGE DE LA PORTE (regresseur x calibration x alpha x seuil) — CV UNIQUEMENT")
print("=" * 118)
base_acc = float(np.mean([np.mean(np.abs(o["y"] - o["n"]) <= TOLERANCE) for o in oof_common]))
rows = []
for rname, rpred in reg_oof.items():
    for cname, cprob in clf_oof.items():
        for a in ALPHAS:
            for pth in PROBS:
                per_fold = [np.mean(np.abs(o["y"] - (o["n"] + a * dp * (pp > pth))) <= TOLERANCE)
                            for o, dp, pp in zip(oof_common, rpred, cprob)]
                rows.append(dict(regresseur=rname, calibration=cname, alpha=a, seuil=pth,
                                 cv_acc=float(np.mean(per_fold)), cv_std=float(np.std(per_fold))))
grid = pd.DataFrame(rows).sort_values("cv_acc", ascending=False).reset_index(drop=True)
grid.to_csv(os.path.join(HERE, "sweep_porte_complet.csv"), index=False)
print(f"  persistance (CV) = {base_acc:.4f}\n")
print(grid.head(15).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

best = grid.iloc[0]
print(f"\n  >>> RETENU EN CV : {best['regresseur']} + calibration {best['calibration']}, "
      f"alpha={best['alpha']}, seuil={best['seuil']}  ->  CV acc={best['cv_acc']:.4f} "
      f"(persistance {base_acc:.4f}, gain {best['cv_acc']-base_acc:+.4f})")

# --------------------------------------------------------------- test gele
print("\n" + "=" * 118)
print("TEST GELE — configuration retenue")
print("=" * 118)
Xa, Xb = CLEAN(Xtr, Xte)
na, nb = mtr["hi_now"].to_numpy(), mte["hi_now"].to_numpy()
rp = REGRESSORS[best["regresseur"]](); rp.fit(Xa, ytr.to_numpy() - na)
dte = rp.predict(Xb)
cp = CLASSIFIERS[best["calibration"]](); cp.fit(Xa, (np.abs(ytr.to_numpy() - na) > TOLERANCE).astype(int))
pte = cp.predict_proba(Xb)[:, 1]
gate = pte > best["seuil"]
yhat = np.clip(nb + best["alpha"] * dte * gate, 0, 1)

lab_te = (np.abs(yte.to_numpy() - nb) > TOLERANCE).astype(int)
test = regression_metrics(yte, yhat, Xa.shape[1], hi_now=nb)
sel_cv = [regression_metrics(o["y"], o["n"] + best["alpha"] * dp * (pp > best["seuil"]), hi_now=o["n"])
          for o, dp, pp in zip(oof_common, reg_oof[best["regresseur"]], clf_oof[best["calibration"]])]
cvdf = pd.DataFrame(sel_cv)

row = {"iteration": 25,
       "name": f"Porte P2 reglee : {best['regresseur']} + {best['calibration']}",
       "change": f"Correction selective ; regresseur={best['regresseur']}, "
                 f"calibration={best['calibration']}, alpha={best['alpha']}, seuil={best['seuil']} "
                 f"(grille de {len(grid)} combinaisons, choisie en CV)",
       "n_features": int(Xa.shape[1]), "delta_form": True,
       "cv_acc_tol_mean": cvdf["acc_tol"].mean(), "cv_acc_tol_std": cvdf["acc_tol"].std(),
       "cv_r2_mean": cvdf["r2"].mean(), "cv_r2_std": cvdf["r2"].std(),
       "cv_rmse_mean": cvdf["rmse"].mean(), "cv_mae_mean": cvdf["mae"].mean(),
       "cv_skill_mean": cvdf["skill_vs_persist"].mean(),
       **{f"test_{k}": v for k, v in test.items()},
       "gap_acc_tol": test["acc_tol"] - cvdf["acc_tol"].mean(),
       "gap_r2": test["r2"] - cvdf["r2"].mean(), "seconds": 0.0,
       "notes": f"AUC test={roc_auc_score(lab_te, pte):.4f} Brier={brier_score_loss(lab_te, pte):.5f} "
                f"lignes corrigees={gate.mean():.1%}"}
for asset, col in ASSET_COLS.items():
    m = (mte[col] == 1).to_numpy()
    row[f"test_acc_{asset[-1]}"] = float(np.mean(np.abs(yte[m] - yhat[m]) <= TOLERANCE))
    row[f"test_r2_{asset[-1]}"] = float(regression_metrics(yte[m], yhat[m])["r2"])
append_log(row)

print(f"  CV   acc = {row['cv_acc_tol_mean']:.4f} ± {row['cv_acc_tol_std']:.4f}  R² = {row['cv_r2_mean']:.4f}")
print(f"  TEST acc = {row['test_acc_tol']:.4f}  R² = {row['test_r2']:.4f}  RMSE = {row['test_rmse']:.5f}"
      f"  skill = {row['test_skill_vs_persist']:+.4f}")
print(f"  gap = {row['gap_acc_tol']:+.4f}   lignes corrigees = {gate.mean():.1%}")
print(f"  per-machine acc : A={row['test_acc_A']:.4f}  B={row['test_acc_B']:.4f}")
print(f"  classifieur test : AUC={roc_auc_score(lab_te, pte):.4f}  "
      f"Brier={brier_score_loss(lab_te, pte):.5f}")

np.save(os.path.join(HERE, "cache", "oof_best_delta.npy"),
        np.concatenate(reg_oof[best["regresseur"]]))
np.save(os.path.join(HERE, "cache", "oof_best_prob.npy"),
        np.concatenate(clf_oof[best["calibration"]]))
print("\n" + "=" * 118)
show_log(sort_by="cv_acc_tol_mean")
