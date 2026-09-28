"""PHASE 3 / DELIVRABLE 2 — classement de toutes les familles de modeles.

Chaque famille est evaluee sous le protocole gele, avec :
  - imputation mediane ajustee par pli,
  - mise a l'echelle quand la famille l'exige (lineaires, SVM, KNN, MLP),
  - facteur de retrecissement alpha choisi EN CV pour cette famille,
  - meme formulation en delta pour tout le monde.

Naive Bayes n'apparait pas : c'est un classifieur, sans equivalent de regression
applicable ici (la cible est continue). Note explicitement dans le rapport.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import catboost as cb
import lightgbm as lgb
from xgboost import XGBRegressor

from harness import append_log, get_data, show_log
from protocol import (ASSET_COLS, N_SPLITS, SEED, TOLERANCE, PurgedTimeSeriesSplit,
                      regression_metrics)
from transforms import chain, drop_constant_and_dupes, prune_correlated

CLEAN = chain(drop_constant_and_dupes, prune_correlated(0.995))
ALPHAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]

d = get_data()
Xtr, ytr, mtr, Xte, yte, mte = d["Xtr"], d["ytr"], d["mtr"], d["Xte"], d["yte"], d["mte"]


def imp(*steps, scale=False):
    pre = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        pre.append(("sc", StandardScaler()))
    return Pipeline(pre + list(steps))


FAMILIES = {
    "Linear (OLS)":      (lambda: imp(("m", LinearRegression()), scale=True), None),
    "Ridge":             (lambda: imp(("m", Ridge(alpha=10.0, random_state=SEED)), scale=True), None),
    "Lasso":             (lambda: imp(("m", Lasso(alpha=1e-4, max_iter=5000, random_state=SEED)), scale=True), None),
    "ElasticNet":        (lambda: imp(("m", ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=5000, random_state=SEED)), scale=True), None),
    "Random Forest":     (lambda: imp(("m", RandomForestRegressor(n_estimators=300, max_depth=14, max_features="sqrt", random_state=SEED, n_jobs=-1))), None),
    "Extra Trees":       (lambda: imp(("m", ExtraTreesRegressor(n_estimators=300, max_depth=14, max_features="sqrt", random_state=SEED, n_jobs=-1))), None),
    "Gradient Boosting": (lambda: imp(("m", GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=SEED))), None),
    "XGBoost":           (lambda: imp(("m", XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1))), None),
    "LightGBM":          (lambda: imp(("m", lgb.LGBMRegressor(objective="l2", n_estimators=400, learning_rate=0.05, num_leaves=31, random_state=SEED, n_jobs=-1, verbose=-1))), None),
    "CatBoost":          (lambda: imp(("m", cb.CatBoostRegressor(iterations=400, depth=6, learning_rate=0.05, random_seed=SEED, verbose=0, thread_count=-1))), None),
    "SVM (RBF, Nystroem)": (lambda: imp(("ny", Nystroem(gamma=0.01, n_components=300, random_state=SEED)), ("m", Ridge(alpha=1.0)), scale=True), None),
    "SVM (RBF exact)":   (lambda: imp(("m", SVR(kernel="rbf", C=1.0, gamma="scale", cache_size=1000)), scale=True), 6000),
    "KNN (k=25)":        (lambda: imp(("m", KNeighborsRegressor(n_neighbors=25, weights="distance", n_jobs=-1)), scale=True), None),
    "MLP (128,64)":      (lambda: imp(("m", MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=300, early_stopping=True, random_state=SEED)), scale=True), None),
}

cv = PurgedTimeSeriesSplit(n_splits=N_SPLITS)
folds = list(cv.split(Xtr, times=mtr["created_at"]))
rows, it = [], 10

print("=" * 120)
print("CLASSEMENT DES FAMILLES DE MODELES — alpha choisi en CV pour chacune")
print("=" * 120)

for fname, (make, sub_n) in FAMILIES.items():
    t0 = time.time()
    try:
        oof, fold_metrics = [], []
        for itr, iva in folds:
            if sub_n and len(itr) > sub_n:      # sous-echantillonnage (SVM exact seulement)
                rng = np.random.default_rng(SEED)
                itr = rng.choice(itr, sub_n, replace=False)
            Xa, Xb = CLEAN(Xtr.iloc[itr], Xtr.iloc[iva])
            na, nb = mtr["hi_now"].iloc[itr], mtr["hi_now"].iloc[iva]
            ya, yb = ytr.iloc[itr], ytr.iloc[iva]
            p = make(); p.fit(Xa, ya - na)
            oof.append((yb.to_numpy(), nb.to_numpy(), p.predict(Xb)))

        accs = {a: float(np.mean([np.mean(np.abs(yb - (nb + a * dp)) <= TOLERANCE)
                                  for yb, nb, dp in oof])) for a in ALPHAS}
        best_a = max(accs, key=accs.get)
        # Le classement des FAMILLES se lit a alpha=1 (modele brut, sans
        # retrecissement) : c'est la seule facon de les comparer entre elles,
        # puisque le alpha optimal en CV vaut 0 pour presque toutes et les
        # ramenerait toutes a la persistance, donc a un classement plat.
        per_fold = [regression_metrics(yb, nb + dp, hi_now=nb) for yb, nb, dp in oof]
        cvdf = pd.DataFrame(per_fold)

        itr_full = np.arange(len(Xtr))
        if sub_n and len(itr_full) > sub_n:
            itr_full = np.random.default_rng(SEED).choice(itr_full, sub_n, replace=False)
        Xa, Xb = CLEAN(Xtr.iloc[itr_full], Xte)
        p = make(); p.fit(Xa, ytr.iloc[itr_full] - mtr["hi_now"].iloc[itr_full])
        dte = p.predict(Xb)
        yhat = np.clip(mte["hi_now"] + dte, 0, 1)
        test = regression_metrics(yte, yhat, Xa.shape[1], hi_now=mte["hi_now"])
        yhat_a = np.clip(mte["hi_now"] + best_a * dte, 0, 1)
        test_best_alpha_acc = float(np.mean(np.abs(yte - yhat_a) <= TOLERANCE))

        row = {
            "iteration": it, "name": fname,
            "change": f"Famille {fname} brute (alpha=1) ; alpha optimal en CV = {best_a}",
            "n_features": int(Xa.shape[1]), "delta_form": True,
            "alpha_cv_best": best_a, "cv_acc_at_best_alpha": accs[best_a],
            "test_acc_at_best_alpha": test_best_alpha_acc,
            "cv_acc_tol_mean": cvdf["acc_tol"].mean(), "cv_acc_tol_std": cvdf["acc_tol"].std(),
            "cv_r2_mean": cvdf["r2"].mean(), "cv_r2_std": cvdf["r2"].std(),
            "cv_rmse_mean": cvdf["rmse"].mean(), "cv_mae_mean": cvdf["mae"].mean(),
            "cv_skill_mean": cvdf["skill_vs_persist"].mean(),
            **{f"test_{k}": v for k, v in test.items()},
            "gap_acc_tol": test["acc_tol"] - cvdf["acc_tol"].mean(),
            "gap_r2": test["r2"] - cvdf["r2"].mean(),
            "seconds": round(time.time() - t0, 1),
            "notes": "leaderboard familles" + (f" (sous-echantillon {sub_n})" if sub_n else ""),
        }
        for asset, col in ASSET_COLS.items():
            m = (mte[col] == 1).to_numpy()
            row[f"test_acc_{asset[-1]}"] = float(np.mean(np.abs(yte[m] - yhat[m]) <= TOLERANCE))
            row[f"test_r2_{asset[-1]}"] = float(regression_metrics(yte[m], yhat[m])["r2"])
        rows.append(row); append_log(row); it += 1
        print(f"  {fname:<22} CV acc={row['cv_acc_tol_mean']:.4f}±{row['cv_acc_tol_std']:.4f}"
              f"  CV R²={row['cv_r2_mean']:>9.4f}  TEST acc={row['test_acc_tol']:.4f}"
              f"  TEST R²={row['test_r2']:>7.4f}  skill={row['test_skill_vs_persist']:+.4f}"
              f"  | a*={best_a:<4} CVacc*={accs[best_a]:.4f}  [{row['seconds']:>5.0f}s]")
    except Exception as e:
        print(f"  {fname:<22} ECHEC : {type(e).__name__}: {e}")

pd.DataFrame(rows).to_csv(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "leaderboard_familles.csv"),
    index=False, encoding="utf-8-sig")

print("\n" + "=" * 120)
print("JOURNAL COMPLET (trie par CV acc)")
print("=" * 120)
show_log(sort_by="cv_acc_tol_mean")
