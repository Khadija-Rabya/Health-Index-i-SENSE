"""PHASE 3 — iterations 26 et 27 : valeurs manquantes et mise a l'echelle,
evaluees DANS l'architecture a porte et DANS le pipeline de CV.

Levier 2 (valeurs manquantes) : mediane, moyenne, mediane + indicateur de
manquant, MICE (IterativeImputer), KNN. Le KNN est evalue avec un jeu de
reference sous-echantillonne : a 25 000 lignes x 150 colonnes, un KNNImputer
complet coute O(n^2 d) par pli, ce qui est hors budget — le sous-echantillonnage
est indique explicitement dans le journal plutot que d'omettre le test.

Levier 5 (mise a l'echelle) : standard, robuste, min-max, quantile, Yeo-Johnson.
Sans effet sur les arbres, determinant pour le regresseur lineaire retenu.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import Lasso
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

CLEAN = chain(drop_constant_and_dupes, prune_correlated(0.995))
d, folds = get_folds()


class SubsampledKNNImputer(BaseEstimator, TransformerMixin):
    """KNNImputer dont le jeu de reference est sous-echantillonne, pour rendre
    le cout O(n_ref * n * d) au lieu de O(n^2 d)."""

    def __init__(self, n_neighbors=5, n_ref=4000, random_state=SEED):
        self.n_neighbors, self.n_ref, self.random_state = n_neighbors, n_ref, random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(X), min(self.n_ref, len(X)), replace=False)
        self.imp_ = KNNImputer(n_neighbors=self.n_neighbors).fit(X[idx])
        return self

    def transform(self, X):
        return self.imp_.transform(np.asarray(X, dtype=float))


IMPUTERS = {
    "mediane":            lambda: SimpleImputer(strategy="median"),
    "moyenne":            lambda: SimpleImputer(strategy="mean"),
    "mediane+indicateur": lambda: SimpleImputer(strategy="median", add_indicator=True),
    "MICE (iterative)":   lambda: IterativeImputer(max_iter=5, n_nearest_features=10,
                                                   random_state=SEED, initial_strategy="median"),
    "KNN (ref 4000)":     lambda: SubsampledKNNImputer(n_neighbors=5, n_ref=4000),
}

SCALERS = {
    "standard":  StandardScaler,
    "robuste":   RobustScaler,
    "min-max":   MinMaxScaler,
    "quantile":  lambda: QuantileTransformer(output_distribution="normal", random_state=SEED),
    "yeo-johnson": lambda: PowerTransformer(method="yeo-johnson", standardize=True),
}


def clf_fn(imputer):
    return lambda: Pipeline([("imp", imputer()),
                             ("m", lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                                      num_leaves=63, min_child_samples=40,
                                                      subsample=0.8, subsample_freq=1,
                                                      colsample_bytree=0.7, random_state=SEED,
                                                      n_jobs=-1, verbose=-1))])


def reg_fn(imputer, scaler=StandardScaler):
    return lambda: Pipeline([("imp", imputer()), ("sc", scaler()),
                             ("m", Lasso(alpha=1e-4, max_iter=5000, random_state=SEED))])


print("=" * 118)
print("ITERATION 26 — strategies de valeurs manquantes (regresseur Lasso, porte LGBM)")
print("=" * 118)
best_imp, best_imp_score, results = None, -1, []
for iname, ifn in IMPUTERS.items():
    t0 = time.time()
    try:
        oof = compute_oof(reg_fn(ifn), clf_fn(ifn), CLEAN, d, folds)
        grid, b = sweep_gate(oof)
        results.append(dict(imputation=iname, alpha=b["alpha"], seuil=b["seuil"],
                            cv_acc=b["cv_acc"], cv_std=b["cv_std"], secondes=round(time.time() - t0, 1)))
        print(f"  {iname:<20} CV acc={b['cv_acc']:.4f}±{b['cv_std']:.4f}  "
              f"(alpha={b['alpha']}, seuil={b['seuil']})  [{time.time()-t0:>5.0f}s]")
        if b["cv_acc"] > best_imp_score:
            best_imp, best_imp_score, best_imp_cfg, best_oof = iname, b["cv_acc"], b, oof
    except Exception as e:
        print(f"  {iname:<20} ECHEC : {type(e).__name__}: {e}")
pd.DataFrame(results).to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "sweep_imputation.csv"), index=False)
print(f"\n  >>> retenu en CV : {best_imp}  (acc={best_imp_score:.4f})")

res = fit_and_test(reg_fn(IMPUTERS[best_imp]), clf_fn(IMPUTERS[best_imp]), CLEAN,
                   best_imp_cfg["alpha"], best_imp_cfg["seuil"], d)
row = make_row(26, f"Porte + imputation {best_imp}",
               f"Comparaison de 5 strategies d'imputation dans le pipeline de CV ; "
               f"retenue : {best_imp}", best_oof, best_imp_cfg["alpha"], best_imp_cfg["seuil"], res, d,
               notes=f"alpha={best_imp_cfg['alpha']} seuil={best_imp_cfg['seuil']}")
append_log(row); summarise(row, f"it26 {best_imp}")

print("\n" + "=" * 118)
print("ITERATION 27 — mise a l'echelle (regresseur lineaire)")
print("=" * 118)
best_sc, best_sc_score, results2 = None, -1, []
for sname, sfn in SCALERS.items():
    t0 = time.time()
    try:
        oof = compute_oof(reg_fn(IMPUTERS[best_imp], sfn), clf_fn(IMPUTERS[best_imp]), CLEAN, d, folds)
        grid, b = sweep_gate(oof)
        results2.append(dict(echelle=sname, alpha=b["alpha"], seuil=b["seuil"],
                             cv_acc=b["cv_acc"], cv_std=b["cv_std"], secondes=round(time.time() - t0, 1)))
        print(f"  {sname:<14} CV acc={b['cv_acc']:.4f}±{b['cv_std']:.4f}  "
              f"(alpha={b['alpha']}, seuil={b['seuil']})  [{time.time()-t0:>5.0f}s]")
        if b["cv_acc"] > best_sc_score:
            best_sc, best_sc_score, best_sc_cfg, best_sc_oof = sname, b["cv_acc"], b, oof
    except Exception as e:
        print(f"  {sname:<14} ECHEC : {type(e).__name__}: {e}")
pd.DataFrame(results2).to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "sweep_echelle.csv"), index=False)
print(f"\n  >>> retenu en CV : {best_sc}  (acc={best_sc_score:.4f})")

res2 = fit_and_test(reg_fn(IMPUTERS[best_imp], SCALERS[best_sc]), clf_fn(IMPUTERS[best_imp]),
                    CLEAN, best_sc_cfg["alpha"], best_sc_cfg["seuil"], d)
row2 = make_row(27, f"Porte + echelle {best_sc}",
                f"Comparaison de 5 mises a l'echelle ; retenue : {best_sc} "
                f"(imputation {best_imp})", best_sc_oof, best_sc_cfg["alpha"], best_sc_cfg["seuil"],
                res2, d)
append_log(row2); summarise(row2, f"it27 {best_sc}")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "best_pre.txt"),
          "w", encoding="utf-8") as f:
    f.write(f"{best_imp}\n{best_sc}\n{best_sc_cfg['alpha']}\n{best_sc_cfg['seuil']}\n")

print("\n" + "=" * 118)
show_log(sort_by="cv_acc_tol_mean")
