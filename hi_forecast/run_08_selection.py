"""PHASE 3 — iteration 28 : selection de features (levier 6).

Quatre criteres compares, tous calcules SUR LE PLI D'ENTRAINEMENT uniquement :
  - elagage par correlation (3 seuils)
  - information mutuelle (top-k)
  - importance par permutation (top-k)
  - importance SHAP (top-k)
Le jeu de features retenu est celui qui maximise l'accuracy en CV.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import lightgbm as lgb

from gated import compute_oof, fit_and_test, get_folds, make_row, summarise, sweep_gate
from harness import append_log, show_log
from protocol import SEED, TOLERANCE
from transforms import chain, drop_constant_and_dupes, prune_correlated

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = chain(drop_constant_and_dupes, prune_correlated(0.995))
d, folds = get_folds()
Xtr, ytr, mtr = d["Xtr"], d["ytr"], d["mtr"]


def reg():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", lgb.LGBMRegressor(objective="huber", alpha=0.005, n_estimators=500,
                                             learning_rate=0.05, num_leaves=63,
                                             min_child_samples=40, subsample=0.8,
                                             subsample_freq=1, colsample_bytree=0.7,
                                             random_state=SEED, n_jobs=-1, verbose=-1))])


def clf():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                              num_leaves=63, min_child_samples=40,
                                              subsample=0.8, subsample_freq=1,
                                              colsample_bytree=0.7, random_state=SEED,
                                              n_jobs=-1, verbose=-1))])


# ------------------------------------------------ classements calcules sur le train
print("=" * 118)
print("CLASSEMENTS D'IMPORTANCE (calcules sur la fenetre d'entrainement uniquement)")
print("=" * 118)
Xa, _ = CLEAN(Xtr, Xtr)
delta_tr = ytr.to_numpy() - mtr["hi_now"].to_numpy()
Xa_i = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(Xa), columns=Xa.columns)

t0 = time.time()
rng = np.random.default_rng(SEED)
sub = rng.choice(len(Xa_i), min(8000, len(Xa_i)), replace=False)
mi = mutual_info_regression(Xa_i.iloc[sub], delta_tr[sub], random_state=SEED)
mi_rank = pd.Series(mi, index=Xa.columns).sort_values(ascending=False)
print(f"  information mutuelle       [{time.time()-t0:>5.0f}s]  top5 : {list(mi_rank.head(5).index)}")

t0 = time.time()
m = lgb.LGBMRegressor(objective="huber", alpha=0.005, n_estimators=300, learning_rate=0.05,
                      num_leaves=63, random_state=SEED, n_jobs=-1, verbose=-1).fit(Xa_i, delta_tr)
pi = permutation_importance(m, Xa_i.iloc[sub], delta_tr[sub], n_repeats=3,
                            random_state=SEED, n_jobs=-1, scoring="neg_mean_absolute_error")
pi_rank = pd.Series(pi.importances_mean, index=Xa.columns).sort_values(ascending=False)
print(f"  importance par permutation [{time.time()-t0:>5.0f}s]  top5 : {list(pi_rank.head(5).index)}")

t0 = time.time()
import shap
sv = shap.TreeExplainer(m).shap_values(Xa_i.iloc[sub])
sh_rank = pd.Series(np.abs(sv).mean(axis=0), index=Xa.columns).sort_values(ascending=False)
print(f"  importance SHAP            [{time.time()-t0:>5.0f}s]  top5 : {list(sh_rank.head(5).index)}")

pd.DataFrame({"mutual_info": mi_rank, "permutation": pi_rank, "shap": sh_rank}).to_csv(
    os.path.join(HERE, "classements_importance.csv"), encoding="utf-8-sig")


def topk(rank, k):
    """Garde les k features les mieux classees. L'intersection avec les colonnes
    reellement presentes est necessaire : l'elagage en amont est refait par pli et
    ne retire pas exactement les memes colonnes d'un pli a l'autre."""
    wanted = list(rank.head(k).index)

    def _t(Xa_, Xb_):
        cols = [c for c in wanted if c in Xa_.columns]
        return Xa_[cols], Xb_[cols]
    return _t


CANDIDATES = {
    "correlation 0.995 (reference)": CLEAN,
    "correlation 0.99":  chain(drop_constant_and_dupes, prune_correlated(0.99)),
    "correlation 0.95":  chain(drop_constant_and_dupes, prune_correlated(0.95)),
    "info. mutuelle top30":  chain(CLEAN, topk(mi_rank, 30)),
    "info. mutuelle top60":  chain(CLEAN, topk(mi_rank, 60)),
    "permutation top30":     chain(CLEAN, topk(pi_rank, 30)),
    "permutation top60":     chain(CLEAN, topk(pi_rank, 60)),
    "SHAP top20":            chain(CLEAN, topk(sh_rank, 20)),
    "SHAP top40":            chain(CLEAN, topk(sh_rank, 40)),
    "SHAP top80":            chain(CLEAN, topk(sh_rank, 80)),
}

print("\n" + "=" * 118)
print("ITERATION 28 — jeux de features compares en CV")
print("=" * 118)
best, best_score, rows = None, -1, []
for name, tf in CANDIDATES.items():
    t0 = time.time()
    try:
        oof = compute_oof(reg, clf, tf, d, folds)
        grid, b = sweep_gate(oof)
        n_feat = tf(Xtr.head(50), Xtr.head(5))[0].shape[1]
        rows.append(dict(jeu=name, n_features=n_feat, alpha=b["alpha"], seuil=b["seuil"],
                         cv_acc=b["cv_acc"], cv_std=b["cv_std"], secondes=round(time.time() - t0, 1)))
        print(f"  {name:<30} p={n_feat:>4}  CV acc={b['cv_acc']:.4f}±{b['cv_std']:.4f}"
              f"  (alpha={b['alpha']}, seuil={b['seuil']})  [{time.time()-t0:>5.0f}s]")
        if b["cv_acc"] > best_score:
            best, best_score, best_cfg, best_oof, best_tf = name, b["cv_acc"], b, oof, tf
    except Exception as e:
        print(f"  {name:<30} ECHEC : {type(e).__name__}: {e}")

pd.DataFrame(rows).to_csv(os.path.join(HERE, "sweep_selection.csv"), index=False,
                          encoding="utf-8-sig")
print(f"\n  >>> retenu en CV : {best}  (acc={best_score:.4f})")

res = fit_and_test(reg, clf, best_tf, best_cfg["alpha"], best_cfg["seuil"], d)
row = make_row(28, f"Porte + selection : {best}",
               f"Comparaison de 10 jeux de features (correlation / info. mutuelle / "
               f"permutation / SHAP) ; retenu : {best}",
               best_oof, best_cfg["alpha"], best_cfg["seuil"], res, d)
append_log(row); summarise(row, f"it28 {best}")

with open(os.path.join(HERE, "cache", "best_features.txt"), "w", encoding="utf-8") as f:
    f.write(f"{best}\n{best_cfg['alpha']}\n{best_cfg['seuil']}\n")

print("\n" + "=" * 118)
show_log(sort_by="cv_acc_tol_mean")
