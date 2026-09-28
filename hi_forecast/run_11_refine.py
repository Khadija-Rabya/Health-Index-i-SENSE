"""PHASE 3 — iteration 32 : combinaison finale.

Les iterations 28 et 30 ont ameliore la CV separement :
  - it28 : jeu de features reduit (SHAP top20)   -> CV 0.7995
  - it30 : hyperparametres Optuna (toutes features) -> CV 0.8057
Cette iteration croise les deux et balaye la porte une derniere fois. Selection
en CV, comme partout ailleurs.
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import lightgbm as lgb

from gated import compute_oof, fit_and_test, get_folds, make_row, summarise, sweep_gate
from harness import append_log, show_log
from protocol import SEED
from transforms import chain, drop_constant_and_dupes, prune_correlated

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = chain(drop_constant_and_dupes, prune_correlated(0.995))
d, folds = get_folds()

with open(os.path.join(HERE, "cache", "best_params.json"), encoding="utf-8") as f:
    P = json.load(f)["params"]
rank = pd.read_csv(os.path.join(HERE, "classements_importance.csv"), index_col=0)


def build():
    reg = lambda: Pipeline([("imp", SimpleImputer(strategy="median")),
                            ("m", lgb.LGBMRegressor(
                                objective="huber", alpha=P["reg_huber_alpha"],
                                n_estimators=P["reg_n_estimators"],
                                learning_rate=P["reg_learning_rate"],
                                num_leaves=P["reg_num_leaves"],
                                min_child_samples=P["reg_min_child"],
                                subsample=P["reg_subsample"], subsample_freq=1,
                                colsample_bytree=P["reg_colsample"],
                                reg_lambda=P["reg_reg_lambda"],
                                random_state=SEED, n_jobs=-1, verbose=-1))])
    clf = lambda: Pipeline([("imp", SimpleImputer(strategy="median")),
                            ("m", lgb.LGBMClassifier(
                                n_estimators=P["clf_n_estimators"],
                                learning_rate=P["clf_learning_rate"],
                                num_leaves=P["clf_num_leaves"],
                                min_child_samples=P["clf_min_child"],
                                subsample=P["clf_subsample"], subsample_freq=1,
                                colsample_bytree=P["clf_colsample"],
                                reg_lambda=P["clf_reg_lambda"],
                                random_state=SEED, n_jobs=-1, verbose=-1))])
    return reg, clf


def topk(col, k):
    wanted = list(rank[col].sort_values(ascending=False).head(k).index)

    def _t(Xa_, Xb_):
        cols = [c for c in wanted if c in Xa_.columns]
        return Xa_[cols], Xb_[cols]
    return _t


CANDIDATES = {
    "toutes (correlation 0.995)": CLEAN,
    "SHAP top15":  chain(CLEAN, topk("shap", 15)),
    "SHAP top20":  chain(CLEAN, topk("shap", 20)),
    "SHAP top30":  chain(CLEAN, topk("shap", 30)),
    "SHAP top50":  chain(CLEAN, topk("shap", 50)),
    "permutation top30": chain(CLEAN, topk("permutation", 30)),
}

print("=" * 118)
print("ITERATION 32 — hyperparametres Optuna x jeu de features (selection en CV)")
print("=" * 118)
best, best_score, rows = None, -1, []
for name, tf in CANDIDATES.items():
    t0 = time.time()
    reg, clf = build()
    oof = compute_oof(reg, clf, tf, d, folds)
    grid, b = sweep_gate(oof)
    n_feat = tf(d["Xtr"].head(50), d["Xtr"].head(5))[0].shape[1]
    rows.append(dict(jeu=name, n_features=n_feat, alpha=b["alpha"], seuil=b["seuil"],
                     cv_acc=b["cv_acc"], cv_std=b["cv_std"], secondes=round(time.time() - t0, 1)))
    print(f"  {name:<28} p={n_feat:>4}  CV acc={b['cv_acc']:.4f}±{b['cv_std']:.4f}"
          f"  (alpha={b['alpha']}, seuil={b['seuil']})  [{time.time()-t0:>5.0f}s]")
    if b["cv_acc"] > best_score:
        best, best_score, best_cfg, best_oof, best_tf = name, b["cv_acc"], b, oof, tf

pd.DataFrame(rows).to_csv(os.path.join(HERE, "sweep_refine.csv"), index=False, encoding="utf-8-sig")
print(f"\n  >>> retenu en CV : {best}  (acc={best_score:.4f}, alpha={best_cfg['alpha']}, "
      f"seuil={best_cfg['seuil']})")

reg, clf = build()
res = fit_and_test(reg, clf, best_tf, best_cfg["alpha"], best_cfg["seuil"], d)
row = make_row(32, f"FINAL — Optuna + {best}",
               f"Hyperparametres Optuna (it30) croises avec 6 jeux de features ; "
               f"retenu : {best}, alpha={best_cfg['alpha']}, seuil={best_cfg['seuil']}",
               best_oof, best_cfg["alpha"], best_cfg["seuil"], res, d)
append_log(row); summarise(row, f"it32 {best}")

with open(os.path.join(HERE, "cache", "final_config.json"), "w", encoding="utf-8") as f:
    json.dump({"params": P, "feature_set": best, "alpha": float(best_cfg["alpha"]),
               "seuil": float(best_cfg["seuil"]), "cv_acc": float(best_score),
               "n_features": int(res["n_features"])}, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 118)
show_log(sort_by="cv_acc_tol_mean")
