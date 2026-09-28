"""PHASE 3 — iteration 31 : ensembles (levier 8, fin de la liste).

Assemblage des meilleures familles du classement DANS l'architecture a porte :
  - vote (moyenne simple des deltas predits)
  - vote pondere (poids proportionnels a la qualite en CV)
  - empilement (stacking) : une regression Ridge apprend a combiner les deltas
    hors-pli des modeles de base — entrainee UNIQUEMENT sur les predictions
    hors-pli, jamais sur des predictions en resubstitution.

Le classifieur de porte reste le LGBM retenu a l'iteration 25.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import lightgbm as lgb

from gated import ALPHAS, PROBS, compute_oof, get_folds, make_row, summarise
from harness import append_log, show_log
from protocol import ASSET_COLS, SEED, TOLERANCE, regression_metrics
from transforms import chain, drop_constant_and_dupes, prune_correlated

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = chain(drop_constant_and_dupes, prune_correlated(0.995))
d, folds = get_folds()
Xtr, ytr, mtr, Xte, yte, mte = d["Xtr"], d["ytr"], d["mtr"], d["Xte"], d["yte"], d["mte"]


def sc(*s):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())] + list(s))


def ns(*s):
    return Pipeline([("imp", SimpleImputer(strategy="median"))] + list(s))


BASE = {
    "LGBM huber": lambda: ns(("m", lgb.LGBMRegressor(objective="huber", alpha=0.005,
                                                     n_estimators=500, learning_rate=0.05,
                                                     num_leaves=63, min_child_samples=40,
                                                     subsample=0.8, subsample_freq=1,
                                                     colsample_bytree=0.7, random_state=SEED,
                                                     n_jobs=-1, verbose=-1))),
    "ElasticNet": lambda: sc(("m", ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=5000,
                                              random_state=SEED))),
    "Lasso":      lambda: sc(("m", Lasso(alpha=1e-4, max_iter=5000, random_state=SEED))),
    "RF sqrt":    lambda: ns(("m", RandomForestRegressor(n_estimators=300, max_depth=14,
                                                         max_features="sqrt", random_state=SEED,
                                                         n_jobs=-1))),
    "ExtraTrees": lambda: ns(("m", ExtraTreesRegressor(n_estimators=300, max_depth=14,
                                                       max_features="sqrt", random_state=SEED,
                                                       n_jobs=-1))),
}


def clf_fn():
    return ns(("m", lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
                                       min_child_samples=40, subsample=0.8, subsample_freq=1,
                                       colsample_bytree=0.7, random_state=SEED,
                                       n_jobs=-1, verbose=-1)))


# ------------------------------------------------- deltas hors-pli des modeles de base
print("=" * 118)
print("DELTAS HORS-PLI DES MODELES DE BASE")
print("=" * 118)
oof_delta, gate_oof = {}, None
for name, fn in BASE.items():
    t0 = time.time()
    o = compute_oof(fn, clf_fn, CLEAN, d, folds)
    oof_delta[name] = [f["d"] for f in o]
    if gate_oof is None:
        gate_oof = o
    print(f"  {name:<12} [{time.time()-t0:>5.0f}s]")

common = [dict(y=f["y"], n=f["n"], p=f["p"]) for f in gate_oof]


def sweep(deltas):
    best = None
    for a in ALPHAS:
        for pth in PROBS:
            per_fold = [np.mean(np.abs(o["y"] - (o["n"] + a * dp * (o["p"] > pth))) <= TOLERANCE)
                        for o, dp in zip(common, deltas)]
            s = float(np.mean(per_fold))
            if best is None or s > best["cv_acc"]:
                best = dict(alpha=a, seuil=pth, cv_acc=s, cv_std=float(np.std(per_fold)))
    return best


print("\n" + "=" * 118)
print("ITERATION 31 — ensembles (selection en CV)")
print("=" * 118)
results = {}
for name in BASE:
    results[name] = sweep(oof_delta[name])
    print(f"  seul  {name:<14} CV acc={results[name]['cv_acc']:.4f}  "
          f"(a={results[name]['alpha']}, s={results[name]['seuil']})")

names = list(BASE)
top = sorted(names, key=lambda n: results[n]["cv_acc"], reverse=True)[:4]
print(f"\n  top 4 retenus pour l'assemblage : {top}")

vote = [np.mean([oof_delta[n][k] for n in top], axis=0) for k in range(len(folds))]
results["Vote (moyenne, top4)"] = sweep(vote)

w = np.array([max(results[n]["cv_acc"] - 0.78, 1e-6) for n in top]); w /= w.sum()
vote_w = [np.average([oof_delta[n][k] for n in top], axis=0, weights=w) for k in range(len(folds))]
results["Vote pondere (top4)"] = sweep(vote_w)

# Empilement : Ridge ajustee sur les deltas HORS-PLI uniquement
Zoof = np.column_stack([np.concatenate(oof_delta[n]) for n in top])
ytrue = np.concatenate([o["y"] - o["n"] for o in common])
meta = Ridge(alpha=1.0).fit(Zoof, ytrue)
sizes = [len(o["y"]) for o in common]
off = np.concatenate([[0], np.cumsum(sizes)])
stack = [meta.predict(Zoof[off[k]:off[k + 1]]) for k in range(len(folds))]
results["Empilement Ridge (top4)"] = sweep(stack)
print(f"  poids de l'empilement : " +
      ", ".join(f"{n}={c:+.3f}" for n, c in zip(top, meta.coef_)))

for name in ["Vote (moyenne, top4)", "Vote pondere (top4)", "Empilement Ridge (top4)"]:
    r = results[name]
    print(f"  {name:<28} CV acc={r['cv_acc']:.4f}±{r['cv_std']:.4f}  "
          f"(a={r['alpha']}, s={r['seuil']})")

pd.DataFrame([{"modele": k, **v} for k, v in results.items()]).to_csv(
    os.path.join(HERE, "sweep_ensembles.csv"), index=False, encoding="utf-8-sig")

best_name = max(results, key=lambda k: results[k]["cv_acc"])
bc = results[best_name]
print(f"\n  >>> retenu en CV : {best_name}  acc={bc['cv_acc']:.4f}")

# ------------------------------------------------------------------ test gele
print("\n" + "=" * 118)
print("TEST GELE — ensemble retenu")
print("=" * 118)
Xa, Xb = CLEAN(Xtr, Xte)
na, nb = mtr["hi_now"].to_numpy(), mte["hi_now"].to_numpy()
dte = {}
for n in top:
    m = BASE[n](); m.fit(Xa, ytr.to_numpy() - na); dte[n] = m.predict(Xb)
c = clf_fn(); c.fit(Xa, (np.abs(ytr.to_numpy() - na) > TOLERANCE).astype(int))
pte = c.predict_proba(Xb)[:, 1]

if best_name == "Vote (moyenne, top4)":
    delta_te = np.mean([dte[n] for n in top], axis=0); sel_oof = vote
elif best_name == "Vote pondere (top4)":
    delta_te = np.average([dte[n] for n in top], axis=0, weights=w); sel_oof = vote_w
elif best_name == "Empilement Ridge (top4)":
    delta_te = meta.predict(np.column_stack([dte[n] for n in top])); sel_oof = stack
else:
    delta_te = dte[best_name] if best_name in dte else None; sel_oof = oof_delta[best_name]
    if delta_te is None:
        m = BASE[best_name](); m.fit(Xa, ytr.to_numpy() - na); delta_te = m.predict(Xb)

gate = pte > bc["seuil"]
yhat = np.clip(nb + bc["alpha"] * delta_te * gate, 0, 1)
test = regression_metrics(yte, yhat, Xa.shape[1], hi_now=nb)
cvdf = pd.DataFrame([regression_metrics(o["y"], o["n"] + bc["alpha"] * dp * (o["p"] > bc["seuil"]),
                                        hi_now=o["n"]) for o, dp in zip(common, sel_oof)])

row = {"iteration": 31, "name": f"Porte + {best_name}",
       "change": f"Ensembles (vote / vote pondere / empilement Ridge) des 4 meilleures "
                 f"familles {top} ; retenu : {best_name}",
       "n_features": int(Xa.shape[1]), "delta_form": True,
       "cv_acc_tol_mean": cvdf["acc_tol"].mean(), "cv_acc_tol_std": cvdf["acc_tol"].std(),
       "cv_r2_mean": cvdf["r2"].mean(), "cv_r2_std": cvdf["r2"].std(),
       "cv_rmse_mean": cvdf["rmse"].mean(), "cv_mae_mean": cvdf["mae"].mean(),
       "cv_skill_mean": cvdf["skill_vs_persist"].mean(),
       **{f"test_{k}": v for k, v in test.items()},
       "gap_acc_tol": test["acc_tol"] - cvdf["acc_tol"].mean(),
       "gap_r2": test["r2"] - cvdf["r2"].mean(), "seconds": 0.0,
       "notes": f"alpha={bc['alpha']} seuil={bc['seuil']} corrigees={gate.mean():.1%}"}
for asset, col in ASSET_COLS.items():
    m_ = (mte[col] == 1).to_numpy()
    row[f"test_acc_{asset[-1]}"] = float(np.mean(np.abs(yte[m_] - yhat[m_]) <= TOLERANCE))
    row[f"test_r2_{asset[-1]}"] = float(regression_metrics(yte[m_], yhat[m_])["r2"])
append_log(row); summarise(row, f"it31 {best_name}")

print("\n" + "=" * 118)
show_log(sort_by="cv_acc_tol_mean")
