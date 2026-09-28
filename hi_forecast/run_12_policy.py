"""PHASE 3 — iteration 33 : politique PAR MACHINE, choisie en CV.

Le modele final gagne globalement mais les metriques par machine du livrable
suggerent que le gain vient entierement de la Motosoufflante B. Choisir la
politique machine par machine d'apres le score de TEST serait exactement la
faute auditee en F6. On tranche donc EN CV : pour chaque machine on compare, sur
les predictions hors-pli, "modele a porte" contre "persistance", puis on
n'applique la politique retenue au test gele qu'une seule fois.
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

import lightgbm as lgb

from gated import compute_oof, get_folds
from harness import append_log, show_log
from protocol import ASSET_COLS, SEED, TOLERANCE, regression_metrics
from transforms import chain, drop_constant_and_dupes, prune_correlated

with open(os.path.join(HERE, "cache", "final_config.json"), encoding="utf-8") as f:
    cfg = json.load(f)
P, ALPHA, SEUIL = cfg["params"], cfg["alpha"], cfg["seuil"]

CLEAN = chain(drop_constant_and_dupes, prune_correlated(0.995))
rank = pd.read_csv(os.path.join(HERE, "classements_importance.csv"), index_col=0)
wanted = list(rank["shap"].sort_values(ascending=False).head(20).index)


def TF(Xa_, Xb_):
    Xa_, Xb_ = CLEAN(Xa_, Xb_)
    cols = [c for c in wanted if c in Xa_.columns]
    return Xa_[cols], Xb_[cols]


def make_reg():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", lgb.LGBMRegressor(objective="huber", alpha=P["reg_huber_alpha"],
                                             n_estimators=P["reg_n_estimators"],
                                             learning_rate=P["reg_learning_rate"],
                                             num_leaves=P["reg_num_leaves"],
                                             min_child_samples=P["reg_min_child"],
                                             subsample=P["reg_subsample"], subsample_freq=1,
                                             colsample_bytree=P["reg_colsample"],
                                             reg_lambda=P["reg_reg_lambda"],
                                             random_state=SEED, n_jobs=-1, verbose=-1))])


def make_clf():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", lgb.LGBMClassifier(n_estimators=P["clf_n_estimators"],
                                              learning_rate=P["clf_learning_rate"],
                                              num_leaves=P["clf_num_leaves"],
                                              min_child_samples=P["clf_min_child"],
                                              subsample=P["clf_subsample"], subsample_freq=1,
                                              colsample_bytree=P["clf_colsample"],
                                              reg_lambda=P["clf_reg_lambda"],
                                              random_state=SEED, n_jobs=-1, verbose=-1))])


d, folds = get_folds()
Xtr, ytr, mtr, Xte, yte, mte = d["Xtr"], d["ytr"], d["mtr"], d["Xte"], d["yte"], d["mte"]

print("=" * 100)
print("ITERATION 33 — choix de la politique par machine, EN CV")
print("=" * 100)
oof = compute_oof(make_reg, make_clf, TF, d, folds)
asset_col = {a: c for a, c in ASSET_COLS.items()}

policy, cv_table = {}, []
for a, col in asset_col.items():
    accs_model, accs_pers = [], []
    for (itr, iva), o in zip(folds, oof):
        m = (mtr[col].iloc[iva] == 1).to_numpy()
        if m.sum() < 30:
            continue
        yv, nv = o["y"][m], o["n"][m]
        pred = nv + ALPHA * o["d"][m] * (o["p"][m] > SEUIL)
        accs_model.append(np.mean(np.abs(yv - pred) <= TOLERANCE))
        accs_pers.append(np.mean(np.abs(yv - nv) <= TOLERANCE))
    mm, pp = float(np.mean(accs_model)), float(np.mean(accs_pers))
    policy[a] = "modele" if mm > pp else "persistance"
    cv_table.append(dict(machine=a, cv_acc_modele=mm, cv_acc_persistance=pp,
                         n_plis=len(accs_model), politique=policy[a]))
    print(f"  {a:<20} CV modele={mm:.4f}  CV persistance={pp:.4f}  -> {policy[a].upper()}")

pd.DataFrame(cv_table).to_csv(os.path.join(HERE, "sweep_politique.csv"), index=False,
                              encoding="utf-8-sig")

# CV de la politique mixte (recomposee sur les memes plis)
cv_mixed = []
for (itr, iva), o in zip(folds, oof):
    pred = o["n"].copy()
    for a, col in asset_col.items():
        m = (mtr[col].iloc[iva] == 1).to_numpy()
        if policy[a] == "modele":
            pred[m] = o["n"][m] + ALPHA * o["d"][m] * (o["p"][m] > SEUIL)
    cv_mixed.append(regression_metrics(o["y"], pred, hi_now=o["n"]))
cvdf = pd.DataFrame(cv_mixed)
print(f"\n  CV politique mixte : acc={cvdf['acc_tol'].mean():.4f} ± {cvdf['acc_tol'].std():.4f}"
      f"   (modele partout : {cfg['cv_acc']:.4f})")

# ---------------------------------------------------------------- test gele
print("\n" + "=" * 100)
print("TEST GELE — politique mixte appliquee une seule fois")
print("=" * 100)
Xa, Xb = TF(Xtr, Xte)
na, nb = mtr["hi_now"].to_numpy(), mte["hi_now"].to_numpy()
reg = make_reg(); reg.fit(Xa, ytr.to_numpy() - na)
clf = make_clf(); clf.fit(Xa, (np.abs(ytr.to_numpy() - na) > TOLERANCE).astype(int))
dte, pte = reg.predict(Xb), clf.predict_proba(Xb)[:, 1]

yhat = nb.copy()
for a, col in asset_col.items():
    m = (mte[col] == 1).to_numpy()
    if policy[a] == "modele":
        yhat[m] = np.clip(nb[m] + ALPHA * dte[m] * (pte[m] > SEUIL), 0, 1)

test = regression_metrics(yte, yhat, Xa.shape[1], hi_now=nb)
print(f"  acc@±{TOLERANCE} = {test['acc_tol']:.4f}   (modele partout 0.8259, persistance {test['acc_tol_persist']:.4f})")
print(f"  R² = {test['r2']:.4f}   RMSE = {test['rmse']:.5f}   MAE = {test['mae']:.5f}"
      f"   skill = {test['skill_vs_persist']:+.4f}")
for a, col in asset_col.items():
    m = (mte[col] == 1).to_numpy()
    mm = regression_metrics(yte[m], yhat[m], hi_now=nb[m])
    print(f"  {a:<20} politique={policy[a]:<12} acc={mm['acc_tol']:.4f}  R²={mm['r2']:.4f}")

row = {"iteration": 33, "name": "FINAL — politique par machine",
       "change": f"Politique choisie EN CV machine par machine : "
                 f"{', '.join(f'{a}={p}' for a, p in policy.items())}",
       "n_features": int(Xa.shape[1]), "delta_form": True,
       "cv_acc_tol_mean": cvdf["acc_tol"].mean(), "cv_acc_tol_std": cvdf["acc_tol"].std(),
       "cv_r2_mean": cvdf["r2"].mean(), "cv_r2_std": cvdf["r2"].std(),
       "cv_rmse_mean": cvdf["rmse"].mean(), "cv_mae_mean": cvdf["mae"].mean(),
       "cv_skill_mean": cvdf["skill_vs_persist"].mean(),
       **{f"test_{k}": v for k, v in test.items()},
       "gap_acc_tol": test["acc_tol"] - cvdf["acc_tol"].mean(),
       "gap_r2": test["r2"] - cvdf["r2"].mean(), "seconds": 0.0,
       "notes": f"politique CV : {policy}"}
for a, col in ASSET_COLS.items():
    m = (mte[col] == 1).to_numpy()
    row[f"test_acc_{a[-1]}"] = float(np.mean(np.abs(yte[m] - yhat[m]) <= TOLERANCE))
    row[f"test_r2_{a[-1]}"] = float(regression_metrics(yte[m], yhat[m])["r2"])
append_log(row)

with open(os.path.join(HERE, "cache", "policy.json"), "w", encoding="utf-8") as f:
    json.dump(policy, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 100)
show_log(sort_by="cv_acc_tol_mean")
