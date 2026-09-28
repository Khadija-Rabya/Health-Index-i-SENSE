"""
SCRIPT D'ENTRAINEMENT FINAL — reproduit de bout en bout le meilleur resultat.

    python hi_forecast/train_best.py

Part de `isense_oil_data_health_index.csv` et refait tout : correction des
fuites, reconstruction du label sur la fenetre d'entrainement seule, decoupage
gele, validation croisee temporelle purgee, entrainement, score unique sur le
test gele, graphiques de diagnostic, SHAP, et sauvegarde d'un artefact joblib
unique contenant TOUT le pretraitement.

Sorties :
  hi_forecast/artifacts/health_index_t18_pipeline.joblib   (artefact reutilisable)
  hi_forecast/artifacts/metriques_finales.json
  hi_forecast/figures/01_residus_final.png
  hi_forecast/figures/03_shap.png
  hi_forecast/importance_shap_final.csv
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             roc_auc_score)
from sklearn.pipeline import Pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

import lightgbm as lgb

from gated import cv_metrics, get_folds, sweep_gate, compute_oof
from protocol import (ASSET_COLS, HORIZON, SEED, TOLERANCE, regression_metrics)
from report_assets import iterations_overview, residual_diagnostics, shap_summary
from transforms import chain, drop_constant_and_dupes, prune_correlated

CFG_PATH = os.path.join(HERE, "cache", "final_config.json")
ART_DIR = os.path.join(HERE, "artifacts")
os.makedirs(ART_DIR, exist_ok=True)


class GatedHealthIndexForecaster:
    """Prevision de health_index a t+18 (~3h) par correction selective.

        prediction = health_index[t] + alpha * delta_predit * 1[P(mouvement) > seuil]

    Deux modeles : un regresseur du delta et un classifieur qui decide s'il faut
    corriger. Sans la porte, la correction degrade la persistance (verifie en CV
    sur 5 familles de modeles : le facteur alpha optimal est 0 sans porte).
    """

    def __init__(self, reg, clf, columns, alpha, seuil, horizon=HORIZON,
                 tolerance=TOLERANCE, metadata=None):
        self.reg, self.clf = reg, clf
        self.columns = list(columns)
        self.alpha, self.seuil = float(alpha), float(seuil)
        self.horizon, self.tolerance = horizon, tolerance
        self.metadata = metadata or {}

    def predict(self, X, hi_now):
        """X : DataFrame contenant au moins self.columns (features a l'instant t).
        hi_now : health_index observe a l'instant t."""
        Xc = X[self.columns]
        hi_now = np.asarray(hi_now, dtype=float)
        delta = self.reg.predict(Xc)
        prob = self.clf.predict_proba(Xc)[:, 1]
        gate = prob > self.seuil
        return np.clip(hi_now + self.alpha * delta * gate, 0.0, 1.0)

    def predict_detail(self, X, hi_now):
        Xc = X[self.columns]
        delta = self.reg.predict(Xc)
        prob = self.clf.predict_proba(Xc)[:, 1]
        gate = prob > self.seuil
        return pd.DataFrame({
            "health_index_now": np.asarray(hi_now, dtype=float),
            "delta_predit": delta,
            "proba_mouvement": prob,
            "porte_ouverte": gate,
            "health_index_predit": self.predict(X, hi_now),
        })


def main():
    t_start = time.time()
    with open(CFG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    P, alpha, seuil = cfg["params"], cfg["alpha"], cfg["seuil"]
    print("=" * 100)
    print("ENTRAINEMENT FINAL — prevision de health_index a t+18 (~3h)")
    print("=" * 100)
    print(f"  configuration : jeu de features = {cfg['feature_set']}, "
          f"alpha = {alpha}, seuil = {seuil}")

    d, folds = get_folds()
    Xtr, ytr, mtr = d["Xtr"], d["ytr"], d["mtr"]
    Xte, yte, mte = d["Xte"], d["yte"], d["mte"]
    na, nb = mtr["hi_now"].to_numpy(), mte["hi_now"].to_numpy()

    CLEAN = chain(drop_constant_and_dupes, prune_correlated(0.995))
    rank = pd.read_csv(os.path.join(HERE, "classements_importance.csv"), index_col=0)
    fs = cfg["feature_set"]
    if fs.startswith("SHAP top") or fs.startswith("permutation top"):
        col = "shap" if fs.startswith("SHAP") else "permutation"
        k = int(fs.split("top")[1])
        wanted = list(rank[col].sort_values(ascending=False).head(k).index)

        def TF(Xa_, Xb_):
            Xa_, Xb_ = CLEAN(Xa_, Xb_)
            cols = [c for c in wanted if c in Xa_.columns]
            return Xa_[cols], Xb_[cols]
    else:
        TF = CLEAN

    def make_reg():
        return Pipeline([("imp", SimpleImputer(strategy="median")),
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

    def make_clf():
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("m", lgb.LGBMClassifier(
                             n_estimators=P["clf_n_estimators"],
                             learning_rate=P["clf_learning_rate"],
                             num_leaves=P["clf_num_leaves"],
                             min_child_samples=P["clf_min_child"],
                             subsample=P["clf_subsample"], subsample_freq=1,
                             colsample_bytree=P["clf_colsample"],
                             reg_lambda=P["clf_reg_lambda"],
                             random_state=SEED, n_jobs=-1, verbose=-1))])

    # ------------------------------------------------------------------ CV
    print("\n  validation croisee (5 blocs temporels purges, embargo 3h)...")
    oof = compute_oof(make_reg, make_clf, TF, d, folds)
    cvdf = cv_metrics(oof, alpha, seuil)
    print(f"    acc@±{TOLERANCE} = {cvdf['acc_tol'].mean():.4f} ± {cvdf['acc_tol'].std():.4f}")
    print(f"    R²               = {cvdf['r2'].mean():.4f} ± {cvdf['r2'].std():.4f}")
    print(f"    par pli : {[f'{v:.4f}' for v in cvdf['acc_tol']]}")

    # ------------------------------------------------ entrainement + test gele
    print("\n  entrainement sur les 80% d'entrainement, score unique sur le test gele...")
    Xa, Xb = TF(Xtr, Xte)
    reg = make_reg(); reg.fit(Xa, ytr.to_numpy() - na)
    lab_tr = (np.abs(ytr.to_numpy() - na) > TOLERANCE).astype(int)
    clf = make_clf(); clf.fit(Xa, lab_tr)

    model = GatedHealthIndexForecaster(
        reg, clf, Xa.columns, alpha, seuil,
        metadata=dict(horizon_steps=HORIZON, horizon_label="3h", tolerance=TOLERANCE,
                      seed=SEED, feature_set=fs, params=P,
                      cutoffs={k: str(v) for k, v in d["cutoffs"].items()},
                      n_train=len(Xa), n_test=len(Xb),
                      sklearn_note="predict(X, hi_now) -> health_index a t+18"))

    yhat = model.predict(Xte, nb)
    test = regression_metrics(yte, yhat, Xa.shape[1], hi_now=nb)
    lab_te = (np.abs(yte.to_numpy() - nb) > TOLERANCE).astype(int)
    prob_te = clf.predict_proba(Xb)[:, 1]

    print(f"\n  {'':<26}{'TEST GELE':>12}{'PERSISTANCE':>14}")
    print(f"    {'accuracy @ ±0.01':<24}{test['acc_tol']:>12.4f}{test['acc_tol_persist']:>14.4f}")
    print(f"    {'R²':<24}{test['r2']:>12.4f}{test['r2_persist']:>14.4f}")
    print(f"    {'R² ajuste':<24}{test['adj_r2']:>12.4f}{'':>14}")
    print(f"    {'RMSE':<24}{test['rmse']:>12.5f}")
    print(f"    {'MAE':<24}{test['mae']:>12.5f}")
    print(f"    {'MedAE':<24}{test['medae']:>12.5f}")
    print(f"    {'MAPE (%)':<24}{test['mape']:>12.4f}")
    print(f"    {'variance expliquee':<24}{test['explained_var']:>12.4f}")
    print(f"    {'erreur max':<24}{test['max_error']:>12.5f}")
    print(f"    {'biais':<24}{test['bias']:>+12.5f}")
    print(f"    {'skill vs persistance':<24}{test['skill_vs_persist']:>+12.4f}")
    print(f"\n    ecart test - CV (acc) : {test['acc_tol'] - cvdf['acc_tol'].mean():+.4f}")
    print(f"    classifieur de porte  : AUC={roc_auc_score(lab_te, prob_te):.4f}  "
          f"PR-AUC={average_precision_score(lab_te, prob_te):.4f}  "
          f"Brier={brier_score_loss(lab_te, prob_te):.5f}  "
          f"logloss={log_loss(lab_te, np.clip(prob_te, 1e-6, 1-1e-6)):.5f}")

    per_asset = {}
    for asset, colname in ASSET_COLS.items():
        m = (mte[colname] == 1).to_numpy()
        per_asset[asset] = regression_metrics(yte[m], yhat[m], hi_now=nb[m])
        print(f"    {asset}: acc={per_asset[asset]['acc_tol']:.4f} "
              f"(persistance {per_asset[asset]['acc_tol_persist']:.4f})  "
              f"R²={per_asset[asset]['r2']:.4f}  n={m.sum()}")

    # ------------------------------------------------------------- graphiques
    print("\n  graphiques de diagnostic...")
    p1 = residual_diagnostics(yte, yhat, TOLERANCE,
                              "Modele final — residus sur le test gele (horizon 3h)",
                              "01_residus_final.png")
    p0 = iterations_overview(os.path.join(HERE, "results_log.csv"), TOLERANCE)
    print(f"    {p1}")
    print(f"    {p0}")
    try:
        p3, imp = shap_summary(reg.named_steps["m"],
                               pd.DataFrame(reg.named_steps["imp"].transform(Xb),
                                            columns=Xa.columns),
                               list(Xa.columns))
        imp.to_csv(os.path.join(HERE, "importance_shap_final.csv"), index=False,
                   encoding="utf-8-sig")
        print(f"    {p3}")
        print("\n  top 12 features (SHAP) :")
        for _, r in imp.head(12).iterrows():
            print(f"    {r['mean_abs_shap']:.6f}  {r['feature']}")
    except Exception as e:
        print(f"    SHAP indisponible : {type(e).__name__}: {e}")

    # --------------------------------------------------------------- artefact
    art = os.path.join(ART_DIR, "health_index_t18_pipeline.joblib")
    joblib.dump(model, art, compress=3)
    out = {
        "horizon_steps": HORIZON, "horizon_label": "3h", "tolerance": TOLERANCE, "seed": SEED,
        "feature_set": fs, "n_features": int(Xa.shape[1]), "alpha": alpha, "seuil": seuil,
        "cv": {"acc_tol_mean": float(cvdf["acc_tol"].mean()),
               "acc_tol_std": float(cvdf["acc_tol"].std()),
               "r2_mean": float(cvdf["r2"].mean()), "r2_std": float(cvdf["r2"].std()),
               "acc_par_pli": [float(v) for v in cvdf["acc_tol"]]},
        "test": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                 for k, v in test.items()},
        "test_par_machine": {a: {k: float(v) for k, v in m.items()} for a, m in per_asset.items()},
        "classifieur_porte": {"auc": float(roc_auc_score(lab_te, prob_te)),
                              "pr_auc": float(average_precision_score(lab_te, prob_te)),
                              "brier": float(brier_score_loss(lab_te, prob_te)),
                              "part_lignes_corrigees": float((prob_te > seuil).mean())},
        "artefact": art,
    }
    with open(os.path.join(ART_DIR, "metriques_finales.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n  artefact enregistre : {art}  ({os.path.getsize(art)/1e6:.1f} Mo)")
    print(f"  metriques           : {os.path.join(ART_DIR, 'metriques_finales.json')}")
    print(f"  duree totale        : {time.time()-t_start:.0f}s")
    return model, out


if __name__ == "__main__":
    main()
