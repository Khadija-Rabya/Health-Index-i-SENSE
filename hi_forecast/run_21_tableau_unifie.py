"""TABLEAU DE COMPARAISON UNIFIE — tous les modeles, protocole strictement identique.

Chaque modele fournit une prediction d'ecart ; l'architecture a porte et le facteur
alpha sont regles EN CV pour chacun, de la meme facon. Les autoencodeurs sont
traites exactement comme LightGBM, ElasticNet, Random Forest ou la persistance :
memes plis, meme test gele, meme bootstrap par blocs. Aucun traitement de faveur.

Colonnes produites : CV acc ± sd, CV skill, test acc, test R2, RMSE, skill,
et IC 95 % bootstrap par blocs (session) du gain de skill.
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, "C:\\pylib")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

import catboost as cb
import lightgbm as lgb
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from ae_models import ConvAE, DenseAE, DenseVAE, SeqAE
from run_16_ae_roleA import (ALPHAS, PROBS, TOP20, bootstrap_ci, dense_builder,
                             evaluate, load, make_clf, make_reg, seq_builder)

MASKED = os.environ.get("UNIF_MASKED", "0") == "1"
LAB = "viscosite masquee" if MASKED else "etiquette actuelle"


def sc(*s):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler())] + list(s))


def ns(*s):
    return Pipeline([("imp", SimpleImputer(strategy="median"))] + list(s))


def top20_builder(make_model):
    """Meme squelette que les AE, mais SANS variables d'autoencodeur : le modele
    recoit uniquement les 20 variables SHAP. C'est le point de comparaison."""
    def _b(d, itr, iap):
        sb = iap if iap is not None else np.arange(len(d["Tte"]))
        Ta = d["Ttr"].to_numpy()[itr]
        Tb = (d["Ttr"].to_numpy()[iap] if iap is not None else d["Tte"].to_numpy())
        return Ta, Tb, itr, sb, None
    return _b


CLASSIQUES = {
    "Ridge":            lambda: sc(("m", Ridge(alpha=10.0, random_state=42))),
    "Lasso":            lambda: sc(("m", Lasso(alpha=1e-4, max_iter=5000, random_state=42))),
    "ElasticNet":       lambda: sc(("m", ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=5000, random_state=42))),
    "Random Forest":    lambda: ns(("m", RandomForestRegressor(n_estimators=300, max_depth=14, max_features="sqrt", random_state=42, n_jobs=-1))),
    "Extra Trees":      lambda: ns(("m", ExtraTreesRegressor(n_estimators=300, max_depth=14, max_features="sqrt", random_state=42, n_jobs=-1))),
    "XGBoost":          lambda: ns(("m", XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1))),
    "LightGBM (l2)":    lambda: ns(("m", lgb.LGBMRegressor(objective="l2", n_estimators=400, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1))),
    "CatBoost":         lambda: ns(("m", cb.CatBoostRegressor(iterations=400, depth=6, learning_rate=0.05, random_seed=42, verbose=0, thread_count=-1))),
    "HistGB (MAE)":     lambda: ns(("m", HistGradientBoostingRegressor(loss="absolute_error", max_iter=400, learning_rate=0.05, random_state=42))),
    "MLP (128,64)":     lambda: sc(("m", MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=300, early_stopping=True, random_state=42))),
    "LightGBM Huber (retenu)": lambda: make_reg(),
}

AUTOENCODEURS = {
    "AE-1 Dense (latent 8)": (lambda: dense_builder(lambda: DenseAE(latent=8, hidden=64)), False),
    "AE-1 Dense debruiteur (lat 16, bruit 0,3)":
        (lambda: dense_builder(lambda: DenseAE(latent=16, hidden=96, noise=0.3)), False),
    "AE-2 GRU sequentiel (W=18)":
        (lambda: seq_builder(lambda: SeqAE(latent=16, hidden=32, cell="gru")), False),
    "AE-2 LSTM sequentiel (W=18)":
        (lambda: seq_builder(lambda: SeqAE(latent=16, hidden=32, cell="lstm")), False),
    "AE-2 GRU seq2seq direct (W=18)":
        (lambda: seq_builder(lambda: SeqAE(latent=16, hidden=32, cell="gru", forecast=True, fc_weight=5.0), direct=True), True),
    "AE-3 Conv1D (W=18)":
        (lambda: seq_builder(lambda: ConvAE(latent=16, ch=32)), False),
    "AE-3 VAE dense (latent 8)":
        (lambda: dense_builder(lambda: DenseVAE(latent=8, hidden=64)), False),
}

print("=" * 104)
print(f"TABLEAU UNIFIE — {LAB}")
print("=" * 104)
d = load(MASKED)
ROWS = []

# ------------------------------------------------------------- persistance
from protocol import TOLERANCE, regression_metrics
from sklearn.metrics import mean_squared_error
yte, nte = d["yte"], d["nte"]
sess = d["mte"]["session_id"].to_numpy()
m0 = regression_metrics(yte, nte, hi_now=nte)
from run_16_ae_roleA import N_SPLITS
from protocol import PurgedTimeSeriesSplit
cv = PurgedTimeSeriesSplit(n_splits=N_SPLITS)
folds = list(cv.split(d["Xtr"], times=d["mtr"]["created_at"]))
pers_cv = float(np.mean([np.mean(np.abs(d["ytr"][b] - d["ntr"][b]) <= TOLERANCE) for _, b in folds]))
pers_sd = float(np.std([np.mean(np.abs(d["ytr"][b] - d["ntr"][b]) <= TOLERANCE) for _, b in folds]))
ROWS.append(dict(famille="Persistance (reference)", categorie="reference",
                 cv_acc=pers_cv, cv_acc_sd=pers_sd, cv_skill=0.0,
                 test_acc=m0["acc_tol"], test_r2=m0["r2"], test_rmse=m0["rmse"],
                 test_skill=0.0, ic_bas=0.0, ic_haut=0.0, significatif=False,
                 n_test=int(len(yte)), secondes=0.0))
print(f"  {'Persistance (reference)':<44} CV {pers_cv:.4f}  test acc {m0['acc_tol']:.4f}  "
      f"R2 {m0['r2']:.4f}  skill 0,0000")

# ------------------------------------------------------------ familles classiques
for nom, mk in CLASSIQUES.items():
    t0 = time.time()
    try:
        r = evaluate(nom, d, top20_builder(mk), verbose=False, reg_factory=mk)
    except Exception as e:
        print(f"  {nom:<44} ECHEC : {type(e).__name__}: {e}")
        continue
    g = r["scopes"].get("global", {})
    mc = r.get("metriques_completes", {})
    ROWS.append(dict(famille=nom, categorie="classique",
                     cv_acc=r["cv_acc"], cv_acc_sd=r["cv_acc_sd"], cv_skill=r["cv_skill"],
                     test_acc=r["test_acc"], test_r2=r["test_r2"], test_rmse=r["test_rmse"],
                     test_skill=r["test_skill"], ic_bas=g.get("ci_skill", [np.nan])[0],
                     ic_haut=g.get("ci_skill", [np.nan, np.nan])[1],
                     significatif=g.get("skill_clears_zero", False),
                     adj_r2=mc.get("adj_r2"), mae=mc.get("mae"), medae=mc.get("medae"),
                     mape=mc.get("mape"), var_expliquee=mc.get("explained_var"),
                     erreur_max=mc.get("max_error"), biais=mc.get("bias"),
                     etat_accuracy=r.get("etat_accuracy"),
                     etat_bal_accuracy=r.get("etat_bal_accuracy"),
                     etat_f1_macro=r.get("etat_f1_macro"), etat_mcc=r.get("etat_mcc"),
                     etat_kappa=r.get("etat_kappa"), alerte_rappel=r.get("alerte_rappel"),
                     alerte_precision=r.get("alerte_precision"), alerte_f1=r.get("alerte_f1"),
                     n_test=r["n_test"], secondes=round(time.time() - t0, 1)))
    print(f"  {nom:<44} CV {r['cv_acc']:.4f}±{r['cv_acc_sd']:.4f}  test acc {r['test_acc']:.4f}  "
          f"R2 {r['test_r2']:.4f}  skill {r['test_skill']:+.4f}  "
          f"IC[{g.get('ci_skill',[np.nan,np.nan])[0]:+.4f},{g.get('ci_skill',[np.nan,np.nan])[1]:+.4f}]"
          f"  [{time.time()-t0:.0f}s]")

# ---------------------------------------------------------------- autoencodeurs
for nom, (mk, direct) in AUTOENCODEURS.items():
    t0 = time.time()
    try:
        r = evaluate(nom, d, mk(), direct=direct, verbose=False)
    except Exception as e:
        print(f"  {nom:<44} ECHEC : {type(e).__name__}: {e}")
        continue
    g = r["scopes"].get("global", {})
    mc = r.get("metriques_completes", {})
    ROWS.append(dict(famille=nom, categorie="autoencodeur",
                     cv_acc=r["cv_acc"], cv_acc_sd=r["cv_acc_sd"], cv_skill=r["cv_skill"],
                     test_acc=r["test_acc"], test_r2=r["test_r2"], test_rmse=r["test_rmse"],
                     test_skill=r["test_skill"], ic_bas=g.get("ci_skill", [np.nan])[0],
                     ic_haut=g.get("ci_skill", [np.nan, np.nan])[1],
                     significatif=g.get("skill_clears_zero", False),
                     adj_r2=mc.get("adj_r2"), mae=mc.get("mae"), medae=mc.get("medae"),
                     mape=mc.get("mape"), var_expliquee=mc.get("explained_var"),
                     erreur_max=mc.get("max_error"), biais=mc.get("bias"),
                     etat_accuracy=r.get("etat_accuracy"),
                     etat_bal_accuracy=r.get("etat_bal_accuracy"),
                     etat_f1_macro=r.get("etat_f1_macro"), etat_mcc=r.get("etat_mcc"),
                     etat_kappa=r.get("etat_kappa"), alerte_rappel=r.get("alerte_rappel"),
                     alerte_precision=r.get("alerte_precision"), alerte_f1=r.get("alerte_f1"),
                     n_test=r["n_test"], secondes=round(time.time() - t0, 1)))
    print(f"  {nom:<44} CV {r['cv_acc']:.4f}±{r['cv_acc_sd']:.4f}  test acc {r['test_acc']:.4f}  "
          f"R2 {r['test_r2']:.4f}  skill {r['test_skill']:+.4f}  "
          f"IC[{g.get('ci_skill',[np.nan,np.nan])[0]:+.4f},{g.get('ci_skill',[np.nan,np.nan])[1]:+.4f}]"
          f"  [{time.time()-t0:.0f}s]")

df = pd.DataFrame(ROWS).sort_values("test_skill", ascending=False)
suffix = "masquee" if MASKED else "actuelle"
df.to_csv(os.path.join(HERE, f"tableau_unifie_{suffix}.csv"), index=False, encoding="utf-8-sig")
print("\n" + "=" * 104)
print(df[["famille", "categorie", "cv_acc", "cv_skill", "test_acc", "test_r2",
          "test_skill", "significatif"]].to_string(index=False,
                                                   float_format=lambda v: f"{v:.4f}"))
print(f"\n  -> hi_forecast/tableau_unifie_{suffix}.csv")
