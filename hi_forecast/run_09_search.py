"""PHASE 3 — iterations 29 et 30 : recherche d'hyperparametres (levier 9).

Etape 1 : recherche ALEATOIRE sur un espace large (equivalent RandomizedSearchCV,
          reimplementee car l'architecture a porte comporte DEUX modeles et une
          regle de decision, que sklearn ne sait pas optimiser conjointement).
Etape 2 : raffinement Optuna (TPE) autour de la meilleure region.

Chaque essai = 5 ajustements du regresseur + 5 du classifieur + balayage de la
porte. Selection sur cv_acc uniquement. L'espace de recherche est journalise.
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
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from gated import compute_oof, fit_and_test, get_folds, make_row, summarise, sweep_gate
from harness import append_log, show_log
from protocol import SEED
from transforms import chain, drop_constant_and_dupes, prune_correlated

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = chain(drop_constant_and_dupes, prune_correlated(0.995))
d, folds = get_folds()

N_RANDOM = int(os.environ.get("N_RANDOM", 18))
N_OPTUNA = int(os.environ.get("N_OPTUNA", 22))

SPACE = {
    "reg_n_estimators":    [200, 300, 400, 600, 800],
    "reg_learning_rate":   (0.01, 0.15),
    "reg_num_leaves":      [15, 31, 63, 127],
    "reg_min_child":       [20, 40, 80, 150, 300],
    "reg_subsample":       (0.6, 1.0),
    "reg_colsample":       (0.4, 1.0),
    "reg_reg_lambda":      (1e-3, 30.0),
    "reg_huber_alpha":     (0.001, 0.02),
    "clf_n_estimators":    [200, 300, 400, 600],
    "clf_learning_rate":   (0.01, 0.15),
    "clf_num_leaves":      [15, 31, 63, 127],
    "clf_min_child":       [20, 40, 80, 150, 300],
    "clf_subsample":       (0.6, 1.0),
    "clf_colsample":       (0.4, 1.0),
    "clf_reg_lambda":      (1e-3, 30.0),
}
with open(os.path.join(HERE, "espace_recherche.json"), "w", encoding="utf-8") as f:
    json.dump({k: (list(v) if isinstance(v, list) else {"min": v[0], "max": v[1]})
               for k, v in SPACE.items()}, f, indent=2, ensure_ascii=False)


def build(p):
    reg = lambda: Pipeline([("imp", SimpleImputer(strategy="median")),
                            ("m", lgb.LGBMRegressor(
                                objective="huber", alpha=p["reg_huber_alpha"],
                                n_estimators=p["reg_n_estimators"],
                                learning_rate=p["reg_learning_rate"],
                                num_leaves=p["reg_num_leaves"],
                                min_child_samples=p["reg_min_child"],
                                subsample=p["reg_subsample"], subsample_freq=1,
                                colsample_bytree=p["reg_colsample"],
                                reg_lambda=p["reg_reg_lambda"],
                                random_state=SEED, n_jobs=-1, verbose=-1))])
    clf = lambda: Pipeline([("imp", SimpleImputer(strategy="median")),
                            ("m", lgb.LGBMClassifier(
                                n_estimators=p["clf_n_estimators"],
                                learning_rate=p["clf_learning_rate"],
                                num_leaves=p["clf_num_leaves"],
                                min_child_samples=p["clf_min_child"],
                                subsample=p["clf_subsample"], subsample_freq=1,
                                colsample_bytree=p["clf_colsample"],
                                reg_lambda=p["clf_reg_lambda"],
                                random_state=SEED, n_jobs=-1, verbose=-1))])
    return reg, clf


def score(p):
    reg, clf = build(p)
    oof = compute_oof(reg, clf, CLEAN, d, folds)
    grid, b = sweep_gate(oof)
    return b["cv_acc"], b, oof


# ------------------------------------------------------------- recherche aleatoire
print("=" * 118)
print(f"ETAPE 1 — RECHERCHE ALEATOIRE ({N_RANDOM} essais, espace large)")
print("=" * 118)
rng = np.random.default_rng(SEED)


def sample(rng):
    p = {}
    for k, v in SPACE.items():
        p[k] = float(rng.uniform(*v)) if isinstance(v, tuple) else int(rng.choice(v))
    return p


trials = []
best_p, best_s, best_b, best_oof = None, -1, None, None
for i in range(N_RANDOM):
    p = sample(rng)
    t0 = time.time()
    try:
        s, b, oof = score(p)
        trials.append({**p, "cv_acc": s, "alpha": b["alpha"], "seuil": b["seuil"]})
        flag = ""
        if s > best_s:
            best_p, best_s, best_b, best_oof = p, s, b, oof; flag = "  <-- meilleur"
        print(f"  essai {i+1:>2}/{N_RANDOM}  cv_acc={s:.4f}  (a={b['alpha']}, s={b['seuil']})"
              f"  [{time.time()-t0:>4.0f}s]{flag}")
    except Exception as e:
        print(f"  essai {i+1:>2}/{N_RANDOM}  ECHEC : {type(e).__name__}: {e}")

print(f"\n  meilleur apres recherche aleatoire : cv_acc={best_s:.4f}")

res = fit_and_test(*build(best_p), CLEAN, best_b["alpha"], best_b["seuil"], d)
row = make_row(29, "Porte + recherche aleatoire",
               f"RandomizedSearch maison, {N_RANDOM} essais sur 15 hyperparametres "
               f"(espace journalise dans espace_recherche.json)",
               best_oof, best_b["alpha"], best_b["seuil"], res, d)
append_log(row); summarise(row, "it29 recherche aleatoire")

# ------------------------------------------------------------------ Optuna (TPE)
print("\n" + "=" * 118)
print(f"ETAPE 2 — RAFFINEMENT OPTUNA/TPE ({N_OPTUNA} essais, amorce = meilleur aleatoire)")
print("=" * 118)
store = {}


def objective(trial):
    p = {
        "reg_n_estimators": trial.suggest_categorical("reg_n_estimators", [300, 400, 600, 800]),
        "reg_learning_rate": trial.suggest_float("reg_learning_rate", 0.01, 0.12, log=True),
        "reg_num_leaves": trial.suggest_categorical("reg_num_leaves", [15, 31, 63, 127]),
        "reg_min_child": trial.suggest_int("reg_min_child", 20, 300, log=True),
        "reg_subsample": trial.suggest_float("reg_subsample", 0.6, 1.0),
        "reg_colsample": trial.suggest_float("reg_colsample", 0.4, 1.0),
        "reg_reg_lambda": trial.suggest_float("reg_reg_lambda", 1e-3, 30.0, log=True),
        "reg_huber_alpha": trial.suggest_float("reg_huber_alpha", 0.001, 0.02, log=True),
        "clf_n_estimators": trial.suggest_categorical("clf_n_estimators", [300, 400, 600]),
        "clf_learning_rate": trial.suggest_float("clf_learning_rate", 0.01, 0.12, log=True),
        "clf_num_leaves": trial.suggest_categorical("clf_num_leaves", [15, 31, 63, 127]),
        "clf_min_child": trial.suggest_int("clf_min_child", 20, 300, log=True),
        "clf_subsample": trial.suggest_float("clf_subsample", 0.6, 1.0),
        "clf_colsample": trial.suggest_float("clf_colsample", 0.4, 1.0),
        "clf_reg_lambda": trial.suggest_float("clf_reg_lambda", 1e-3, 30.0, log=True),
    }
    s, b, oof = score(p)
    store[trial.number] = (p, b, oof)
    return s


study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=SEED))
study.enqueue_trial({k: v for k, v in best_p.items()})
t0 = time.time()
study.optimize(objective, n_trials=N_OPTUNA, show_progress_bar=False)
print(f"  {N_OPTUNA} essais en {time.time()-t0:.0f}s")
print(f"  meilleur cv_acc Optuna = {study.best_value:.4f}")
print(f"  meilleurs parametres :")
for k, v in study.best_params.items():
    print(f"    {k:<22} {v}")

bp, bb, boof = store[study.best_trial.number]
res2 = fit_and_test(*build(bp), CLEAN, bb["alpha"], bb["seuil"], d)
row2 = make_row(30, "Porte + Optuna (TPE)",
                f"Raffinement TPE, {N_OPTUNA} essais, amorce = meilleur de la recherche aleatoire",
                boof, bb["alpha"], bb["seuil"], res2, d)
append_log(row2); summarise(row2, "it30 Optuna")

pd.DataFrame(trials).to_csv(os.path.join(HERE, "recherche_aleatoire.csv"), index=False)
study.trials_dataframe().to_csv(os.path.join(HERE, "recherche_optuna.csv"), index=False)
with open(os.path.join(HERE, "cache", "best_params.json"), "w", encoding="utf-8") as f:
    json.dump({"params": bp, "alpha": float(bb["alpha"]), "seuil": float(bb["seuil"]),
               "cv_acc": float(bb["cv_acc"])}, f, indent=2)

print("\n" + "=" * 118)
show_log(sort_by="cv_acc_tol_mean")
