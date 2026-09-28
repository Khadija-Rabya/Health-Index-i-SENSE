"""PREDICTION D'ALERTES a t+18 (~3 h) — reformulation en CLASSIFICATION.

L'objectif reel du systeme n'est pas de suivre la valeur de l'indice mais de
declencher une alerte au bon moment. La regression n'y arrive pas : elle fait
match nul avec la persistance sur la prediction d'etat. On change donc de cible.

TROIS CADRAGES, par valeur operationnelle croissante :

  C1  ETAT a t+18 (3 classes Normal / Surveillance / Alarme).
      Reference : persistance (l'etat dans 3 h = l'etat actuel).

  C2  ALERTE a t+18 (binaire : etat != Normal).
      Reference : persistance.

  C3  NOUVELLE ALERTE a t+18 (transition : Normal maintenant -> alerte dans 3 h).
      *** La persistance ne peut STRUCTURELLEMENT jamais predire une transition :
      son rappel est 0 par construction. Tout rappel non nul est un gain net. ***
      C'est le cadrage qui correspond au besoin : prevenir AVANT que ca casse.

Protocole inchange : etiquette gelee, test gele, CV 5 blocs purges + embargo 3 h,
graine 42, seuil de decision regle EN CV (jamais sur le test).
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, "C:\\pylib")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss, cohen_kappa_score,
                             confusion_matrix, f1_score, log_loss,
                             matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline

from ae_models import DenseAE
from protocol import (ASSET_COLS, HORIZON, N_SPLITS, SEED, PurgedTimeSeriesSplit,
                      build_xy, feature_columns, prepare, split_frozen)
from transforms import chain, drop_constant_and_dupes, prune_correlated

CLEAN = chain(drop_constant_and_dupes, prune_correlated(0.995))
STATES = ["Normal", "Surveillance", "Alarme"]
N_BOOT = 2000


def load(masked):
    """Construit X a t, et les cibles de classification a t+18."""
    df, _, cutoffs, _ = prepare(verbose=False, mask_off_viscosity=masked)
    d = df.sort_values(["session_id", "created_at"]).reset_index(drop=True)
    st_next = d.groupby("session_id", sort=False)["health_state_lf"].shift(-HORIZON)
    valid = st_next.notna().to_numpy()

    feats = feature_columns(d)
    X = d.loc[valid, feats].reset_index(drop=True).astype(np.float64)
    meta = d.loc[valid, ["created_at", "session_id", "health_state_lf", "health_index_lf",
                         "state_OFF"] + list(ASSET_COLS.values())].reset_index(drop=True)
    meta["etat_futur"] = st_next[valid].to_numpy()
    meta["etat_actuel"] = meta["health_state_lf"]

    is_tr = np.zeros(len(meta), dtype=bool)
    for a, c in ASSET_COLS.items():
        is_tr |= ((meta[c] == 1) & (meta["created_at"] < cutoffs[a])).to_numpy()
    itr, ite = np.flatnonzero(is_tr), np.flatnonzero(~is_tr)
    return dict(X=X, meta=meta, itr=itr, ite=ite)


def ae_features(Xa_raw, Xb_raw):
    """Latent + erreur de reconstruction de l'AE dense debruiteur (ajuste sur Xa)."""
    Ca = np.nan_to_num(Xa_raw.to_numpy(), nan=0.0)
    Cb = np.nan_to_num(Xb_raw.to_numpy(), nan=0.0)
    ae = DenseAE(latent=16, hidden=96, noise=0.3).fit(Ca)
    Fa = np.hstack([Ca, ae.encode(Ca), ae.reconstruction_error(Ca)[:, None]])
    Fb = np.hstack([Cb, ae.encode(Cb), ae.reconstruction_error(Cb)[:, None]])
    return Fa, Fb


def make_clf(scale_pos_weight=1.0, n=600):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", lgb.LGBMClassifier(
                         n_estimators=n, learning_rate=0.03, num_leaves=63,
                         min_child_samples=60, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.7, reg_lambda=1.0,
                         scale_pos_weight=scale_pos_weight,
                         random_state=SEED, n_jobs=-1, verbose=-1))])


def binary_metrics(y, p, thr):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(
        seuil=float(thr), n=int(len(y)), n_pos=int(y.sum()),
        VP=int(tp), FP=int(fp), VN=int(tn), FN=int(fn),
        rappel=float(recall_score(y, yh, zero_division=0)),
        precision=float(precision_score(y, yh, zero_division=0)),
        f1=float(f1_score(y, yh, zero_division=0)),
        accuracy=float(accuracy_score(y, yh)),
        bal_accuracy=float(balanced_accuracy_score(y, yh)),
        mcc=float(matthews_corrcoef(y, yh)) if len(np.unique(yh)) > 1 else 0.0,
        kappa=float(cohen_kappa_score(y, yh)),
        specificite=float(tn / max(tn + fp, 1)),
        taux_fausse_alerte=float(fp / max(fp + tn, 1)),
    )


def proba_metrics(y, p):
    if len(np.unique(y)) < 2:
        return {}
    return dict(roc_auc=float(roc_auc_score(y, p)),
                pr_auc=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                log_loss=float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
                prevalence=float(y.mean()))


def tune_threshold(y, p, critere="f1", precision_min=0.50):
    """Regle le seuil EN CV. `critere` :
    - 'f1'          : maximise le F1 ;
    - 'rappel_prec' : maximise le rappel sous contrainte de precision >= seuil
                      (le cadrage maintenance : ne pas rater, sans crier au loup)."""
    grid = np.unique(np.quantile(p, np.linspace(0.50, 0.999, 220)))
    best, bs = 0.5, -1
    for t in grid:
        yh = (p >= t).astype(int)
        if yh.sum() == 0:
            continue
        rec = recall_score(y, yh, zero_division=0)
        pre = precision_score(y, yh, zero_division=0)
        if critere == "f1":
            s = f1_score(y, yh, zero_division=0)
        else:
            s = rec if pre >= precision_min else -1 + pre
        if s > bs:
            bs, best = s, float(t)
    return best


def bootstrap_recall_gain(y, p, thr, y_persist, sess, n_boot=N_BOOT, seed=SEED):
    """IC 95 % par blocs (session) du gain de rappel et de F1 contre la persistance."""
    rng = np.random.default_rng(seed)
    yh = (p >= thr).astype(int)
    uniq = np.unique(sess)
    groups = {u: np.flatnonzero(sess == u) for u in uniq}
    dr, df1 = [], []
    for _ in range(n_boot):
        ii = np.concatenate([groups[u] for u in rng.choice(uniq, len(uniq), replace=True)])
        if y[ii].sum() == 0:
            continue
        dr.append(recall_score(y[ii], yh[ii], zero_division=0)
                  - recall_score(y[ii], y_persist[ii], zero_division=0))
        df1.append(f1_score(y[ii], yh[ii], zero_division=0)
                   - f1_score(y[ii], y_persist[ii], zero_division=0))
    dr, df1 = np.array(dr), np.array(df1)
    return dict(n_blocs=int(len(uniq)),
                gain_rappel=float(recall_score(y, yh, zero_division=0)
                                  - recall_score(y, y_persist, zero_division=0)),
                ic_rappel=[float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5))],
                gain_f1=float(f1_score(y, yh, zero_division=0)
                              - f1_score(y, y_persist, zero_division=0)),
                ic_f1=[float(np.percentile(df1, 2.5)), float(np.percentile(df1, 97.5))],
                rappel_significatif=bool(np.percentile(dr, 2.5) > 0))
