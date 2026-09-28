"""ALERTE PAR SEUIL SUR L'INDICE PREDIT — la logique du systeme, suivie jusqu'au bout.

Le Health Index porte deja des seuils qui definissent l'etat. Predire l'indice
predit donc l'alerte : il n'y a pas besoin d'un classifieur separe. Ce qui manquait
est le REGLAGE DU SEUIL DE DECISION.

    alerte_predite(t+3h)  <=>  indice_predit(t+3h) <= tau

tau est regle EN CV, jamais sur le test. La reference de persistance recoit
EXACTEMENT le meme traitement — son propre tau_p regle en CV sur l'indice
COURANT — sinon la comparaison serait truquee en faveur du modele.

Deux decisions evaluees :
  D1  ALERTE  : l'etat sera != Normal dans 3 h (toutes les lignes)
  D2  NOUVELLE ALERTE : Normal maintenant -> alerte dans 3 h (lignes Normales)
      la persistance ne peut structurellement jamais la predire.

Trois criteres de reglage, correspondant a trois politiques de maintenance :
  - F1                       : equilibre
  - rappel >= 0,80           : ne pas rater, quitte a se deplacer pour rien
  - precision >= 0,50        : ne pas crier au loup
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

import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline

from ae_models import DenseAE
from protocol import (ASSET_COLS, HORIZON, N_SPLITS, SEED, TOLERANCE,
                      PurgedTimeSeriesSplit, feature_columns, prepare)
from transforms import chain, drop_constant_and_dupes, prune_correlated

CLEAN = chain(drop_constant_and_dupes, prune_correlated(0.995))
with open(os.path.join(HERE, "cache", "final_config.json"), encoding="utf-8") as f:
    P = json.load(f)["params"]
rank = pd.read_csv(os.path.join(HERE, "classements_importance.csv"), index_col=0)
TOP20 = list(rank["shap"].sort_values(ascending=False).head(20).index)
N_BOOT = 2000


def reg():
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


def gate():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", lgb.LGBMClassifier(n_estimators=P["clf_n_estimators"],
                                              learning_rate=P["clf_learning_rate"],
                                              num_leaves=P["clf_num_leaves"],
                                              min_child_samples=P["clf_min_child"],
                                              subsample=P["clf_subsample"], subsample_freq=1,
                                              colsample_bytree=P["clf_colsample"],
                                              reg_lambda=P["clf_reg_lambda"],
                                              random_state=SEED, n_jobs=-1, verbose=-1))])


def load(masked):
    df, _, cutoffs, _ = prepare(verbose=False, mask_off_viscosity=masked)
    d = df.sort_values(["session_id", "created_at"]).reset_index(drop=True)
    y_next = d.groupby("session_id", sort=False)["health_index_lf"].shift(-HORIZON)
    s_next = d.groupby("session_id", sort=False)["health_state_lf"].shift(-HORIZON)
    ok = y_next.notna().to_numpy()

    feats = feature_columns(d)
    X = d.loc[ok, feats].reset_index(drop=True).astype(np.float64)
    m = d.loc[ok, ["created_at", "session_id", "health_index_lf", "health_state_lf",
                   "state_OFF"] + list(ASSET_COLS.values())].reset_index(drop=True)
    m["y"] = y_next[ok].to_numpy()
    m["etat_futur"] = s_next[ok].to_numpy()
    is_tr = np.zeros(len(m), dtype=bool)
    for a, c in ASSET_COLS.items():
        is_tr |= ((m[c] == 1) & (m["created_at"] < cutoffs[a])).to_numpy()
    return dict(X=X, m=m, itr=np.flatnonzero(is_tr), ite=np.flatnonzero(~is_tr))


def predict_index(d, itr, iap):
    """Indice predit a t+18 par le modele AE + porte. iap=None -> test gele."""
    X, m = d["X"], d["m"]
    Xa_raw = X.iloc[itr]
    Xb_raw = X.iloc[iap] if iap is not None else X.iloc[d["ite"]]
    Ca, Cb = CLEAN(Xa_raw, Xb_raw)
    Ca_n = np.nan_to_num(Ca.to_numpy(), nan=0.0)
    Cb_n = np.nan_to_num(Cb.to_numpy(), nan=0.0)
    ae = DenseAE(latent=16, hidden=96, noise=0.3).fit(Ca_n)
    Ta = Xa_raw[[c for c in TOP20 if c in Xa_raw.columns]].to_numpy()
    Tb = Xb_raw[[c for c in TOP20 if c in Xb_raw.columns]].to_numpy()
    Fa = np.hstack([Ta, ae.encode(Ca_n), ae.reconstruction_error(Ca_n)[:, None]])
    Fb = np.hstack([Tb, ae.encode(Cb_n), ae.reconstruction_error(Cb_n)[:, None]])
    na = m["health_index_lf"].to_numpy()[itr]
    ya = m["y"].to_numpy()[itr]
    nb = m["health_index_lf"].to_numpy()[iap if iap is not None else d["ite"]]
    r = reg(); r.fit(Fa, ya - na)
    g = gate(); g.fit(Fa, (np.abs(ya - na) > TOLERANCE).astype(int))
    delta = r.predict(Fb)
    p = g.predict_proba(Fb)[:, 1]
    return np.clip(nb + 1.0 * delta * (p > 0.35), 0, 1), nb


def sweep_tau(score, y, critere, cible=0.80):
    """Regle tau : alerte si score <= tau. `score` est un indice (bas = mauvais)."""
    grid = np.unique(np.quantile(score, np.linspace(0.001, 0.60, 300)))
    best, bs = grid[0], -np.inf
    for t in grid:
        yh = (score <= t).astype(int)
        if yh.sum() == 0:
            continue
        rec = recall_score(y, yh, zero_division=0)
        pre = precision_score(y, yh, zero_division=0)
        if critere == "f1":
            s = f1_score(y, yh, zero_division=0)
        elif critere == "rappel":
            s = pre if rec >= cible else -1 + rec        # max precision sous rappel>=cible
        else:
            s = rec if pre >= cible else -1 + pre        # max rappel sous precision>=cible
        if s > bs:
            bs, best = s, float(t)
    return best


def evaluate(score, y, tau, sess=None, ref=None, n_boot=N_BOOT):
    yh = (score <= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    out = dict(tau=float(tau), n=int(len(y)), n_pos=int(y.sum()),
               VP=int(tp), FP=int(fp), VN=int(tn), FN=int(fn),
               rappel=float(recall_score(y, yh, zero_division=0)),
               precision=float(precision_score(y, yh, zero_division=0)),
               f1=float(f1_score(y, yh, zero_division=0)),
               taux_fausse_alerte=float(fp / max(fp + tn, 1)))
    if len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, -score))
        out["pr_auc"] = float(average_precision_score(y, -score))
    if ref is not None and sess is not None:
        rng = np.random.default_rng(SEED)
        uniq = np.unique(sess)
        groups = {u: np.flatnonzero(sess == u) for u in uniq}
        dr, df = [], []
        for _ in range(n_boot):
            ii = np.concatenate([groups[u] for u in rng.choice(uniq, len(uniq), replace=True)])
            if y[ii].sum() == 0:
                continue
            dr.append(recall_score(y[ii], yh[ii], zero_division=0)
                      - recall_score(y[ii], ref[ii], zero_division=0))
            df.append(f1_score(y[ii], yh[ii], zero_division=0)
                      - f1_score(y[ii], ref[ii], zero_division=0))
        if dr:
            out["gain_rappel"] = float(np.mean(dr))
            out["ic_rappel"] = [float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5))]
            out["gain_f1"] = float(np.mean(df))
            out["ic_f1"] = [float(np.percentile(df, 2.5)), float(np.percentile(df, 97.5))]
            out["significatif"] = bool(np.percentile(dr, 2.5) > 0)
            out["n_blocs"] = int(len(uniq))
    return out


RES = []
for masked, lab in [(False, "etiquette actuelle"), (True, "viscosite masquee")]:
    print("\n" + "#" * 100)
    print(f"#  {lab.upper()}")
    print("#" * 100)
    t0 = time.time()
    d = load(masked)
    m, itr, ite = d["m"], d["itr"], d["ite"]

    # ---- CV : indice predit hors-pli ----
    cv = PurgedTimeSeriesSplit(n_splits=N_SPLITS)
    Xtr = d["X"].iloc[itr]
    oof_pred, oof_pers, oof_idx = [], [], []
    for a, b in cv.split(Xtr, times=m["created_at"].iloc[itr]):
        pa, pb = predict_index(d, itr[a], itr[b])
        oof_pred.append(pa); oof_pers.append(pb); oof_idx.append(itr[b])
    Pcv = np.concatenate(oof_pred); Ncv = np.concatenate(oof_pers)
    Icv = np.concatenate(oof_idx)

    # ---- test gele : une seule fois ----
    Pte, Nte = predict_index(d, itr, None)
    sess_te = m["session_id"].to_numpy()[ite]

    for nom, sel_cv, sel_te, ycv, yte in [
        ("D1 ALERTE (etat != Normal a t+3h)",
         np.ones(len(Icv), bool), np.ones(len(ite), bool),
         (m["etat_futur"].to_numpy()[Icv] != "Normal").astype(int),
         (m["etat_futur"].to_numpy()[ite] != "Normal").astype(int)),
        ("D2 NOUVELLE ALERTE (Normal -> alerte)",
         (m["health_state_lf"].to_numpy()[Icv] == "Normal"),
         (m["health_state_lf"].to_numpy()[ite] == "Normal"),
         ((m["health_state_lf"].to_numpy()[Icv] == "Normal") &
          (m["etat_futur"].to_numpy()[Icv] != "Normal")).astype(int),
         ((m["health_state_lf"].to_numpy()[ite] == "Normal") &
          (m["etat_futur"].to_numpy()[ite] != "Normal")).astype(int))]:
        print(f"\n  {nom}")
        yc, yt = ycv[sel_cv], yte[sel_te]
        pc, nc = Pcv[sel_cv], Ncv[sel_cv]
        pt, nt = Pte[sel_te], Nte[sel_te]
        st = sess_te[sel_te]
        if yt.sum() < 15 or yc.sum() < 15:
            print(f"    trop peu de positifs (CV {int(yc.sum())}, test {int(yt.sum())}) — ECARTE")
            continue
        print(f"    prevalence : CV {yc.mean():.1%}  test {yt.mean():.1%}  "
              f"(n test = {len(yt)}, positifs = {int(yt.sum())})")
        for crit, cible, libelle in [("f1", 0, "F1 maximal"),
                                     ("rappel", 0.80, "rappel >= 0,80"),
                                     ("precision", 0.50, "precision >= 0,50")]:
            tau_m = sweep_tau(pc, yc, crit, cible)
            tau_p = sweep_tau(nc, yc, crit, cible)       # persistance reglee IDENTIQUEMENT
            ref = (nt <= tau_p).astype(int)
            em = evaluate(pt, yt, tau_m, st, ref)
            ep = evaluate(nt, yt, tau_p)
            print(f"    [{libelle:<16}] tau_modele={tau_m:.4f}  tau_persist={tau_p:.4f}")
            print(f"      MODELE      rappel {em['rappel']:.3f}  precision {em['precision']:.3f}"
                  f"  F1 {em['f1']:.3f}  fausses alertes {em['taux_fausse_alerte']:.1%}"
                  f"  (VP={em['VP']} FP={em['FP']} FN={em['FN']})")
            print(f"      PERSISTANCE rappel {ep['rappel']:.3f}  precision {ep['precision']:.3f}"
                  f"  F1 {ep['f1']:.3f}  fausses alertes {ep['taux_fausse_alerte']:.1%}")
            if "gain_rappel" in em:
                print(f"      gain rappel {em['gain_rappel']:+.3f} "
                      f"IC[{em['ic_rappel'][0]:+.3f},{em['ic_rappel'][1]:+.3f}]  "
                      f"gain F1 {em['gain_f1']:+.3f} "
                      f"IC[{em['ic_f1'][0]:+.3f},{em['ic_f1'][1]:+.3f}]  "
                      f"-> {'SIGNIFICATIF' if em['significatif'] else 'non significatif'}")
            RES.append(dict(etiquette=lab, decision=nom, critere=libelle,
                            modele=em, persistance=ep))
    print(f"\n  [{time.time()-t0:.0f}s]")

with open(os.path.join(HERE, "alerte_seuil_resultats.json"), "w", encoding="utf-8") as f:
    json.dump(RES, f, indent=2, ensure_ascii=False, default=float)
rows = [dict(etiquette=r["etiquette"], decision=r["decision"][:30], critere=r["critere"],
             rappel=r["modele"]["rappel"], precision=r["modele"]["precision"],
             f1=r["modele"]["f1"], rappel_p=r["persistance"]["rappel"],
             f1_p=r["persistance"]["f1"], gain_rappel=r["modele"].get("gain_rappel"),
             signif=r["modele"].get("significatif")) for r in RES]
pd.DataFrame(rows).to_csv(os.path.join(HERE, "alerte_seuil_resultats.csv"), index=False,
                          encoding="utf-8-sig")
print("\n" + "=" * 100)
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\n  -> hi_forecast/alerte_seuil_resultats.json / .csv")
