"""AUTOENCODEURS — ROLE A : prevision de health_index a t+18.

Le latent et l'erreur de reconstruction de l'autoencodeur deviennent des
variables supplementaires du regresseur d'ecart, dans l'architecture a porte
deja retenue. Les architectures sequentielles sont en outre testees en prevision
DIRECTE de l'ecart (seq2seq).

Protocole IDENTIQUE a l'audit :
  - etiquette gelee, test gele, CV 5 blocs purges + embargo 3 h, graine 42 ;
  - autoencodeur ET mise a l'echelle ajustes DANS chaque pli, sur les seules
    lignes d'entrainement du pli (les AE sont sensibles a l'echelle) ;
  - test score UNE SEULE FOIS par configuration.

Les fenetres sont construites une fois sur les lignes triees (session, temps) et
ne franchissent JAMAIS une frontiere de session. Une fenetre dont l'ancre tombe
dans le bloc de validation peut remonter sur des lignes du bloc d'entrainement :
ce sont des observations PASSEES de la meme session, ce qu'un deploiement reel
possede aussi. La purge et l'embargo protegent la CIBLE, qui est la seule chose
qui puisse fuir.

Critere de reussite (« > 80 % » ne compte pas : la persistance fait deja 81,6 %) :
  (a) battre la persistance en CV sur acc@±0.01 ET sur le skill ;
  (b) IC 95 % bootstrap par blocs du gain de skill au-dessus de zero, en global,
      par machine, et par etat ON/OFF.
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
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from ae_models import ConvAE, DenseAE, DenseVAE, SeqAE, build_windows
from protocol import (ASSET_COLS, HORIZON, N_SPLITS, SEED, TOLERANCE,
                      PurgedTimeSeriesSplit, build_xy, prepare, regression_metrics,
                      split_frozen)
from transforms import chain, drop_constant_and_dupes, prune_correlated

CLEAN = chain(drop_constant_and_dupes, prune_correlated(0.995))
W = int(os.environ.get("AE_WINDOW", 18))
N_BOOT = int(os.environ.get("AE_NBOOT", 2000))
ALPHAS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0]
PROBS = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]

with open(os.path.join(HERE, "cache", "final_config.json"), encoding="utf-8") as f:
    P = json.load(f)["params"]
rank = pd.read_csv(os.path.join(HERE, "classements_importance.csv"), index_col=0)
TOP20 = list(rank["shap"].sort_values(ascending=False).head(20).index)


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


# ------------------------------------------------------------------ donnees
def load(masked):
    df, _, cutoffs, _ = prepare(verbose=False, mask_off_viscosity=masked)
    X, y, meta = build_xy(df, HORIZON)
    Xtr, Xte, ytr, yte, mtr, mte = split_frozen(X, y, meta, cutoffs)
    Xa_c, Xb_c = CLEAN(Xtr, Xte)                       # jeu large, entree de l'AE
    d = dict(Xtr=Xtr, Xte=Xte, ytr=ytr.to_numpy(), yte=yte.to_numpy(),
             mtr=mtr, mte=mte, ntr=mtr["hi_now"].to_numpy(), nte=mte["hi_now"].to_numpy(),
             Ctr=Xa_c, Cte=Xb_c,
             Ttr=Xtr[[c for c in TOP20 if c in Xtr.columns]],
             Tte=Xte[[c for c in TOP20 if c in Xte.columns]])
    # fenetres construites une fois, sans franchir de session
    d["Str"], d["Atr"] = build_windows(None, mtr["session_id"].to_numpy(),
                                       d["Ttr"].to_numpy(), W)
    d["Ste"], d["Ate"] = build_windows(None, mte["session_id"].to_numpy(),
                                       d["Tte"].to_numpy(), W)
    return d


# ------------------------------------------------------------- significativite
def bootstrap_ci(y, n, yhat, sess, idx, n_boot=N_BOOT, seed=SEED):
    if len(idx) < 30:
        return None
    rng = np.random.default_rng(seed)
    hm = (np.abs(y - yhat) <= TOLERANCE).astype(int)
    hp = (np.abs(y - n) <= TOLERANCE).astype(int)
    s = sess[idx]; uniq = np.unique(s)
    groups = {u: idx[s == u] for u in uniq}
    da, ds = [], []
    for _ in range(n_boot):
        ii = np.concatenate([groups[u] for u in rng.choice(uniq, len(uniq), replace=True)])
        da.append(hm[ii].mean() - hp[ii].mean())
        mp = mean_squared_error(y[ii], n[ii])
        ds.append(1 - mean_squared_error(y[ii], yhat[ii]) / mp if mp > 0 else np.nan)
    da, ds = np.array(da), np.array(ds)
    mp = mean_squared_error(y[idx], n[idx])
    lo_s = float(np.nanpercentile(ds, 2.5))
    return dict(n=int(len(idx)), n_blocks=int(len(uniq)),
                delta_acc=float(hm[idx].mean() - hp[idx].mean()),
                ci_acc=[float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))],
                skill=float(1 - mean_squared_error(y[idx], yhat[idx]) / mp) if mp > 0 else np.nan,
                ci_skill=[lo_s, float(np.nanpercentile(ds, 97.5))],
                skill_clears_zero=bool(lo_s > 0),
                acc_clears_zero=bool(np.percentile(da, 2.5) > 0))


def scopes_for(d, sel, y, n, yhat):
    mte = d["mte"].iloc[sel].reset_index(drop=True)
    off = d["Xte"]["state_OFF"].to_numpy()[sel] if "state_OFF" in d["Xte"] else np.zeros(len(sel))
    A = (mte[ASSET_COLS["Motosoufflante A"]] == 1).to_numpy()
    B = (mte[ASSET_COLS["Motosoufflante B"]] == 1).to_numpy()
    sess = mte["session_id"].to_numpy()
    sc = {"global": np.arange(len(y)), "A": np.flatnonzero(A), "B": np.flatnonzero(B),
          "ON": np.flatnonzero(off == 0), "OFF": np.flatnonzero(off == 1),
          "A/ON": np.flatnonzero(A & (off == 0)), "A/OFF": np.flatnonzero(A & (off == 1)),
          "B/ON": np.flatnonzero(B & (off == 0)), "B/OFF": np.flatnonzero(B & (off == 1))}
    out = {}
    for k, ii in sc.items():
        r = bootstrap_ci(y, n, yhat, sess, ii)
        if r:
            out[k] = r
    return out


# ------------------------------------------------------------ moteur commun
def evaluate(name, d, build_feats, direct=False, verbose=True, reg_factory=None):
    """build_feats(d, idx_tr, idx_ap, fit) -> (F_tr, F_ap, sel_tr, sel_ap, delta_ap|None)
    `idx_ap` = None signifie « appliquer au TEST gele ».

    reg_factory : fabrique du regresseur d'ecart. Par defaut le LightGBM Huber
    retenu ; on peut injecter n'importe quelle famille pour que la comparaison
    porte sur le MODELE et non seulement sur les variables."""
    make_r = reg_factory or make_reg
    cv = PurgedTimeSeriesSplit(n_splits=N_SPLITS)
    folds = list(cv.split(d["Xtr"], times=d["mtr"]["created_at"]))
    t0 = time.time()

    oof = []
    for itr, iva in folds:
        Fa, Fb, sa, sb, dpred = build_feats(d, itr, iva)
        ya, yb = d["ytr"][sa], d["ytr"][sb]
        na_, nb_ = d["ntr"][sa], d["ntr"][sb]
        if not direct:
            r = make_r(); r.fit(Fa, ya - na_); dpred = r.predict(Fb)
        c = make_clf(); c.fit(Fa, (np.abs(ya - na_) > TOLERANCE).astype(int))
        oof.append(dict(y=yb, n=nb_, d=dpred, p=c.predict_proba(Fb)[:, 1]))

    persist_cv = float(np.mean([np.mean(np.abs(o["y"] - o["n"]) <= TOLERANCE) for o in oof]))
    best = None
    for a in ALPHAS:
        for pth in PROBS:
            accs, sks = [], []
            for o in oof:
                yh = o["n"] + a * o["d"] * (o["p"] > pth)
                accs.append(np.mean(np.abs(o["y"] - yh) <= TOLERANCE))
                mp = mean_squared_error(o["y"], o["n"])
                sks.append(1 - mean_squared_error(o["y"], yh) / mp if mp > 0 else np.nan)
            s = float(np.mean(accs))
            if best is None or s > best["cv_acc"]:
                best = dict(alpha=a, seuil=pth, cv_acc=s, cv_acc_sd=float(np.std(accs)),
                            cv_skill=float(np.nanmean(sks)))

    Fa, Fb, sa, sb, dte = build_feats(d, np.arange(len(d["Xtr"])), None)
    ya, na_ = d["ytr"][sa], d["ntr"][sa]
    if not direct:
        r = make_r(); r.fit(Fa, ya - na_); dte = r.predict(Fb)
    c = make_clf(); c.fit(Fa, (np.abs(ya - na_) > TOLERANCE).astype(int))
    pte = c.predict_proba(Fb)[:, 1]
    y_te, n_te = d["yte"][sb], d["nte"][sb]
    yhat = np.clip(n_te + best["alpha"] * dte * (pte > best["seuil"]), 0, 1)

    m = regression_metrics(y_te, yhat, hi_now=n_te)
    sc = scopes_for(d, sb, y_te, n_te, yhat)
    crit_a = (best["cv_acc"] > persist_cv) and (best["cv_skill"] > 0)
    keys = [k for k in ("global", "A", "B", "ON", "OFF") if k in sc]
    crit_b = all(sc[k]["skill_clears_zero"] for k in keys)

    # --- metriques de CLASSIFICATION de l'etat, derivees de l'indice predit ---
    # (les seuils viennent des centiles du train : meme convention que le projet)
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                                 f1_score, matthews_corrcoef, precision_score, recall_score)
    thr_s = float(np.quantile(d["ntr"], 0.05)); thr_a = float(np.quantile(d["ntr"], 0.01))
    to_state = lambda v: np.where(v <= thr_a, "Alarme",
                                  np.where(v <= thr_s, "Surveillance", "Normal"))
    s_true, s_mod = to_state(y_te), to_state(yhat)
    LB = ["Normal", "Surveillance", "Alarme"]
    deg_t, deg_p = (s_true != "Normal").astype(int), (s_mod != "Normal").astype(int)
    clf_m = dict(
        etat_accuracy=float(accuracy_score(s_true, s_mod)),
        etat_bal_accuracy=float(balanced_accuracy_score(s_true, s_mod)),
        etat_f1_macro=float(f1_score(s_true, s_mod, average="macro", labels=LB, zero_division=0)),
        etat_mcc=float(matthews_corrcoef(deg_t, deg_p)) if len(np.unique(deg_p)) > 1 else 0.0,
        etat_kappa=float(cohen_kappa_score(s_true, s_mod)),
        alerte_rappel=float(recall_score(deg_t, deg_p, zero_division=0)),
        alerte_precision=float(precision_score(deg_t, deg_p, zero_division=0)),
        alerte_f1=float(f1_score(deg_t, deg_p, zero_division=0)))

    res = dict(nom=name, n_test=int(len(y_te)), alpha=best["alpha"], seuil=best["seuil"],
               metriques_completes={k: float(v) for k, v in m.items()}, **clf_m,
               persist_cv_acc=persist_cv, cv_acc=best["cv_acc"], cv_acc_sd=best["cv_acc_sd"],
               cv_skill=best["cv_skill"], test_acc=float(m["acc_tol"]),
               test_acc_persist=float(m["acc_tol_persist"]), test_r2=float(m["r2"]),
               test_rmse=float(m["rmse"]), test_skill=float(m["skill_vs_persist"]),
               scopes=sc, critere_a=bool(crit_a), critere_b=bool(crit_b),
               passe=bool(crit_a and crit_b), secondes=round(time.time() - t0, 1))
    if verbose:
        print(f"    CV   acc={best['cv_acc']:.4f}±{best['cv_acc_sd']:.4f} "
              f"(persist {persist_cv:.4f})  skill={best['cv_skill']:+.4f}"
              f"  -> (a) {'OUI' if crit_a else 'NON'}")
        print(f"    TEST acc={m['acc_tol']:.4f} (persist {m['acc_tol_persist']:.4f})  "
              f"R²={m['r2']:.4f}  skill={m['skill_vs_persist']:+.4f}  n={len(y_te)}  "
              f"[{res['secondes']}s]")
        for k in keys:
            s = sc[k]
            print(f"      {k:<7} skill={s['skill']:+.4f} "
                  f"IC[{s['ci_skill'][0]:+.4f},{s['ci_skill'][1]:+.4f}] "
                  f"blocs={s['n_blocks']:>2}  >0 : {'OUI' if s['skill_clears_zero'] else 'non'}")
        print(f"    -> (b) {'OUI' if crit_b else 'NON'}   VERDICT : "
              f"{'*** PASSE ***' if res['passe'] else 'ECHEC'}")
    return res


# ================================================== constructeurs de variables
def dense_builder(make_ae):
    """AE-1 : AE tabulaire sur le jeu large ; latent + erreur ajoutes au top-20."""
    def _b(d, itr, iap):
        Ca = d["Ctr"].to_numpy()[itr]
        ae = make_ae().fit(np.nan_to_num(Ca, nan=0.0))
        if iap is None:
            Cb, Tb, sb = d["Cte"].to_numpy(), d["Tte"].to_numpy(), np.arange(len(d["Cte"]))
        else:
            Cb, Tb, sb = d["Ctr"].to_numpy()[iap], d["Ttr"].to_numpy()[iap], iap
        Ta = d["Ttr"].to_numpy()[itr]
        Fa = np.hstack([Ta, ae.encode(np.nan_to_num(Ca, nan=0.0)),
                        ae.reconstruction_error(np.nan_to_num(Ca, nan=0.0))[:, None]])
        Fb = np.hstack([Tb, ae.encode(np.nan_to_num(Cb, nan=0.0)),
                        ae.reconstruction_error(np.nan_to_num(Cb, nan=0.0))[:, None]])
        return Fa, Fb, itr, sb, None
    return _b


def seq_builder(make_ae, direct=False):
    """AE-2 / AE-3 : AE sequentiel sur une fenetre de W pas.
    `direct=True` utilise la tete de prevision seq2seq au lieu du regresseur."""
    def _b(d, itr, iap):
        pos_tr = np.isin(d["Atr"], itr)
        Sa, sa = d["Str"][pos_tr], d["Atr"][pos_tr]
        delta_a = d["ytr"][sa] - d["ntr"][sa]
        ae = make_ae()
        ae.fit(Sa, delta_a) if direct else ae.fit(Sa)
        if iap is None:
            Sb, sb, Tb = d["Ste"], d["Ate"], d["Tte"].to_numpy()[d["Ate"]]
        else:
            pos = np.isin(d["Atr"], iap)
            Sb, sb = d["Str"][pos], d["Atr"][pos]
            Tb = d["Ttr"].to_numpy()[sb]
        Ta = d["Ttr"].to_numpy()[sa]
        Fa = np.hstack([Ta, ae.encode(Sa), ae.reconstruction_error(Sa)[:, None]])
        Fb = np.hstack([Tb, ae.encode(Sb), ae.reconstruction_error(Sb)[:, None]])
        dpred = ae.predict_delta(Sb) if direct else None
        return Fa, Fb, sa, sb, dpred
    return _b
