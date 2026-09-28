"""ENTRAINEMENT FINAL PAR MACHINE + COMPARAISON EXHAUSTIVE.

Trois configurations de decision, evaluees pour chaque famille et chaque mode :
  G0  alpha = 0            -> persistance pure (borne basse, aucun apprentissage)
  G1  alpha = 1, sans porte-> correction brute, toujours appliquee
  G*  porte reglee en CV   -> alpha et seuil choisis sur la CV de la machine

Metriques adaptees au TYPE de modele :
  - regression        : 11 metriques (acc±0,01, R², R² aj., RMSE, MAE, MedAE,
                        MAPE, var. expliquee, err. max, biais, skill)
  - decision d'etat   : 8 metriques (accuracy, acc. equilibree, F1 macro, MCC,
                        kappa, rappel/precision/F1 d'alerte)
  - porte (classifieur) : ROC-AUC, PR-AUC, Brier, log loss — uniquement pour les
                        configurations qui en utilisent une
  - significativite   : IC 95 % bootstrap par blocs de session

Les deux modeles recommandes sont en outre ENTRAINES et SAUVEGARDES.
"""
import json
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "C:\\pylib")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

import catboost as cb
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             cohen_kappa_score, f1_score, log_loss,
                             matthews_corrcoef, mean_squared_error, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ae_models import DenseAE
from protocol import (ASSET_COLS, N_SPLITS, SEED, TOLERANCE, PurgedTimeSeriesSplit,
                      regression_metrics)
from run_16_ae_roleA import ALPHAS, PROBS, load, make_clf, make_reg

N_BOOT = 2000
ART = os.path.join(HERE, "artifacts")
os.makedirs(ART, exist_ok=True)


def sc(*s):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler())] + list(s))


def ns(*s):
    return Pipeline([("imp", SimpleImputer(strategy="median"))] + list(s))


FAMILLES = {
    "Ridge":                 dict(reg=lambda: sc(("m", Ridge(alpha=10.0, random_state=SEED))), ae=False),
    "Lasso":                 dict(reg=lambda: sc(("m", Lasso(alpha=1e-4, max_iter=5000, random_state=SEED))), ae=False),
    "ElasticNet":            dict(reg=lambda: sc(("m", ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=5000, random_state=SEED))), ae=False),
    "Random Forest":         dict(reg=lambda: ns(("m", RandomForestRegressor(n_estimators=300, max_depth=14, max_features="sqrt", random_state=SEED, n_jobs=-1))), ae=False),
    "CatBoost":              dict(reg=lambda: ns(("m", cb.CatBoostRegressor(iterations=400, depth=6, learning_rate=0.05, random_seed=SEED, verbose=0, thread_count=-1))), ae=False),
    "LightGBM Huber":        dict(reg=make_reg, ae=False),
    "AE-1 dense debruiteur": dict(reg=make_reg, ae=True),
    "Ridge + AE":            dict(reg=lambda: sc(("m", Ridge(alpha=10.0, random_state=SEED))), ae=True),
}

d = load(False)
mtr, mte = d["mtr"], d["mte"]
MACH = {a: (np.flatnonzero((mtr[c] == 1).to_numpy()),
            np.flatnonzero((mte[c] == 1).to_numpy())) for a, c in ASSET_COLS.items()}
THR_S = float(np.quantile(d["ntr"], 0.05))
THR_A = float(np.quantile(d["ntr"], 0.01))
LB = ["Normal", "Surveillance", "Alarme"]


def to_state(v):
    return np.where(v <= THR_A, "Alarme", np.where(v <= THR_S, "Surveillance", "Normal"))


def feats(itr, iap, use_ae, keep_ae=False):
    Ta = d["Ttr"].to_numpy()[itr]
    Tb = d["Ttr"].to_numpy()[iap] if iap is not None else d["Tte"].to_numpy()
    if not use_ae:
        return Ta, Tb, None
    Ca = np.nan_to_num(d["Ctr"].to_numpy()[itr], nan=0.0)
    Cb = np.nan_to_num(d["Ctr"].to_numpy()[iap] if iap is not None else d["Cte"].to_numpy(), nan=0.0)
    ae = DenseAE(latent=16, hidden=96, noise=0.3).fit(Ca)
    return (np.hstack([Ta, ae.encode(Ca), ae.reconstruction_error(Ca)[:, None]]),
            np.hstack([Tb, ae.encode(Cb), ae.reconstruction_error(Cb)[:, None]]),
            ae if keep_ae else None)


def oof_predictions(itr_rows, cfg):
    cv = PurgedTimeSeriesSplit(n_splits=N_SPLITS)
    out = []
    for a, b in cv.split(np.zeros((len(itr_rows), 1)),
                         times=mtr["created_at"].iloc[itr_rows]):
        ia, ib = itr_rows[a], itr_rows[b]
        Fa, Fb, _ = feats(ia, ib, cfg["ae"])
        ya, na = d["ytr"][ia], d["ntr"][ia]
        r = cfg["reg"](); r.fit(Fa, ya - na)
        c = make_clf(); c.fit(Fa, (np.abs(ya - na) > TOLERANCE).astype(int))
        out.append(dict(idx=ib, y=d["ytr"][ib], n=d["ntr"][ib],
                        d=r.predict(Fb), p=c.predict_proba(Fb)[:, 1]))
    return out


def cv_stats(oof, alpha, prob, restrict=None):
    accs, sks = [], []
    for o in oof:
        m = np.ones(len(o["y"]), bool) if restrict is None else np.isin(o["idx"], restrict)
        if m.sum() < 20:
            continue
        yh = o["n"][m] + alpha * o["d"][m] * (o["p"][m] > prob)
        accs.append(np.mean(np.abs(o["y"][m] - yh) <= TOLERANCE))
        mp = mean_squared_error(o["y"][m], o["n"][m])
        sks.append(1 - mean_squared_error(o["y"][m], yh) / mp if mp > 0 else np.nan)
    if not accs:
        return None
    return dict(cv_acc=float(np.mean(accs)), cv_sd=float(np.std(accs)),
                cv_skill=float(np.nanmean(sks)))


def tune(oof, restrict=None):
    best = None
    for al in ALPHAS:
        for pth in PROBS:
            s = cv_stats(oof, al, pth, restrict)
            if s and (best is None or s["cv_acc"] > best["cv_acc"]):
                best = dict(alpha=al, seuil=pth, **s)
    return best


def full_metrics(y, n, yhat, prob, sess, uses_gate):
    m = regression_metrics(y, yhat, n_features=20, hi_now=n)
    st, sm = to_state(y), to_state(yhat)
    dt, dp = (st != "Normal").astype(int), (sm != "Normal").astype(int)
    out = dict(
        acc_tol=m["acc_tol"], acc_persist=m["acc_tol_persist"], r2=m["r2"],
        adj_r2=m["adj_r2"], rmse=m["rmse"], mae=m["mae"], medae=m["medae"],
        mape=m["mape"], var_expl=m["explained_var"], err_max=m["max_error"],
        biais=m["bias"], skill=m["skill_vs_persist"],
        etat_acc=float(accuracy_score(st, sm)),
        etat_bal=float(balanced_accuracy_score(st, sm)),
        etat_f1m=float(f1_score(st, sm, average="macro", labels=LB, zero_division=0)),
        etat_mcc=float(matthews_corrcoef(dt, dp)) if len(np.unique(dp)) > 1 else 0.0,
        etat_kappa=float(cohen_kappa_score(st, sm)),
        alerte_rappel=float(recall_score(dt, dp, zero_division=0)),
        alerte_prec=float(precision_score(dt, dp, zero_division=0)),
        alerte_f1=float(f1_score(dt, dp, zero_division=0)))
    if uses_gate and prob is not None:
        lab = (np.abs(y - n) > TOLERANCE).astype(int)
        if len(np.unique(lab)) > 1:
            out.update(porte_auc=float(roc_auc_score(lab, prob)),
                       porte_pr_auc=float(average_precision_score(lab, prob)),
                       porte_brier=float(brier_score_loss(lab, prob)),
                       porte_logloss=float(log_loss(lab, np.clip(prob, 1e-6, 1 - 1e-6))))
    rng = np.random.default_rng(SEED)
    uniq = np.unique(sess); groups = {u: np.flatnonzero(sess == u) for u in uniq}
    ds = []
    for _ in range(N_BOOT):
        ii = np.concatenate([groups[u] for u in rng.choice(uniq, len(uniq), replace=True)])
        mp = mean_squared_error(y[ii], n[ii])
        ds.append(1 - mean_squared_error(y[ii], yhat[ii]) / mp if mp > 0 else np.nan)
    lo = float(np.nanpercentile(ds, 2.5))
    out.update(ic_bas=lo, ic_haut=float(np.nanpercentile(ds, 97.5)),
               significatif=bool(lo > 0), n_blocs=int(len(uniq)))
    return out


ROWS = []
FIT_CACHE = {}
print("=" * 108)
print("COMPARAISON EXHAUSTIVE — 8 familles × 2 modes × 3 configurations de porte")
print("=" * 108)

for fam, cfg in FAMILLES.items():
    t0 = time.time()
    for mode in ("commun", "separe"):
        for asset, (tr, te) in MACH.items():
            itr = np.arange(len(d["Xtr"])) if mode == "commun" else tr
            restrict = tr if mode == "commun" else None
            key = (fam, mode, asset if mode == "separe" else "TOUS")
            if key not in FIT_CACHE:
                FIT_CACHE[key] = oof_predictions(itr, cfg)
            oof = FIT_CACHE[key]

            best = tune(oof, restrict)
            Fa, Fb, _ = feats(itr, None, cfg["ae"])
            ya, na = d["ytr"][itr], d["ntr"][itr]
            r = cfg["reg"](); r.fit(Fa, ya - na)
            c = make_clf(); c.fit(Fa, (np.abs(ya - na) > TOLERANCE).astype(int))
            dte = r.predict(Fb); pte = c.predict_proba(Fb)[:, 1]
            y, n = d["yte"][te], d["nte"][te]
            sess = mte["session_id"].to_numpy()[te]

            for gname, al, pth, gate in [("G0 persistance", 0.0, 1.1, False),
                                         ("G1 sans porte", 1.0, -0.1, False),
                                         ("G* porte reglee", best["alpha"], best["seuil"], True)]:
                yhat = np.clip(d["nte"] + al * dte * (pte > pth), 0, 1)[te]
                s = cv_stats(oof, al, pth, restrict) or {}
                m = full_metrics(y, n, yhat, pte[te] if gate else None, sess, gate)
                ROWS.append(dict(famille=fam, mode=mode, machine=asset, config=gname,
                                 alpha=al, seuil=pth if gate else np.nan,
                                 n_test=len(te), **s, **m))
    print(f"  {fam:<24} [{time.time()-t0:.0f}s]")

df = pd.DataFrame(ROWS)
df.to_csv(os.path.join(HERE, "comparaison_finale_complete.csv"), index=False,
          encoding="utf-8-sig")

# ------------------------------------------------- classement par machine
RECO = {}
for asset in MACH:
    print("\n" + "=" * 108)
    print(f"{asset.upper()} — classement (selection sur CV accuracy, porte reglee)")
    print("=" * 108)
    s = df[(df.machine == asset) & (df.config == "G* porte reglee")].sort_values(
        "cv_acc", ascending=False)
    print(f"{'famille':<24}{'mode':>8}{'CV acc':>9}{'CV skill':>10}{'acc':>8}{'pers':>8}"
          f"{'R²':>8}{'RMSE':>8}{'MAE':>8}{'skill':>9}{'MCC':>8}{'F1 al.':>8}{'sig':>5}")
    for _, r in s.iterrows():
        print(f"{r['famille']:<24}{r['mode']:>8}{r['cv_acc']:>9.4f}{r['cv_skill']:>+10.4f}"
              f"{r['acc_tol']:>8.4f}{r['acc_persist']:>8.4f}{r['r2']:>8.4f}{r['rmse']:>8.4f}"
              f"{r['mae']:>8.4f}{r['skill']:>+9.4f}{r['etat_mcc']:>8.4f}"
              f"{r['alerte_f1']:>8.4f}{'OUI' if r['significatif'] else 'non':>5}")
    sig = s[s["significatif"]]
    w = sig.iloc[0] if len(sig) else s.iloc[0]
    RECO[asset] = dict(famille=w["famille"], mode=w["mode"], alpha=float(w["alpha"]),
                       seuil=float(w["seuil"]), cv_acc=float(w["cv_acc"]),
                       test_acc=float(w["acc_tol"]), skill=float(w["skill"]))
    print(f"\n  >>> RETENU (1er en CV parmi les significatifs) : {w['famille']} ({w['mode']})")
    print(f"      CV {w['cv_acc']:.4f}  test acc {w['acc_tol']:.4f} (pers {w['acc_persist']:.4f})"
          f"  skill {w['skill']:+.4f}  IC[{w['ic_bas']:+.4f},{w['ic_haut']:+.4f}]")

# ------------------------------------------------- entrainement des artefacts
print("\n" + "=" * 108)
print("ENTRAINEMENT ET SAUVEGARDE DES ARTEFACTS FINAUX")
print("=" * 108)
bundle = {}
for asset, rec in RECO.items():
    cfg = FAMILLES[rec["famille"]]
    tr, te = MACH[asset]
    itr = np.arange(len(d["Xtr"])) if rec["mode"] == "commun" else tr
    Fa, Fb, ae = feats(itr, None, cfg["ae"], keep_ae=True)
    ya, na = d["ytr"][itr], d["ntr"][itr]
    r = cfg["reg"](); r.fit(Fa, ya - na)
    c = make_clf(); c.fit(Fa, (np.abs(ya - na) > TOLERANCE).astype(int))
    bundle[asset] = dict(famille=rec["famille"], mode=rec["mode"], regresseur=r,
                         porte=c, autoencodeur=ae, utilise_ae=cfg["ae"],
                         alpha=rec["alpha"], seuil=rec["seuil"],
                         colonnes=list(d["Ttr"].columns),
                         colonnes_ae=list(d["Ctr"].columns) if cfg["ae"] else None,
                         seuil_surveillance=THR_S, seuil_alarme=THR_A,
                         tolerance=TOLERANCE, horizon_pas=18)
    print(f"  {asset:<20} {rec['famille']} ({rec['mode']})  alpha={rec['alpha']} "
          f"seuil={rec['seuil']}  -> entraine sur {len(itr)} lignes")

path = os.path.join(ART, "modeles_par_machine.joblib")
joblib.dump(bundle, path, compress=3)
with open(os.path.join(ART, "recommandation_par_machine.json"), "w", encoding="utf-8") as f:
    json.dump(RECO, f, indent=2, ensure_ascii=False)
print(f"\n  artefact : {path}  ({os.path.getsize(path)/1e6:.2f} Mo)")
print(f"  -> hi_forecast/comparaison_finale_complete.csv "
      f"({len(df)} lignes = 8 familles × 2 modes × 2 machines × 3 configurations)")
