"""ENTRAINEMENT SEPARE PAR MACHINE — deux schemas de validation, toutes les metriques.

Chaque motosoufflante recoit ses PROPRES modeles, entraines uniquement sur ses
lignes, valides sur sa propre chronologie.

DEUX SCHEMAS DE VALIDATION, compares :

  1. VALIDATION CROISEE PAR BLOCS EXPANSIFS (« cross-validation »)
     Le bloc d'entrainement grandit a chaque pli : [0..i] -> valider sur i+1.
     Repond a : « avec TOUT l'historique disponible, que vaut le modele ? »

  2. VALIDATION GLISSANTE (« loop / walk-forward validation »)
     Fenetre d'entrainement de TAILLE FIXE qui avance : [i..i+w] -> valider sur
     le bloc suivant. Repond a : « avec un historique RECENT et borne, que vaut
     le modele ? » — c'est le regime reel d'un systeme en production qui
     reentraine periodiquement sur une fenetre glissante.

Les deux appliquent la meme purge et le meme embargo de 3 h.

TROIS CONFIGURATIONS DE DECISION :
  G0  alpha = 0             persistance pure (borne basse)
  G1  alpha = 1, sans porte correction systematique
  G*  porte reglee          alpha et seuil choisis sur la validation
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
                             cohen_kappa_score, f1_score, log_loss, matthews_corrcoef,
                             mean_squared_error, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ae_models import DenseAE
from protocol import (ASSET_COLS, HORIZON_TD, N_SPLITS, SEED, TOLERANCE,
                      PurgedTimeSeriesSplit, regression_metrics)
from run_16_ae_roleA import ALPHAS, PROBS, load, make_clf, make_reg

N_BOOT = 2000
ART = os.path.join(HERE, "artifacts")


# ----------------------------------------------------- validation glissante
class WalkForwardSplit:
    """Fenetre d'entrainement de TAILLE FIXE qui avance dans le temps.

    Contrairement aux blocs expansifs, la taille du train reste constante : on
    mesure ce que vaut le modele avec un historique recent et borne, ce qui est
    le regime d'un systeme reentraine periodiquement. Meme purge et meme embargo
    de 3 h que la validation croisee."""

    def __init__(self, n_splits=N_SPLITS, horizon_td=HORIZON_TD, embargo_td=HORIZON_TD):
        self.n_splits, self.horizon_td, self.embargo_td = n_splits, horizon_td, embargo_td

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None, times=None):
        times = pd.Series(pd.to_datetime(np.asarray(times)))
        order = np.argsort(times.to_numpy(), kind="mergesort")
        blocks = np.array_split(order, self.n_splits + 1)
        for i in range(self.n_splits):
            val = blocks[i + 1]
            train = blocks[i]                      # UNE SEULE fenetre, taille fixe
            val_start = times.iloc[val].min()
            keep = times.iloc[train].to_numpy() < (val_start - self.horizon_td - self.embargo_td)
            if keep.sum() < 100:                   # fenetre trop courte : on elargit
                train = np.concatenate(blocks[max(0, i - 1): i + 1])
                keep = times.iloc[train].to_numpy() < (val_start - self.horizon_td - self.embargo_td)
            yield train[keep], val


SCHEMAS = {"croisee (blocs expansifs)": PurgedTimeSeriesSplit,
           "glissante (walk-forward)": WalkForwardSplit}


def sc(*s):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler())] + list(s))


def ns(*s):
    return Pipeline([("imp", SimpleImputer(strategy="median"))] + list(s))


FAMILLES = {
    "Ridge":                 dict(reg=lambda: sc(("m", Ridge(alpha=10.0, random_state=SEED))), ae=False,
                                  desc="Régression linéaire pénalisée L2 : rétrécit tous les coefficients sans en annuler aucun. Stabilise les variables colinéaires."),
    "Lasso":                 dict(reg=lambda: sc(("m", Lasso(alpha=1e-4, max_iter=5000, random_state=SEED))), ae=False,
                                  desc="Pénalisation L1 pure : annule les coefficients inutiles, donc sélectionne un sous-ensemble parcimonieux de variables."),
    "ElasticNet":            dict(reg=lambda: sc(("m", ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=5000, random_state=SEED))), ae=False,
                                  desc="Combinaison L1 + L2 : sélection de variables du Lasso, stabilité du Ridge sur les groupes corrélés."),
    "Random Forest":         dict(reg=lambda: ns(("m", RandomForestRegressor(n_estimators=300, max_depth=14, max_features="sqrt", random_state=SEED, n_jobs=-1))), ae=False,
                                  desc="Forêt d'arbres décorrélés (bagging), agrégés par moyenne. Capture les non-linéarités et les interactions sans réglage fin."),
    "CatBoost":              dict(reg=lambda: ns(("m", cb.CatBoostRegressor(iterations=400, depth=6, learning_rate=0.05, random_seed=SEED, verbose=0, thread_count=-1))), ae=False,
                                  desc="Boosting de gradient à arbres symétriques, réputé robuste sans réglage. Corrige séquentiellement le résidu."),
    "LightGBM Huber":        dict(reg=make_reg, ae=False,
                                  desc="Boosting par croissance en profondeur d'abord, avec perte de Huber : robuste aux valeurs extrêmes, estime une médiane conditionnelle plutôt qu'une moyenne."),
    "AE-1 dense débruiteur": dict(reg=make_reg, ae=True,
                                  desc="Autoencodeur dense débruiteur (latent 16, bruit 0,3) : compresse l'état capteur en 16 dimensions robustes ; latent + erreur de reconstruction alimentent le régresseur."),
    "Ridge + AE":            dict(reg=lambda: sc(("m", Ridge(alpha=10.0, random_state=SEED))), ae=True,
                                  desc="Variables de l'autoencodeur passées à une régression Ridge : compression non linéaire, décision linéaire régularisée."),
}

d = load(False)
mtr, mte = d["mtr"], d["mte"]
MACH = {a: (np.flatnonzero((mtr[c] == 1).to_numpy()),
            np.flatnonzero((mte[c] == 1).to_numpy())) for a, c in ASSET_COLS.items()}
THR_S, THR_A = float(np.quantile(d["ntr"], 0.05)), float(np.quantile(d["ntr"], 0.01))
LB = ["Normal", "Surveillance", "Alarme"]
to_state = lambda v: np.where(v <= THR_A, "Alarme", np.where(v <= THR_S, "Surveillance", "Normal"))


def feats(itr, iap, use_ae, keep=False):
    Ta = d["Ttr"].to_numpy()[itr]
    Tb = d["Ttr"].to_numpy()[iap] if iap is not None else d["Tte"].to_numpy()
    if not use_ae:
        return Ta, Tb, None
    Ca = np.nan_to_num(d["Ctr"].to_numpy()[itr], nan=0.0)
    Cb = np.nan_to_num(d["Ctr"].to_numpy()[iap] if iap is not None else d["Cte"].to_numpy(), nan=0.0)
    ae = DenseAE(latent=16, hidden=96, noise=0.3).fit(Ca)
    return (np.hstack([Ta, ae.encode(Ca), ae.reconstruction_error(Ca)[:, None]]),
            np.hstack([Tb, ae.encode(Cb), ae.reconstruction_error(Cb)[:, None]]),
            ae if keep else None)


def oof(itr_rows, cfg, splitter):
    out = []
    for a, b in splitter().split(np.zeros((len(itr_rows), 1)),
                                 times=mtr["created_at"].iloc[itr_rows]):
        ia, ib = itr_rows[a], itr_rows[b]
        if len(ia) < 100:
            continue
        Fa, Fb, _ = feats(ia, ib, cfg["ae"])
        ya, na = d["ytr"][ia], d["ntr"][ia]
        r = cfg["reg"](); r.fit(Fa, ya - na)
        c = make_clf(); c.fit(Fa, (np.abs(ya - na) > TOLERANCE).astype(int))
        out.append(dict(y=d["ytr"][ib], n=d["ntr"][ib], d=r.predict(Fb),
                        p=c.predict_proba(Fb)[:, 1], n_train=len(ia)))
    return out


def val_stats(o_list, al, pth):
    accs, sks = [], []
    for o in o_list:
        yh = o["n"] + al * o["d"] * (o["p"] > pth)
        accs.append(np.mean(np.abs(o["y"] - yh) <= TOLERANCE))
        mp = mean_squared_error(o["y"], o["n"])
        sks.append(1 - mean_squared_error(o["y"], yh) / mp if mp > 0 else np.nan)
    if not accs:
        return None
    return dict(val_acc=float(np.mean(accs)), val_sd=float(np.std(accs)),
                val_skill=float(np.nanmean(sks)),
                n_train_moyen=float(np.mean([o["n_train"] for o in o_list])))


def tune(o_list):
    best = None
    for al in ALPHAS:
        for pth in PROBS:
            s = val_stats(o_list, al, pth)
            if s and (best is None or s["val_acc"] > best["val_acc"]):
                best = dict(alpha=al, seuil=pth, **s)
    return best


def metrics_all(y, n, yhat, prob, sess, gate):
    m = regression_metrics(y, yhat, n_features=20, hi_now=n)
    st, sm = to_state(y), to_state(yhat)
    dt, dp = (st != "Normal").astype(int), (sm != "Normal").astype(int)
    out = dict(acc_tol=m["acc_tol"], acc_persist=m["acc_tol_persist"], r2=m["r2"],
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
    if gate and prob is not None:
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


ROWS, PRED = [], {}
print("=" * 110)
print("ENTRAINEMENT SEPARE PAR MACHINE — 8 familles × 2 schémas de validation × 3 configurations")
print("=" * 110)

for asset, (tr, te) in MACH.items():
    print(f"\n### {asset}  (train {len(tr)} lignes, test {len(te)})")
    y, n = d["yte"][te], d["nte"][te]
    sess = mte["session_id"].to_numpy()[te]
    for fam, cfg in FAMILLES.items():
        t0 = time.time()
        Fa, Fb, _ = feats(tr, None, cfg["ae"])
        ya, na = d["ytr"][tr], d["ntr"][tr]
        r = cfg["reg"](); r.fit(Fa, ya - na)
        c = make_clf(); c.fit(Fa, (np.abs(ya - na) > TOLERANCE).astype(int))
        dte, pte = r.predict(Fb)[te], c.predict_proba(Fb)[:, 1][te]

        for sname, splitter in SCHEMAS.items():
            o = oof(tr, cfg, splitter)
            if not o:
                continue
            best = tune(o)
            for gname, al, pth, gate in [("G0 persistance", 0.0, 1.1, False),
                                         ("G1 sans porte", 1.0, -0.1, False),
                                         ("G* porte réglée", best["alpha"], best["seuil"], True)]:
                yhat = np.clip(n + al * dte * (pte > pth), 0, 1)
                v = val_stats(o, al, pth) or {}
                m = metrics_all(y, n, yhat, pte if gate else None, sess, gate)
                ROWS.append(dict(machine=asset, famille=fam, validation=sname,
                                 config=gname, alpha=al,
                                 seuil=pth if gate else np.nan, n_test=len(te),
                                 n_train=len(tr), **v, **m))
                if gname == "G* porte réglée" and sname.startswith("croisee"):
                    PRED[(asset, fam)] = yhat
        print(f"  {fam:<24} [{time.time()-t0:.0f}s]")

df = pd.DataFrame(ROWS)
df.to_csv(os.path.join(HERE, "separe_par_machine_complet.csv"), index=False,
          encoding="utf-8-sig")
np.save(os.path.join(HERE, "cache", "pred_separe.npy"),
        np.array([PRED[k] for k in sorted(PRED)], dtype=object), allow_pickle=True)
with open(os.path.join(HERE, "cache", "pred_separe_cles.json"), "w", encoding="utf-8") as f:
    json.dump([list(k) for k in sorted(PRED)], f, ensure_ascii=False)

# --------------------------------------------------- classement et artefacts
RECO = {}
for asset in MACH:
    for sname in SCHEMAS:
        print("\n" + "=" * 110)
        print(f"{asset.upper()} — validation {sname} — porte réglée")
        print("=" * 110)
        s = df[(df.machine == asset) & (df.validation == sname) &
               (df.config == "G* porte réglée")].sort_values("val_acc", ascending=False)
        print(f"{'famille':<24}{'val acc':>9}{'sd':>8}{'val skill':>10}{'test acc':>9}"
              f"{'pers':>8}{'R²':>8}{'RMSE':>8}{'skill':>9}{'MCC':>8}{'sig':>5}")
        for _, r in s.iterrows():
            print(f"{r['famille']:<24}{r['val_acc']:>9.4f}{r['val_sd']:>8.4f}"
                  f"{r['val_skill']:>+10.4f}{r['acc_tol']:>9.4f}{r['acc_persist']:>8.4f}"
                  f"{r['r2']:>8.4f}{r['rmse']:>8.4f}{r['skill']:>+9.4f}{r['etat_mcc']:>8.4f}"
                  f"{'OUI' if r['significatif'] else 'non':>5}")
        if sname.startswith("croisee"):
            sig = s[s["significatif"]]
            w = sig.iloc[0] if len(sig) else s.iloc[0]
            RECO[asset] = dict(famille=w["famille"], alpha=float(w["alpha"]),
                               seuil=float(w["seuil"]), val_acc=float(w["val_acc"]),
                               test_acc=float(w["acc_tol"]), skill=float(w["skill"]),
                               significatif=bool(w["significatif"]))
            print(f"\n  >>> RETENU : {w['famille']}  val {w['val_acc']:.4f}  "
                  f"test acc {w['acc_tol']:.4f}  skill {w['skill']:+.4f}  "
                  f"{'significatif' if w['significatif'] else 'NON significatif'}")

bundle = {}
for asset, rec in RECO.items():
    cfg = FAMILLES[rec["famille"]]
    tr, te = MACH[asset]
    Fa, Fb, ae = feats(tr, None, cfg["ae"], keep=True)
    ya, na = d["ytr"][tr], d["ntr"][tr]
    r = cfg["reg"](); r.fit(Fa, ya - na)
    c = make_clf(); c.fit(Fa, (np.abs(ya - na) > TOLERANCE).astype(int))
    bundle[asset] = dict(famille=rec["famille"], regresseur=r, porte=c, autoencodeur=ae,
                         utilise_ae=cfg["ae"], alpha=rec["alpha"], seuil=rec["seuil"],
                         colonnes=list(d["Ttr"].columns),
                         colonnes_ae=list(d["Ctr"].columns) if cfg["ae"] else None,
                         seuil_surveillance=THR_S, seuil_alarme=THR_A,
                         tolerance=TOLERANCE, horizon_pas=18, n_train=len(tr))
    print(f"  {asset:<20} {rec['famille']:<24} entraîné sur {len(tr)} lignes de CETTE machine")

p = os.path.join(ART, "modeles_separes_par_machine.joblib")
joblib.dump(bundle, p, compress=3)
with open(os.path.join(ART, "reco_separe.json"), "w", encoding="utf-8") as f:
    json.dump(RECO, f, indent=2, ensure_ascii=False)
print(f"\n  artefact : {p}  ({os.path.getsize(p)/1e6:.2f} Mo)")
print(f"  -> hi_forecast/separe_par_machine_complet.csv ({len(df)} lignes)")
