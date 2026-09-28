"""LE MEILLEUR MODELE POUR CHAQUE MOTOSOUFFLANTE — comparaison a armes egales.

Pour repondre honnetement il faut comparer, POUR CHAQUE MACHINE :
  - le modele COMMUN, dont la CV est restreinte aux lignes de validation de CETTE
    machine (et non la CV globale, qui melange les deux) ;
  - le modele SEPARE, entraine et valide sur cette seule machine.

Sans cette restriction on comparerait une CV globale a une CV par machine : ce
n'est pas la meme quantite. C'est le chiffre qui manquait.

Selection sur la CV (critere verrouille : accuracy @ ±0,01), skill rapporte a
cote. Test gele score une seule fois par configuration.
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ae_models import DenseAE
from protocol import (ASSET_COLS, N_SPLITS, SEED, TOLERANCE, PurgedTimeSeriesSplit,
                      regression_metrics)
from run_16_ae_roleA import ALPHAS, PROBS, load, make_clf, make_reg

N_BOOT = 2000


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


def feats(itr, iap, use_ae):
    Ta = d["Ttr"].to_numpy()[itr]
    Tb = d["Ttr"].to_numpy()[iap] if iap is not None else d["Tte"].to_numpy()
    if not use_ae:
        return Ta, Tb
    Ca = np.nan_to_num(d["Ctr"].to_numpy()[itr], nan=0.0)
    Cb = np.nan_to_num(d["Ctr"].to_numpy()[iap] if iap is not None else d["Cte"].to_numpy(), nan=0.0)
    ae = DenseAE(latent=16, hidden=96, noise=0.3).fit(Ca)
    return (np.hstack([Ta, ae.encode(Ca), ae.reconstruction_error(Ca)[:, None]]),
            np.hstack([Tb, ae.encode(Cb), ae.reconstruction_error(Cb)[:, None]]))


def oof_predictions(itr_rows, cfg):
    """Predictions hors-pli + index des lignes concernees."""
    cv = PurgedTimeSeriesSplit(n_splits=N_SPLITS)
    times = mtr["created_at"].iloc[itr_rows]
    out = []
    for a, b in cv.split(np.zeros((len(itr_rows), 1)), times=times):
        ia, ib = itr_rows[a], itr_rows[b]
        Fa, Fb = feats(ia, ib, cfg["ae"])
        ya, na = d["ytr"][ia], d["ntr"][ia]
        r = cfg["reg"](); r.fit(Fa, ya - na)
        c = make_clf(); c.fit(Fa, (np.abs(ya - na) > TOLERANCE).astype(int))
        out.append(dict(idx=ib, y=d["ytr"][ib], n=d["ntr"][ib],
                        d=r.predict(Fb), p=c.predict_proba(Fb)[:, 1]))
    return out


def tune(oof, restrict=None):
    """Regle (alpha, seuil) sur la CV. `restrict` : masque de lignes a ne garder
    (pour restreindre la CV a une machine)."""
    best = None
    for al in ALPHAS:
        for pth in PROBS:
            accs, sks = [], []
            for o in oof:
                m = np.ones(len(o["y"]), bool) if restrict is None else np.isin(o["idx"], restrict)
                if m.sum() < 20:
                    continue
                yh = o["n"][m] + al * o["d"][m] * (o["p"][m] > pth)
                accs.append(np.mean(np.abs(o["y"][m] - yh) <= TOLERANCE))
                mp = mean_squared_error(o["y"][m], o["n"][m])
                sks.append(1 - mean_squared_error(o["y"][m], yh) / mp if mp > 0 else np.nan)
            if not accs:
                continue
            s = float(np.mean(accs))
            if best is None or s > best["cv_acc"]:
                best = dict(alpha=al, seuil=pth, cv_acc=s, cv_sd=float(np.std(accs)),
                            cv_skill=float(np.nanmean(sks)))
    return best


def test_score(itr_rows, cfg, best, te_rows):
    Fa, Fb = feats(itr_rows, None, cfg["ae"])
    ya, na = d["ytr"][itr_rows], d["ntr"][itr_rows]
    r = cfg["reg"](); r.fit(Fa, ya - na)
    c = make_clf(); c.fit(Fa, (np.abs(ya - na) > TOLERANCE).astype(int))
    yhat = np.clip(d["nte"] + best["alpha"] * r.predict(Fb)
                   * (c.predict_proba(Fb)[:, 1] > best["seuil"]), 0, 1)
    y, n = d["yte"][te_rows], d["nte"][te_rows]
    yh = yhat[te_rows]
    m = regression_metrics(y, yh, hi_now=n)
    sess = mte["session_id"].to_numpy()[te_rows]
    rng = np.random.default_rng(SEED)
    uniq = np.unique(sess); groups = {u: np.flatnonzero(sess == u) for u in uniq}
    ds = []
    for _ in range(N_BOOT):
        ii = np.concatenate([groups[u] for u in rng.choice(uniq, len(uniq), replace=True)])
        mp = mean_squared_error(y[ii], n[ii])
        ds.append(1 - mean_squared_error(y[ii], yh[ii]) / mp if mp > 0 else np.nan)
    lo = float(np.nanpercentile(ds, 2.5))
    return m, lo, float(np.nanpercentile(ds, 97.5)), bool(lo > 0), int(len(uniq))


ROWS = []
print("=" * 110)
print("MEILLEUR MODELE PAR MACHINE — CV du modele commun RESTREINTE a chaque machine")
print("=" * 110)

for fam, cfg in FAMILLES.items():
    t0 = time.time()
    # --- COMMUN : un seul entrainement, CV restreinte ensuite ---
    oof_c = oof_predictions(np.arange(len(d["Xtr"])), cfg)
    for asset, (tr, te) in MACH.items():
        b = tune(oof_c, restrict=tr)
        if b is None:
            continue
        m, lo, hi, sig, nb = test_score(np.arange(len(d["Xtr"])), cfg, b, te)
        ROWS.append(dict(famille=fam, mode="commun", machine=asset, n_test=len(te),
                         cv_acc=b["cv_acc"], cv_sd=b["cv_sd"], cv_skill=b["cv_skill"],
                         test_acc=m["acc_tol"], acc_persist=m["acc_tol_persist"],
                         test_r2=m["r2"], test_rmse=m["rmse"], test_mae=m["mae"],
                         skill=m["skill_vs_persist"], ic_bas=lo, ic_haut=hi,
                         significatif=sig, n_blocs=nb, alpha=b["alpha"], seuil=b["seuil"]))
    # --- SEPARE ---
    for asset, (tr, te) in MACH.items():
        if len(tr) < 500:
            continue
        try:
            oof_s = oof_predictions(tr, cfg)
            b = tune(oof_s)
            m, lo, hi, sig, nb = test_score(tr, cfg, b, te)
            ROWS.append(dict(famille=fam, mode="separe", machine=asset, n_test=len(te),
                             cv_acc=b["cv_acc"], cv_sd=b["cv_sd"], cv_skill=b["cv_skill"],
                             test_acc=m["acc_tol"], acc_persist=m["acc_tol_persist"],
                             test_r2=m["r2"], test_rmse=m["rmse"], test_mae=m["mae"],
                             skill=m["skill_vs_persist"], ic_bas=lo, ic_haut=hi,
                             significatif=sig, n_blocs=nb, alpha=b["alpha"], seuil=b["seuil"]))
        except Exception as e:
            print(f"  {fam} / {asset} SEPARE ECHEC : {type(e).__name__}: {e}")
    print(f"  {fam:<24} [{time.time()-t0:.0f}s]")

df = pd.DataFrame(ROWS)
df.to_csv(os.path.join(HERE, "meilleur_par_machine.csv"), index=False, encoding="utf-8-sig")

for asset in MACH:
    print("\n" + "=" * 110)
    print(f"{asset.upper()} — classement par CV accuracy (critere verrouille)")
    print("=" * 110)
    s = df[df.machine == asset].sort_values("cv_acc", ascending=False)
    print(f"{'famille':<24}{'mode':>9}{'CV acc':>10}{'CV sd':>8}{'CV skill':>10}"
          f"{'test acc':>10}{'persist':>9}{'test R2':>9}{'skill':>9}{'IC bas':>9}{'signif':>8}")
    for _, r in s.iterrows():
        print(f"{r['famille']:<24}{r['mode']:>9}{r['cv_acc']:>10.4f}{r['cv_sd']:>8.4f}"
              f"{r['cv_skill']:>+10.4f}{r['test_acc']:>10.4f}{r['acc_persist']:>9.4f}"
              f"{r['test_r2']:>9.4f}{r['skill']:>+9.4f}{r['ic_bas']:>+9.4f}"
              f"{'OUI' if r['significatif'] else 'non':>8}")
    w = s.iloc[0]
    ws = s.sort_values("cv_skill", ascending=False).iloc[0]
    print(f"\n  >>> meilleur par CV ACCURACY : {w['famille']} ({w['mode']})  "
          f"CV {w['cv_acc']:.4f}  test acc {w['test_acc']:.4f}  skill {w['skill']:+.4f}")
    print(f"  >>> meilleur par CV SKILL    : {ws['famille']} ({ws['mode']})  "
          f"CV skill {ws['cv_skill']:+.4f}  test skill {ws['skill']:+.4f}  "
          f"IC[{ws['ic_bas']:+.4f},{ws['ic_haut']:+.4f}]")
print(f"\n  -> hi_forecast/meilleur_par_machine.csv")
