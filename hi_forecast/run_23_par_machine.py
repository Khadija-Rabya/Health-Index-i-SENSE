"""MODELE PAR MACHINE contre MODELE COMMUN — avec l'architecture FINALE.

Le test D6 du debut du projet comparait des Random Forest bruts sur 163 variables
et concluait que separer les machines etait pire. Ce test n'a jamais ete rejoue
avec l'architecture retenue (porte + autoencodeur + 20 variables SHAP). C'est ce
que fait ce script.

Pour chaque machine :
  - COMMUN  : le modele entraine sur les DEUX machines, evalue sur les lignes de
              cette machine ;
  - SEPARE  : un modele entraine UNIQUEMENT sur cette machine, CV sur sa propre
              chronologie, porte et alpha regles sur SA CV.
La persistance sert de reference dans les deux cas.

Trois familles testees : Ridge (meilleur skill du tableau unifie), AE-1 dense
debruiteur (meilleure accuracy), LightGBM Huber (le modele actuellement retenu).
Protocole inchange : test gele, CV purgee + embargo, graine 42, selection en CV.
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

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             mean_squared_error, precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ae_models import DenseAE
from protocol import (ASSET_COLS, N_SPLITS, SEED, TOLERANCE, PurgedTimeSeriesSplit,
                      regression_metrics)
from run_16_ae_roleA import (ALPHAS, PROBS, CLEAN, load, make_clf, make_reg)

N_BOOT = 2000


def ridge_reg():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()), ("m", Ridge(alpha=10.0, random_state=SEED))])


FAMILLES = {
    "Ridge": dict(reg=ridge_reg, ae=False),
    "AE-1 dense debruiteur": dict(reg=make_reg, ae=True),
    "LightGBM Huber (retenu)": dict(reg=make_reg, ae=False),
}


def build_features(d, itr, iap, use_ae):
    """Variables du regresseur : top-20, plus latent + erreur de reconstruction
    si `use_ae`. L'autoencodeur est ajuste sur itr uniquement."""
    Ta = d["Ttr"].to_numpy()[itr]
    Tb = d["Ttr"].to_numpy()[iap] if iap is not None else d["Tte"].to_numpy()
    if not use_ae:
        return Ta, Tb
    Ca = np.nan_to_num(d["Ctr"].to_numpy()[itr], nan=0.0)
    Cb = np.nan_to_num(d["Ctr"].to_numpy()[iap] if iap is not None
                       else d["Cte"].to_numpy(), nan=0.0)
    ae = DenseAE(latent=16, hidden=96, noise=0.3).fit(Ca)
    return (np.hstack([Ta, ae.encode(Ca), ae.reconstruction_error(Ca)[:, None]]),
            np.hstack([Tb, ae.encode(Cb), ae.reconstruction_error(Cb)[:, None]]))


def fit_predict(d, itr_rows, ite_rows, cfg, cv_times):
    """Ajuste sur itr_rows, regle (alpha, seuil) sur SA propre CV, predit ite_rows."""
    cv = PurgedTimeSeriesSplit(n_splits=N_SPLITS)
    sub_times = cv_times.iloc[itr_rows] if hasattr(cv_times, "iloc") else cv_times[itr_rows]
    oof = []
    for a, b in cv.split(np.zeros((len(itr_rows), 1)), times=sub_times):
        ia, ib = itr_rows[a], itr_rows[b]
        Fa, Fb = build_features(d, ia, ib, cfg["ae"])
        ya, yb = d["ytr"][ia], d["ytr"][ib]
        na, nb = d["ntr"][ia], d["ntr"][ib]
        r = cfg["reg"](); r.fit(Fa, ya - na)
        c = make_clf(); c.fit(Fa, (np.abs(ya - na) > TOLERANCE).astype(int))
        oof.append(dict(y=yb, n=nb, d=r.predict(Fb), p=c.predict_proba(Fb)[:, 1]))

    best = None
    for al in ALPHAS:
        for pth in PROBS:
            accs = [np.mean(np.abs(o["y"] - (o["n"] + al * o["d"] * (o["p"] > pth))) <= TOLERANCE)
                    for o in oof]
            s = float(np.mean(accs))
            if best is None or s > best["cv_acc"]:
                sk = []
                for o in oof:
                    yh = o["n"] + al * o["d"] * (o["p"] > pth)
                    mp = mean_squared_error(o["y"], o["n"])
                    sk.append(1 - mean_squared_error(o["y"], yh) / mp if mp > 0 else np.nan)
                best = dict(alpha=al, seuil=pth, cv_acc=s, cv_sd=float(np.std(accs)),
                            cv_skill=float(np.nanmean(sk)))

    Fa, Fb = build_features(d, itr_rows, None, cfg["ae"])
    ya, na = d["ytr"][itr_rows], d["ntr"][itr_rows]
    r = cfg["reg"](); r.fit(Fa, ya - na)
    c = make_clf(); c.fit(Fa, (np.abs(ya - na) > TOLERANCE).astype(int))
    dte = r.predict(Fb); pte = c.predict_proba(Fb)[:, 1]
    yhat = np.clip(d["nte"] + best["alpha"] * dte * (pte > best["seuil"]), 0, 1)
    return yhat[ite_rows], best


def boot(y, n, yhat, sess, seed=SEED):
    rng = np.random.default_rng(seed)
    uniq = np.unique(sess)
    groups = {u: np.flatnonzero(sess == u) for u in uniq}
    ds = []
    for _ in range(N_BOOT):
        ii = np.concatenate([groups[u] for u in rng.choice(uniq, len(uniq), replace=True)])
        mp = mean_squared_error(y[ii], n[ii])
        ds.append(1 - mean_squared_error(y[ii], yhat[ii]) / mp if mp > 0 else np.nan)
    lo = float(np.nanpercentile(ds, 2.5))
    return lo, float(np.nanpercentile(ds, 97.5)), bool(lo > 0), int(len(uniq))


d = load(False)
mtr, mte = d["mtr"], d["mte"]
ROWS = []

print("=" * 106)
print("MODELE COMMUN contre MODELE PAR MACHINE — architecture finale")
print("=" * 106)

for fam, cfg in FAMILLES.items():
    print(f"\n### {fam}")
    t0 = time.time()

    # ---------- COMMUN : entraine sur tout, evalue par machine ----------
    yhat_all, best_all = fit_predict(d, np.arange(len(d["Xtr"])),
                                     np.arange(len(d["Xte"])), cfg, mtr["created_at"])
    print(f"  COMMUN   (alpha={best_all['alpha']}, seuil={best_all['seuil']})  "
          f"CV {best_all['cv_acc']:.4f}±{best_all['cv_sd']:.4f}  skill CV {best_all['cv_skill']:+.4f}")

    for asset, col in ASSET_COLS.items():
        te = np.flatnonzero((mte[col] == 1).to_numpy())
        y, n = d["yte"][te], d["nte"][te]
        s = mte["session_id"].to_numpy()[te]
        m = regression_metrics(y, yhat_all[te], hi_now=n)
        lo, hi, sig, nb = boot(y, n, yhat_all[te], s)
        ROWS.append(dict(famille=fam, mode="commun", machine=asset[-1], n_test=len(te),
                         cv_acc=best_all["cv_acc"], cv_sd=best_all["cv_sd"],
                         cv_skill=best_all["cv_skill"], test_acc=m["acc_tol"],
                         acc_persist=m["acc_tol_persist"], test_r2=m["r2"],
                         test_rmse=m["rmse"], test_mae=m["mae"], skill=m["skill_vs_persist"],
                         ic_bas=lo, ic_haut=hi, significatif=sig, n_blocs=nb,
                         alpha=best_all["alpha"], seuil=best_all["seuil"]))
        print(f"    -> {asset:<20} acc {m['acc_tol']:.4f} (pers {m['acc_tol_persist']:.4f})  "
              f"R² {m['r2']:.4f}  skill {m['skill_vs_persist']:+.4f}  "
              f"IC[{lo:+.4f},{hi:+.4f}] {'OUI' if sig else 'non'}")

    # ---------- SEPARE : un modele par machine ----------
    for asset, col in ASSET_COLS.items():
        tr = np.flatnonzero((mtr[col] == 1).to_numpy())
        te = np.flatnonzero((mte[col] == 1).to_numpy())
        if len(tr) < 500:
            print(f"    {asset} : trop peu de lignes d'entrainement ({len(tr)}) — ECARTE")
            continue
        try:
            yh, b = fit_predict(d, tr, te, cfg, mtr["created_at"])
        except Exception as e:
            print(f"    {asset} SEPARE ECHEC : {type(e).__name__}: {e}")
            continue
        y, n = d["yte"][te], d["nte"][te]
        s = mte["session_id"].to_numpy()[te]
        m = regression_metrics(y, yh, hi_now=n)
        lo, hi, sig, nb = boot(y, n, yh, s)
        ROWS.append(dict(famille=fam, mode="separe", machine=asset[-1], n_test=len(te),
                         cv_acc=b["cv_acc"], cv_sd=b["cv_sd"], cv_skill=b["cv_skill"],
                         test_acc=m["acc_tol"], acc_persist=m["acc_tol_persist"],
                         test_r2=m["r2"], test_rmse=m["rmse"], test_mae=m["mae"],
                         skill=m["skill_vs_persist"], ic_bas=lo, ic_haut=hi,
                         significatif=sig, n_blocs=nb, alpha=b["alpha"], seuil=b["seuil"]))
        print(f"  SEPARE   {asset:<20} n_train={len(tr):<6} CV {b['cv_acc']:.4f}±{b['cv_sd']:.4f}"
              f"  acc {m['acc_tol']:.4f}  R² {m['r2']:.4f}  skill {m['skill_vs_persist']:+.4f}  "
              f"IC[{lo:+.4f},{hi:+.4f}] {'OUI' if sig else 'non'}")
    print(f"  [{time.time()-t0:.0f}s]")

df = pd.DataFrame(ROWS)
df.to_csv(os.path.join(HERE, "par_machine_resultats.csv"), index=False, encoding="utf-8-sig")

print("\n" + "=" * 106)
print("COMMUN contre SEPARE — verdict par famille et par machine")
print("=" * 106)
print(f"{'famille':<26}{'machine':>8}{'CV commun':>11}{'CV separe':>11}"
      f"{'skill commun':>14}{'skill separe':>14}{'meilleur':>12}")
for fam in FAMILLES:
    for mach in ["A", "B"]:
        c = df[(df.famille == fam) & (df["mode"] == "commun") & (df.machine == mach)]
        s = df[(df.famille == fam) & (df["mode"] == "separe") & (df.machine == mach)]
        if c.empty or s.empty:
            continue
        c, s = c.iloc[0], s.iloc[0]
        win = "COMMUN" if c["cv_acc"] >= s["cv_acc"] else "SEPARE"
        print(f"{fam:<26}{mach:>8}{c['cv_acc']:>11.4f}{s['cv_acc']:>11.4f}"
              f"{c['skill']:>+14.4f}{s['skill']:>+14.4f}{win:>12}")
print("\n  (le choix se lit sur la CV, jamais sur le skill de test)")
print(f"  -> hi_forecast/par_machine_resultats.csv")
