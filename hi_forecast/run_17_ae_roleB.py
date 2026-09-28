"""AUTOENCODEURS — ROLE B : reconstruction de l'ETIQUETTE elle-meme.

L'erreur de reconstruction d'un autoencodeur entraine sur les seules lignes
SAINES sert de mesure de severite, en remplacement (ou en complement) de
PCA T2/SPE + Isolation Forest :

    severite_AE  = erreur_reconstruction / quantile_99(erreur sur lignes saines)
    health_index = 1 / (1 + severite)

Contraintes :
  - l'AE est ajuste sur les lignes SAINES (aucun flag_high_* actif) de la SEULE
    fenetre d'entrainement, puis applique a toutes les lignes ;
  - le seuil d'alarme est le 99e centile des lignes saines d'entrainement.

AVERTISSEMENT METHODOLOGIQUE, repete dans toutes les sorties :
les scores obtenus sur une etiquette reconstruite ne sont PAS comparables a ceux
de l'etiquette gelee. Une etiquette plus lisse fait mecaniquement monter
l'accuracy sans que la prevision se soit amelioree. Ces resultats vivent dans un
tableau separe et ne rejoignent jamais celui du Role A.
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

from ae_models import DenseAE, DenseVAE
from protocol import (ASSET_COLS, FLAG_COLS, HORIZON, N_SPLITS, PCA_VARS, SEED,
                      TOLERANCE, PurgedTimeSeriesSplit, feature_columns,
                      frozen_cutoffs, prepare, regression_metrics,
                      train_mask_from_cutoffs)
from run_16_ae_roleA import (ALPHAS, PROBS, TOP20, bootstrap_ci, make_clf, make_reg)
from transforms import chain, drop_constant_and_dupes, prune_correlated

CLEAN = chain(drop_constant_and_dupes, prune_correlated(0.995))
ALARM_PCTL = 0.99

AE_VARIANTS = {
    "AE dense (latent 6)": lambda: DenseAE(latent=6, hidden=48, noise=0.0),
    "AE debruiteur (latent 6, bruit 0,2)": lambda: DenseAE(latent=6, hidden=48, noise=0.2),
    "VAE dense (latent 6)": lambda: DenseVAE(latent=6, hidden=48, beta=1e-3),
}


def build_ae_label(df, train_mask, make_ae, combine):
    """Construit health_index_ae. `combine` : 'ae_seul' ou 'ae_ou_pca'
    (logique OR avec la severite PCA/IsoForest existante, comme la Phase 5)."""
    vars_lf = [("vi_proxy_lf" if v == "vi_proxy" else v) for v in PCA_VARS]
    out = np.full(len(df), np.nan)
    info = {}
    for asset, col in ASSET_COLS.items():
        m = (df[col] == 1).to_numpy()
        healthy_tr = m & train_mask.to_numpy() & (~df[FLAG_COLS].any(axis=1)).to_numpy()
        Xh = np.nan_to_num(df.loc[healthy_tr, vars_lf].to_numpy(dtype=float), nan=0.0)
        ae = make_ae().fit(Xh)
        err_h = ae.reconstruction_error(Xh)
        alarm = float(np.quantile(err_h, ALARM_PCTL))
        err_all = ae.reconstruction_error(
            np.nan_to_num(df.loc[m, vars_lf].to_numpy(dtype=float), nan=0.0))
        sev_ae = err_all / max(alarm, 1e-12)
        if combine == "ae_ou_pca":
            sev_pca = (1.0 / df.loc[m, "health_index_lf"].to_numpy()) - 1.0
            sev = np.maximum(sev_ae, sev_pca)
        else:
            sev = sev_ae
        out[m] = 1.0 / (1.0 + sev)
        info[asset] = dict(n_healthy_train=int(healthy_tr.sum()), alarm=alarm,
                           val_loss=float(ae.val_loss_), epochs=int(ae.epochs_ran_))
    return out, info


def run_label(df, cutoffs, train_mask, label_col, tag):
    """Protocole gele complet sur une etiquette donnee."""
    d = df.sort_values(["session_id", "created_at"]).reset_index(drop=True)
    y = d.groupby("session_id", sort=False)[label_col].shift(-HORIZON)
    valid = y.notna().to_numpy()
    feats = [c for c in feature_columns(d) if c != label_col]
    X = d.loc[valid, feats].reset_index(drop=True).astype(np.float64)
    yv = y[valid].reset_index(drop=True).to_numpy()
    meta = d.loc[valid, ["created_at", "session_id", label_col]
                 + list(ASSET_COLS.values())].reset_index(drop=True)
    hi_now = meta[label_col].to_numpy()

    is_tr = np.zeros(len(meta), dtype=bool)
    for a, c in ASSET_COLS.items():
        is_tr |= ((meta[c] == 1) & (meta["created_at"] < cutoffs[a])).to_numpy()
    itr_all, ite_all = np.flatnonzero(is_tr), np.flatnonzero(~is_tr)

    Xtr, Xte = X.iloc[itr_all].reset_index(drop=True), X.iloc[ite_all].reset_index(drop=True)
    ytr, yte = yv[itr_all], yv[ite_all]
    ntr, nte = hi_now[itr_all], hi_now[ite_all]
    mtr, mte = meta.iloc[itr_all].reset_index(drop=True), meta.iloc[ite_all].reset_index(drop=True)

    Ta = Xtr[[c for c in TOP20 if c in Xtr.columns]]
    Tb = Xte[[c for c in TOP20 if c in Xte.columns]]

    cv = PurgedTimeSeriesSplit(n_splits=N_SPLITS)
    oof = []
    for itr, iva in cv.split(Xtr, times=mtr["created_at"]):
        r = make_reg(); r.fit(Ta.iloc[itr], ytr[itr] - ntr[itr])
        c = make_clf(); c.fit(Ta.iloc[itr], (np.abs(ytr[itr] - ntr[itr]) > TOLERANCE).astype(int))
        oof.append(dict(y=ytr[iva], n=ntr[iva], d=r.predict(Ta.iloc[iva]),
                        p=c.predict_proba(Ta.iloc[iva])[:, 1]))

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

    r = make_reg(); r.fit(Ta, ytr - ntr)
    c = make_clf(); c.fit(Ta, (np.abs(ytr - ntr) > TOLERANCE).astype(int))
    dte, pte = r.predict(Tb), c.predict_proba(Tb)[:, 1]
    yhat = np.clip(nte + best["alpha"] * dte * (pte > best["seuil"]), 0, 1)
    m = regression_metrics(yte, yhat, hi_now=nte)

    sess = mte["session_id"].to_numpy()
    glob = bootstrap_ci(yte, nte, yhat, sess, np.arange(len(yte)))
    return dict(tag=tag, n_test=int(len(yte)), sd_label=float(np.std(yv)),
                persist_cv_acc=persist_cv, cv_acc=best["cv_acc"],
                cv_acc_sd=best["cv_acc_sd"], cv_skill=best["cv_skill"],
                test_acc=float(m["acc_tol"]), test_acc_persist=float(m["acc_tol_persist"]),
                test_r2=float(m["r2"]), test_rmse=float(m["rmse"]),
                test_skill=float(m["skill_vs_persist"]),
                skill_ci=glob["ci_skill"], skill_clears_zero=glob["skill_clears_zero"],
                alpha=best["alpha"], seuil=best["seuil"])


def main():
    results = []
    for masked, lab in [(False, "etiquette actuelle"), (True, "viscosite masquee")]:
        print("\n" + "#" * 100)
        print(f"#  BASE : {lab.upper()}")
        print("#" * 100)
        df, train_mask, cutoffs, _ = prepare(verbose=False, mask_off_viscosity=masked)

        ref = run_label(df, cutoffs, train_mask, "health_index_lf",
                        f"REFERENCE PCA+IsoForest ({lab})")
        ref["famille"] = "reference"; ref["base"] = lab
        results.append(ref)
        print(f"  reference (etiquette gelee)      : ecart-type {ref['sd_label']:.4f}  "
              f"CV acc {ref['cv_acc']:.4f}  test acc {ref['test_acc']:.4f} "
              f"(persist {ref['test_acc_persist']:.4f})  skill {ref['test_skill']:+.4f}")

        for aname, mk in AE_VARIANTS.items():
            for combine in ("ae_seul", "ae_ou_pca"):
                t0 = time.time()
                tag = f"{aname} [{combine}] ({lab})"
                try:
                    lab_vals, info = build_ae_label(df, train_mask, mk, combine)
                    df2 = df.copy()
                    df2["health_index_ae"] = lab_vals
                    r = run_label(df2, cutoffs, train_mask, "health_index_ae", tag)
                    r["famille"] = aname; r["combinaison"] = combine; r["base"] = lab
                    r["info"] = info; r["secondes"] = round(time.time() - t0, 1)
                    results.append(r)
                    print(f"  {aname:<36} [{combine:<9}] ecart-type {r['sd_label']:.4f}  "
                          f"CV acc {r['cv_acc']:.4f}  test acc {r['test_acc']:.4f} "
                          f"(persist {r['test_acc_persist']:.4f})  skill {r['test_skill']:+.4f}"
                          f"  [{r['secondes']}s]")
                except Exception as e:
                    print(f"  {tag} ECHEC : {type(e).__name__}: {e}")

    with open(os.path.join(HERE, "ae_roleB_resultats.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)
    pd.DataFrame([{k: v for k, v in r.items() if k != "info"} for r in results]).to_csv(
        os.path.join(HERE, "ae_roleB_resultats.csv"), index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("AVERTISSEMENT — LECTURE DU TABLEAU ROLE B")
    print("=" * 100)
    print("  Ces accuracies NE SONT PAS comparables a celles du Role A : la cible n'est pas")
    print("  la meme. Une etiquette plus lisse (ecart-type plus faible) fait monter")
    print("  mecaniquement l'accuracy sans qu'aucune prevision se soit amelioree. La colonne")
    print("  'ecart-type etiquette' est la pour rendre cet effet visible ; c'est le SKILL,")
    print("  mesure contre la persistance de la MEME etiquette, qui reste interpretable.")
    print(f"\n  -> hi_forecast/ae_roleB_resultats.json / .csv")


if __name__ == "__main__":
    main()
