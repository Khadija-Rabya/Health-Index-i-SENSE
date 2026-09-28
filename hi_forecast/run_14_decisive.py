"""EXPERIENCE DECISIVE + analyses complementaires (aucun reglage nouveau).

1. Etiquette reconstruite avec la viscosite masquee (manquante) sur toutes les
   lignes a l'arret, puis protocole gele rejoue de bout en bout : persistance,
   selection en CV de la porte, test gele, bootstrap par blocs.
   HYPOTHESE TESTEE : le gain de +0,0589 sur la Motosoufflante B vient du modele
   qui suit l'artefact du viscosimetre, pas la machine.
2. Decoupage 2x2 {A, B} x {ON, OFF}.
3. Decomposition de la variance du bootstrap (inter-machine vs intra-machine).

Les hyperparametres du regresseur et du classifieur sont ceux deja retenus
(cache/final_config.json) : seuls alpha et le seuil de porte sont reselectionnes
en CV, puisque l'etiquette change.
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

import lightgbm as lgb

from gated import ALPHAS, PROBS
from protocol import (ASSET_COLS, HORIZON, N_SPLITS, SEED, TOLERANCE,
                      PurgedTimeSeriesSplit, build_xy, prepare,
                      regression_metrics, split_frozen)
from transforms import chain, drop_constant_and_dupes, prune_correlated

CLEAN = chain(drop_constant_and_dupes, prune_correlated(0.995))
with open(os.path.join(HERE, "cache", "final_config.json"), encoding="utf-8") as f:
    CFG = json.load(f)
P = CFG["params"]
rank = pd.read_csv(os.path.join(HERE, "classements_importance.csv"), index_col=0)
WANTED = list(rank["shap"].sort_values(ascending=False).head(20).index)


def TF(Xa, Xb):
    Xa, Xb = CLEAN(Xa, Xb)
    cols = [c for c in WANTED if c in Xa.columns]
    return Xa[cols], Xb[cols]


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


def run_protocol(masked: bool, tag: str):
    """Protocole gele complet sous une version de l'etiquette."""
    print("\n" + "=" * 104)
    print(f"PROTOCOLE COMPLET — {tag}")
    print("=" * 104)
    df, train_mask, cutoffs, _ = prepare(verbose=True, mask_off_viscosity=masked)
    X, y, meta = build_xy(df, HORIZON)
    Xtr, Xte, ytr, yte, mtr, mte = split_frozen(X, y, meta, cutoffs)
    na, nb = mtr["hi_now"].to_numpy(), mte["hi_now"].to_numpy()
    yte_a, ytr_a = yte.to_numpy(), ytr.to_numpy()
    print(f"  train/test : {len(Xtr)}/{len(Xte)}   y_test mean={yte.mean():.4f} sd={yte.std():.4f}")

    # --- CV : predictions hors-pli, puis selection de (alpha, seuil) ---
    cv = PurgedTimeSeriesSplit(n_splits=N_SPLITS)
    folds = list(cv.split(Xtr, times=mtr["created_at"]))
    oof = []
    for itr, iva in folds:
        Xa, Xb = TF(Xtr.iloc[itr], Xtr.iloc[iva])
        na_, nb_ = na[itr], na[iva]
        ya, yb = ytr_a[itr], ytr_a[iva]
        r = make_reg(); r.fit(Xa, ya - na_)
        c = make_clf(); c.fit(Xa, (np.abs(ya - na_) > TOLERANCE).astype(int))
        oof.append(dict(y=yb, n=nb_, d=r.predict(Xb), p=c.predict_proba(Xb)[:, 1]))

    persist_cv = float(np.mean([np.mean(np.abs(o["y"] - o["n"]) <= TOLERANCE) for o in oof]))
    best = None
    for a in ALPHAS:
        for pth in PROBS:
            pf = [np.mean(np.abs(o["y"] - (o["n"] + a * o["d"] * (o["p"] > pth))) <= TOLERANCE)
                  for o in oof]
            s = float(np.mean(pf))
            if best is None or s > best["cv_acc"]:
                best = dict(alpha=a, seuil=pth, cv_acc=s, cv_std=float(np.std(pf)))
    print(f"  persistance CV = {persist_cv:.4f}   |   porte CV = {best['cv_acc']:.4f} "
          f"± {best['cv_std']:.4f}  (alpha={best['alpha']}, seuil={best['seuil']})")

    # --- test gele, une seule fois ---
    Xa, Xb = TF(Xtr, Xte)
    reg = make_reg(); reg.fit(Xa, ytr_a - na)
    clf = make_clf(); clf.fit(Xa, (np.abs(ytr_a - na) > TOLERANCE).astype(int))
    dte, pte = reg.predict(Xb), clf.predict_proba(Xb)[:, 1]
    yhat = np.clip(nb + best["alpha"] * dte * (pte > best["seuil"]), 0, 1)

    return dict(tag=tag, masked=masked, df=df, yte=yte_a, nb=nb, yhat=yhat, mte=mte,
                Xte=Xte, best=best, persist_cv=persist_cv,
                test=regression_metrics(yte_a, yhat, hi_now=nb))


def block_bootstrap(yte, nb, yhat, sess, idx, n_boot=4000, seed=SEED):
    rng = np.random.default_rng(seed)
    hit_m = (np.abs(yte - yhat) <= TOLERANCE).astype(int)
    hit_p = (np.abs(yte - nb) <= TOLERANCE).astype(int)
    s = sess[idx]
    uniq = np.unique(s)
    groups = {u: idx[s == u] for u in uniq}
    da, ds = [], []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        ii = np.concatenate([groups[u] for u in pick])
        da.append(hit_m[ii].mean() - hit_p[ii].mean())
        mp = mean_squared_error(yte[ii], nb[ii])
        ds.append(1 - mean_squared_error(yte[ii], yhat[ii]) / mp if mp > 0 else np.nan)
    da, ds = np.array(da), np.array(ds)
    diff = hit_m[idx] - hit_p[idx]
    var_iid = diff.var(ddof=1) / len(idx)
    deff = da.var(ddof=1) / var_iid if var_iid > 0 else np.nan
    return dict(n=len(idx), n_blocks=int(len(uniq)),
                delta_acc=float(hit_m[idx].mean() - hit_p[idx].mean()),
                ci_acc=[float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))],
                skill=float(1 - mean_squared_error(yte[idx], yhat[idx])
                            / mean_squared_error(yte[idx], nb[idx])),
                ci_skill=[float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))],
                deff=float(deff), n_eff=float(len(idx) / deff) if np.isfinite(deff) else np.nan)


# =============================================================== 1. AVANT/APRES
base = run_protocol(False, "ETIQUETTE ACTUELLE (viscosite telle quelle)")
mask = run_protocol(True, "ETIQUETTE CORRIGEE (viscosite masquee a l'arret)")

print("\n" + "=" * 104)
print("1. EXPERIENCE DECISIVE — Motosoufflante B, avant / apres masquage")
print("=" * 104)
OUT = {}
for r in (base, mask):
    sess = r["mte"]["session_id"].to_numpy()
    r["boot"] = {}
    scopes = [("global", np.arange(len(r["yte"])))]
    for a, col in ASSET_COLS.items():
        scopes.append((a, np.flatnonzero((r["mte"][col] == 1).to_numpy())))
    for lab, idx in scopes:
        r["boot"][lab] = block_bootstrap(r["yte"], r["nb"], r["yhat"], sess, idx)

hdr = f"{'':<26}{'AVANT':>26}{'APRES':>26}"
print(hdr)
for lab in ["global", "Motosoufflante A", "Motosoufflante B"]:
    b0, b1 = base["boot"][lab], mask["boot"][lab]
    print(f"\n  {lab}")
    print(f"    {'delta accuracy':<22}{b0['delta_acc']:>+12.4f}{'':>14}{b1['delta_acc']:>+12.4f}")
    print(f"    {'IC95 accuracy':<22}"
          f"[{b0['ci_acc'][0]:+.4f},{b0['ci_acc'][1]:+.4f}]{'':>4}"
          f"[{b1['ci_acc'][0]:+.4f},{b1['ci_acc'][1]:+.4f}]")
    print(f"    {'skill':<22}{b0['skill']:>+12.4f}{'':>14}{b1['skill']:>+12.4f}")
    print(f"    {'IC95 skill':<22}"
          f"[{b0['ci_skill'][0]:+.4f},{b0['ci_skill'][1]:+.4f}]{'':>4}"
          f"[{b1['ci_skill'][0]:+.4f},{b1['ci_skill'][1]:+.4f}]")
    print(f"    {'blocs / n_eff':<22}{b0['n_blocks']:>5} / {b0['n_eff']:>6.0f}{'':>13}"
          f"{b1['n_blocks']:>5} / {b1['n_eff']:>6.0f}")

bB0, bB1 = base["boot"]["Motosoufflante B"], mask["boot"]["Motosoufflante B"]
disparu = bB1["ci_acc"][0] <= 0 <= bB1["ci_acc"][1]
print(f"\n  >>> VERDICT : gain B accuracy {bB0['delta_acc']:+.4f} -> {bB1['delta_acc']:+.4f}"
      f"   ({100*(bB1['delta_acc']-bB0['delta_acc'])/abs(bB0['delta_acc']):+.0f} %)")
print(f"      skill B {bB0['skill']:+.4f} -> {bB1['skill']:+.4f}")
print(f"      l'IC de B croise-t-il zero apres correction ? "
      f"{'OUI — le gain disparait' if disparu else 'NON — le gain survit'}")
OUT["decisive"] = dict(avant={k: v for k, v in base["boot"].items()},
                       apres={k: v for k, v in mask["boot"].items()},
                       cv_avant=base["best"], cv_apres=mask["best"],
                       persist_cv_avant=base["persist_cv"],
                       persist_cv_apres=mask["persist_cv"],
                       test_avant=base["test"], test_apres=mask["test"])

# ===================================================== 2. DECOUPAGE 2x2
print("\n" + "=" * 104)
print("2. DECOUPAGE {A, B} x {ON, OFF} — modele contre persistance")
print("=" * 104)
rows2 = []
for r in (base, mask):
    for a, col in ASSET_COLS.items():
        for st, lab in [(0, "ON"), (1, "OFF")]:
            m = ((r["mte"][col] == 1).to_numpy()
                 & (r["Xte"]["state_OFF"].to_numpy() == st))
            if m.sum() < 20:
                continue
            mm = regression_metrics(r["yte"][m], r["yhat"][m], hi_now=r["nb"][m])
            rows2.append(dict(etiquette="avant" if not r["masked"] else "apres",
                              machine=a[-1], etat=lab, n=int(m.sum()),
                              acc=mm["acc_tol"], acc_pers=mm["acc_tol_persist"],
                              r2=mm["r2"], r2_pers=mm["r2_persist"],
                              rmse=mm["rmse"], skill=mm["skill_vs_persist"]))
t2 = pd.DataFrame(rows2)
for etq in ("avant", "apres"):
    print(f"\n  --- etiquette {etq} ---")
    print(f"  {'cell':<8}{'n':>6}{'acc':>9}{'acc_pers':>10}{'R2':>10}{'R2_pers':>10}"
          f"{'RMSE':>9}{'skill':>9}")
    for _, x in t2[t2.etiquette == etq].iterrows():
        print(f"  {x['machine']+'/'+x['etat']:<8}{x['n']:>6}{x['acc']:>9.4f}"
              f"{x['acc_pers']:>10.4f}{x['r2']:>10.4f}{x['r2_pers']:>10.4f}"
              f"{x['rmse']:>9.5f}{x['skill']:>+9.4f}")
t2.to_csv(os.path.join(HERE, "verif_2x2_machine_etat.csv"), index=False, encoding="utf-8-sig")

# ------------------ B/OFF : excursions et bascule du viscosimetre
print("\n" + "=" * 104)
print("2b. MOTOSOUFFLANTE B A L'ARRET — excursions du health index et bascule du viscosimetre")
print("=" * 104)
df0 = base["df"].sort_values(["session_id", "created_at"]).reset_index(drop=True)
Bc = ASSET_COLS["Motosoufflante B"]
g = df0.groupby("session_id", sort=False)
df0["_y_next"] = g["health_index_lf"].shift(-HORIZON)
df0["_v_next"] = g["Viscosity at 40°C_filled"].shift(-HORIZON)
sel = (df0[Bc] == 1) & (df0["state_OFF"] == 1) & df0["_y_next"].notna()
sub = df0.loc[sel]
delta = (sub["_y_next"] - sub["health_index_lf"]).abs()
exc = delta > TOLERANCE
MID = 25.0                      # point median entre les deux modes (~12 et ~42 cSt)
cross = ((sub["Viscosity at 40°C_filled"] < MID) != (sub["_v_next"] < MID))
print(f"  lignes B/OFF avec cible valide : {len(sub)}")
print(f"  part avec |delta health_index| > {TOLERANCE} sur 3 h : {exc.mean():.2%} "
      f"({int(exc.sum())} lignes)")
print(f"  part ou la viscosite bascule entre les deux modes (seuil {MID} cSt) : "
      f"{cross.mean():.2%} ({int(cross.sum())} lignes)")
ct = pd.crosstab(exc, cross, normalize=False)
print(f"\n  table de contingence (lignes = excursion, colonnes = bascule) :")
print(ct.to_string())
p_exc_if_cross = exc[cross].mean() if cross.sum() else np.nan
p_exc_if_not = exc[~cross].mean() if (~cross).sum() else np.nan
print(f"\n  P(excursion | bascule)      = {p_exc_if_cross:.2%}")
print(f"  P(excursion | pas bascule)  = {p_exc_if_not:.2%}")
if np.isfinite(p_exc_if_not) and p_exc_if_not > 0:
    print(f"  rapport de risque           = {p_exc_if_cross/p_exc_if_not:.2f}x")
print(f"  part des excursions accompagnees d'une bascule : "
      f"{(exc & cross).sum() / max(int(exc.sum()), 1):.2%}")
OUT["b_off"] = dict(n=int(len(sub)), pct_excursion=float(exc.mean()),
                    pct_bascule=float(cross.mean()),
                    p_exc_si_bascule=float(p_exc_if_cross),
                    p_exc_si_pas=float(p_exc_if_not),
                    part_exc_avec_bascule=float((exc & cross).sum() / max(int(exc.sum()), 1)))

# ============================== 4. DECOMPOSITION DE LA VARIANCE DU BOOTSTRAP
print("\n" + "=" * 104)
print("4. POURQUOI n_eff GLOBAL (98) << n_eff DE B SEULE (2255) ?")
print("=" * 104)
sess = base["mte"]["session_id"].to_numpy()
hit_m = (np.abs(base["yte"] - base["yhat"]) <= TOLERANCE).astype(int)
hit_p = (np.abs(base["yte"] - base["nb"]) <= TOLERANCE).astype(int)
diff = hit_m - hit_p
rows = []
for s in np.unique(sess):
    i = sess == s
    mach = "A" if base["mte"][ASSET_COLS["Motosoufflante A"]].to_numpy()[i][0] == 1 else "B"
    rows.append(dict(session=s, machine=mach, n=int(i.sum()), delta=float(diff[i].mean())))
sd = pd.DataFrame(rows)
print("  delta d'accuracy par session (unite de reechantillonnage) :")
print(sd.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
gm = sd.groupby("machine")["delta"].agg(["count", "mean", "std"])
print(f"\n{gm.to_string(float_format=lambda v: f'{v:.4f}')}")
grand = sd["delta"].mean()
ssb = ((sd.groupby("machine")["delta"].mean() - grand) ** 2
       * sd.groupby("machine")["delta"].count()).sum()
ssw = ((sd["delta"] - sd.groupby("machine")["delta"].transform("mean")) ** 2).sum()
print(f"\n  variance INTER-machine (SSB) = {ssb:.6f}   ({ssb/(ssb+ssw):.1%} du total)")
print(f"  variance INTRA-machine (SSW) = {ssw:.6f}   ({ssw/(ssb+ssw):.1%} du total)")
print(f"  -> les deux machines ont des deltas de signes OPPOSES "
      f"(A={gm.loc['A','mean']:+.4f}, B={gm.loc['B','mean']:+.4f}) ; le bootstrap global "
      f"tire\n     des melanges A/B variables, ce qui domine la variance. Ce n'est PAS "
      f"l'autocorrelation\n     intra-session : a l'interieur de B seule, deff = "
      f"{base['boot']['Motosoufflante B']['deff']:.1f} (quasi 1).")
OUT["variance"] = dict(ssb=float(ssb), ssw=float(ssw),
                       part_inter=float(ssb / (ssb + ssw)),
                       delta_A=float(gm.loc["A", "mean"]), delta_B=float(gm.loc["B", "mean"]),
                       n_sessions_A=int(gm.loc["A", "count"]),
                       n_sessions_B=int(gm.loc["B", "count"]))

with open(os.path.join(HERE, "verif_decisive.json"), "w", encoding="utf-8") as f:
    json.dump(OUT, f, indent=2, ensure_ascii=False, default=float)
print(f"\n  -> hi_forecast/verif_decisive.json")
