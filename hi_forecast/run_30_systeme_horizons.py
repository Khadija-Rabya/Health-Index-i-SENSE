"""Prediction du Health Index SYSTEME — comparaison de familles de modeles.

Pendant : run_28_multi_horizon.py fait la meme chose pour le Health Index HUILE.
Ici la cible est HI_systeme_lf (cf. systeme_hi.py), aux memes quatre horizons :
10 min (t+1), 20 min (t+2), 3 h (t+18), 24 h (t+144).

FAMILLES COMPAREES
    persistance          « rien ne change » — la reference a battre, pas un modele
    ridge                lineaire regularise L2
    lasso_porte          Lasso sur le delta + porte de mouvement (config du HI huile)
    foret_aleatoire      RandomForest — arbres en bagging
    gradient_histogramme HistGradientBoosting (scikit-learn)
    xgboost              XGBoost
    lightgbm             LightGBM, perte de Huber — la configuration retenue pour
                         l'indice HUILE, reprise ici pour que les deux indices
                         soient juges sur le meme jeu de familles
    catboost             CatBoost

Les trois derniers ont ete ajoutes apres coup : la premiere version ne comparait
que quatre familles, alors que le pipeline de l'indice huile avait ete evalue
contre LightGBM et CatBoost. Les deux indices etaient donc juges sur des jeux de
familles differents — une incoherence, corrigee ici.

PROTOCOLE — identique au protocole gele du projet :
    - decoupage gele par machine (cutoffs sur horodatages, jamais aleatoire) ;
    - selection de la famille par validation croisee PURGEE sur l'entrainement
      seul, avec embargo proportionnel a l'horizon ;
    - le test n'est touche qu'une fois, apres la selection. La famille gagnante
      est designee par la CV, jamais par le score de test.

METRIQUE DE DECISION : le SKILL contre la persistance, pas le R2. La cible est
un indice construit, sans verite terrain : un R2 flatteur peut n'etre que la
reconduction de la valeur courante. Un modele qui ne bat pas la persistance
n'apporte rien, quel que soit son R2.

LIMITE CONNUE — Motosoufflante B : la machine ne tourne plus apres le 31 janvier
et sa fenetre de test ne contient aucune ligne en marche. Son indice systeme est
donc calculable mais NON EVALUABLE. Le script l'ecrit explicitement au lieu de
produire un score trompeur.
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

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from protocol import (ASSET_COLS, N_SPLITS, SEED, TOLERANCE, PurgedTimeSeriesSplit,
                      feature_columns, frozen_cutoffs, prepare, regression_metrics,
                      split_frozen, train_mask_from_cutoffs)
from run_16_ae_roleA import make_clf
from systeme_hi import build_systeme_hi, resume_zones

HORIZONS = {1: "10 min", 2: "20 min", 18: "3 h", 144: "24 h"}
ART = os.path.join(HERE, "artifacts")
CIBLE = "HI_systeme_lf"

# Colonnes a retirer des features : sorties textuelles ou booleennes du module
# systeme, et la cible elle-meme (elle revient comme ancre de persistance).
HORS_FEATURES = {"zone_iso_lf", "etat_systeme_lf", "systeme_evaluable"}


def p_ridge():
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("m", Ridge(alpha=1.0, random_state=SEED))])


def p_lasso():
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("m", Lasso(alpha=1e-4, max_iter=5000, random_state=SEED))])


def p_foret():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                                 random_state=SEED, n_jobs=-1))])


def p_hgb():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                                         random_state=SEED))])


def p_xgb():
    from xgboost import XGBRegressor
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", XGBRegressor(n_estimators=300, learning_rate=0.06,
                                        max_depth=6, subsample=0.8,
                                        colsample_bytree=0.8, random_state=SEED,
                                        n_jobs=-1, verbosity=0))])


def p_lgbm():
    from lightgbm import LGBMRegressor
    # Perte de Huber : elle estime une MEDIANE conditionnelle, pas une moyenne.
    # Elle predit donc de petits ecarts, ce que la bande de tolerance recompense.
    # C'est la configuration retenue pour l'indice HUILE.
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", LGBMRegressor(objective="huber", n_estimators=300,
                                         learning_rate=0.06, num_leaves=31,
                                         random_state=SEED, n_jobs=-1, verbose=-1))])


def p_cat():
    from catboost import CatBoostRegressor
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", CatBoostRegressor(iterations=300, learning_rate=0.06,
                                             depth=6, random_seed=SEED,
                                             verbose=0, allow_writing_files=False))])


def build_xy_systeme(df: pd.DataFrame, horizon: int):
    """X a t, y = HI_systeme_lf a t+horizon, dans la MEME session.

    On exige que l'indice existe aux DEUX instants : la machine doit tourner a t
    et a t+horizon. Un arret intercale rend la cible indefinie, on ecarte la
    fenetre plutot que d'inventer une valeur."""
    d = df.sort_values(["session_id", "created_at"]).reset_index(drop=True)
    y = d.groupby("session_id", sort=False)[CIBLE].shift(-horizon)
    valid = (y.notna() & d[CIBLE].notna()).to_numpy()

    feats = [c for c in feature_columns(d) if c not in HORS_FEATURES]
    X = d.loc[valid, feats].reset_index(drop=True).astype(np.float64)
    y = y[valid].reset_index(drop=True)
    meta = d.loc[valid, ["created_at", "session_id", CIBLE] + list(ASSET_COLS.values())]
    meta = meta.reset_index(drop=True).rename(columns={CIBLE: "hi_now"})
    return X, y, meta


def acc_tol(y, yhat):
    return float(np.mean(np.abs(y - yhat) <= TOLERANCE))


print("=" * 96)
print("HEALTH INDEX SYSTEME — MULTI-HORIZONS, COMPARAISON DE MODELES")
print("=" * 96)

df, _, cutoffs, _ = prepare(verbose=False, mask_off_viscosity=False)
tm = train_mask_from_cutoffs(df, cutoffs)
df, calibrage = build_systeme_hi(df, tm)
print("\nCalibrage (sous-indices monotones, maillon faible) :")
for a, c in calibrage.items():
    if "erreur" in c:
        print(f"  {a:20} ECARTE — {c['erreur']}")
        continue
    v, pr, ct = c["vibration"], c["pression"], c["controle"]
    print(f"  {a:20} vibration : sain <= {v['reference']:.3f}  alarme a "
          f"{v['seuil_alarme']:.3f}  ({(v['pct_impute'] or 0)*100:.0f} % imputee)")
    print(f"  {'':20} pression  : cible {pr['cible_bar']:.2f} bar  alarme a "
          f"±{pr['seuil_alarme']:.2f}")
    print(f"  {'':20} controle  : {ct['taux_alarme_population_saine']*100:.2f} % de la "
          f"population saine en alarme  |  pilote par la vibration "
          f"{ct['pilote_vibration']*100:.0f} %, par la pression "
          f"{ct['pilote_pression']*100:.0f} %")
zones = resume_zones(df)
print("\nRepartition des zones :")
print(zones.to_string(index=False))

# Correlation entre les deux indices : elle justifie qu'on en garde DEUX plutot
# qu'un. Trop haute, le second serait redondant ; nulle, ils ne mesureraient pas
# la meme machine.
_m = df["systeme_evaluable"] & df["HI_systeme_lf"].notna() & df["health_index_lf"].notna()
_correl = float(df.loc[_m, ["HI_systeme_lf", "health_index_lf"]].corr().iloc[0, 1])
os.makedirs(ART, exist_ok=True)
with open(os.path.join(ART, "systeme_zones.json"), "w", encoding="utf-8") as f:
    json.dump({"zones": zones.to_dict("records"),
               "correlation_huile_systeme": _correl,
               "n_lignes_comparees": int(_m.sum())},
              f, indent=2, ensure_ascii=False, default=float)
print(f"  correlation huile/systeme : {_correl:.4f} sur {int(_m.sum())} lignes")

FAMILLES = {"ridge": p_ridge, "lasso_porte": p_lasso,
            "foret_aleatoire": p_foret, "gradient_histogramme": p_hgb,
            "xgboost": p_xgb, "lightgbm_huber": p_lgbm, "catboost": p_cat}

BUNDLE, LEADERBOARD, METRICS = {}, [], []

for h, label in HORIZONS.items():
    X, y, meta = build_xy_systeme(df, h)
    Xtr, Xte, ytr, yte, mtr, mte = split_frozen(X, y, meta, cutoffs)
    cols = list(Xtr.columns)
    Ttr, Tte = Xtr.to_numpy(), Xte.to_numpy()
    ntr, nte = mtr["hi_now"].to_numpy(), mte["hi_now"].to_numpy()
    ytr_a, yte_a = ytr.to_numpy(), yte.to_numpy()
    emb = pd.Timedelta(minutes=10 * h)

    print(f"\n{'='*96}\n### Horizon t+{h} ({label}) — train {len(Xtr)}, test {len(Xte)}, "
          f"embargo {emb}, {len(cols)} variables")

    for asset, col in ASSET_COLS.items():
        tr = np.flatnonzero((mtr[col] == 1).to_numpy())
        te = np.flatnonzero((mte[col] == 1).to_numpy())
        if len(tr) < 300:
            print(f"  {asset:<20} ECARTE — {len(tr)} lignes d'entrainement")
            continue
        if len(te) == 0:
            print(f"  {asset:<20} entrainable ({len(tr)} lignes) mais NON EVALUABLE — "
                  f"0 ligne de test en marche")
            METRICS.append(dict(horizon=h, horizon_label=label, machine=asset,
                                evaluable=False, n_train=int(len(tr)), n_test=0,
                                motif="aucune ligne en marche dans la fenetre de test"))

        # ---------- selection de la famille par CV purgee sur l'entrainement ----------
        cv = PurgedTimeSeriesSplit(n_splits=N_SPLITS, horizon_td=emb, embargo_td=emb)
        plis = list(cv.split(np.zeros((len(tr), 1)), times=mtr["created_at"].iloc[tr]))
        scores = {k: [] for k in list(FAMILLES) + ["persistance"]}
        for a, b in plis:
            ia, ib = tr[a], tr[b]
            if len(ia) < 100 or len(ib) < 20:
                continue
            scores["persistance"].append(acc_tol(ytr_a[ib], ntr[ib]))
            for nom, fab in FAMILLES.items():
                t0 = time.time()
                if nom == "lasso_porte":
                    r = fab(); r.fit(Ttr[ia], ytr_a[ia] - ntr[ia])
                    c = make_clf()
                    c.fit(Ttr[ia], (np.abs(ytr_a[ia] - ntr[ia]) > TOLERANCE).astype(int))
                    pr = c.predict_proba(Ttr[ib])[:, 1]
                    yh = np.clip(ntr[ib] + r.predict(Ttr[ib]) * (pr > 0.5), 0, 1)
                else:
                    m = fab(); m.fit(Ttr[ia], ytr_a[ia])
                    yh = np.clip(m.predict(Ttr[ib]), 0, 1)
                scores[nom].append(acc_tol(ytr_a[ib], yh))
        cv_moy = {k: float(np.mean(v)) if v else np.nan for k, v in scores.items()}
        cv_sd = {k: float(np.std(v)) if v else np.nan for k, v in scores.items()}
        if all(not np.isfinite(cv_moy[k]) for k in FAMILLES):
            # Aucun pli exploitable : fenetre trop courte pour cet horizon. On
            # l'ecrit plutot que de designer un gagnant sur des scores absents.
            print(f"  {asset:<20} ECARTE — aucun pli de CV exploitable a t+{h} "
                  f"({len(plis)} plis tentes, fenetre trop courte)")
            METRICS.append(dict(horizon=h, horizon_label=label, machine=asset,
                                evaluable=False, n_train=int(len(tr)), n_test=int(len(te)),
                                motif="aucun pli de validation croisee exploitable"))
            continue
        gagnant = max(FAMILLES, key=lambda k: cv_moy[k])

        print(f"\n  {asset} — CV purgee ({len(plis)} plis) :")
        for k in ["persistance"] + list(FAMILLES):
            marque = "  <-- retenu" if k == gagnant else ""
            print(f"      {k:<22} acc {cv_moy[k]:.4f} ± {cv_sd[k]:.4f}{marque}")

        # ---------- refit sur tout l'entrainement, test touche une seule fois ----------
        modeles_fit = {}
        for nom, fab in FAMILLES.items():
            if nom == "lasso_porte":
                r = fab(); r.fit(Ttr[tr], ytr_a[tr] - ntr[tr])
                c = make_clf()
                c.fit(Ttr[tr], (np.abs(ytr_a[tr] - ntr[tr]) > TOLERANCE).astype(int))
                modeles_fit[nom] = ("porte", r, c)
            else:
                m = fab(); m.fit(Ttr[tr], ytr_a[tr])
                modeles_fit[nom] = ("simple", m, None)

        for nom, (kind, m1, m2) in modeles_fit.items():
            ligne = dict(horizon=h, horizon_label=label, machine=asset, famille=nom,
                         cv_acc=cv_moy[nom], cv_sd=cv_sd[nom],
                         cv_acc_persistance=cv_moy["persistance"],
                         retenu_par_cv=(nom == gagnant), n_train=int(len(tr)),
                         n_test=int(len(te)))
            if len(te):
                if kind == "porte":
                    pr = m2.predict_proba(Tte[te])[:, 1]
                    yh = np.clip(nte[te] + m1.predict(Tte[te]) * (pr > 0.5), 0, 1)
                else:
                    yh = np.clip(m1.predict(Tte[te]), 0, 1)
                mt = regression_metrics(yte_a[te], yh, n_features=len(cols),
                                        hi_now=nte[te])
                ligne.update(test_acc=mt["acc_tol"], acc_persist=mt["acc_tol_persist"],
                             skill=mt["skill_vs_persist"], r2=mt["r2"],
                             rmse=mt["rmse"], mae=mt["mae"], evaluable=True)
            else:
                ligne.update(evaluable=False)
            LEADERBOARD.append(ligne)

        kind, m1, m2 = modeles_fit[gagnant]
        BUNDLE[f"{h}|{asset}"] = dict(horizon=h, horizon_label=label, machine=asset,
                                      famille=gagnant, type=kind, modele=m1, porte=m2,
                                      colonnes=cols, n_train=int(len(tr)),
                                      cv_acc=cv_moy[gagnant])
        if len(te):
            g = [r for r in LEADERBOARD if r["machine"] == asset and r["horizon"] == h
                 and r["famille"] == gagnant][0]
            METRICS.append(dict(horizon=h, horizon_label=label, machine=asset,
                                evaluable=True, famille=gagnant, cv_acc=g["cv_acc"],
                                test_acc=g["test_acc"], acc_persist=g["acc_persist"],
                                skill=g["skill"], r2=g["r2"], rmse=g["rmse"],
                                mae=g["mae"], n_train=g["n_train"], n_test=g["n_test"]))
            print(f"      TEST ({gagnant}) : acc {g['test_acc']:.4f} "
                  f"(persistance {g['acc_persist']:.4f})  skill {g['skill']:+.4f}  "
                  f"R2 {g['r2']:.4f}")

# ------------------------------------------------------------------- sorties
os.makedirs(ART, exist_ok=True)
p = os.path.join(ART, "modeles_systeme_multi_horizon.joblib")
joblib.dump(BUNDLE, p, compress=3)

lb = pd.DataFrame(LEADERBOARD)
lb.to_csv(os.path.join(HERE, "systeme_leaderboard.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(ART, "systeme_metriques.json"), "w", encoding="utf-8") as f:
    json.dump(METRICS, f, indent=2, ensure_ascii=False, default=float)
with open(os.path.join(ART, "systeme_calibrage.json"), "w", encoding="utf-8") as f:
    json.dump(calibrage, f, indent=2, ensure_ascii=False, default=float)
with open(os.path.join(ART, "systeme_leaderboard.json"), "w", encoding="utf-8") as f:
    json.dump(LEADERBOARD, f, indent=2, ensure_ascii=False, default=float)

print("\n" + "=" * 96)
print("CLASSEMENT FINAL (skill contre la persistance, sur le test)")
print("=" * 96)
ev = lb[lb["evaluable"] == True] if "evaluable" in lb else lb
if len(ev):
    vue = ev[["horizon_label", "machine", "famille", "cv_acc", "test_acc",
              "acc_persist", "skill", "r2"]].sort_values(["machine", "horizon_label",
                                                          "skill"], ascending=[1, 1, 0])
    print(vue.to_string(index=False))
else:
    print("  aucune ligne evaluable")
print(f"\n  artefact : {p}  ({os.path.getsize(p)/1e6:.2f} Mo)  — {len(BUNDLE)} modeles")
print(f"  -> hi_forecast/systeme_leaderboard.csv")
print(f"  -> hi_forecast/artifacts/systeme_metriques.json")
