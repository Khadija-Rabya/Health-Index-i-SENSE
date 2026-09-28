"""VERIFICATION (aucun reglage) — repond aux 4 demandes de controle.

A. Comparaison EN CV, plis identiques, du RF d'origine et du modele final a porte.
B. Intervalles : McNemar apparie + bootstrap par blocs (sessions) sur le delta
   d'accuracy et sur le skill ; taille d'echantillon effective.
C. Arbitrage entre metriques : porte / RF sans fuite / ElasticNet.
D. Controles de coherence (AUC, valeurs identiques, definitions).
E. Preuves chiffrees pour la note viscosite.
"""
import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

import lightgbm as lgb

from gated import compute_oof, get_folds
from protocol import ASSET_COLS, SEED, TOLERANCE, regression_metrics
from train_best import GatedHealthIndexForecaster  # noqa: F401 (unpickle)
from transforms import chain, drop_constant_and_dupes, prune_correlated

OUT = {}
d, folds = get_folds()
Xtr, ytr, mtr = d["Xtr"], d["ytr"], d["mtr"]
Xte, yte, mte = d["Xte"], d["yte"], d["mte"]
na, nb = mtr["hi_now"].to_numpy(), mte["hi_now"].to_numpy()
yte_a = yte.to_numpy()

CLEAN = chain(drop_constant_and_dupes, prune_correlated(0.995))
with open(os.path.join(HERE, "cache", "final_config.json"), encoding="utf-8") as f:
    cfg = json.load(f)
P = cfg["params"]
rank = pd.read_csv(os.path.join(HERE, "classements_importance.csv"), index_col=0)
wanted = list(rank["shap"].sort_values(ascending=False).head(20).index)


def TF20(Xa_, Xb_):
    Xa_, Xb_ = CLEAN(Xa_, Xb_)
    cols = [c for c in wanted if c in Xa_.columns]
    return Xa_[cols], Xb_[cols]


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


def rf_origine():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", RandomForestRegressor(n_estimators=300, max_depth=14,
                                                 random_state=SEED, n_jobs=-1))])


def enet():
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("m", ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=5000,
                                      random_state=SEED))])


# ============================================================== A — CV comparee
print("=" * 108)
print("A — RF D'ORIGINE vs MODELE FINAL A PORTE, EN CV, PLIS IDENTIQUES")
print("=" * 108)
oof_gate = compute_oof(make_reg, make_clf, TF20, d, folds)
ALPHA, SEUIL = cfg["alpha"], cfg["seuil"]

rows_a = []
print(f"{'pli':<5}{'n_val':>7} | {'RF orig acc':>12}{'RF orig R2':>12} | "
      f"{'porte acc':>11}{'porte R2':>11} | {'persist acc':>12}")
rf_acc, rf_r2, g_acc, g_r2, p_acc = [], [], [], [], []
for k, ((itr, iva), o) in enumerate(zip(folds, oof_gate), 1):
    # RF d'origine : toutes les features, pas de transformation, formulation delta
    Xa, Xb = Xtr.iloc[itr], Xtr.iloc[iva]
    ya, yb = ytr.iloc[itr].to_numpy(), ytr.iloc[iva].to_numpy()
    na_, nb_ = mtr["hi_now"].iloc[itr].to_numpy(), mtr["hi_now"].iloc[iva].to_numpy()
    m = rf_origine(); m.fit(Xa, ya - na_)
    yhat_rf = nb_ + m.predict(Xb)
    a1 = float(np.mean(np.abs(yb - yhat_rf) <= TOLERANCE)); r1 = r2_score(yb, yhat_rf)
    yhat_g = o["n"] + ALPHA * o["d"] * (o["p"] > SEUIL)
    a2 = float(np.mean(np.abs(o["y"] - yhat_g) <= TOLERANCE)); r2v = r2_score(o["y"], yhat_g)
    a3 = float(np.mean(np.abs(yb - nb_) <= TOLERANCE))
    rf_acc.append(a1); rf_r2.append(r1); g_acc.append(a2); g_r2.append(r2v); p_acc.append(a3)
    rows_a.append(dict(pli=k, n_val=len(yb), rf_acc=a1, rf_r2=r1, gate_acc=a2,
                       gate_r2=r2v, persist_acc=a3))
    print(f"{k:<5}{len(yb):>7} | {a1:>12.4f}{r1:>12.4f} | {a2:>11.4f}{r2v:>11.4f} | {a3:>12.4f}")
print(f"{'moy':<5}{'':>7} | {np.mean(rf_acc):>12.4f}{np.mean(rf_r2):>12.4f} | "
      f"{np.mean(g_acc):>11.4f}{np.mean(g_r2):>11.4f} | {np.mean(p_acc):>12.4f}")
print(f"{'ecart-type':<12} | {np.std(rf_acc):>5.4f}{'':>19} | {np.std(g_acc):>11.4f}")
print(f"\n  VERDICT CV : porte {np.mean(g_acc):.4f} vs RF d'origine {np.mean(rf_acc):.4f}"
      f"  -> ecart {np.mean(g_acc)-np.mean(rf_acc):+.4f}")
print(f"  la porte gagne sur {sum(1 for a,b in zip(g_acc, rf_acc) if a>b)}/5 plis")
pd.DataFrame(rows_a).to_csv(os.path.join(HERE, "verif_A_cv_comparee.csv"), index=False)
OUT["A"] = dict(rf_cv_acc=float(np.mean(rf_acc)), rf_cv_sd=float(np.std(rf_acc)),
                rf_cv_r2=float(np.mean(rf_r2)), gate_cv_acc=float(np.mean(g_acc)),
                gate_cv_sd=float(np.std(g_acc)), gate_cv_r2=float(np.mean(g_r2)),
                persist_cv_acc=float(np.mean(p_acc)),
                plis_gagnes=int(sum(1 for a, b in zip(g_acc, rf_acc) if a > b)))

# ================================================ predictions sur le test gele
model = joblib.load(os.path.join(HERE, "artifacts", "health_index_t18_pipeline.joblib"))
yhat_gate = model.predict(Xte, nb)

m_rf = rf_origine(); m_rf.fit(Xtr, ytr.to_numpy() - na)
yhat_rf_te = np.clip(nb + m_rf.predict(Xte), 0, 1)

Xa_c, Xb_c = CLEAN(Xtr, Xte)
m_en = enet(); m_en.fit(Xa_c, ytr.to_numpy() - na)
yhat_en_te = np.clip(nb + m_en.predict(Xb_c), 0, 1)

# ======================================================= B — McNemar + bootstrap
print("\n" + "=" * 108)
print("B — SIGNIFICATIVITE : McNEMAR APPARIE + BOOTSTRAP PAR BLOCS (SESSIONS)")
print("=" * 108)
from scipy import stats

hit_g = (np.abs(yte_a - yhat_gate) <= TOLERANCE).astype(int)
hit_p = (np.abs(yte_a - nb) <= TOLERANCE).astype(int)
sess = mte["session_id"].to_numpy()


def mcnemar(h1, h0, label):
    b = int(np.sum((h1 == 1) & (h0 == 0)))   # modele gagne
    c = int(np.sum((h1 == 0) & (h0 == 1)))   # persistance gagne
    n_disc = b + c
    if n_disc == 0:
        return dict(label=label, b=b, c=c, p=np.nan, stat=np.nan)
    p_exact = float(stats.binomtest(b, n_disc, 0.5).pvalue)
    stat = (abs(b - c) - 1) ** 2 / n_disc
    print(f"  {label:<22} b(modele seul)={b:>5}  c(persist seul)={c:>5}  "
          f"discordants={n_disc:>5}  chi2_cc={stat:>7.3f}  p_exact={p_exact:.3e}")
    return dict(label=label, b=b, c=c, n_disc=n_disc, chi2=float(stat), p=p_exact)


print("  McNemar exact (binomial sur les paires discordantes) :")
mc = {"pooled": mcnemar(hit_g, hit_p, "global")}
for a, col in ASSET_COLS.items():
    msk = (mte[col] == 1).to_numpy()
    mc[a] = mcnemar(hit_g[msk], hit_p[msk], a)


def block_bootstrap(idx, n_boot=4000, seed=SEED):
    """Reechantillonne les SESSIONS avec remise (les lignes d'une meme session
    sont fortement correlees : un bootstrap ligne a ligne sous-estimerait la
    variance)."""
    rng = np.random.default_rng(seed)
    s = sess[idx]
    uniq = np.unique(s)
    groups = {u: idx[s == u] for u in uniq}
    d_acc, d_skill = [], []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        ii = np.concatenate([groups[u] for u in pick])
        d_acc.append(hit_g[ii].mean() - hit_p[ii].mean())
        mse_m = mean_squared_error(yte_a[ii], yhat_gate[ii])
        mse_p = mean_squared_error(yte_a[ii], nb[ii])
        d_skill.append(1 - mse_m / mse_p if mse_p > 0 else np.nan)
    return np.array(d_acc), np.array(d_skill), len(uniq)


print("\n  Bootstrap par blocs (4000 tirages, unite = session) :")
boot = {}
scopes = [("global", np.arange(len(yte_a)))]
for a, col in ASSET_COLS.items():
    scopes.append((a, np.flatnonzero((mte[col] == 1).to_numpy())))
for label, idx in scopes:
    da, ds, n_sess = block_bootstrap(idx)
    pt_acc = hit_g[idx].mean() - hit_p[idx].mean()
    pt_sk = 1 - mean_squared_error(yte_a[idx], yhat_gate[idx]) / mean_squared_error(yte_a[idx], nb[idx])
    ci_a = np.percentile(da, [2.5, 97.5]); ci_s = np.percentile(ds, [2.5, 97.5])
    # taille effective : deff = var_bloc / var_iid  (sur le delta d'accuracy)
    diff = hit_g[idx] - hit_p[idx]
    var_iid = diff.var(ddof=1) / len(idx)
    deff = da.var(ddof=1) / var_iid if var_iid > 0 else np.nan
    n_eff = len(idx) / deff if deff and np.isfinite(deff) else np.nan
    crosses = ci_a[0] <= 0 <= ci_a[1]
    boot[label] = dict(n=len(idx), n_sessions=int(n_sess), delta_acc=float(pt_acc),
                       ci_acc=[float(ci_a[0]), float(ci_a[1])], skill=float(pt_sk),
                       ci_skill=[float(ci_s[0]), float(ci_s[1])],
                       deff=float(deff), n_eff=float(n_eff), ci_crosses_zero=bool(crosses))
    print(f"  {label:<22} n={len(idx):>5} sessions={n_sess:>3} | "
          f"delta acc = {pt_acc:+.4f}  IC95 [{ci_a[0]:+.4f}, {ci_a[1]:+.4f}] "
          f"{'** CROISE 0 **' if crosses else ''}")
    print(f"  {'':<22} skill = {pt_sk:+.4f}  IC95 [{ci_s[0]:+.4f}, {ci_s[1]:+.4f}]  |  "
          f"deff={deff:.1f}  n_eff={n_eff:.0f}")
OUT["B"] = dict(mcnemar=mc, bootstrap=boot)

# ================================================== C — arbitrage des metriques
print("\n" + "=" * 108)
print("C — ARBITRAGE ENTRE METRIQUES (memes lignes de test, meme protocole)")
print("=" * 108)
cands = {"Modele final a porte": yhat_gate, "RF sans fuite (it00)": yhat_rf_te,
         "ElasticNet (it13)": yhat_en_te, "Persistance": nb}
rows_c = []
print(f"{'configuration':<24}{'acc@±0.01':>11}{'R2':>9}{'RMSE':>9}{'MAE':>9}{'skill':>9}")
for name, yh in cands.items():
    mm = regression_metrics(yte_a, yh, hi_now=nb)
    rows_c.append(dict(configuration=name, acc=mm["acc_tol"], r2=mm["r2"], rmse=mm["rmse"],
                       mae=mm["mae"], skill=mm["skill_vs_persist"]))
    print(f"{name:<24}{mm['acc_tol']:>11.4f}{mm['r2']:>9.4f}{mm['rmse']:>9.5f}"
          f"{mm['mae']:>9.5f}{mm['skill_vs_persist']:>+9.4f}")
pd.DataFrame(rows_c).to_csv(os.path.join(HERE, "verif_C_arbitrage.csv"), index=False)
OUT["C"] = rows_c

# ==================================================== D — controles de coherence
print("\n" + "=" * 108)
print("D — CONTROLES DE COHERENCE")
print("=" * 108)
lab_te = (np.abs(yte_a - nb) > TOLERANCE).astype(int)
prob_final = model.clf.predict_proba(Xte[model.columns])[:, 1]
auc_final = roc_auc_score(lab_te, prob_final)
print(f"  AUC de la porte, modele FINAL (Optuna), sur le test  : {auc_final:.4f}")
print(f"  AUC de la porte, configuration it24 (LGBM d'origine) : 0.7426  (rapportee dans run_05)")
print(f"  -> deux classifieurs DIFFERENTS ; le 0.74 du §8 designe la configuration it24, "
      f"pas le modele final.")

lb = pd.read_csv(os.path.join(HERE, "leaderboard_familles.csv"))
rf_test_acc = float(lb.loc[lb["name"] == "Random Forest", "test_acc_tol"].iloc[0])
pol = pd.read_csv(os.path.join(HERE, "sweep_politique.csv"))
a_cv = float(pol.loc[pol["machine"] == "Motosoufflante A", "cv_acc_modele"].iloc[0])
print(f"\n  Random Forest, acc de TEST (leaderboard)  : {rf_test_acc:.10f}")
print(f"  Motosoufflante A, acc de CV (politique)   : {a_cv:.10f}")
print(f"  -> identiques ? {'OUI (suspect)' if abs(rf_test_acc-a_cv) < 1e-9 else 'NON — coincidence a 4 decimales seulement'}")

sd = lb["cv_acc_tol_std"]
print(f"\n  Leaderboard, colonne CV acc : ecart-type de {sd.min():.2f} a {sd.max():.2f} ; "
      f"etendue des moyennes = {lb['cv_acc_tol_mean'].max()-lb['cv_acc_tol_mean'].min():.2f}")
print(f"  -> l'etendue ({lb['cv_acc_tol_mean'].max()-lb['cv_acc_tol_mean'].min():.2f}) "
      f"est du meme ordre que l'ecart-type median ({sd.median():.2f}) : cette colonne "
      f"ne separe pas les familles.")
OUT["D"] = dict(auc_final=float(auc_final), auc_it24=0.7426,
                rf_test_acc=rf_test_acc, a_cv_acc=a_cv,
                identiques=bool(abs(rf_test_acc - a_cv) < 1e-9),
                lb_sd_min=float(sd.min()), lb_sd_max=float(sd.max()),
                lb_sd_median=float(sd.median()),
                lb_range=float(lb["cv_acc_tol_mean"].max() - lb["cv_acc_tol_mean"].min()))

# ============================================== E — preuves pour la note viscosite
print("\n" + "=" * 108)
print("E — PREUVES CHIFFREES : VISCOSITE HORS PLAGE")
print("=" * 108)
raw = pd.read_csv(os.path.join(os.path.dirname(HERE), "isense_oil_data_health_index.csv"),
                  parse_dates=["created_at"], low_memory=False)
v = raw["Viscosity at 40°C_filled"]
low = v < 20
tot = len(raw)
print(f"  lignes totales : {tot}")
print(f"  Viscosity@40C < 20 cSt : {int(low.sum())} ({low.mean():.2%})")
print(f"  plage de ces valeurs : min={v[low].min():.2f}  med={v[low].median():.2f}  max={v[low].max():.2f} cSt")
print(f"  (reference TD46 / ISO VG 46 = 46,0 cSt ; plage normale du reste : "
      f"{v[~low].min():.2f}-{v[~low].max():.2f})")

tab = []
for a, col in ASSET_COLS.items():
    for st, lab in [(1, "OFF"), (0, "ON")]:
        m = (raw[col] == 1) & (raw["state_OFF"] == st)
        n = int(m.sum()); nl = int((m & low).sum())
        tab.append(dict(machine=a, etat=lab, n=n, n_basse=nl,
                        pct=(nl / n if n else np.nan)))
        print(f"    {a:<20} {lab:<4} n={n:>6}  viscosite<20 : {nl:>6}  ({nl/n if n else 0:.1%})")
ev = pd.DataFrame(tab)
ev.to_csv(os.path.join(HERE, "verif_E_viscosite.csv"), index=False, encoding="utf-8-sig")

mb = raw[ASSET_COLS["Motosoufflante B"]] == 1
print(f"\n  Motosoufflante B seule : viscosite mediane OFF = "
      f"{raw.loc[mb & (raw['state_OFF']==1), 'Viscosity at 40°C_filled'].median():.2f} cSt, "
      f"ON = {raw.loc[mb & (raw['state_OFF']==0), 'Viscosity at 40°C_filled'].median():.2f} cSt")
print(f"  Motosoufflante A seule : viscosite mediane OFF = "
      f"{raw.loc[~mb & (raw['state_OFF']==1), 'Viscosity at 40°C_filled'].median():.2f} cSt, "
      f"ON = {raw.loc[~mb & (raw['state_OFF']==0), 'Viscosity at 40°C_filled'].median():.2f} cSt")
print(f"\n  health_index moyen, Motosoufflante B : OFF = "
      f"{raw.loc[mb & (raw['state_OFF']==1), 'health_index'].mean():.4f}, "
      f"ON = {raw.loc[mb & (raw['state_OFF']==0), 'health_index'].mean():.4f}")
print(f"  part des etats Alarme/Surveillance de B en OFF : "
      f"{(raw.loc[mb & (raw['state_OFF']==1), 'health_state'] != 'Normal').mean():.1%}"
      f"  vs en ON : {(raw.loc[mb & (raw['state_OFF']==0), 'health_state'] != 'Normal').mean():.1%}")
OUT["E"] = dict(n_total=int(tot), n_low=int(low.sum()), pct_low=float(low.mean()),
                low_min=float(v[low].min()), low_med=float(v[low].median()),
                low_max=float(v[low].max()), par_groupe=ev.to_dict("records"))

with open(os.path.join(HERE, "verif_resultats.json"), "w", encoding="utf-8") as f:
    json.dump(OUT, f, indent=2, ensure_ascii=False, default=float)
print(f"\n  -> hi_forecast/verif_resultats.json")
