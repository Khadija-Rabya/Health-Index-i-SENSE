"""Execution des trois cadrages de prediction d'alertes, sur les deux etiquettes."""
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

from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, recall_score, precision_score)

from protocol import ASSET_COLS, N_SPLITS, SEED, PurgedTimeSeriesSplit
from run_19_alertes import (CLEAN, STATES, ae_features, binary_metrics,
                            bootstrap_recall_gain, load, make_clf, proba_metrics,
                            tune_threshold)

RESULTS = []


def run_binary(d, cadrage, y_all, mask_all, persist_all, critere, precision_min, lab):
    """Entraine, regle le seuil en CV, score le test une seule fois."""
    X, meta, itr, ite = d["X"], d["meta"], d["itr"], d["ite"]
    sel_tr = itr[mask_all[itr]]
    sel_te = ite[mask_all[ite]]
    if len(sel_tr) < 200 or len(sel_te) < 50 or y_all[sel_tr].sum() < 30:
        print(f"    {cadrage} : pas assez de donnees (train {len(sel_tr)}, "
              f"test {len(sel_te)}, positifs {int(y_all[sel_tr].sum())}) — ECARTE")
        return None

    Xtr, Xte = X.iloc[sel_tr], X.iloc[sel_te]
    ytr, yte = y_all[sel_tr], y_all[sel_te]
    mtr, mte = meta.iloc[sel_tr], meta.iloc[sel_te]
    spw = float((len(ytr) - ytr.sum()) / max(ytr.sum(), 1))

    # ---- CV : probabilites hors-pli, puis reglage du seuil ----
    cv = PurgedTimeSeriesSplit(n_splits=N_SPLITS)
    oof_p, oof_y = [], []
    for a, b in cv.split(Xtr, times=mtr["created_at"]):
        Xa, Xb = CLEAN(Xtr.iloc[a], Xtr.iloc[b])
        Fa, Fb = ae_features(Xa, Xb)
        m = make_clf(spw); m.fit(Fa, ytr[a])
        oof_p.append(m.predict_proba(Fb)[:, 1]); oof_y.append(ytr[b])
    P, Y = np.concatenate(oof_p), np.concatenate(oof_y)
    thr = tune_threshold(Y, P, critere, precision_min)
    cv_m = binary_metrics(Y, P, thr) | proba_metrics(Y, P)

    # ---- test gele, une seule fois ----
    Xa, Xb = CLEAN(Xtr, Xte)
    Fa, Fb = ae_features(Xa, Xb)
    m = make_clf(spw); m.fit(Fa, ytr)
    pte = m.predict_proba(Fb)[:, 1]
    te_m = binary_metrics(yte, pte, thr) | proba_metrics(yte, pte)

    pers_te = persist_all[sel_te]
    pers_m = binary_metrics(yte, pers_te.astype(float), 0.5)
    boot = bootstrap_recall_gain(yte, pte, thr, pers_te.astype(int),
                                 mte["session_id"].to_numpy())

    r = dict(cadrage=cadrage, etiquette=lab, critere=critere, seuil=thr,
             n_train=int(len(ytr)), n_test=int(len(yte)),
             prevalence_test=float(yte.mean()), cv=cv_m, test=te_m,
             persistance=pers_m, bootstrap=boot)
    RESULTS.append(r)

    print(f"    {cadrage:<34} seuil={thr:.3f}  prevalence={yte.mean():.1%}")
    print(f"      CV   : rappel {cv_m['rappel']:.3f}  precision {cv_m['precision']:.3f}  "
          f"F1 {cv_m['f1']:.3f}  PR-AUC {cv_m.get('pr_auc', float('nan')):.3f}")
    print(f"      TEST : rappel {te_m['rappel']:.3f}  precision {te_m['precision']:.3f}  "
          f"F1 {te_m['f1']:.3f}  ROC-AUC {te_m.get('roc_auc', float('nan')):.3f}  "
          f"PR-AUC {te_m.get('pr_auc', float('nan')):.3f}")
    print(f"      PERS : rappel {pers_m['rappel']:.3f}  precision {pers_m['precision']:.3f}  "
          f"F1 {pers_m['f1']:.3f}")
    print(f"      gain rappel {boot['gain_rappel']:+.3f} "
          f"IC[{boot['ic_rappel'][0]:+.3f},{boot['ic_rappel'][1]:+.3f}]  "
          f"gain F1 {boot['gain_f1']:+.3f}  "
          f"-> {'SIGNIFICATIF' if boot['rappel_significatif'] else 'non significatif'}")
    print(f"      VP={te_m['VP']} FP={te_m['FP']} FN={te_m['FN']} VN={te_m['VN']}  "
          f"taux de fausse alerte {te_m['taux_fausse_alerte']:.1%}")
    return r


for masked, lab in [(False, "etiquette actuelle"), (True, "viscosite masquee")]:
    print("\n" + "#" * 100)
    print(f"#  {lab.upper()}")
    print("#" * 100)
    t0 = time.time()
    d = load(masked)
    meta = d["meta"]
    n = len(meta)
    fut, act = meta["etat_futur"].to_numpy(), meta["etat_actuel"].to_numpy()

    # --- C1 : etat a t+18 (3 classes) — evalue a part, multi-classe ---
    print("\n  C1 — ETAT a t+18 (3 classes)")
    Xtr, Xte = d["X"].iloc[d["itr"]], d["X"].iloc[d["ite"]]
    ytr3, yte3 = act[d["itr"]], fut[d["ite"]]
    ytr3 = fut[d["itr"]]
    Xa, Xb = CLEAN(Xtr, Xte)
    Fa, Fb = ae_features(Xa, Xb)
    from sklearn.utils.class_weight import compute_sample_weight
    import lightgbm as lgb
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    p3 = Pipeline([("imp", SimpleImputer(strategy="median")),
                   ("m", lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03,
                                            num_leaves=63, min_child_samples=60,
                                            subsample=0.8, subsample_freq=1,
                                            colsample_bytree=0.7, class_weight="balanced",
                                            random_state=SEED, n_jobs=-1, verbose=-1))])
    p3.fit(Fa, ytr3)
    pred3 = p3.predict(Fb)
    pers3 = act[d["ite"]]
    for nm, s in [("MODELE", pred3), ("PERSISTANCE", pers3)]:
        print(f"    {nm:<12} accuracy {accuracy_score(yte3, s):.4f}  "
              f"bal_acc {balanced_accuracy_score(yte3, s):.4f}  "
              f"F1 macro {f1_score(yte3, s, average='macro', labels=STATES):.4f}")
    cm = confusion_matrix(yte3, pred3, labels=STATES)
    print("    matrice (lignes=vrai, colonnes=predit) : " +
          " | ".join(f"{STATES[i]}:{list(cm[i])}" for i in range(3)))
    RESULTS.append(dict(cadrage="C1 etat 3 classes", etiquette=lab,
                        test=dict(accuracy=float(accuracy_score(yte3, pred3)),
                                  bal_accuracy=float(balanced_accuracy_score(yte3, pred3)),
                                  f1_macro=float(f1_score(yte3, pred3, average="macro",
                                                          labels=STATES))),
                        persistance=dict(accuracy=float(accuracy_score(yte3, pers3)),
                                         bal_accuracy=float(balanced_accuracy_score(yte3, pers3)),
                                         f1_macro=float(f1_score(yte3, pers3, average="macro",
                                                                 labels=STATES)))))

    # --- C2 : alerte a t+18 (binaire) ---
    print("\n  C2 — ALERTE a t+18 (etat != Normal)")
    y2 = (fut != "Normal").astype(int)
    persist2 = (act != "Normal").astype(int)
    all_rows = np.ones(n, dtype=bool)
    for crit, pmin in [("f1", 0.0), ("rappel_prec", 0.60)]:
        run_binary(d, f"C2 alerte [{crit}]", y2, all_rows, persist2, crit, pmin, lab)

    # --- C3 : NOUVELLE alerte (transition Normal -> alerte) ---
    print("\n  C3 — NOUVELLE ALERTE (Normal maintenant -> alerte dans 3 h)")
    print("       la persistance predit STRUCTURELLEMENT zero transition : rappel 0 par construction")
    mask3 = (act == "Normal")
    y3 = ((act == "Normal") & (fut != "Normal")).astype(int)
    persist3 = np.zeros(n, dtype=int)
    for crit, pmin in [("f1", 0.0), ("rappel_prec", 0.30), ("rappel_prec", 0.20)]:
        run_binary(d, f"C3 nouvelle alerte [{crit}"
                      + (f" p>={pmin}" if crit != "f1" else "") + "]",
                   y3, mask3, persist3, crit, pmin, lab)
    print(f"\n  [{time.time()-t0:.0f}s]")

with open(os.path.join(HERE, "alertes_resultats.json"), "w", encoding="utf-8") as f:
    json.dump(RESULTS, f, indent=2, ensure_ascii=False, default=float)

print("\n" + "=" * 100)
print("SYNTHESE — PREDICTION D'ALERTES")
print("=" * 100)
rows = []
for r in RESULTS:
    if "bootstrap" not in r:
        continue
    rows.append(dict(etiquette=r["etiquette"], cadrage=r["cadrage"],
                     prevalence=r["prevalence_test"], seuil=r["seuil"],
                     rappel=r["test"]["rappel"], precision=r["test"]["precision"],
                     f1=r["test"]["f1"], pr_auc=r["test"].get("pr_auc"),
                     rappel_pers=r["persistance"]["rappel"],
                     f1_pers=r["persistance"]["f1"],
                     gain_rappel=r["bootstrap"]["gain_rappel"],
                     signif=r["bootstrap"]["rappel_significatif"]))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(HERE, "alertes_resultats.csv"), index=False, encoding="utf-8-sig")
print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\n  -> hi_forecast/alertes_resultats.json / .csv")
