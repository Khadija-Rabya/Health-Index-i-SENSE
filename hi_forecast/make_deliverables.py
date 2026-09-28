"""Verifie l'artefact sauvegarde (aller-retour joblib) et produit les tableaux
Markdown des livrables 1 et 2."""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gated import get_folds
from protocol import ASSET_COLS, TOLERANCE, regression_metrics
from train_best import GatedHealthIndexForecaster  # noqa: F401  (necessaire au unpickle)

# ------------------------------------------------------- verification artefact
print("=" * 100)
print("VERIFICATION DE L'ARTEFACT (aller-retour joblib)")
print("=" * 100)
art = os.path.join(HERE, "artifacts", "health_index_t18_pipeline.joblib")
model = joblib.load(art)
d, _ = get_folds()
Xte, yte, mte = d["Xte"], d["yte"], d["mte"]
yhat = model.predict(Xte, mte["hi_now"].to_numpy())
m = regression_metrics(yte, yhat, len(model.columns), hi_now=mte["hi_now"].to_numpy())
with open(os.path.join(HERE, "artifacts", "metriques_finales.json"), encoding="utf-8") as f:
    saved = json.load(f)
print(f"  artefact recharge : {len(model.columns)} colonnes, alpha={model.alpha}, seuil={model.seuil}")
print(f"  accuracy recalculee = {m['acc_tol']:.6f}   (enregistree {saved['test']['acc_tol']:.6f})")
print(f"  R² recalcule        = {m['r2']:.6f}   (enregistre {saved['test']['r2']:.6f})")
ok = abs(m["acc_tol"] - saved["test"]["acc_tol"]) < 1e-9 and abs(m["r2"] - saved["test"]["r2"]) < 1e-9
print(f"  -> reproduction {'EXACTE' if ok else 'DIVERGENTE'}")
print(f"\n  detail des 5 premieres lignes :")
print(model.predict_detail(Xte.head(5), mte["hi_now"].head(5).to_numpy()).to_string(index=False))

# ------------------------------------------------- livrable 1 : iterations
log = pd.read_csv(os.path.join(HERE, "results_log.csv"))
LEVIER = {
    -1: "Reference", 0: "Baseline", 1: "1. Qualite des donnees", 2: "9. Hyperparametres",
    3: "9. Hyperparametres", 4: "Formulation", 5: "Formulation",
    6: "Perte + retrecissement", 7: "Perte + retrecissement", 8: "Perte + retrecissement",
    9: "Perte + retrecissement", 24: "Correction selective", 25: "Correction selective",
    26: "2. Valeurs manquantes", 27: "5. Mise a l'echelle", 28: "6. Selection de features",
    29: "9. Hyperparametres", 30: "9. Hyperparametres", 31: "8. Ensembles",
    32: "FINAL (combinaison)",
}
for i in range(10, 24):
    LEVIER[i] = "8. Familles de modeles"
log["levier"] = log["iteration"].map(LEVIER).fillna("-")


def verdict(r):
    if r["iteration"] == -1:
        return "garde-fou : a battre"
    g = r["cv_acc_tol_mean"] - 0.7912
    if g > 0.005:
        return "RETENU — gain net en CV"
    if g > 0.0005:
        return "gain marginal en CV"
    if g > -0.005:
        return "neutre en CV"
    return "ECARTE — degrade la CV"


log["verdict"] = log.apply(verdict, axis=1)
cols = ["iteration", "levier", "name", "cv_acc_tol_mean", "cv_acc_tol_std", "cv_r2_mean",
        "test_acc_tol", "test_r2", "test_rmse", "test_mae", "test_skill_vs_persist",
        "gap_acc_tol", "verdict"]
tab = log[cols].sort_values("test_acc_tol", ascending=False)
tab.to_csv(os.path.join(HERE, "livrable1_iterations.csv"), index=False, encoding="utf-8-sig")


def md(df, floatfmt="{:.4f}"):
    h = "| " + " | ".join(df.columns) + " |"
    s = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = []
    for _, r in df.iterrows():
        vals = [floatfmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v)
                else ("" if pd.isna(v) else str(v)) for v in r]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([h, s] + rows)


with open(os.path.join(HERE, "livrable1_iterations.md"), "w", encoding="utf-8") as f:
    f.write("# Livrable 1 — toutes les iterations, triees par accuracy sur le test gele\n\n")
    f.write(f"Tolerance : ±{TOLERANCE}. Selection faite sur `cv_acc_tol_mean` uniquement.\n\n")
    f.write(md(tab.rename(columns={
        "iteration": "it", "name": "configuration", "cv_acc_tol_mean": "CV acc",
        "cv_acc_tol_std": "CV sd", "cv_r2_mean": "CV R2", "test_acc_tol": "TEST acc",
        "test_r2": "TEST R2", "test_rmse": "RMSE", "test_mae": "MAE",
        "test_skill_vs_persist": "skill", "gap_acc_tol": "gap"})))

# ------------------------------------------------- livrable 2 : leaderboard
lb = pd.read_csv(os.path.join(HERE, "leaderboard_familles.csv"))
lb_cols = ["name", "cv_acc_tol_mean", "cv_acc_tol_std", "cv_r2_mean", "test_acc_tol",
           "test_r2", "test_rmse", "test_mae", "test_medae", "test_mape",
           "test_explained_var", "test_max_error", "test_skill_vs_persist",
           "alpha_cv_best", "seconds"]
lb2 = lb[lb_cols].sort_values("test_r2", ascending=False)
lb2.to_csv(os.path.join(HERE, "livrable2_leaderboard.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(HERE, "livrable2_leaderboard.md"), "w", encoding="utf-8") as f:
    f.write("# Livrable 2 — classement des familles de modeles\n\n")
    f.write("Toutes evaluees a alpha=1 (correction brute, sans porte) pour etre comparables "
            "entre elles ; `alpha*` est le facteur de retrecissement optimal en CV pour "
            "chaque famille. Naive Bayes est absent : c'est un classifieur, sans equivalent "
            "de regression applicable a une cible continue.\n\n")
    f.write(md(lb2.rename(columns={
        "name": "famille", "cv_acc_tol_mean": "CV acc", "cv_acc_tol_std": "CV sd",
        "cv_r2_mean": "CV R2", "test_acc_tol": "TEST acc", "test_r2": "TEST R2",
        "test_rmse": "RMSE", "test_mae": "MAE", "test_medae": "MedAE", "test_mape": "MAPE%",
        "test_explained_var": "var.expl", "test_max_error": "err.max",
        "test_skill_vs_persist": "skill", "alpha_cv_best": "alpha*", "seconds": "s"})))

print("\n" + "=" * 100)
print("LIVRABLES GENERES")
print("=" * 100)
for p in ["livrable1_iterations.md", "livrable1_iterations.csv",
          "livrable2_leaderboard.md", "livrable2_leaderboard.csv"]:
    print(f"  {p}")
print("\nTop 8 par accuracy sur le test gele :")
print(tab.head(8)[["iteration", "name", "cv_acc_tol_mean", "test_acc_tol", "test_r2",
                   "test_skill_vs_persist"]].to_string(index=False,
                                                       float_format=lambda v: f"{v:.4f}"))
