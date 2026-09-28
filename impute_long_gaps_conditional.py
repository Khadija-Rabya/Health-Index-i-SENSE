"""
Evalue la mediane/moyenne CONDITIONNELLE comme methode de remplissage pour les
30,9 % de `Oil System Vibration` volontairement laisses `NaN` par fill_vibration.py
(trous >= 20 mesures du point connu le plus proche, ou aucune des 4 methodes
comparees dans impute_vibration.py ne faisait mieux qu'une prediction proche de
la moyenne globale - R2 negatif pour toutes).

Idee : si le probleme sur les trous longs est que le modele ne peut de toute
facon pas mieux faire que "predire une valeur moyenne", alors autant utiliser
directement une moyenne/mediane CONDITIONNEE par le regime de fonctionnement
(machine, etat ON/OFF) plutot qu'un modele ML complexe qui n'apporte rien de
plus dans ce regime.

Reutilise EXACTEMENT la meme selection de segments "trous longs" que
impute_vibration.py (meme random_state=42, meme min_run=20) pour une
comparaison directe avec les resultats deja obtenus (HistGradBoost, Random
Forest, etc. - voir rapport_imputation_vibration.md).

Sortie :
  - eda_output/vibration_imputation/06_metrics_conditional_long_gaps.png
  - rapport_imputation_conditionnelle.md
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

INPUT_FILE = "isense_oil_data_cleaned.csv"
OUTPUT_DIR = "eda_output/vibration_imputation"
TARGET = "Oil System Vibration"
RANDOM_STATE = 42
MIN_RUN = 20

PREDICTORS = [
    "DC", "Dynamic Viscosity", "ISO 4", "ISO 6", "ISO 14",
    "Oil Conductivity", "Oil H2O Saturation", "Oil H2O ppm",
    "Oil Pressure", "Oil Temperature", "Viscosity at 40°C",
]

METHOD_COLORS = {
    "Médiane globale": "#a0aec0",
    "Moyenne globale": "#718096",
    "Médiane cond. (machine)": "#68d391",
    "Moyenne cond. (machine)": "#38a169",
    "Médiane cond. (machine+état)": "#f6ad55",
    "Moyenne cond. (machine+état)": "#c05621",
    "HistGradBoost (référence)": "#3182ce",
}


def load_data():
    df = pd.read_csv(INPUT_FILE, parse_dates=["created_at"])
    return df.sort_values(["asset_name", "created_at"]).reset_index(drop=True)


def metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def select_long_gap_segments(df, min_run=MIN_RUN):
    """Reproduit exactement la selection de impute_vibration.run_long_gap_stress_test."""
    work = df.copy()
    work["_valid"] = work[TARGET].notna()
    work["_run_id"] = (work["_valid"] != work.groupby("session_id")["_valid"].shift()).cumsum()

    runs = (
        work[work["_valid"]]
        .groupby(["session_id", "_run_id"])
        .size()
        .reset_index(name="length")
    )
    long_runs = runs[runs["length"] >= min_run]
    chosen = long_runs.sample(min(15, len(long_runs)), random_state=RANDOM_STATE)

    mask_idx = []
    gap_distance = {}
    for _, row in chosen.iterrows():
        seg = work[(work["session_id"] == row["session_id"]) & (work["_run_id"] == row["_run_id"])]
        seg_idx = seg.index.tolist()
        middle = seg_idx[1:-1]
        mask_idx.extend(middle)
        for pos, idx in enumerate(middle, start=1):
            gap_distance[idx] = min(pos, len(middle) - pos + 1)

    mask_idx = pd.Index(mask_idx)
    distances = np.array([gap_distance[i] for i in mask_idx])
    return mask_idx, distances


def predict_conditional(df, mask_idx, group_cols, stat):
    train = df.drop(index=mask_idx)
    agg = "median" if stat == "median" else "mean"
    global_val = train[TARGET].median() if stat == "median" else train[TARGET].mean()
    grouped = train.groupby(group_cols)[TARGET].agg(agg)

    def lookup(row):
        key = tuple(row[c] for c in group_cols) if len(group_cols) > 1 else row[group_cols[0]]
        return grouped.get(key, global_val)

    return df.loc[mask_idx, group_cols].apply(lookup, axis=1).values


def predict_histgb_reference(df, mask_idx):
    """Reference : la methode ML la plus robuste sur trous longs (voir impute_vibration.py)."""
    X = df[PREDICTORS].copy()
    X["hour_sin"] = np.sin(2 * np.pi * df["created_at"].dt.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * df["created_at"].dt.hour / 24)
    X = pd.get_dummies(X.join(df[["asset_name", "machine_state"]]), columns=["asset_name", "machine_state"])
    X = X.fillna(X.median())

    train_idx = df.index[df[TARGET].notna() & ~df.index.isin(mask_idx)]
    model = HistGradientBoostingRegressor(max_depth=8, random_state=RANDOM_STATE)
    model.fit(X.loc[train_idx], df.loc[train_idx, TARGET])
    return model.predict(X.loc[mask_idx])


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()

    mask_idx, distances = select_long_gap_segments(df)
    y_true = df.loc[mask_idx, TARGET].values

    train = df.drop(index=mask_idx)
    results = {}
    results["Médiane globale"] = metrics(y_true, np.full(len(y_true), train[TARGET].median()))
    results["Moyenne globale"] = metrics(y_true, np.full(len(y_true), train[TARGET].mean()))
    results["Médiane cond. (machine)"] = metrics(y_true, predict_conditional(df, mask_idx, ["asset_name"], "median"))
    results["Moyenne cond. (machine)"] = metrics(y_true, predict_conditional(df, mask_idx, ["asset_name"], "mean"))
    results["Médiane cond. (machine+état)"] = metrics(
        y_true, predict_conditional(df, mask_idx, ["asset_name", "machine_state"], "median")
    )
    results["Moyenne cond. (machine+état)"] = metrics(
        y_true, predict_conditional(df, mask_idx, ["asset_name", "machine_state"], "mean")
    )
    results["HistGradBoost (référence)"] = metrics(y_true, predict_histgb_reference(df, mask_idx))

    # --- Graphique ---
    methods = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, metric_name in zip(axes, ["MAE", "RMSE", "R2"]):
        values = [results[m][metric_name] for m in methods]
        colors = [METHOD_COLORS[m] for m in methods]
        ax.bar(methods, values, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(metric_name)
        ax.tick_params(axis="x", rotation=35)
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    fig.suptitle("Médiane/moyenne conditionnelle vs référence ML — test trous longs (mêmes segments masqués)")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/06_metrics_conditional_long_gaps.png", dpi=120)
    plt.close(fig)

    # --- Rapport ---
    report = ["# Rapport — Médiane/moyenne conditionnelle pour les trous longs de `Oil System Vibration`\n"]
    report.append(f"- Points masqués (mêmes segments que le test « trous longs » de impute_vibration.py) : {len(y_true)}\n")
    report.append("| Méthode | MAE | RMSE | R² |")
    report.append("|---|---|---|---|")
    for m, r in results.items():
        report.append(f"| {m} | {r['MAE']:.4f} | {r['RMSE']:.4f} | {r['R2']:.4f} |")

    best = min(results, key=lambda m: results[m]["RMSE"])
    report.append(f"\n**Meilleure méthode (RMSE la plus basse) : {best}**")

    with open("rapport_imputation_conditionnelle.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    print("\n".join(report))
    print(f"\nGraphique : {OUTPUT_DIR}/06_metrics_conditional_long_gaps.png")


if __name__ == "__main__":
    main()
