"""
Modeles baseline (Random Forest, XGBoost) pour la prediction de health_index
a l'horizon t+n (plan_prediction_health_index.md).

Un modele distinct par horizon et par machine (pas un seul modele avec
l'horizon en entree - plus simple a entrainer/evaluer separement, cf. plan
section 2.2). Decoupage train/test TEMPOREL par machine (health_index_prediction_data.py).

Entree : isense_oil_data_health_index.csv
Sortie : rapport_prediction_health_index_baseline.md
         + eda_output/health_index_prediction/01_r2_vs_horizon.png
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from health_index_prediction_data import (
    ASSET_COLS, HORIZONS, HORIZON_LABELS, build_dataset_for_horizon, load_data,
    temporal_train_test_split,
)

warnings.filterwarnings("ignore")

OUTPUT_DIR = "eda_output/health_index_prediction"
RANDOM_STATE = 42

MODELS = {
    "Random Forest": lambda: RandomForestRegressor(n_estimators=300, max_depth=14, random_state=RANDOM_STATE, n_jobs=-1),
    "XGBoost": lambda: XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE, n_jobs=-1),
}


def metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()

    report = [
        "# Rapport — Prédiction de `health_index` à t+n (baselines Random Forest / XGBoost)\n",
        "**Formulation retenue : prédiction du delta** `health_index[t+n] - health_index[t]`, "
        "puis reconstruction (`health_index[t] + delta_prédit`) — la prédiction directe de la "
        "valeur absolue donnait des résultats nettement moins bons (health_index est fortement "
        "autocorrélé : forcer le modèle à réapprendre cette persistance lui-même ajoute du "
        "bruit). Cf. `explication_prediction_health_index.md`, section 3, pour l'analyse "
        "complète qui a motivé ce choix.\n",
    ]
    report.append("| Horizon | Machine | Modèle | MAE | RMSE | R² | Lignes train | Lignes test |")
    report.append("|---|---|---|---|---|---|---|---|")

    results = []  # (horizon, asset, model_name, mae, rmse, r2)

    for horizon in HORIZONS:
        X, y, meta = build_dataset_for_horizon(df, horizon)
        imputer = SimpleImputer(strategy="median")
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

        for asset, col in ASSET_COLS.items():
            X_train, X_test, y_train, y_test = temporal_train_test_split(X_imputed, y, meta, col)
            if len(X_train) < 50 or len(X_test) < 20:
                continue

            # Formulation en DELTA (y_{t+n} - y_t) plutot qu'en valeur absolue : health_index
            # est fortement autocorrele (persistance seule R2=0,94 a 10 min), donc predire la
            # valeur absolue force le modele a re-apprendre cette persistance lui-meme, ce qui
            # ajoute du bruit. Predire l'ecart isole le signal reellement nouveau a apprendre -
            # verifie experimentalement systematiquement meilleur ou egal a la prediction directe.
            delta_train = y_train - X_train["health_index"]
            for model_name, model_fn in MODELS.items():
                model = model_fn()
                model.fit(X_train, delta_train)
                y_pred = X_test["health_index"] + model.predict(X_test)
                m = metrics(y_test, y_pred)
                results.append((horizon, asset, model_name, m["MAE"], m["RMSE"], m["R2"]))
                report.append(
                    f"| {HORIZON_LABELS[horizon]} | {asset} | {model_name} | {m['MAE']:.4f} | "
                    f"{m['RMSE']:.4f} | {m['R2']:.4f} | {len(X_train)} | {len(X_test)} |"
                )

    # --- Graphique : R2 vs horizon, par machine et modele ---
    results_df = pd.DataFrame(results, columns=["horizon", "asset", "model", "mae", "rmse", "r2"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    colors = {"Random Forest": "#2b6cb0", "XGBoost": "#dd6b20"}
    for ax, asset in zip(axes, ASSET_COLS.keys()):
        sub = results_df[results_df["asset"] == asset]
        for model_name in MODELS:
            m = sub[sub["model"] == model_name].sort_values("horizon")
            ax.plot([HORIZON_LABELS[h] for h in m["horizon"]], m["r2"], marker="o",
                    label=model_name, color=colors[model_name])
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(asset)
        ax.set_xlabel("Horizon")
        ax.legend()
    axes[0].set_ylabel("R² (test, découpage temporel)")
    fig.suptitle("Dégradation de la performance en fonction de l'horizon de prédiction")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/01_r2_vs_horizon.png", dpi=120)
    plt.close(fig)

    with open("rapport_prediction_health_index_baseline.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    results_df.to_csv("rapport_prediction_health_index_baseline_resultats.csv", index=False)

    print("\n".join(report))
    print(f"\nFichiers générés : rapport_prediction_health_index_baseline.md, "
          f"rapport_prediction_health_index_baseline_resultats.csv, {OUTPUT_DIR}/01_r2_vs_horizon.png")


if __name__ == "__main__":
    main()
