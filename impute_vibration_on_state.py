"""
Comparaison de methodes pour predire `Oil System Vibration` UNIQUEMENT sur les
lignes ou la machine est ON (etat cree a partir de Oil Pressure, cf.
data_quality_isense.py / clean_isense_data.py).

Decision retenue en amont (non testee ici, appliquee directement) :
  - machine_state == OFF & vibration NaN -> remplace par 0 (la machine ne
    tourne pas, donc pas de vibration mecanique du systeme).
  - machine_state == ON  & vibration NaN -> a PREDIRE, car la machine tourne
    et vibre reellement ; c'est l'objet de ce script.

8 methodes comparees sur le sous-ensemble ON uniquement :
  - 1 methode mathematique : interpolation temporelle (spline/lineaire) par session
  - 3 modeles ML : Random Forest, XGBoost, K-Nearest Neighbors
  - 4 methodes statistiques simples : moyenne globale, mediane globale,
    moyenne conditionnelle (par machine), mediane conditionnelle (par machine)

Methodologie : masquage aleatoire de 20% des valeurs connues en etat ON,
comparaison MAE/RMSE/R2, AUCUNE modification du dataset (evaluation seule).

Entree : isense_oil_data_cleaned.csv
Sortie : eda_output/vibration_on_state/*.png + rapport_imputation_vibration_on_state.md
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

INPUT_FILE = "isense_oil_data_cleaned.csv"
OUTPUT_DIR = "eda_output/vibration_on_state"
TARGET = "Oil System Vibration"

PREDICTORS = [
    "DC", "Density", "Dynamic Viscosity", "Kinematic Viscosity",
    "ISO 4", "ISO 6", "ISO 14", "Oil Conductivity",
    "Oil H2O Saturation", "Oil H2O ppm", "Oil Pressure", "Oil Temperature",
    "Viscosity at 40°C",
]  # machine_state exclue : constante (= ON) sur ce sous-ensemble, aucune variance

RANDOM_STATE = 42
TEST_FRACTION = 0.20

METHOD_COLORS = {
    "1. Interpolation": "#718096",
    "2. Random Forest": "#2b6cb0",
    "3. XGBoost": "#dd6b20",
    "4. KNN": "#38a169",
    "5. Moyenne globale": "#a0aec0",
    "6. Médiane globale": "#718096",
    "7. Moyenne cond. (machine)": "#805ad5",
    "8. Médiane cond. (machine)": "#d53f8c",
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


def build_feature_matrix(df):
    X = df[PREDICTORS].copy()
    X["hour_sin"] = np.sin(2 * np.pi * df["created_at"].dt.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * df["created_at"].dt.hour / 24)
    X = pd.get_dummies(X.join(df[["asset_name"]]), columns=["asset_name"])
    imputer = SimpleImputer(strategy="median")
    return pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)


def interpolate_method(df_full, mask_idx):
    """Interpole sur la serie temporelle COMPLETE par session (methode
    mathematique), y compris les points OFF, pour ne pas casser la continuite
    temporelle - seule l'erreur sur les points ON masques est ensuite retenue."""
    work = df_full[["session_id", "created_at", TARGET]].copy()
    work.loc[mask_idx, TARGET] = np.nan

    def interp_session(group):
        n_valid = group.notna().sum()
        if n_valid >= 4:
            return group.interpolate(method="spline", order=3, limit_direction="both")
        elif n_valid >= 2:
            return group.interpolate(method="linear", limit_direction="both")
        return group

    work[TARGET] = work.groupby("session_id", group_keys=False)[TARGET].apply(interp_session)
    global_mean = df_full.loc[~df_full.index.isin(mask_idx), TARGET].mean()
    work[TARGET] = work[TARGET].fillna(global_mean)
    return work.loc[mask_idx, TARGET].values


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_full = load_data()  # dataset complet (ON+OFF), necessaire pour l'interpolation par session
    df_on = df_full[df_full["machine_state"] == "ON"].copy()

    known_idx = df_on.index[df_on[TARGET].notna()]
    train_idx, test_idx = train_test_split(known_idx, test_size=TEST_FRACTION, random_state=RANDOM_STATE)

    report = ["# Rapport — Comparaison de 8 méthodes pour `Oil System Vibration` (état ON uniquement)\n"]
    report.append(f"- Lignes ON avec vibration connue : {len(known_idx)}")
    report.append(f"- Points masqués pour le test (20%) : {len(test_idx)}")
    report.append(f"- Lignes ON avec vibration NaN à prédire (hors test) : "
                   f"{(df_on[TARGET].isna()).sum()}\n")

    y_true = df_on.loc[test_idx, TARGET].values
    results = {}
    predictions = {}

    # 1. Interpolation (mathematique)
    pred_interp = interpolate_method(df_full, test_idx)
    results["1. Interpolation"] = metrics(y_true, pred_interp)
    predictions["1. Interpolation"] = pred_interp

    # Matrice de features (sur le sous-ensemble ON uniquement)
    X_on = build_feature_matrix(df_on)
    X_train, y_train = X_on.loc[train_idx], df_on.loc[train_idx, TARGET]
    X_test = X_on.loc[test_idx]
    scaler = StandardScaler().fit(X_train)

    # 2. Random Forest
    rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    results["2. Random Forest"] = metrics(y_true, pred_rf)
    predictions["2. Random Forest"] = pred_rf

    # 3. XGBoost
    xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                        random_state=RANDOM_STATE, n_jobs=-1)
    xgb.fit(X_train, y_train)
    pred_xgb = xgb.predict(X_test)
    results["3. XGBoost"] = metrics(y_true, pred_xgb)
    predictions["3. XGBoost"] = pred_xgb

    # 4. KNN
    knn = KNeighborsRegressor(n_neighbors=15, weights="distance")
    knn.fit(scaler.transform(X_train), y_train)
    pred_knn = knn.predict(scaler.transform(X_test))
    results["4. KNN"] = metrics(y_true, pred_knn)
    predictions["4. KNN"] = pred_knn

    # 5-6. Moyenne / mediane globale (sur le pool d'entrainement ON)
    train_target = df_on.loc[train_idx, TARGET]
    pred_mean = np.full(len(test_idx), train_target.mean())
    pred_median = np.full(len(test_idx), train_target.median())
    results["5. Moyenne globale"] = metrics(y_true, pred_mean)
    predictions["5. Moyenne globale"] = pred_mean
    results["6. Médiane globale"] = metrics(y_true, pred_median)
    predictions["6. Médiane globale"] = pred_median

    # 7-8. Moyenne / mediane conditionnelle par machine (asset_name)
    train_df = df_on.loc[train_idx, ["asset_name", TARGET]]
    grouped_mean = train_df.groupby("asset_name")[TARGET].mean()
    grouped_median = train_df.groupby("asset_name")[TARGET].median()
    test_assets = df_on.loc[test_idx, "asset_name"]
    pred_cond_mean = test_assets.map(grouped_mean).fillna(train_target.mean()).values
    pred_cond_median = test_assets.map(grouped_median).fillna(train_target.median()).values
    results["7. Moyenne cond. (machine)"] = metrics(y_true, pred_cond_mean)
    predictions["7. Moyenne cond. (machine)"] = pred_cond_mean
    results["8. Médiane cond. (machine)"] = metrics(y_true, pred_cond_median)
    predictions["8. Médiane cond. (machine)"] = pred_cond_median

    # --- Rapport chiffre ---
    report.append("| Méthode | MAE | RMSE | R² |")
    report.append("|---|---|---|---|")
    for m, r in results.items():
        report.append(f"| {m} | {r['MAE']:.4f} | {r['RMSE']:.4f} | {r['R2']:.4f} |")

    best_method = min(results, key=lambda m: results[m]["RMSE"])
    report.append(f"\n**Meilleure méthode (RMSE la plus basse) : {best_method}**")

    # --- Graphique 1 : comparaison des metriques ---
    methods = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, metric_name in zip(axes, ["MAE", "RMSE", "R2"]):
        values = [results[m][metric_name] for m in methods]
        colors = [METHOD_COLORS[m] for m in methods]
        ax.bar(methods, values, color=colors)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(metric_name)
        ax.tick_params(axis="x", rotation=40)
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    fig.suptitle("Comparaison des 8 méthodes — Oil System Vibration, état machine ON uniquement")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/01_metrics_comparison.png", dpi=120)
    plt.close(fig)

    # --- Graphique 2 : predit vs reel (grille) ---
    fig, axes = plt.subplots(3, 3, figsize=(15, 14))
    lims = [0, max(y_true.max(), max(p.max() for p in predictions.values())) * 1.05]
    for ax, (method, pred) in zip(axes.flat, predictions.items()):
        ax.scatter(y_true, pred, s=8, alpha=0.35, color=METHOD_COLORS[method])
        ax.plot(lims, lims, "k--", linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Valeur réelle")
        ax.set_ylabel("Valeur prédite")
        ax.set_title(method, fontsize=9)
    axes.flat[-1].axis("off")
    fig.suptitle("Valeurs prédites vs réelles — 8 méthodes (état ON)")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/02_scatter_grid.png", dpi=120)
    plt.close(fig)

    with open("rapport_imputation_vibration_on_state.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    print("\n".join(report))
    print(f"\nGraphiques dans {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
