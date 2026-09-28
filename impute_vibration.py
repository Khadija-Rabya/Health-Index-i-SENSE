"""
Comparaison de 4 methodes pour imputer les valeurs manquantes de
`Oil System Vibration` (65.9 % de NaN) dans le dataset i-SENSE nettoye.

Methodes comparees :
  1. Mathematique   : interpolation temporelle (spline cubique) par session
  2. ML - Modele 1  : Random Forest Regressor
  3. ML - Modele 2  : HistGradientBoosting Regressor
  4. ML - Modele 3  : K-Nearest Neighbors Regressor

Comme la vraie valeur des points manquants est inconnue, l'evaluation se fait
en masquant artificiellement une partie des points CONNUS, en appliquant
chaque methode, puis en comparant aux vraies valeurs (MAE / RMSE / R2).

Un second test ("longs trous") masque des segments entiers pour verifier la
robustesse de chaque methode quand l'ecart temporel au point connu le plus
proche est grand (l'analyse des runs a montre des trous NaN allant jusqu'a
~2000 points consecutifs, largement au-dela de ce qu'une interpolation locale
peut raisonnablement combler).

Sortie :
  - eda_output/vibration_imputation/*.png (graphiques comparatifs)
  - rapport_imputation_vibration.md (metriques + conclusion)
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

INPUT_FILE = "isense_oil_data_cleaned.csv"
OUTPUT_DIR = "eda_output/vibration_imputation"
TARGET = "Oil System Vibration"

PREDICTORS = [
    "DC", "Dynamic Viscosity", "ISO 4", "ISO 6", "ISO 14",
    "Oil Conductivity", "Oil H2O Saturation", "Oil H2O ppm",
    "Oil Pressure", "Oil Temperature", "Viscosity at 40°C",
]  # Density exclue : redondante avec Oil Temperature (r=-1.00, cf. investigation_correlations.md)

RANDOM_STATE = 42
TEST_FRACTION = 0.20

METHOD_COLORS = {
    "1. Interpolation": "#718096",
    "2. Random Forest": "#2b6cb0",
    "3. HistGradBoost": "#dd6b20",
    "4. KNN": "#38a169",
}


def load_data():
    df = pd.read_csv(INPUT_FILE, parse_dates=["created_at"])
    df = df.sort_values(["asset_name", "created_at"]).reset_index(drop=True)
    return df


def build_feature_matrix(df):
    X = df[PREDICTORS].copy()
    X["hour_sin"] = np.sin(2 * np.pi * df["created_at"].dt.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * df["created_at"].dt.hour / 24)
    X = pd.get_dummies(X.join(df[["asset_name", "machine_state"]]), columns=["asset_name", "machine_state"])
    # petites proportions de NaN residuelles (sentinelles) -> imputation simple par mediane
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    return X_imputed


def metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Methode 1 : interpolation temporelle par session
# ---------------------------------------------------------------------------
def interpolate_method(df, mask_idx):
    """Masque mask_idx dans une copie de la serie, interpole par session (spline),
    et renvoie les valeurs predites a ces index."""
    work = df[["session_id", "created_at", TARGET]].copy()
    work.loc[mask_idx, TARGET] = np.nan

    def interp_session(group):
        n_valid = group.notna().sum()
        if n_valid >= 4:
            return group.interpolate(method="spline", order=3, limit_direction="both")
        elif n_valid >= 2:
            return group.interpolate(method="linear", limit_direction="both")
        else:
            return group  # pas assez de points -> reste NaN (fallback plus bas)

    work[TARGET] = work.groupby("session_id", group_keys=False)[TARGET].apply(interp_session)
    # fallback pour sessions sans assez de points : moyenne globale d'entrainement
    global_mean = df.loc[~df.index.isin(mask_idx), TARGET].mean()
    work[TARGET] = work[TARGET].fillna(global_mean)
    return work.loc[mask_idx, TARGET]


# ---------------------------------------------------------------------------
# Evaluation principale : masquage aleatoire de 20% des points connus
# ---------------------------------------------------------------------------
def run_random_masking_evaluation(df, X):
    known_idx = df.index[df[TARGET].notna()]
    train_idx, test_idx = train_test_split(
        known_idx, test_size=TEST_FRACTION, random_state=RANDOM_STATE
    )

    y_true = df.loc[test_idx, TARGET].values
    results = {}
    predictions = {}

    # 1. Interpolation
    pred_interp = interpolate_method(df, test_idx)
    results["1. Interpolation"] = metrics(y_true, pred_interp.values)
    predictions["1. Interpolation"] = pred_interp.values

    # Jeu ML : entrainement sur train_idx uniquement
    X_train, y_train = X.loc[train_idx], df.loc[train_idx, TARGET]
    X_test = X.loc[test_idx]

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 2. Random Forest
    rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    results["2. Random Forest"] = metrics(y_true, pred_rf)
    predictions["2. Random Forest"] = pred_rf

    # 3. HistGradientBoosting
    hgb = HistGradientBoostingRegressor(max_depth=8, random_state=RANDOM_STATE)
    hgb.fit(X_train, y_train)
    pred_hgb = hgb.predict(X_test)
    results["3. HistGradBoost"] = metrics(y_true, pred_hgb)
    predictions["3. HistGradBoost"] = pred_hgb

    # 4. KNN (sur donnees standardisees)
    knn = KNeighborsRegressor(n_neighbors=15, weights="distance")
    knn.fit(X_train_s, y_train)
    pred_knn = knn.predict(X_test_s)
    results["4. KNN"] = metrics(y_true, pred_knn)
    predictions["4. KNN"] = pred_knn

    return results, predictions, y_true, test_idx


# ---------------------------------------------------------------------------
# Test de robustesse : masquage de segments longs (simule les grands trous)
# ---------------------------------------------------------------------------
def run_long_gap_stress_test(df, X, min_run=20):
    df = df.copy()
    df["_valid"] = df[TARGET].notna()
    df["_run_id"] = (df["_valid"] != df.groupby("session_id")["_valid"].shift()).cumsum()

    runs = (
        df[df["_valid"]]
        .groupby(["session_id", "_run_id"])
        .size()
        .reset_index(name="length")
    )
    long_runs = runs[runs["length"] >= min_run]

    if long_runs.empty:
        return None

    rng = np.random.RandomState(RANDOM_STATE)
    chosen = long_runs.sample(min(15, len(long_runs)), random_state=RANDOM_STATE)

    mask_idx = []
    gap_distance = {}
    for _, row in chosen.iterrows():
        seg = df[(df["session_id"] == row["session_id"]) & (df["_run_id"] == row["_run_id"])]
        seg_idx = seg.index.tolist()
        # garde le premier et dernier point connus, masque tout le milieu
        middle = seg_idx[1:-1]
        mask_idx.extend(middle)
        for pos, idx in enumerate(middle, start=1):
            dist_to_edge = min(pos, len(middle) - pos + 1)
            gap_distance[idx] = dist_to_edge

    mask_idx = pd.Index(mask_idx)
    y_true = df.loc[mask_idx, TARGET].values
    distances = np.array([gap_distance[i] for i in mask_idx])

    results = {}
    predictions = {}

    pred_interp = interpolate_method(df, mask_idx)
    results["1. Interpolation"] = metrics(y_true, pred_interp.values)
    predictions["1. Interpolation"] = pred_interp.values

    train_idx = df.index[df[TARGET].notna() & ~df.index.isin(mask_idx)]
    X_train, y_train = X.loc[train_idx], df.loc[train_idx, TARGET]
    X_test = X.loc[mask_idx]

    rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    results["2. Random Forest"] = metrics(y_true, pred_rf)
    predictions["2. Random Forest"] = pred_rf

    hgb = HistGradientBoostingRegressor(max_depth=8, random_state=RANDOM_STATE)
    hgb.fit(X_train, y_train)
    pred_hgb = hgb.predict(X_test)
    results["3. HistGradBoost"] = metrics(y_true, pred_hgb)
    predictions["3. HistGradBoost"] = pred_hgb

    scaler = StandardScaler().fit(X_train)
    knn = KNeighborsRegressor(n_neighbors=15, weights="distance")
    knn.fit(scaler.transform(X_train), y_train)
    pred_knn = knn.predict(scaler.transform(X_test))
    results["4. KNN"] = metrics(y_true, pred_knn)
    predictions["4. KNN"] = pred_knn

    return results, predictions, y_true, distances


# ---------------------------------------------------------------------------
# Graphiques
# ---------------------------------------------------------------------------
def plot_metrics_bar(results, filename, title):
    methods = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric_name in zip(axes, ["MAE", "RMSE", "R2"]):
        values = [results[m][metric_name] for m in methods]
        colors = [METHOD_COLORS[m] for m in methods]
        ax.bar(methods, values, color=colors)
        ax.set_title(metric_name)
        ax.tick_params(axis="x", rotation=30)
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/{filename}", dpi=120)
    plt.close(fig)


def plot_scatter_grid(predictions, y_true, filename, title):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    lims = [min(y_true.min(), 0), y_true.max() * 1.05]
    for ax, (method, pred) in zip(axes.flat, predictions.items()):
        ax.scatter(y_true, pred, s=8, alpha=0.4, color=METHOD_COLORS[method])
        ax.plot(lims, lims, "k--", linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Valeur réelle")
        ax.set_ylabel("Valeur prédite")
        ax.set_title(method)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/{filename}", dpi=120)
    plt.close(fig)


def plot_gap_distance_vs_error(predictions, y_true, distances, filename):
    bins = np.array([0, 2, 4, 6, 9, 13, 20, distances.max() + 1])
    bin_labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins) - 1)]
    bin_idx = np.digitize(distances, bins) - 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Gauche : nuage de points brut (scatter, pas de lignes trompeuses)
    ax = axes[0]
    for method, pred in predictions.items():
        abs_error = np.abs(pred - y_true)
        jitter = np.random.RandomState(0).uniform(-0.15, 0.15, size=len(distances))
        ax.scatter(distances + jitter, abs_error, s=10, alpha=0.35,
                   label=method, color=METHOD_COLORS[method])
    ax.set_xlabel("Distance au point connu le plus proche (nb. de mesures)")
    ax.set_ylabel("Erreur absolue")
    ax.set_title("Erreurs individuelles (nuage de points)")
    ax.legend(fontsize=7)

    # Droite : erreur moyenne par palier de distance (lecture claire de la tendance)
    ax = axes[1]
    x_pos = np.arange(len(bin_labels))
    width = 0.15
    for i, (method, pred) in enumerate(predictions.items()):
        abs_error = np.abs(pred - y_true)
        means = [abs_error[bin_idx == b].mean() if (bin_idx == b).any() else np.nan
                 for b in range(len(bin_labels))]
        ax.bar(x_pos + i * width, means, width=width, label=method, color=METHOD_COLORS[method])
    ax.set_xticks(x_pos + width * 2)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("Distance au point connu le plus proche (palier)")
    ax.set_ylabel("Erreur absolue moyenne")
    ax.set_title("Erreur moyenne par palier de distance")
    ax.legend(fontsize=7)

    fig.suptitle("Dégradation de l'erreur selon la distance au point connu (test longs trous)")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/{filename}", dpi=120)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()
    X = build_feature_matrix(df)

    report = ["# Rapport — Comparaison de 4 méthodes d'imputation pour `Oil System Vibration`\n"]

    # --- Evaluation principale ---
    results_rand, preds_rand, y_true_rand, test_idx = run_random_masking_evaluation(df, X)
    plot_metrics_bar(results_rand, "01_metrics_random_masking.png",
                      "Comparaison des 4 méthodes — masquage aléatoire (20% des points connus)")
    plot_scatter_grid(preds_rand, y_true_rand, "02_scatter_random_masking.png",
                       "Valeurs prédites vs réelles — masquage aléatoire")

    report.append("## 1. Évaluation principale (masquage aléatoire de 20% des points connus)\n")
    report.append(f"- Points connus disponibles : {df[TARGET].notna().sum()}")
    report.append(f"- Points masqués pour le test : {len(test_idx)}\n")
    report.append("| Méthode | MAE | RMSE | R² |")
    report.append("|---|---|---|---|")
    for m, r in results_rand.items():
        report.append(f"| {m} | {r['MAE']:.4f} | {r['RMSE']:.4f} | {r['R2']:.4f} |")

    best_random = min(results_rand, key=lambda m: results_rand[m]["RMSE"])
    report.append(f"\n**Meilleure méthode (masquage aléatoire, RMSE la plus basse) : {best_random}**")

    # --- Test de robustesse (longs trous) ---
    stress = run_long_gap_stress_test(df, X)
    if stress is not None:
        results_long, preds_long, y_true_long, distances = stress
        plot_metrics_bar(results_long, "03_metrics_long_gaps.png",
                          "Comparaison des 4 méthodes — segments masqués en entier (trous longs)")
        plot_scatter_grid(preds_long, y_true_long, "04_scatter_long_gaps.png",
                           "Valeurs prédites vs réelles — trous longs")
        plot_gap_distance_vs_error(preds_long, y_true_long, distances, "05_gap_distance_vs_error.png")

        report.append("\n## 2. Test de robustesse (segments longs masqués en entier)\n")
        report.append(
            "Simule les vrais trous du dataset (jusqu'à ~2000 points consécutifs) en masquant "
            "le milieu de segments de mesures connues d'au moins 20 points."
        )
        report.append(f"- Points masqués : {len(y_true_long)}\n")
        report.append("| Méthode | MAE | RMSE | R² |")
        report.append("|---|---|---|---|")
        for m, r in results_long.items():
            report.append(f"| {m} | {r['MAE']:.4f} | {r['RMSE']:.4f} | {r['R2']:.4f} |")

        best_long = min(results_long, key=lambda m: results_long[m]["RMSE"])
        report.append(f"\n**Meilleure méthode (trous longs, RMSE la plus basse) : {best_long}**")
    else:
        report.append("\n## 2. Test de robustesse : pas assez de segments longs disponibles pour ce test.")

    report.append("\n## 3. Graphiques générés\n")
    report.append("- `01_metrics_random_masking.png` — MAE/RMSE/R² par méthode (masquage aléatoire)")
    report.append("- `02_scatter_random_masking.png` — prédit vs réel, 4 méthodes")
    report.append("- `03_metrics_long_gaps.png` — MAE/RMSE/R² par méthode (trous longs)")
    report.append("- `04_scatter_long_gaps.png` — prédit vs réel, trous longs")
    report.append("- `05_gap_distance_vs_error.png` — dégradation de l'erreur selon la distance")

    with open("rapport_imputation_vibration.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    print("\n".join(report))
    print(f"\nGraphiques dans {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
