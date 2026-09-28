"""
Comparaison de 7 methodes pour remplir les ~2,36 % de NaN restants sur 7 variables
"sante" du dataset i-SENSE, causes par UN SEUL episode d'indisponibilite capteur
synchrone et contigu (478 mesures d'affilee pour Motosoufflante A, 07/2026 ;
516 pour Motosoufflante B, 07/2026 - voir investigation menee en conversation).

Variables concernees (NaN simultanes sur les memes lignes) :
  DC, Dynamic Viscosity, ISO 4, Oil H2O Saturation, Oil H2O ppm,
  Oil Temperature, Viscosity at 40°C

Important : contrairement a Oil System Vibration, CE gap-la laisse `ISO 6`,
`ISO 14`, `Oil Conductivity` et `Oil Pressure` disponibles en continu -> un
modele multivarie peut donc s'appuyer sur ces capteurs encore actifs.

Methodes comparees :
  1. Forward-fill (mathematique, au niveau de la machine)
  2. Interpolation temporelle (mathematique, spline/lineaire, niveau machine)
  3. Moyenne globale
  4. Mediane globale
  5. Moyenne conditionnelle (par machine x etat ON/OFF)
  6. Mediane conditionnelle (par machine x etat ON/OFF)
  7. Random Forest (predicteurs : ISO 6, ISO 14, Oil Conductivity, Oil Pressure,
     Oil System Vibration_filled, heure, machine, etat)

Methodologie : un segment synthetique de meme longueur que le vrai trou est
masque dans une zone par ailleurs complete du dataset (pas de vraie valeur
connue dans le vrai trou -> impossible d'evaluer directement dessus), pour
chaque machine, puis les 7 methodes sont comparees sur ce segment masque.

Sortie :
  - eda_output/sensor_gap_imputation/*.png
  - rapport_imputation_capteurs.md
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

warnings.filterwarnings("ignore")

INPUT_FILE = "isense_oil_data_vibration_filled.csv"
OUTPUT_DIR = "eda_output/sensor_gap_imputation"

TARGETS = [
    "DC", "Dynamic Viscosity", "ISO 4",
    "Oil H2O Saturation", "Oil H2O ppm",
    "Oil Temperature", "Viscosity at 40°C",
]

# Predicteurs qui restent disponibles PENDANT le vrai trou (verifie sur les donnees)
AVAILABLE_PREDICTORS = ["ISO 6", "ISO 14", "Oil Conductivity", "Oil Pressure", "Oil System Vibration_filled"]

# Longueur reelle du trou observe (nb. de mesures consecutives), par machine
REAL_GAP_LENGTH = {"Motosoufflante A": 478, "Motosoufflante B": 516}

RANDOM_STATE = 42

METHOD_COLORS = {
    "1. Forward-fill": "#a0aec0",
    "2. Interpolation": "#718096",
    "3. Moyenne globale": "#63b3ed",
    "4. Médiane globale": "#3182ce",
    "5. Moyenne cond.": "#68d391",
    "6. Médiane cond.": "#38a169",
    "7. Random Forest": "#c05621",
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


def select_synthetic_gap(df, asset):
    """Choisit, dans la plus longue plage sans NaN sur les 7 cibles, un segment
    de la meme longueur que le vrai trou observe pour cette machine."""
    sub = df[df["asset_name"] == asset].sort_values("created_at")
    valid = sub[TARGETS].notna().all(axis=1).values
    idx_array = sub.index.values

    runs = []
    cur = valid[0]
    start = 0
    for i in range(1, len(valid)):
        if valid[i] != cur:
            runs.append((cur, start, i))
            cur = valid[i]
            start = i
    runs.append((cur, start, len(valid)))

    valid_runs = [(s, e) for v, s, e in runs if v]
    longest = max(valid_runs, key=lambda t: t[1] - t[0])

    length = REAL_GAP_LENGTH[asset]
    mid = (longest[0] + longest[1]) // 2
    gap_start = max(longest[0], mid - length // 2)
    gap_end = min(longest[1], gap_start + length)

    return pd.Index(idx_array[gap_start:gap_end])


def forward_fill_method(df, target, mask_idx):
    work = df[["asset_name", "created_at", target]].copy()
    work.loc[mask_idx, target] = np.nan
    work[target] = work.groupby("asset_name")[target].ffill()
    global_median = df.loc[~df.index.isin(mask_idx), target].median()
    work[target] = work[target].fillna(global_median)  # au cas ou le trou commence des la 1ere mesure
    return work.loc[mask_idx, target].values


def interpolate_method(df, target, mask_idx):
    work = df[["asset_name", "created_at", target]].copy()
    work.loc[mask_idx, target] = np.nan

    def interp_asset(group):
        n_valid = group.notna().sum()
        if n_valid >= 4:
            return group.interpolate(method="spline", order=3, limit_direction="both")
        elif n_valid >= 2:
            return group.interpolate(method="linear", limit_direction="both")
        return group

    work[target] = work.groupby("asset_name", group_keys=False)[target].apply(interp_asset)
    global_median = df.loc[~df.index.isin(mask_idx), target].median()
    work[target] = work[target].fillna(global_median)
    return work.loc[mask_idx, target].values


def global_stat_method(df, target, mask_idx, stat):
    train = df.loc[~df.index.isin(mask_idx), target]
    value = train.median() if stat == "median" else train.mean()
    return np.full(len(mask_idx), value)


def conditional_stat_method(df, target, mask_idx, stat):
    train = df.loc[~df.index.isin(mask_idx)]
    agg = "median" if stat == "median" else "mean"
    global_val = train[target].median() if stat == "median" else train[target].mean()
    grouped = train.groupby(["asset_name", "machine_state"])[target].agg(agg)

    def lookup(row):
        return grouped.get((row["asset_name"], row["machine_state"]), global_val)

    return df.loc[mask_idx, ["asset_name", "machine_state"]].apply(lookup, axis=1).values


def build_predictor_matrix(df):
    X = df[AVAILABLE_PREDICTORS].copy()
    X["hour_sin"] = np.sin(2 * np.pi * df["created_at"].dt.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * df["created_at"].dt.hour / 24)
    X = pd.get_dummies(X.join(df[["asset_name", "machine_state"]]), columns=["asset_name", "machine_state"])
    imputer = SimpleImputer(strategy="median")
    return pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)


def random_forest_method(df, X, target, mask_idx):
    train_idx = df.index[df[target].notna() & ~df.index.isin(mask_idx)]
    model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X.loc[train_idx], df.loc[train_idx, target])
    return model.predict(X.loc[mask_idx])


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()
    X = build_predictor_matrix(df)

    mask_idx_by_asset = {asset: select_synthetic_gap(df, asset) for asset in df["asset_name"].unique()}
    all_mask_idx = pd.Index(np.concatenate([idx.values for idx in mask_idx_by_asset.values()]))

    report = ["# Rapport — Comparaison de 7 méthodes pour les 7 variables santé en panne simultanée\n"]
    for asset, idx in mask_idx_by_asset.items():
        report.append(f"- Segment synthétique masqué pour {asset} : {len(idx)} mesures "
                       f"(longueur du vrai trou observé)")
    report.append("")

    per_target_results = {}  # target -> {method: metrics}
    for target in TARGETS:
        y_true = df.loc[all_mask_idx, target].values
        results = {}
        results["1. Forward-fill"] = metrics(y_true, forward_fill_method(df, target, all_mask_idx))
        results["2. Interpolation"] = metrics(y_true, interpolate_method(df, target, all_mask_idx))
        results["3. Moyenne globale"] = metrics(y_true, global_stat_method(df, target, all_mask_idx, "mean"))
        results["4. Médiane globale"] = metrics(y_true, global_stat_method(df, target, all_mask_idx, "median"))
        results["5. Moyenne cond."] = metrics(y_true, conditional_stat_method(df, target, all_mask_idx, "mean"))
        results["6. Médiane cond."] = metrics(y_true, conditional_stat_method(df, target, all_mask_idx, "median"))
        results["7. Random Forest"] = metrics(y_true, random_forest_method(df, X, target, all_mask_idx))
        per_target_results[target] = results

    # --- Tableau detaille par variable ---
    report.append("## Résultats détaillés par variable\n")
    for target, results in per_target_results.items():
        report.append(f"### {target}\n")
        report.append("| Méthode | MAE | RMSE | R² |")
        report.append("|---|---|---|---|")
        for m, r in results.items():
            report.append(f"| {m} | {r['MAE']:.4f} | {r['RMSE']:.4f} | {r['R2']:.4f} |")
        best = min(results, key=lambda m: results[m]["RMSE"])
        report.append(f"\n**Meilleure méthode : {best}**\n")

    # --- Agregation : R2 moyen (comparable entre variables) + RMSE normalisee (RMSE/ecart-type) ---
    methods = list(next(iter(per_target_results.values())).keys())
    avg_r2 = {m: np.mean([per_target_results[t][m]["R2"] for t in TARGETS]) for m in methods}
    std_by_target = {t: df[t].std() for t in TARGETS}
    avg_nrmse = {
        m: np.mean([per_target_results[t][m]["RMSE"] / std_by_target[t] for t in TARGETS])
        for m in methods
    }

    report.append("## Synthèse agrégée (moyenne sur les 7 variables)\n")
    report.append("| Méthode | R² moyen | RMSE normalisée moyenne (RMSE / écart-type) |")
    report.append("|---|---|---|")
    for m in methods:
        report.append(f"| {m} | {avg_r2[m]:.4f} | {avg_nrmse[m]:.4f} |")

    best_overall = max(methods, key=lambda m: avg_r2[m])
    report.append(f"\n**Meilleure méthode globale (R² moyen le plus élevé) : {best_overall}**")

    # --- Graphique agregat ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].bar(methods, [avg_r2[m] for m in methods], color=[METHOD_COLORS[m] for m in methods])
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("R² moyen (7 variables)")
    axes[0].tick_params(axis="x", rotation=35)
    for i, m in enumerate(methods):
        axes[0].text(i, avg_r2[m], f"{avg_r2[m]:.3f}", ha="center",
                     va="bottom" if avg_r2[m] >= 0 else "top", fontsize=8)

    axes[1].bar(methods, [avg_nrmse[m] for m in methods], color=[METHOD_COLORS[m] for m in methods])
    axes[1].set_title("RMSE normalisée moyenne (RMSE / écart-type de la variable)")
    axes[1].tick_params(axis="x", rotation=35)
    for i, m in enumerate(methods):
        axes[1].text(i, avg_nrmse[m], f"{avg_nrmse[m]:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Comparaison agrégée — 7 méthodes, 7 variables santé, trou synchrone simulé")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/01_metrics_aggregate.png", dpi=120)
    plt.close(fig)

    # --- Graphique par variable (R2) ---
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(TARGETS))
    width = 0.11
    for i, m in enumerate(methods):
        values = [per_target_results[t][m]["R2"] for t in TARGETS]
        ax.bar(x + i * width, values, width=width, label=m, color=METHOD_COLORS[m])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x + width * 3)
    ax.set_xticklabels(TARGETS, rotation=20)
    ax.set_ylabel("R²")
    ax.set_title("R² par méthode et par variable")
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/02_r2_by_variable.png", dpi=120)
    plt.close(fig)

    with open("rapport_imputation_capteurs.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    print("\n".join(report[-15:]))
    print(f"\nGraphiques dans {OUTPUT_DIR}/")
    print("Rapport complet : rapport_imputation_capteurs.md")


if __name__ == "__main__":
    main()
