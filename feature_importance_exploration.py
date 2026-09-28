"""
Exploration de l'importance des features generees par feature_engineering.py.

Contexte : aucun Health Index/label n'existe encore (etape future du pipeline,
modele autoencoder). En attendant, ce script utilise `contamination_index`
(deja calcule, synthese des 3 codes ISO 4406) comme proxy de degradation pour
classer les 225 features par importance (Random Forest), a titre EXPLORATOIRE
uniquement - ce classement sera a refaire une fois le vrai Health Index defini.

Pour eviter un resultat trivial, sont exclus de l'entrainement :
- contamination_index lui-meme et toutes ses features derivees (roll/lag/ewma/
  diff/slope/zscore de contamination_index)
- ISO 4_filled, ISO 6, ISO 14 brutes (composantes directes de contamination_index)
  et leurs features derivees directes

Entree : isense_oil_data_features.csv
Sortie : rapport_feature_importance.md + eda_output/feature_importance/*.png
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

INPUT_FILE = "isense_oil_data_features.csv"
OUTPUT_DIR = "eda_output/feature_importance"
TARGET = "contamination_index"

EXCLUDE_PREFIXES = ("ISO 4", "ISO 6", "ISO 14", "contamination_index")
EXCLUDE_EXACT_COLS = {"created_at", "session_id"}
TOP_N = 25


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_FILE, parse_dates=["created_at"])

    drop_cols = set(EXCLUDE_EXACT_COLS)
    for col in df.columns:
        if col.startswith(EXCLUDE_PREFIXES):
            drop_cols.add(col)

    feature_cols = [c for c in df.columns if c not in drop_cols and c != TARGET]
    X = df[feature_cols].copy()
    y = df[TARGET]

    valid_rows = y.notna()
    X, y = X.loc[valid_rows], y.loc[valid_rows]

    # bool -> int, puis imputation mediane pour les NaN residuels (debut de session : rolling/lag/ewma)
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
    X = X.fillna(X.median(numeric_only=True))
    X = X.select_dtypes(include=[np.number])

    rf = RandomForestRegressor(n_estimators=400, max_depth=14, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    top = importances.head(TOP_N)

    report = [
        "# Rapport — Importance des features (proxy : `contamination_index`)\n",
        f"- Cible proxy : `{TARGET}` (synthèse ISO 4406, pas encore le Health Index final)",
        f"- Lignes utilisées : {len(X)}",
        f"- Features candidates (après exclusion des composantes directes du proxy) : {X.shape[1]}",
        f"- R² du Random Forest (in-sample, exploratoire) : {rf.score(X, y):.4f}\n",
        "| Rang | Feature | Importance |",
        "|---|---|---|",
    ]
    for rank, (feat, imp) in enumerate(top.items(), 1):
        report.append(f"| {rank} | `{feat}` | {imp:.4f} |")

    with open("rapport_feature_importance.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    fig, ax = plt.subplots(figsize=(9, 9))
    top_sorted = top.sort_values()
    ax.barh(top_sorted.index, top_sorted.values, color="#2b6cb0")
    ax.set_xlabel("Importance (Random Forest)")
    ax.set_title(f"Top {TOP_N} features — proxy contamination_index")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/01_top_features.png", dpi=120)
    plt.close(fig)

    print("\n".join(report))


if __name__ == "__main__":
    main()
