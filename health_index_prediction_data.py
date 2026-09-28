"""
Module partage : construction du jeu de donnees supervise pour la prediction
de `health_index` a l'horizon t+n (plan_prediction_health_index.md).

Pour chaque ligne t d'une session, la cible est health_index[t+n] (NaN si
l'horizon depasse la fin de la session - ces lignes sont retirees). Toutes
les 147 colonnes de isense_oil_data_health_index.csv sont des predicteurs
valides au temps t (aucune fuite : elles precedent strictement la cible),
sauf les identifiants et les doublons categoriels en chaine de caracteres
deja encodes numeriquement ailleurs (cf. EXCLUDE_COLS).

Horizons retenus (couverture calculee sur les 352 sessions reelles) :
  t+1 (~10 min, 99,2% couverture) ... t+1008 (~1 semaine, 40,0% couverture)
Au-dela, la couverture continue de baisser progressivement (21,8% a ~20 jours)
mais la prediction a cet horizon devient peu fiable (aucune information sur
des evenements futurs non observables dans l'etat courant) - N_max retenu
= 1008 pas (~1 semaine).

Decoupage train/test : TEMPOREL, par machine (les sessions les plus anciennes
de chaque machine servent a l'entrainement, les plus recentes au test) - un
decoupage aleatoire fuiterait de l'information entre lignes consecutives
tres correlees de la meme session.
"""

import numpy as np
import pandas as pd

INPUT_FILE = "isense_oil_data_health_index.csv"

HORIZONS = [1, 6, 18, 144, 432, 1008]  # ~10min, 1h, 3h, 24h, 3j, 1 semaine
HORIZON_LABELS = {1: "10 min", 6: "1h", 18: "3h", 144: "24h", 432: "3j", 1008: "1 semaine"}

TARGET = "health_index"
ASSET_COLS = {"Motosoufflante A": "asset_Motosoufflante A", "Motosoufflante B": "asset_Motosoufflante B"}

# Colonnes non-predictives (identifiants) ou doublons categoriels deja encodes
# numeriquement ailleurs dans le dataset (vibration_confidence_score, vibsource_*,
# sensor_gap_filled) - cf. feature_engineering.py.
EXCLUDE_COLS = {
    "created_at", "session_id",
    "vibration_confidence", "sensor_gap_source",
}

STATE_COLS = ["health_state_regles", "health_state_pca", "health_state_isoforest", "health_state"]


def load_data():
    df = pd.read_csv(INPUT_FILE, parse_dates=["created_at"])
    df = df.sort_values(["session_id", "created_at"]).reset_index(drop=True)
    return df


def compute_horizon_coverage(df, horizons=HORIZONS):
    """Retourne, pour chaque horizon, le nombre et le % de lignes ayant une
    cible valide (horizon ne depassant pas la fin de sa session)."""
    n = len(df)
    coverage = {}
    countdown = df.groupby("session_id").cumcount(ascending=False)
    for h in horizons:
        valid = int((countdown >= h).sum())
        coverage[h] = (valid, valid / n)
    return coverage


def build_feature_columns(df):
    # health_index (TARGET) AU TEMPS t reste un predicteur valide et informatif (autocorrelation) -
    # seule sa version DECALEE (y, construite a part par build_dataset_for_horizon) est la cible.
    cols = [c for c in df.columns if c not in EXCLUDE_COLS and c not in STATE_COLS]
    X = df[cols].copy()
    # Encodage one-hot des etats categoriels (Normal/Surveillance/Alarme) - predicteurs
    # contemporains valides, pas une fuite (calcules a partir de l'etat au temps t).
    X = pd.get_dummies(X.join(df[STATE_COLS]), columns=STATE_COLS)
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
    return X


def build_dataset_for_horizon(df, horizon):
    """Construit X (features au temps t) et y (health_index au temps t+horizon),
    par session, en retirant les lignes ou la cible depasse la fin de la session."""
    df = df.sort_values(["session_id", "created_at"]).reset_index(drop=True)
    y_shifted = df.groupby("session_id")[TARGET].shift(-horizon)

    X = build_feature_columns(df)
    valid = y_shifted.notna()

    meta = df.loc[valid, ["created_at", "session_id"] + list(ASSET_COLS.values())]
    return X.loc[valid].reset_index(drop=True), y_shifted.loc[valid].reset_index(drop=True), meta.reset_index(drop=True)


def temporal_train_test_split(X, y, meta, asset_col, test_fraction=0.2):
    """Decoupage temporel PAR MACHINE : les (1-test_fraction) premieres lignes
    (par ordre chronologique) de cette machine vont a l'entrainement, le reste
    au test - jamais de melange aleatoire entre lignes."""
    mask = meta[asset_col] == 1
    idx = meta.loc[mask].sort_values("created_at").index
    cut = int(len(idx) * (1 - test_fraction))
    train_idx, test_idx = idx[:cut], idx[cut:]
    return X.loc[train_idx], X.loc[test_idx], y.loc[train_idx], y.loc[test_idx]


if __name__ == "__main__":
    df = load_data()
    coverage = compute_horizon_coverage(df)
    print(f"Lignes totales : {len(df)}\n")
    print("| Horizon | Pas | Lignes valides | Couverture |")
    print("|---|---|---|---|")
    for h in HORIZONS:
        n_valid, pct = coverage[h]
        print(f"| {HORIZON_LABELS[h]} | {h} | {n_valid} | {pct:.1%} |")
