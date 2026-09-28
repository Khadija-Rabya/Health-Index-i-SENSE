"""
3 modeles a base d'autoencodeur pour la prediction de health_index a t+n
(plan_prediction_health_index.md, section 1.2) :

  A. Autoencodeur (compression) + regresseur (Random Forest) sur l'espace latent
  B. LSTM Encoder-Decoder (seq2seq) - Malhotra et al. (2016)
  C. Autoencodeur debruiteur (denoising) + tete de prediction, fine-tune

Testes sur 3 horizons representatifs (court/moyen/long : 10 min, 3h, 24h) plutot
que les 6 du plan, pour rester dans un temps de calcul raisonnable - cf. note
dans le rapport. Meme formulation en DELTA que les baselines (RF/XGBoost,
health_index_prediction_baseline.py), meme decoupage temporel par machine.

Necessite PyTorch (installe dans un venv a chemin court, C:\\pfe_venv, TensorFlow
et l'installation globale de PyTorch echouant sur ce poste a cause de la limite
de longueur de chemin Windows sur l'installation Python du Windows Store).

Entree : isense_oil_data_health_index.csv
Sortie : rapport_prediction_health_index_autoencoder.md
         + eda_output/health_index_prediction/02_autoencoder_comparison.png
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from health_index_prediction_data import (
    ASSET_COLS, build_dataset_for_horizon, load_data, temporal_train_test_split,
)

warnings.filterwarnings("ignore")
torch.manual_seed(42)

OUTPUT_DIR = "eda_output/health_index_prediction"
HORIZONS = [1, 18, 144]
HORIZON_LABELS = {1: "10 min", 18: "3h", 144: "24h"}
SEQ_LEN = 12  # fenetre passee pour le modele B (LSTM), ~2h de contexte
EPOCHS = 60
BATCH_SIZE = 256
DEVICE = "cpu"


def metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


GRAD_CLIP_NORM = 1.0  # cf. instabilite numerique du modele C (R2 divergent jusqu'a -24000
                       # avant correction) - clipping standard pour stabiliser l'entrainement


def train_torch_model(model, X_train, y_train, epochs=EPOCHS, lr=1e-3):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
    n = len(X_t)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            out = model(X_t[idx])
            loss = loss_fn(out, y_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()
    return model


# ===========================================================================
# MODELE A : Autoencodeur (compression) + Random Forest sur l'espace latent
# ===========================================================================

class SimpleAutoencoder(nn.Module):
    def __init__(self, n_features, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 32), nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, n_features),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)


def model_a_autoencoder_rf(X_train, X_test, y_train_delta, y_train_base_col_idx=None):
    ae = SimpleAutoencoder(X_train.shape[1])
    ae.to(DEVICE)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    X_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    n = len(X_t)
    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            out = ae(X_t[idx])
            loss = loss_fn(out, X_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), GRAD_CLIP_NORM)
            opt.step()

    with torch.no_grad():
        latent_train = ae.encode(torch.tensor(X_train, dtype=torch.float32)).numpy()
        latent_test = ae.encode(torch.tensor(X_test, dtype=torch.float32)).numpy()

    rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(latent_train, y_train_delta)
    return rf.predict(latent_test)


# ===========================================================================
# MODELE B : LSTM Encoder-Decoder (seq2seq)
# ===========================================================================

class LSTMEncoderDecoder(nn.Module):
    def __init__(self, n_features, hidden_dim=32):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        return self.head(h[-1])


def build_sequences(X_values, session_ids, seq_len=SEQ_LEN):
    """Construit des sequences (seq_len pas passes) par session.

    X_values : ndarray (n, n_features), DEJA triees chronologiquement et
    regroupees par session de facon contigue (garanti par temporal_train_test_split
    + build_dataset_for_horizon, qui trient par session_id/created_at en amont).
    session_ids : array (n,) aligne PAR POSITION avec X_values (pas par label
    d'index pandas - evite toute confusion entre position dans le tableau et
    label d'index, source d'un bug corrige ici).
    """
    n, n_feat = X_values.shape
    sequences = np.zeros((n, seq_len, n_feat), dtype=np.float32)
    positions_by_session = pd.DataFrame(
        {"pos": np.arange(n), "session_id": np.asarray(session_ids)}
    ).groupby("session_id", sort=False)["pos"]

    for _, pos_group in positions_by_session:
        pos_array = pos_group.values  # deja dans l'ordre chronologique (cf. docstring)
        for local_i, global_pos in enumerate(pos_array):
            start = max(0, local_i - seq_len + 1)
            window = X_values[pos_array[start:local_i + 1]]
            sequences[global_pos, -len(window):] = window
    return sequences


def model_b_lstm_seq2seq(X_train_seq, X_test_seq, y_train_delta):
    model = LSTMEncoderDecoder(X_train_seq.shape[2])
    train_torch_model(model, X_train_seq, y_train_delta)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_test_seq, dtype=torch.float32)).numpy().ravel()
    return pred


# ===========================================================================
# MODELE C : Autoencodeur debruiteur + tete de prediction (fine-tuning)
# ===========================================================================

class DenoisingAutoencoderPredictor(nn.Module):
    def __init__(self, n_features, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 32), nn.ReLU(),
            nn.Linear(32, latent_dim), nn.ReLU(),
            nn.LayerNorm(latent_dim),  # borne l'echelle du latent, independamment de la
                                       # dynamique du pre-entrainement - cf. instabilite
                                       # persistante de la tete de prediction sans cette
                                       # normalisation (R2 jusqu'a -252 malgre gel + clipping)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, n_features),
        )
        self.predictor_head = nn.Sequential(nn.Linear(latent_dim, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        z = self.encoder(x)
        return self.predictor_head(z)


def model_c_denoising_finetune(X_train, X_test, y_train_delta, noise_std=0.1):
    model = DenoisingAutoencoderPredictor(X_train.shape[1])
    model.to(DEVICE)

    # Etape 1 : pre-entrainement non supervise (reconstruction a partir d'une version bruitee)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    X_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    n = len(X_t)
    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            noisy = X_t[idx] + noise_std * torch.randn_like(X_t[idx])
            opt.zero_grad()
            recon = model.decoder(model.encoder(noisy))
            loss = loss_fn(recon, X_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()

    # Etape 2 : fine-tuning supervise de la SEULE tete de prediction, encodeur/decodeur GELES.
    # Une premiere version fine-tunait l'encodeur en meme temps que la tete (memes donnees,
    # meme taux d'apprentissage) : le signal de gradient de la reconstruction (etape 1) et celui
    # de la regression du delta (etape 2) sont de nature trop differente pour etre optimises
    # conjointement sans diverger (R2 jusqu'a -24000 observe) - geler l'encodeur une fois pre-
    # entraine (transfer learning classique) resout l'instabilite.
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = False

    opt2 = torch.optim.Adam(model.predictor_head.parameters(), lr=1e-3)
    y_t = torch.tensor(y_train_delta, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt2.zero_grad()
            out = model(X_t[idx])
            loss = loss_fn(out, y_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.predictor_head.parameters(), GRAD_CLIP_NORM)
            opt2.step()

    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_test, dtype=torch.float32)).numpy().ravel()
    return pred


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()

    report = [
        "# Rapport — Prédiction de `health_index` à t+n (3 modèles à base d'autoencodeur)\n",
        f"Testés sur {len(HORIZONS)} horizons représentatifs (court/moyen/long) plutôt que les 6 "
        "du plan complet, pour un temps de calcul raisonnable. Même formulation en delta et même "
        "découpage temporel par machine que les baselines "
        "(`health_index_prediction_baseline.py`).\n",
        "| Horizon | Machine | Modèle | MAE | RMSE | R² |",
        "|---|---|---|---|---|---|",
    ]

    results = []

    for horizon in HORIZONS:
        X, y, meta = build_dataset_for_horizon(df, horizon)
        imputer = SimpleImputer(strategy="median")
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

        for asset, col in ASSET_COLS.items():
            X_train, X_test, y_train, y_test = temporal_train_test_split(X_imputed, y, meta, col)
            if len(X_train) < 200 or len(X_test) < 50:
                continue

            scaler = StandardScaler().fit(X_train)
            X_train_s = scaler.transform(X_train)
            X_test_s = scaler.transform(X_test)
            delta_train = (y_train - X_train["health_index"]).values.astype(np.float32)
            base_test = X_test["health_index"].values

            # --- Modele A ---
            pred_delta = model_a_autoencoder_rf(X_train_s, X_test_s, delta_train)
            y_pred = base_test + pred_delta
            m = metrics(y_test, y_pred)
            results.append((horizon, asset, "A. Autoencodeur+RF (latent)", m["MAE"], m["RMSE"], m["R2"]))
            report.append(f"| {HORIZON_LABELS[horizon]} | {asset} | A. Autoencodeur+RF (latent) | "
                           f"{m['MAE']:.4f} | {m['RMSE']:.4f} | {m['R2']:.4f} |")

            # --- Modele B (sequences) ---
            # meta.loc[X_train.index] preserve l'ordre de X_train.index (deja chronologique,
            # cf. temporal_train_test_split) -> alignement par POSITION valide avec X_train_s.
            session_ids_train = meta.loc[X_train.index, "session_id"].values
            session_ids_test = meta.loc[X_test.index, "session_id"].values
            X_train_seq = build_sequences(X_train_s, session_ids_train)
            X_test_seq = build_sequences(X_test_s, session_ids_test)
            pred_delta = model_b_lstm_seq2seq(X_train_seq, X_test_seq, delta_train)
            y_pred = base_test + pred_delta
            m = metrics(y_test, y_pred)
            results.append((horizon, asset, "B. LSTM Encoder-Decoder", m["MAE"], m["RMSE"], m["R2"]))
            report.append(f"| {HORIZON_LABELS[horizon]} | {asset} | B. LSTM Encoder-Decoder | "
                           f"{m['MAE']:.4f} | {m['RMSE']:.4f} | {m['R2']:.4f} |")

            # --- Modele C ---
            pred_delta = model_c_denoising_finetune(X_train_s, X_test_s, delta_train)
            y_pred = base_test + pred_delta
            m = metrics(y_test, y_pred)
            results.append((horizon, asset, "C. Débruiteur + fine-tuning", m["MAE"], m["RMSE"], m["R2"]))
            report.append(f"| {HORIZON_LABELS[horizon]} | {asset} | C. Débruiteur + fine-tuning | "
                           f"{m['MAE']:.4f} | {m['RMSE']:.4f} | {m['R2']:.4f} |")

    results_df = pd.DataFrame(results, columns=["horizon", "asset", "model", "mae", "rmse", "r2"])
    colors = {
        "A. Autoencodeur+RF (latent)": "#2b6cb0", "B. LSTM Encoder-Decoder": "#805ad5",
        "C. Débruiteur + fine-tuning": "#dd6b20",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, asset in zip(axes, ASSET_COLS.keys()):
        sub = results_df[results_df["asset"] == asset]
        for model_name, color in colors.items():
            m = sub[sub["model"] == model_name].sort_values("horizon")
            ax.plot([HORIZON_LABELS[h] for h in m["horizon"]], m["r2"], marker="o",
                    label=model_name, color=color)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(asset)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("R² (test, découpage temporel)")
    fig.suptitle("Modèles à base d'autoencodeur — R² par horizon")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/02_autoencoder_comparison.png", dpi=120)
    plt.close(fig)

    with open("rapport_prediction_health_index_autoencoder.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    results_df.to_csv("rapport_prediction_health_index_autoencoder_resultats.csv", index=False)

    print("\n".join(report))
    print(f"\nFichiers générés : rapport_prediction_health_index_autoencoder.md, "
          f"rapport_prediction_health_index_autoencoder_resultats.csv, {OUTPUT_DIR}/02_autoencoder_comparison.png")


if __name__ == "__main__":
    main()
