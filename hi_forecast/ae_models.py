"""Trois architectures d'autoencodeur (PyTorch, CPU), utilisables dans les DEUX roles :

  ROLE A — prevision de health_index a t+18 : le latent et l'erreur de
           reconstruction deviennent des variables du regresseur d'ecart.
  ROLE B — reconstruction de l'etiquette : l'erreur de reconstruction sert de
           mesure de severite, en remplacement de PCA T2/SPE + Isolation Forest.

Toutes les classes exposent la meme interface :
    fit(X)                 -> ajuste (mise a l'echelle comprise, sur X seulement)
    encode(X)              -> vecteur latent
    reconstruction_error(X)-> erreur quadratique moyenne par ligne
Les graines sont fixees a chaque construction pour la reproductibilite.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

SEED = 42
DEVICE = torch.device("cpu")


def _seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


class _Base:
    """Boucle d'entrainement commune : Adam, arret anticipe sur une fraction de
    validation interne decoupee CHRONOLOGIQUEMENT (jamais au hasard : ces donnees
    sont temporelles)."""

    def _train_loop(self, model, tensors, loss_fn, epochs, batch, lr, patience=8,
                    val_frac=0.15, verbose=False):
        n = tensors[0].shape[0]
        cut = int(n * (1 - val_frac))
        tr = [t[:cut] for t in tensors]
        va = [t[cut:] for t in tensors]
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        best, best_state, bad = np.inf, None, 0
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(cut)
            for i in range(0, cut, batch):
                idx = perm[i:i + batch]
                opt.zero_grad()
                loss = loss_fn(model, [t[idx] for t in tr])
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = float(loss_fn(model, va))
            if vl < best - 1e-6:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        self.val_loss_ = best
        self.epochs_ran_ = ep + 1
        return model

    # --- mise a l'echelle interne : ajustee sur le X passe a fit() uniquement ---
    def _fit_scaler(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.mu_ = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
        sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0
        self.sd_ = sd
        return self._apply_scaler(X)

    def _apply_scaler(self, X):
        X = np.asarray(X, dtype=np.float32)
        Z = (X - self.mu_) / self.sd_
        return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# ============================================================== AE-1 : dense
class DenseAE(_Base):
    """Autoencodeur tabulaire dense. `noise` > 0 donne la variante debruiteuse
    (denoising) : l'entree est bruitee, la cible reste propre — adapte a un jeu
    de capteurs bruites."""

    def __init__(self, latent=8, hidden=64, noise=0.0, epochs=120, batch=256,
                 lr=1e-3, seed=SEED):
        self.latent, self.hidden, self.noise = latent, hidden, noise
        self.epochs, self.batch, self.lr, self.seed = epochs, batch, lr, seed

    def fit(self, X):
        _seed(self.seed)
        Z = self._fit_scaler(X)
        p = Z.shape[1]
        self.enc = nn.Sequential(nn.Linear(p, self.hidden), nn.ReLU(),
                                 nn.Linear(self.hidden, self.latent))
        self.dec = nn.Sequential(nn.Linear(self.latent, self.hidden), nn.ReLU(),
                                 nn.Linear(self.hidden, p))
        model = nn.ModuleDict({"enc": self.enc, "dec": self.dec})
        T = torch.from_numpy(Z)

        def loss_fn(m, ts):
            x = ts[0]
            xin = x + self.noise * torch.randn_like(x) if self.noise > 0 else x
            return nn.functional.mse_loss(m["dec"](m["enc"](xin)), x)

        self._train_loop(model, [T], loss_fn, self.epochs, self.batch, self.lr)
        return self

    @torch.no_grad()
    def encode(self, X):
        self.enc.eval()
        return self.enc(torch.from_numpy(self._apply_scaler(X))).numpy()

    @torch.no_grad()
    def reconstruction_error(self, X):
        self.enc.eval(); self.dec.eval()
        Z = torch.from_numpy(self._apply_scaler(X))
        return ((self.dec(self.enc(Z)) - Z) ** 2).mean(dim=1).numpy()


# ================================================================ AE-1b : VAE
class DenseVAE(_Base):
    """Autoencodeur variationnel dense (utilise comme variante d'AE-3)."""

    def __init__(self, latent=8, hidden=64, beta=1e-3, epochs=120, batch=256,
                 lr=1e-3, seed=SEED):
        self.latent, self.hidden, self.beta = latent, hidden, beta
        self.epochs, self.batch, self.lr, self.seed = epochs, batch, lr, seed

    def fit(self, X):
        _seed(self.seed)
        Z = self._fit_scaler(X)
        p = Z.shape[1]
        self.body = nn.Sequential(nn.Linear(p, self.hidden), nn.ReLU())
        self.mu = nn.Linear(self.hidden, self.latent)
        self.lv = nn.Linear(self.hidden, self.latent)
        self.dec = nn.Sequential(nn.Linear(self.latent, self.hidden), nn.ReLU(),
                                 nn.Linear(self.hidden, p))
        model = nn.ModuleDict({"body": self.body, "mu": self.mu, "lv": self.lv,
                               "dec": self.dec})
        T = torch.from_numpy(Z)

        def loss_fn(m, ts):
            x = ts[0]
            h = m["body"](x)
            mu, lv = m["mu"](h), m["lv"](h).clamp(-8, 8)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
            rec = nn.functional.mse_loss(m["dec"](z), x)
            kl = -0.5 * torch.mean(1 + lv - mu ** 2 - lv.exp())
            return rec + self.beta * kl

        self._train_loop(model, [T], loss_fn, self.epochs, self.batch, self.lr)
        return self

    @torch.no_grad()
    def encode(self, X):
        for m in (self.body, self.mu):
            m.eval()
        return self.mu(self.body(torch.from_numpy(self._apply_scaler(X)))).numpy()

    @torch.no_grad()
    def reconstruction_error(self, X):
        for m in (self.body, self.mu, self.dec):
            m.eval()
        Z = torch.from_numpy(self._apply_scaler(X))
        rec = self.dec(self.mu(self.body(Z)))
        return ((rec - Z) ** 2).mean(dim=1).numpy()


# ========================================================== AE-2 : sequentiel
class SeqAE(_Base):
    """Autoencodeur sequentiel GRU/LSTM sur une fenetre de W pas passes.

    L'encodeur comprime la fenetre en un vecteur ; le decodeur la reconstruit.
    `forecast=True` ajoute une tete de prevision directe de l'ecart (seq2seq),
    entrainee conjointement a la reconstruction."""

    def __init__(self, latent=16, hidden=32, cell="gru", forecast=False,
                 fc_weight=1.0, epochs=60, batch=256, lr=2e-3, seed=SEED):
        self.latent, self.hidden, self.cell = latent, hidden, cell
        self.forecast, self.fc_weight = forecast, fc_weight
        self.epochs, self.batch, self.lr, self.seed = epochs, batch, lr, seed

    def _scale3(self, S, fit=False):
        n, w, p = S.shape
        flat = S.reshape(-1, p)
        Z = self._fit_scaler(flat) if fit else self._apply_scaler(flat)
        return Z.reshape(n, w, p)

    def fit(self, S, y_delta=None):
        _seed(self.seed)
        Z = self._scale3(np.asarray(S, dtype=np.float32), fit=True)
        n, w, p = Z.shape
        RNN = nn.GRU if self.cell == "gru" else nn.LSTM
        self.enc = RNN(p, self.hidden, batch_first=True)
        self.to_lat = nn.Linear(self.hidden, self.latent)
        self.from_lat = nn.Linear(self.latent, self.hidden)
        self.dec = RNN(self.hidden, self.hidden, batch_first=True)
        self.out = nn.Linear(self.hidden, p)
        mods = {"enc": self.enc, "to_lat": self.to_lat, "from_lat": self.from_lat,
                "dec": self.dec, "out": self.out}
        if self.forecast:
            self.head = nn.Sequential(nn.Linear(self.latent, 32), nn.ReLU(),
                                      nn.Linear(32, 1))
            mods["head"] = self.head
        model = nn.ModuleDict(mods)

        T = torch.from_numpy(Z)
        tensors = [T]
        if self.forecast:
            tensors.append(torch.from_numpy(
                np.asarray(y_delta, dtype=np.float32).reshape(-1, 1)))

        def loss_fn(m, ts):
            x = ts[0]
            h = m["enc"](x)[0][:, -1, :]
            lat = m["to_lat"](h)
            rep = m["from_lat"](lat).unsqueeze(1).repeat(1, x.shape[1], 1)
            rec = m["out"](m["dec"](rep)[0])
            loss = nn.functional.mse_loss(rec, x)
            if self.forecast:
                loss = loss + self.fc_weight * nn.functional.mse_loss(m["head"](lat), ts[1])
            return loss

        self._train_loop(model, tensors, loss_fn, self.epochs, self.batch, self.lr)
        return self

    @torch.no_grad()
    def _latent(self, S):
        for m in (self.enc, self.to_lat):
            m.eval()
        Z = torch.from_numpy(self._scale3(np.asarray(S, dtype=np.float32)))
        return self.to_lat(self.enc(Z)[0][:, -1, :])

    def encode(self, S):
        return self._latent(S).numpy()

    @torch.no_grad()
    def reconstruction_error(self, S):
        for m in (self.enc, self.to_lat, self.from_lat, self.dec, self.out):
            m.eval()
        Z = torch.from_numpy(self._scale3(np.asarray(S, dtype=np.float32)))
        h = self.enc(Z)[0][:, -1, :]
        lat = self.to_lat(h)
        rep = self.from_lat(lat).unsqueeze(1).repeat(1, Z.shape[1], 1)
        rec = self.out(self.dec(rep)[0])
        return ((rec - Z) ** 2).mean(dim=(1, 2)).numpy()

    @torch.no_grad()
    def predict_delta(self, S):
        if not self.forecast:
            raise RuntimeError("SeqAE construit sans tete de prevision")
        self.head.eval()
        return self.head(self._latent(S)).numpy().ravel()


# ====================================================== AE-3 : convolutif 1D
class ConvAE(_Base):
    """Autoencodeur temporel a convolutions 1D sur une fenetre de W pas.
    `forecast=True` ajoute une tete de prevision directe de l'ecart."""

    def __init__(self, latent=16, ch=32, forecast=False, fc_weight=1.0,
                 epochs=60, batch=256, lr=2e-3, seed=SEED):
        self.latent, self.ch = latent, ch
        self.forecast, self.fc_weight = forecast, fc_weight
        self.epochs, self.batch, self.lr, self.seed = epochs, batch, lr, seed

    def _scale3(self, S, fit=False):
        n, w, p = S.shape
        flat = S.reshape(-1, p)
        Z = self._fit_scaler(flat) if fit else self._apply_scaler(flat)
        return Z.reshape(n, w, p)

    def fit(self, S, y_delta=None):
        _seed(self.seed)
        Z = self._scale3(np.asarray(S, dtype=np.float32), fit=True)
        n, w, p = Z.shape
        self.w_, self.p_ = w, p
        self.enc = nn.Sequential(
            nn.Conv1d(p, self.ch, 3, padding=1), nn.ReLU(),
            nn.Conv1d(self.ch, self.ch, 3, padding=1, stride=2), nn.ReLU(),
            nn.Flatten(), nn.Linear(self.ch * ((w + 1) // 2), self.latent))
        self.dec = nn.Sequential(
            nn.Linear(self.latent, self.ch * ((w + 1) // 2)), nn.ReLU(),
            nn.Unflatten(1, (self.ch, (w + 1) // 2)),
            nn.Upsample(size=w, mode="nearest"),
            nn.Conv1d(self.ch, self.ch, 3, padding=1), nn.ReLU(),
            nn.Conv1d(self.ch, p, 3, padding=1))
        mods = {"enc": self.enc, "dec": self.dec}
        if self.forecast:
            self.head = nn.Sequential(nn.Linear(self.latent, 32), nn.ReLU(),
                                      nn.Linear(32, 1))
            mods["head"] = self.head
        model = nn.ModuleDict(mods)

        T = torch.from_numpy(Z).transpose(1, 2)          # (n, p, w)
        tensors = [T]
        if self.forecast:
            tensors.append(torch.from_numpy(
                np.asarray(y_delta, dtype=np.float32).reshape(-1, 1)))

        def loss_fn(m, ts):
            x = ts[0]
            lat = m["enc"](x)
            loss = nn.functional.mse_loss(m["dec"](lat), x)
            if self.forecast:
                loss = loss + self.fc_weight * nn.functional.mse_loss(m["head"](lat), ts[1])
            return loss

        self._train_loop(model, tensors, loss_fn, self.epochs, self.batch, self.lr)
        return self

    @torch.no_grad()
    def _latent(self, S):
        self.enc.eval()
        Z = torch.from_numpy(self._scale3(np.asarray(S, dtype=np.float32))).transpose(1, 2)
        return self.enc(Z)

    def encode(self, S):
        return self._latent(S).numpy()

    @torch.no_grad()
    def reconstruction_error(self, S):
        self.enc.eval(); self.dec.eval()
        Z = torch.from_numpy(self._scale3(np.asarray(S, dtype=np.float32))).transpose(1, 2)
        return ((self.dec(self.enc(Z)) - Z) ** 2).mean(dim=(1, 2)).numpy()

    @torch.no_grad()
    def predict_delta(self, S):
        if not self.forecast:
            raise RuntimeError("ConvAE construit sans tete de prevision")
        self.head.eval()
        return self.head(self._latent(S)).numpy().ravel()


# ==================================================== fenetrage par session
def build_windows(df_idx, session_ids, X, W):
    """Construit les fenetres de longueur W SANS jamais franchir une frontiere de
    session. Renvoie (S, keep) ou S a la forme (m, W, p) et `keep` donne les
    positions (dans X) des instants t retenus.

    Aucun remplissage : une fenetre incomplete est ecartee."""
    X = np.asarray(X, dtype=np.float32)
    sid = np.asarray(session_ids)
    keep, wins = [], []
    start = 0
    for i in range(1, len(sid) + 1):
        if i == len(sid) or sid[i] != sid[start]:
            L = i - start
            if L >= W:
                for t in range(start + W - 1, i):
                    wins.append(X[t - W + 1:t + 1])
                    keep.append(t)
            start = i
    if not wins:
        return np.empty((0, W, X.shape[1]), dtype=np.float32), np.array([], dtype=int)
    return np.stack(wins).astype(np.float32), np.asarray(keep, dtype=int)
