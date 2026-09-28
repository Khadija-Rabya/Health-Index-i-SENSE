"""Transformations de features composables, TOUJOURS ajustees sur le pli
d'entrainement uniquement (signature `(X_train, X_valid) -> (X_train', X_valid')`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

# Colonnes structurellement redondantes reperees au diagnostic D4 : versions
# brutes dont la version _filled est deja presente, et doublons exacts.
STRUCTURAL_DUPES = [
    "DC", "Density", "Dynamic Viscosity", "ISO 4", "Kinematic Viscosity",
    "Oil Conductivity", "Oil Conductivity_nSm", "Oil Conductivity_filled",
    "Oil H2O Saturation", "Oil H2O ppm", "Oil Temperature", "Viscosity at 40°C",
    "vibration_available",          # == vibsource_mesuré
    "density_deviation_pct",        # transformation affine de density_15C
    "measure_index_in_session",     # r=0.9998 avec time_in_session_h
    "state_ON",                     # complement exact de state_OFF
    "asset_Motosoufflante B",       # complement exact de asset_Motosoufflante A
]


def chain(*fns):
    """Compose plusieurs transformations dans l'ordre."""
    def _t(Xa, Xb):
        for f in fns:
            Xa, Xb = f(Xa, Xb)
        return Xa, Xb
    return _t


def drop_constant_and_dupes(Xa, Xb):
    """Retire les colonnes constantes sur le pli d'entrainement et les doublons
    structurels identifies au diagnostic."""
    nun = Xa.nunique()
    drop = set(nun[nun <= 1].index) | {c for c in STRUCTURAL_DUPES if c in Xa.columns}
    keep = [c for c in Xa.columns if c not in drop]
    return Xa[keep], Xb[keep]


def prune_correlated(threshold=0.995):
    """Elague les paires trop correlees, calcule sur le pli d'entrainement."""
    def _t(Xa, Xb):
        med = Xa.median()
        c = Xa.fillna(med).corr().abs().to_numpy(copy=True)
        np.fill_diagonal(c, 0.0)
        cols, drop = list(Xa.columns), set()
        for i in range(len(cols)):
            if cols[i] in drop:
                continue
            for j in range(i + 1, len(cols)):
                if cols[j] not in drop and c[i, j] > threshold:
                    drop.add(cols[j])
        keep = [c_ for c_ in cols if c_ not in drop]
        return Xa[keep], Xb[keep]
    return _t


def add_regime_features(Xa, Xb):
    """Features metier issues du diagnostic D5 :

    - la machine est a l'arret (OFF) 50% du temps ; a l'arret l'huile ne circule
      pas, le health index bouge a peine (|delta| median 0.00037 contre 0.00439
      en marche) -> le regime est le facteur explicatif numero 1 ;
    - en OFF sur la Motosoufflante B le viscosimetre renvoie 3-6 cSt pour une
      huile ISO VG 46, valeur physiquement impossible : on la marque au lieu de
      la laisser polluer le modele silencieusement.
    """
    def _f(X):
        X = X.copy()
        off = X["state_OFF"] if "state_OFF" in X else pd.Series(0, index=X.index)
        X["visc_implausible"] = (X["Viscosity at 40°C_filled"] < 20).astype(np.int8)
        X["off_x_visc_implausible"] = (off * X["visc_implausible"]).astype(np.int8)
        for c in ("Oil Temperature_filled", "Oil H2O ppm_filled",
                  "Oil System Vibration_filled", "contamination_index"):
            if c in X:
                X[f"off_x_{c}"] = off * X[c]
        if "time_in_session_h" in X:
            X["log_time_in_session"] = np.log1p(X["time_in_session_h"].clip(lower=0))
            X["off_x_time"] = off * X["log_time_in_session"]
        return X
    return _f(Xa), _f(Xb)


def add_dynamics_features(Xa, Xb):
    """Vitesse/acceleration recente du health index et des principaux indicateurs :
    ce qui bouge maintenant est le meilleur predicteur de ce qui bougera a t+3h."""
    def _f(X):
        X = X.copy()
        hi = X.get("health_index_lf_proxy")
        for base in ("Oil Temperature_filled", "Oil H2O ppm_filled",
                     "Viscosity at 40°C_filled", "ISO 4_filled",
                     "Oil System Vibration_filled", "density_15C"):
            d1, ew = f"{base}_diff1", f"{base}_ewma"
            if d1 in X and ew in X and base in X:
                X[f"{base}_dev_ewma"] = X[base] - X[ew]           # ecart a la tendance
            if f"{base}_lag3" in X and base in X:
                X[f"{base}_accel"] = X[base] - 2 * X.get(f"{base}_lag1", X[base]) + X[f"{base}_lag3"]
        if "HI_pca_t2_lf" in X and "HI_pca_spe_lf" in X:
            X["pca_t2_log"] = np.log1p(X["HI_pca_t2_lf"].clip(lower=0))
            X["pca_spe_log"] = np.log1p(X["HI_pca_spe_lf"].clip(lower=0))
            X["pca_ratio"] = X["HI_pca_t2_lf"] / (X["HI_pca_spe_lf"] + 1e-6)
        return X
    return _f(Xa), _f(Xb)


def add_hi_history(hi_col="hi_hist"):
    """Placeholder : l'historique du health index est injecte en amont par
    `harness` via les colonnes hi_lag*/hi_slope construites dans protocol."""
    def _t(Xa, Xb):
        return Xa, Xb
    return _t


class ShrunkRegressor(BaseEstimator, RegressorMixin):
    """Retrecit la prediction vers 0 d'un facteur `alpha`.

    Sur la formulation en delta, prediction finale = hi_now + alpha * delta_predit.
    alpha = 0 redonne exactement la persistance, alpha = 1 le modele brut. Sert
    de garde-fou contre les plis catastrophiques observes au diagnostic D1/D6,
    ou le modele produit un biais systematique lors d'un changement de regime.
    """

    def __init__(self, base=None, alpha=0.5):
        self.base, self.alpha = base, alpha

    def fit(self, X, y):
        self.base_ = clone(self.base).fit(X, y)
        return self

    def predict(self, X):
        return self.alpha * self.base_.predict(X)


class ClippedRegressor(BaseEstimator, RegressorMixin):
    """Borne la prediction de delta aux quantiles observes en entrainement :
    empeche une extrapolation aberrante hors du domaine appris."""

    def __init__(self, base=None, q=0.999):
        self.base, self.q = base, q

    def fit(self, X, y):
        self.base_ = clone(self.base).fit(X, y)
        self.lo_, self.hi_ = np.quantile(y, [1 - self.q, self.q])
        return self

    def predict(self, X):
        return np.clip(self.base_.predict(X), self.lo_, self.hi_)
