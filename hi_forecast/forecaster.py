"""Artefact reutilisable : l'integralite du pipeline retenu dans un seul objet.

Contient tout ce qu'un deploiement doit connaitre :
  - les objets ajustes du LABEL (scaler/PCA/IsolationForest/seuils par machine),
    pour recalculer `health_index` a l'instant t sans jamais utiliser le futur ;
  - les statistiques train-only des features reconstruites (z-scores, vi_proxy,
    references de viscosite) ;
  - la liste exacte des colonnes attendues par les modeles ;
  - le regresseur de delta et le classifieur de mouvement ;
  - les parametres de la porte (alpha, seuil) et la tolerance de reference.

Se charge avec `joblib.load(...)` et s'utilise via `.predict(df)`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class HealthIndexForecaster:
    """Prevision de `health_index` a t+horizon par correction selective de la
    persistance :

        prediction = hi[t] + alpha * delta_predit * 1[P(mouvement) > seuil]
    """

    def __init__(self, *, reg, clf, feature_cols, alpha, prob_threshold,
                 tolerance, horizon, label_fit, zscore_stats, visc_refs,
                 vi_proxy_coefs, cutoffs, pca_vars, asset_cols, metrics=None,
                 trained_at=None, notes=""):
        self.reg = reg
        self.clf = clf
        self.feature_cols = list(feature_cols)
        self.alpha = float(alpha)
        self.prob_threshold = float(prob_threshold)
        self.tolerance = float(tolerance)
        self.horizon = int(horizon)
        self.label_fit = label_fit
        self.zscore_stats = zscore_stats
        self.visc_refs = visc_refs
        self.vi_proxy_coefs = vi_proxy_coefs
        self.cutoffs = cutoffs
        self.pca_vars = list(pca_vars)
        self.asset_cols = dict(asset_cols)
        self.metrics = metrics or {}
        self.trained_at = trained_at
        self.notes = notes

    # ------------------------------------------------------------------ API
    def _check(self, X):
        missing = [c for c in self.feature_cols if c not in X.columns]
        if missing:
            raise KeyError(f"{len(missing)} colonnes attendues absentes, "
                           f"p.ex. {missing[:5]}")
        return X[self.feature_cols]

    def predict_delta(self, X: pd.DataFrame) -> np.ndarray:
        """Variation brute predite du health index sur l'horizon (avant porte)."""
        return self.reg.predict(self._check(X))

    def predict_move_proba(self, X: pd.DataFrame) -> np.ndarray:
        """P(|variation| > tolerance) — probabilite que l'indice bouge vraiment."""
        return self.clf.predict_proba(self._check(X))[:, 1]

    def predict(self, X: pd.DataFrame, hi_now=None) -> np.ndarray:
        """Health index prevu a t+horizon.

        `hi_now` : health index a l'instant t. S'il n'est pas fourni, il est lu
        dans la colonne `health_index_lf` de X.
        """
        if hi_now is None:
            if "health_index_lf" not in X.columns:
                raise KeyError("fournir `hi_now` ou une colonne `health_index_lf`")
            hi_now = X["health_index_lf"]
        hi_now = np.asarray(hi_now, dtype=float)
        gate = self.predict_move_proba(X) > self.prob_threshold
        return np.clip(hi_now + self.alpha * self.predict_delta(X) * gate, 0.0, 1.0)

    def explain(self, X: pd.DataFrame) -> pd.DataFrame:
        """Detail ligne a ligne : persistance, delta brut, porte, prevision."""
        hi_now = np.asarray(X["health_index_lf"], dtype=float)
        delta = self.predict_delta(X)
        proba = self.predict_move_proba(X)
        gate = proba > self.prob_threshold
        return pd.DataFrame({
            "hi_now": hi_now, "delta_brut": delta, "p_mouvement": proba,
            "porte_ouverte": gate,
            "delta_applique": self.alpha * delta * gate,
            "prevision": np.clip(hi_now + self.alpha * delta * gate, 0.0, 1.0),
        }, index=X.index)

    def score_label(self, frame: pd.DataFrame, asset: str) -> pd.DataFrame:
        """Recalcule T2 / SPE / IsolationForest / health_index pour une machine,
        avec les objets ajustes sur la seule fenetre d'entrainement."""
        f = self.label_fit[asset]
        vals = frame[[("vi_proxy_lf" if v == "vi_proxy" else v) for v in self.pca_vars]]
        Xs = f["scaler"].transform(vals)
        sc = f["pca"].transform(Xs)
        t2 = np.sum(sc ** 2 / f["pca"].explained_variance_, axis=1)
        spe = np.sum((Xs - f["pca"].inverse_transform(sc)) ** 2, axis=1)
        iso = -f["iso"].score_samples(f["iso_scaler"].transform(vals))
        sev = np.maximum(np.maximum(t2 / f["t2_alarm"], spe / f["spe_alarm"]),
                         iso / f["iso_alarm"])
        return pd.DataFrame({"HI_pca_t2": t2, "HI_pca_spe": spe, "HI_isoforest": iso,
                             "health_index": 1.0 / (1.0 + sev)}, index=frame.index)

    def __repr__(self):
        m = self.metrics.get("test_acc_tol")
        return (f"HealthIndexForecaster(horizon=t+{self.horizon}, "
                f"tolerance=±{self.tolerance}, alpha={self.alpha}, "
                f"seuil={self.prob_threshold}, features={len(self.feature_cols)}"
                + (f", test_acc={m:.4f}" if m is not None else "") + ")")
