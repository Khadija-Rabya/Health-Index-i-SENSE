"""Graphiques de diagnostic exiges par le protocole (regression) :
residus vs valeurs ajustees, distribution des residus, QQ-plot, et vue
d'ensemble des iterations.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
BLUE, ORANGE, RED, GREEN = "#2b6cb0", "#dd6b20", "#e53e3e", "#38a169"


def residual_diagnostics(y_true, y_pred, tolerance, title, fname):
    """Residus vs ajustes + distribution + QQ-plot + erreur absolue triee."""
    os.makedirs(FIGDIR, exist_ok=True)
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    res = y_true - y_pred

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    ax[0, 0].scatter(y_pred, res, s=4, alpha=0.25, color=BLUE, edgecolors="none")
    ax[0, 0].axhline(0, color="black", lw=1)
    ax[0, 0].axhline(tolerance, color=ORANGE, ls="--", lw=1, label=f"±{tolerance} (bande de tolerance)")
    ax[0, 0].axhline(-tolerance, color=ORANGE, ls="--", lw=1)
    ax[0, 0].set_xlabel("Valeur predite"); ax[0, 0].set_ylabel("Residu (vrai - predit)")
    ax[0, 0].set_title("Residus vs valeurs ajustees"); ax[0, 0].legend(fontsize=8)

    ax[0, 1].hist(res, bins=120, color=BLUE, alpha=0.85)
    ax[0, 1].axvline(0, color="black", lw=1)
    for s in (-1, 1):
        ax[0, 1].axvline(s * tolerance, color=ORANGE, ls="--", lw=1)
    ax[0, 1].set_xlabel("Residu"); ax[0, 1].set_ylabel("Effectif")
    ax[0, 1].set_title(f"Distribution des residus (mediane={np.median(res):+.5f}, "
                       f"ecart-type={res.std():.5f})")

    stats.probplot(res, dist="norm", plot=ax[1, 0])
    ax[1, 0].get_lines()[0].set(markersize=2, color=BLUE, alpha=0.4)
    ax[1, 0].get_lines()[1].set(color=RED, lw=1.2)
    ax[1, 0].set_title("QQ-plot des residus (normalite)")

    a = np.sort(np.abs(res))
    frac = np.arange(1, len(a) + 1) / len(a)
    ax[1, 1].plot(a, frac, color=BLUE, lw=1.5)
    ax[1, 1].axvline(tolerance, color=ORANGE, ls="--", lw=1.2,
                     label=f"tolerance ±{tolerance}")
    hit = float(np.mean(np.abs(res) <= tolerance))
    ax[1, 1].axhline(hit, color=GREEN, ls=":", lw=1.2, label=f"accuracy = {hit:.1%}")
    ax[1, 1].set_xscale("log")
    ax[1, 1].set_xlabel("|residu| (echelle log)"); ax[1, 1].set_ylabel("Part cumulee")
    ax[1, 1].set_title("Fonction de repartition de l'erreur absolue")
    ax[1, 1].legend(fontsize=8)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    path = os.path.join(FIGDIR, fname)
    fig.savefig(path, dpi=120); plt.close(fig)
    return path


def iterations_overview(log_csv, tolerance, fname="00_iterations.png"):
    """Vue d'ensemble : CV vs test par iteration, et ecart CV-test."""
    os.makedirs(FIGDIR, exist_ok=True)
    df = pd.read_csv(log_csv)
    ref = df[df["iteration"] == -1]
    it = df[df["iteration"] >= 0].sort_values("cv_acc_tol_mean", ascending=True)
    if it.empty:
        return None

    fig, ax = plt.subplots(1, 2, figsize=(16, max(6, 0.32 * len(it))))
    y = np.arange(len(it))
    ax[0].barh(y - 0.2, it["cv_acc_tol_mean"], height=0.4, color=BLUE,
               xerr=it["cv_acc_tol_std"], error_kw=dict(lw=0.7, ecolor="#555"),
               label="CV (selection)")
    ax[0].barh(y + 0.2, it["test_acc_tol"], height=0.4, color=ORANGE, label="Test gele")
    if len(ref):
        ax[0].axvline(ref["cv_acc_tol_mean"].iloc[0], color=BLUE, ls="--", lw=1.2)
        ax[0].axvline(ref["test_acc_tol"].iloc[0], color=RED, ls="--", lw=1.4,
                      label=f"persistance test = {ref['test_acc_tol'].iloc[0]:.3f}")
    ax[0].axvline(0.80, color=GREEN, ls=":", lw=1.6, label="objectif 80%")
    ax[0].set_yticks(y); ax[0].set_yticklabels(it["name"], fontsize=7)
    ax[0].set_xlabel(f"accuracy @ ±{tolerance}")
    ax[0].set_title("CV vs test gele, par iteration"); ax[0].legend(fontsize=7, loc="lower right")

    ax[1].barh(y, it["gap_acc_tol"], color=[RED if g > 0.05 else GREEN for g in it["gap_acc_tol"]])
    ax[1].axvline(0, color="black", lw=1)
    ax[1].set_yticks(y); ax[1].set_yticklabels([])
    ax[1].set_xlabel("test - CV  (positif = la CV sous-estime)")
    ax[1].set_title("Ecart CV / test")

    fig.tight_layout()
    path = os.path.join(FIGDIR, fname)
    fig.savefig(path, dpi=120); plt.close(fig)
    return path


def shap_summary(model, X, feature_names, fname="03_shap.png", max_display=25, nsample=3000):
    """Resume SHAP du modele gagnant (echantillonne pour le temps de calcul)."""
    import shap
    os.makedirs(FIGDIR, exist_ok=True)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), min(nsample, len(X)), replace=False)
    Xs = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xs)
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, Xs, feature_names=feature_names, max_display=max_display, show=False)
    plt.title("Importance SHAP — modele retenu", fontsize=12)
    plt.tight_layout()
    path = os.path.join(FIGDIR, fname)
    plt.savefig(path, dpi=120); plt.close(fig)
    imp = pd.DataFrame({"feature": feature_names,
                        "mean_abs_shap": np.abs(sv).mean(axis=0)}).sort_values(
        "mean_abs_shap", ascending=False).reset_index(drop=True)
    return path, imp
