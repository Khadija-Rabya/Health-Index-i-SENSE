"""
Harnais d'evaluation : execute UNE configuration (features + modele) sous le
protocole gele de `protocol.py` et journalise toutes les metriques.

Regle absolue : la selection se fait sur la CV (5 blocs temporels purges) ;
le test gele n'est score qu'une fois, apres coup, et n'influence jamais un
choix. Toute violation de cette regle rendrait le chiffre final inutilisable.
"""

from __future__ import annotations

import json
import os
import time
import warnings

import numpy as np
import pandas as pd

from protocol import (
    ASSET_COLS, HORIZON, N_SPLITS, SEED, TOLERANCE,
    PurgedTimeSeriesSplit, build_xy, prepare, regression_metrics, split_frozen,
)

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_CSV = os.path.join(HERE, "results_log.csv")

_CACHE: dict = {}


def get_data(horizon: int = HORIZON):
    """Prepare une seule fois par process les donnees corrigees des fuites."""
    key = ("data", horizon)
    if key not in _CACHE:
        df, train_mask, cutoffs, meta_fit = prepare(verbose=False)
        X, y, meta = build_xy(df, horizon)
        Xtr, Xte, ytr, yte, mtr, mte = split_frozen(X, y, meta, cutoffs)
        _CACHE[key] = dict(df=df, cutoffs=cutoffs, meta_fit=meta_fit,
                           Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, mtr=mtr, mte=mte)
    return _CACHE[key]


# ------------------------------------------------------------------ evaluation
def evaluate(iteration, name, change, make_model, *, horizon=HORIZON,
             transform=None, delta=True, log=True, verbose=True, notes=""):
    """
    make_model : callable() -> estimateur sklearn (refait a neuf par pli).
    transform  : callable(X_train, X_valid) -> (X_train', X_valid') applique
                 DANS chaque pli (donc ajuste sur le train du pli uniquement).
    delta      : True  -> la cible apprise est y[t+n] - hi_now ; la prediction
                          finale est hi_now + delta_predit.
                 False -> la cible apprise est y[t+n] directement.
    Toutes les metriques sont calculees sur la valeur ABSOLUE reconstruite,
    donc comparables entre les deux formulations.
    """
    d = get_data(horizon)
    Xtr, ytr, mtr = d["Xtr"], d["ytr"], d["mtr"]
    Xte, yte, mte = d["Xte"], d["yte"], d["mte"]
    t0 = time.time()

    # ---------------- CV (selection) ----------------
    cv = PurgedTimeSeriesSplit(n_splits=N_SPLITS)
    fold_rows = []
    for k, (itr, iva) in enumerate(cv.split(Xtr, times=mtr["created_at"]), 1):
        Xa, Xb = Xtr.iloc[itr], Xtr.iloc[iva]
        ya, yb = ytr.iloc[itr], ytr.iloc[iva]
        na, nb = mtr["hi_now"].iloc[itr], mtr["hi_now"].iloc[iva]
        if transform is not None:
            Xa, Xb = transform(Xa, Xb)
        model = make_model()
        model.fit(Xa, (ya - na) if delta else ya)
        pred = model.predict(Xb)
        yhat = (nb + pred) if delta else pred
        fold_rows.append(regression_metrics(yb, yhat, Xa.shape[1], hi_now=nb))
    cvdf = pd.DataFrame(fold_rows)

    # ---------------- test gele (score unique) ----------------
    Xa, Xb = (transform(Xtr, Xte) if transform is not None else (Xtr, Xte))
    model = make_model()
    model.fit(Xa, (ytr - mtr["hi_now"]) if delta else ytr)
    pred = model.predict(Xb)
    yhat_te = (mte["hi_now"] + pred) if delta else pred
    yhat_te = np.clip(np.asarray(yhat_te, dtype=float), 0.0, 1.0)
    test = regression_metrics(yte, yhat_te, Xa.shape[1], hi_now=mte["hi_now"])

    per_asset = {}
    for asset, col in ASSET_COLS.items():
        m = (mte[col] == 1).to_numpy()
        if m.sum() > 10:
            per_asset[asset] = regression_metrics(yte[m], yhat_te[m], Xa.shape[1],
                                                  hi_now=mte["hi_now"][m])

    row = {
        "iteration": iteration, "name": name, "change": change,
        "n_features": int(Xa.shape[1]), "delta_form": delta,
        "cv_acc_tol_mean": cvdf["acc_tol"].mean(), "cv_acc_tol_std": cvdf["acc_tol"].std(),
        "cv_r2_mean": cvdf["r2"].mean(), "cv_r2_std": cvdf["r2"].std(),
        "cv_rmse_mean": cvdf["rmse"].mean(), "cv_mae_mean": cvdf["mae"].mean(),
        "cv_skill_mean": cvdf["skill_vs_persist"].mean(),
        **{f"test_{k}": v for k, v in test.items()},
        "gap_acc_tol": test["acc_tol"] - cvdf["acc_tol"].mean(),
        "gap_r2": test["r2"] - cvdf["r2"].mean(),
        "test_acc_A": per_asset.get("Motosoufflante A", {}).get("acc_tol", np.nan),
        "test_acc_B": per_asset.get("Motosoufflante B", {}).get("acc_tol", np.nan),
        "test_r2_A": per_asset.get("Motosoufflante A", {}).get("r2", np.nan),
        "test_r2_B": per_asset.get("Motosoufflante B", {}).get("r2", np.nan),
        "seconds": round(time.time() - t0, 1), "notes": notes,
    }

    if verbose:
        print(f"\n--- it{iteration:02d} | {name}")
        print(f"    {change}")
        print(f"    CV   acc@±{TOLERANCE} = {row['cv_acc_tol_mean']:.4f} ± {row['cv_acc_tol_std']:.4f}"
              f"   R² = {row['cv_r2_mean']:.4f} ± {row['cv_r2_std']:.4f}"
              f"   skill = {row['cv_skill_mean']:+.4f}")
        print(f"    TEST acc@±{TOLERANCE} = {row['test_acc_tol']:.4f}"
              f"   R² = {row['test_r2']:.4f}   RMSE = {row['test_rmse']:.5f}"
              f"   skill = {row['test_skill_vs_persist']:+.4f}")
        print(f"    gap(test-CV) acc = {row['gap_acc_tol']:+.4f}   R² = {row['gap_r2']:+.4f}"
              f"   [{row['seconds']}s]")
        print(f"    per-machine acc: A={row['test_acc_A']:.4f}  B={row['test_acc_B']:.4f}")

    if log:
        append_log(row)
    return row, model, yhat_te


def persistence_reference(horizon=HORIZON):
    """Reference de persistance (y_pred = health_index a t) sous le meme protocole."""
    d = get_data(horizon)
    mtr, mte, ytr, yte = d["mtr"], d["mte"], d["ytr"], d["yte"]
    cv = PurgedTimeSeriesSplit(n_splits=N_SPLITS)
    folds = [regression_metrics(ytr.iloc[iva], mtr["hi_now"].iloc[iva])
             for _, iva in cv.split(d["Xtr"], times=mtr["created_at"])]
    cvdf = pd.DataFrame(folds)
    test = regression_metrics(yte, mte["hi_now"], hi_now=mte["hi_now"])
    row = {
        "iteration": -1, "name": "Persistance (reference)",
        "change": "y_pred = health_index[t] ; aucun apprentissage",
        "n_features": 1, "delta_form": False,
        "cv_acc_tol_mean": cvdf["acc_tol"].mean(), "cv_acc_tol_std": cvdf["acc_tol"].std(),
        "cv_r2_mean": cvdf["r2"].mean(), "cv_r2_std": cvdf["r2"].std(),
        "cv_rmse_mean": cvdf["rmse"].mean(), "cv_mae_mean": cvdf["mae"].mean(),
        "cv_skill_mean": 0.0,
        **{f"test_{k}": v for k, v in test.items()},
        "gap_acc_tol": test["acc_tol"] - cvdf["acc_tol"].mean(),
        "gap_r2": test["r2"] - cvdf["r2"].mean(),
        "test_acc_A": np.nan, "test_acc_B": np.nan,
        "test_r2_A": np.nan, "test_r2_B": np.nan,
        "seconds": 0.0, "notes": "garde-fou : tout modele doit battre cette ligne",
    }
    for asset, col in ASSET_COLS.items():
        m = (mte[col] == 1).to_numpy()
        row[f"test_acc_{asset[-1]}"] = float(np.mean(np.abs(yte[m] - mte['hi_now'][m]) <= TOLERANCE))
        row[f"test_r2_{asset[-1]}"] = float(regression_metrics(yte[m], mte["hi_now"][m])["r2"])
    return row


def append_log(row: dict):
    df = pd.DataFrame([row])
    if os.path.exists(LOG_CSV):
        old = pd.read_csv(LOG_CSV)
        # une iteration relancee ecrase son ancienne ligne
        old = old[~((old["iteration"] == row["iteration"]) & (old["name"] == row["name"]))]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(LOG_CSV, index=False, encoding="utf-8-sig")


def show_log(sort_by="test_acc_tol", top=None):
    if not os.path.exists(LOG_CSV):
        print("(journal vide)")
        return
    df = pd.read_csv(LOG_CSV).sort_values(sort_by, ascending=False)
    cols = ["iteration", "name", "cv_acc_tol_mean", "cv_acc_tol_std", "cv_r2_mean",
            "test_acc_tol", "test_r2", "test_skill_vs_persist", "gap_acc_tol"]
    out = df[cols].head(top) if top else df[cols]
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
