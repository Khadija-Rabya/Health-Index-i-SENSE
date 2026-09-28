"""
Prediction de health_index a t+n, pour chacun des 6 horizons benchmarques
(health_index_prediction_data.HORIZONS), a partir d'un snapshot temps reel deja
entierement enrichi par realtime/feature_pipeline.build_live_snapshot.

Pour chaque (horizon, machine), utilise EXACTEMENT le modele qui a gagne dans
rapport_prediction_health_index_comparaison_finale.md (lu depuis manifest.json,
cf. realtime/artifacts.py) :
  - "random_forest" -> health_index[t+n] = health_index[t] + RF.predict(delta) (formulation
    en ecart, cf. explication_prediction_health_index.md section 3 - la prediction directe
    de la valeur absolue est nettement moins bonne, health_index etant fortement autocorrele) ;
  - "persistence"    -> health_index[t+n] = health_index[t] (aucun modele ne bat cette
    reference a cet horizon pour cette machine).
"""

import json
import os
import sys
from datetime import timedelta

import joblib
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
sys.path.insert(0, PROJECT_DIR)

import health_index_prediction_data as hpd  # noqa: E402

ASSET_COLS = {"Motosoufflante A": "asset_Motosoufflante A", "Motosoufflante B": "asset_Motosoufflante B"}

# ~10 min/pas (frequence nominale des mesures, cf. plan_prediction_health_index.md)
STEP_MINUTES = 10

_rf_model_cache = {}


def _load_rf_bundle(artifact_name):
    if artifact_name not in _rf_model_cache:
        _rf_model_cache[artifact_name] = joblib.load(os.path.join(ARTIFACTS_DIR, artifact_name))
    return _rf_model_cache[artifact_name]


def _state_from_health_index(health_index, hi_bundle):
    """Reutilise les seuils PCA/IsoForest geles pour classer une valeur predite - a defaut
    de pouvoir recalculer T2/SPE/IsoForest pour un point futur non encore observe, on
    applique le meme decoupage Normal/Surveillance/Alarme que health_index_baseline.py a la
    VALEUR de health_index elle-meme (1=sain), via les percentiles surveillance/alarme
    observes historiquement sur les lignes saines de cette machine (meme convention que
    tout le reste du pipeline : percentile 95/99)."""
    surv, alarm = hi_bundle["hi_surv"], hi_bundle["hi_alarm"]
    if health_index <= alarm:
        return "Alarme"
    if health_index <= surv:
        return "Surveillance"
    return "Normal"


def predict_all_horizons(snapshot_df, artifacts):
    """snapshot_df : sortie de feature_pipeline.build_live_snapshot (toutes les lignes de
    la fenetre live, deja enrichies). Retourne {asset: {"current": {...}, "horizons": [...]}}."""
    manifest = artifacts["manifest"]
    hi_bundles = artifacts["health_index"]

    results = {}
    for asset, col in ASSET_COLS.items():
        sub = snapshot_df[snapshot_df[col] == 1].sort_values("created_at")
        if sub.empty:
            continue
        latest = sub.iloc[[-1]]
        current_hi = float(latest["health_index"].iloc[0])
        current_state = latest["health_state"].iloc[0]
        current_time = latest["created_at"].iloc[0]

        X_row = hpd.build_feature_columns(latest)

        horizons = []
        for horizon in hpd.HORIZONS:
            entry = manifest.get(f"{horizon}|{asset}")
            if entry is None:
                continue

            if entry["model"] == "random_forest":
                bundle = _load_rf_bundle(entry["artifact"])
                X_aligned = X_row.reindex(columns=bundle["columns"], fill_value=0)
                X_imputed = pd.DataFrame(
                    bundle["imputer"].transform(X_aligned), columns=X_aligned.columns
                )
                delta = float(bundle["model"].predict(X_imputed)[0])
                predicted_hi = current_hi + delta
            else:
                predicted_hi = current_hi
            predicted_hi = min(1.0, max(0.0, predicted_hi))

            horizons.append({
                "horizon_steps": horizon,
                "horizon_label": entry["horizon_label"],
                "target_time": current_time + timedelta(minutes=STEP_MINUTES * horizon),
                "predicted_health_index": predicted_hi,
                "predicted_state": _state_from_health_index(predicted_hi, hi_bundles[asset]),
                "model": entry["model"],
                "historical_mae": entry["mae"],
                "historical_rmse": entry["rmse"],
                "historical_r2": entry["r2"],
            })

        results[asset] = {
            "current_health_index": current_hi,
            "current_state": current_state,
            "current_time": current_time,
            "horizons": horizons,
        }
    return results


if __name__ == "__main__":
    from api_client import fetch_recent_wide
    from feature_pipeline import build_live_snapshot, load_artifacts

    wide, fetched_at = fetch_recent_wide()
    artifacts = load_artifacts()
    snapshot = build_live_snapshot(wide, artifacts)
    predictions = predict_all_horizons(snapshot, artifacts)

    for asset, data in predictions.items():
        print(f"\n{asset} — now ({data['current_time']}): "
              f"health_index={data['current_health_index']:.4f} ({data['current_state']})")
        for h in data["horizons"]:
            print(f"  t+{h['horizon_label']:<10} ({h['target_time']}) -> "
                  f"{h['predicted_health_index']:.4f} ({h['predicted_state']}) "
                  f"[{h['model']}, R²hist={h['historical_r2']:.2f}]")
