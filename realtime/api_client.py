"""
Recuperation temps reel des mesures i-SENSE - reutilise Api_to_excel.py (login,
fetch_trend, flatten_measures) telles quelles, mais interroge une fenetre glissante
recente (par defaut 3 jours) plutot que tout l'historique depuis le startDate de
chaque machine : suffisant pour couvrir plusieurs sessions completes et donner du
contexte aux features a fenetre glissante (lags, EWMA span=18, pente sur 18 pas -
cf. feature_engineering.py), tout en gardant chaque appel API leger.

Combine les deux machines dans UN SEUL dataframe large (comme Api_to_excel.main())
- necessaire pour que le one-hot encoding de asset_name en aval produise les deux
colonnes de maniere fiable meme si une machine est peu representee dans la fenetre.
"""

import os
import sys
from datetime import date, datetime, timedelta

import pandas as pd
import requests

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from Api_to_excel import ASSETS, FEATURES, flatten_measures, login  # noqa: E402

API_BASE_URL = "https://v3back-demo.i-sense.io/api/i-sense-v3"
LOOKBACK_DAYS = 3


def fetch_trend_window(session, token, asset_cfg, feature, start_date):
    """Variante de Api_to_excel.fetch_trend avec un startDate explicite (fenetre
    glissante) au lieu de asset_cfg['startDate'] (debut d'historique complet)."""
    url = f"{API_BASE_URL}/tenant/assets/trend/{asset_cfg['asset_id']}/{feature['feature_id']}"

    payload = {
        "assetsIds": asset_cfg["children"],
        "selectedPointId": "",
        "startDate": start_date.isoformat(),
        "endDate": (date.today() + timedelta(days=1)).isoformat(),
    }
    if feature.get("is_grandeur"):
        payload["is_grandeur"] = True

    response = session.post(
        url,
        json=payload,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return flatten_measures(data.get("measures", {}), feature)


def fetch_recent_wide(lookback_days=LOOKBACK_DAYS):
    """Retourne (df_wide, fetched_at) - df_wide au meme format que
    isense_oil_data_combined_wide.csv (une ligne par asset_name x created_at, une
    colonne par variable), pour la fenetre [aujourd'hui - lookback_days, demain]."""
    token = login()
    session = requests.Session()
    start_date = date.today() - timedelta(days=lookback_days)

    all_rows = []
    for asset_name, asset_cfg in ASSETS.items():
        for feature in FEATURES:
            rows = fetch_trend_window(session, token, asset_cfg, feature, start_date)
            for row in rows:
                row["asset_name"] = asset_name
            all_rows.extend(rows)

    fetched_at = datetime.now()
    if not all_rows:
        return pd.DataFrame(), fetched_at

    df = pd.DataFrame(all_rows)
    df["created_at"] = pd.to_datetime(
        df["created_at"], errors="coerce", utc=True, format="mixed"
    ).dt.tz_localize(None)

    wide = df.pivot_table(
        index=["asset_name", "created_at"], columns="variable", values="value", aggfunc="first"
    ).reset_index()
    wide = wide.sort_values(["asset_name", "created_at"]).reset_index(drop=True)
    return wide, fetched_at


if __name__ == "__main__":
    wide, fetched_at = fetch_recent_wide()
    print(f"Fetched at {fetched_at}: {len(wide)} rows, {wide['asset_name'].value_counts().to_dict()}")
    print(wide.tail())
