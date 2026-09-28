import os
import getpass
import requests
import pandas as pd
from datetime import date, timedelta

# Adresse de l'API i-SENSE : variable d'environnement ISENSE_API_URL, avec
# l'adresse du serveur de demonstration comme valeur par defaut.
API_BASE_URL = os.environ.get(
    "ISENSE_API_URL", "https://v3back-demo.i-sense.io/api/i-sense-v3").rstrip("/")

# Identifiants — JAMAIS en dur dans ce fichier : il est suivi en version, et
# tout ce qui y est ecrit finit dans l'historique du depot.
#
#   $env:ISENSE_EMAIL    = "..."
#   $env:ISENSE_PASSWORD = "..."
#
# A defaut de variables d'environnement, ils sont demandes au clavier au moment
# de la connexion — la saisie du mot de passe reste masquee. La demande est
# DIFFEREE dans identifiants() et non faite au chargement du module : le
# connecteur temps reel importe ce fichier, un appel bloquant a l'import
# suspendrait le serveur.
EMAIL = os.environ.get("ISENSE_EMAIL")
PASSWORD = os.environ.get("ISENSE_PASSWORD")


def identifiants():
    """Renvoie (email, mot de passe), depuis l'environnement ou au clavier."""
    email = EMAIL or input("Adresse i-SENSE : ").strip()
    password = PASSWORD or getpass.getpass("Mot de passe i-SENSE : ")
    if not email or not password:
        raise RuntimeError(
            "Identifiants i-SENSE absents. Definir ISENSE_EMAIL et "
            "ISENSE_PASSWORD dans l'environnement.")
    return email, password

ASSETS = {
    "Motosoufflante A": {
        "asset_id": 1420,
        "children": ["1423", "1424"],
        "oil_device": "Oil-AA0005",
        "startDate": "2025-12-29",
    },
    "Motosoufflante B": {
        "asset_id": 1425,
        "children": ["1426", "1427"],
        "oil_device": "Oil-AA0004",
        "startDate": "2026-01-02",
    },
}

FEATURES = [
    {
        "feature_id": 27,
        "api_name": "Oil Contamination",
        "is_grandeur": True,
        "series_names": {
            "iso_14": ("ISO 14", "-"),
            "iso_6": ("ISO 6", "-"),
            "iso_4": ("ISO 4", "-"),
        },
    },
    {
        "feature_id": 38,
        "api_name": "Oil Viscosity",
        "is_grandeur": True,
        "series_names": {
            "Vis_40": ("Viscosity at 40°C", "cSt"),
            "K_viscosity": ("Kinematic Viscosity", "cSt"),
            "D_viscosity": ("Dynamic Viscosity", "cP"),
        },
    },
    {"feature_id": 73, "variable": "Density", "unit": "kg/m3"},
    {"feature_id": 58, "variable": "Oil Temperature", "unit": "°C"},
    {"feature_id": 59, "variable": "Oil H2O Saturation", "unit": "%"},
    {"feature_id": 77, "variable": "Oil H2O ppm", "unit": "ppm"},
    {"feature_id": 61, "variable": "Oil Conductivity", "unit": "nS/m"},
    {"feature_id": 68, "variable": "DC", "unit": "-"},
    {"feature_id": 78, "variable": "Oil System Vibration", "unit": "mm/s²"},
    {"feature_id": 79, "variable": "Oil Pressure", "unit": "Bar"},
]


def login():
    url = f"{API_BASE_URL}/auth/login"
    email, password = identifiants()
    response = requests.post(
        url,
        json={"email": email, "password": password},
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["token"]


def flatten_measures(measures, feature):
    rows = []

    def walk(obj, path):
        if not isinstance(obj, dict):
            return

        if isinstance(obj.get("values"), list):
            series_key = path[0] if path else feature.get("variable")

            if feature.get("is_grandeur"):
                variable, unit = feature["series_names"].get(series_key, (series_key, ""))
            else:
                variable, unit = feature["variable"], feature["unit"]

            for item in obj["values"]:
                rows.append({
                    "variable": variable,
                    "feature_id": feature["feature_id"],
                    "unit": unit,
                    "series_path": "/".join(path),
                    "point_id": obj.get("point_id"),
                    "point_name": obj.get("point_name"),
                    "created_at": item.get("created_at"),
                    "measure_id": item.get("measure_id"),
                    "value": item.get("value"),
                })
            return

        for key, value in obj.items():
            walk(value, path + [key])

    walk(measures, [])
    return rows


def fetch_trend(session, token, asset_cfg, feature):
    url = f"{API_BASE_URL}/tenant/assets/trend/{asset_cfg['asset_id']}/{feature['feature_id']}"

    payload = {
        "assetsIds": asset_cfg["children"],
        "selectedPointId": "",
        "startDate": asset_cfg["startDate"],
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


def main():
    token = login()
    session = requests.Session()

    all_rows = []

    for asset_name, asset_cfg in ASSETS.items():
        for feature in FEATURES:
            print(f"Fetching {asset_name} - {feature.get('api_name') or feature.get('variable')}")

            rows = fetch_trend(session, token, asset_cfg, feature)

            for row in rows:
                row["asset_name"] = asset_name
                row["asset_id"] = asset_cfg["asset_id"]
                row["oil_device"] = asset_cfg["oil_device"]

            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    columns = [
        "asset_name",
        "asset_id",
        "oil_device",
        "point_id",
        "point_name",
        "variable",
        "feature_id",
        "unit",
        "created_at",
        "measure_id",
        "value",
        "series_path",
    ]
    df = df[columns]
    df["created_at"] = pd.to_datetime(
        df["created_at"], errors="coerce", utc=True, format="mixed"
    ).dt.tz_localize(None)
    df = df.sort_values(["asset_name", "variable", "created_at"])

    df.to_csv("isense_oil_data_combined_long.csv", index=False, encoding="utf-8-sig")

    combined_wide = df.pivot_table(
        index=["asset_name", "created_at"],
        columns="variable",
        values="value",
        aggfunc="first",
    ).reset_index()

    combined_wide.to_csv("isense_oil_data_combined_wide.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter("isense_oil_data.xlsx", engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="combined_long", index=False)
        combined_wide.to_excel(writer, sheet_name="combined_wide", index=False)

        for asset_name in ASSETS:
            asset_df = df[df["asset_name"] == asset_name].copy()

            safe_name = asset_name.lower().replace(" ", "_")
            asset_df.to_csv(f"isense_oil_data_{safe_name}_long.csv", index=False, encoding="utf-8-sig")

            wide = asset_df.pivot_table(
                index="created_at",
                columns="variable",
                values="value",
                aggfunc="first",
            ).reset_index()

            wide.to_csv(f"isense_oil_data_{safe_name}_wide.csv", index=False, encoding="utf-8-sig")

            sheet_base = "A" if "A" in asset_name else "B"
            asset_df.to_excel(writer, sheet_name=f"{sheet_base}_long", index=False)
            wide.to_excel(writer, sheet_name=f"{sheet_base}_wide", index=False)

    print("Done.")
    print("Created:")
    print("- isense_oil_data_combined_long.csv")
    print("- isense_oil_data_combined_wide.csv")
    print("- isense_oil_data_motosoufflante_a_long.csv")
    print("- isense_oil_data_motosoufflante_a_wide.csv")
    print("- isense_oil_data_motosoufflante_b_long.csv")
    print("- isense_oil_data_motosoufflante_b_wide.csv")
    print("- isense_oil_data.xlsx")


if __name__ == "__main__":
    main()