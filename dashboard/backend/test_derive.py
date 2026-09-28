"""Compare les mesures QUI ARRIVENT aux mesures d'ENTRAINEMENT, capteur par capteur.

Pourquoi : les modeles et l'etiquette ont ete ajustes sur des donnees arretees au
6 aout 2026. Tout ce qui s'ecarte de ce domaine est lu comme une anomalie — que
la machine se degrade vraiment, ou qu'un capteur ait ete recalibre, ou qu'une
unite ait change cote API. Ce script distingue les trois.

Lecture du resultat :
  - un capteur DANS la plage d'entrainement  -> le modele est en terrain connu
  - un capteur HORS plage, sur UNE machine   -> possible degradation reelle
  - un capteur HORS plage, sur LES DEUX      -> plutot un changement de capteur,
                                                d'unite ou de configuration

Aucune ecriture, aucune ingestion : ce script lit et compare.

    python test_derive.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, ROOT, os.path.join(ROOT, "hi_forecast"), "C:\\pylib"):
    if p not in sys.path:
        sys.path.insert(0, p)
warnings.filterwarnings("ignore")

CAPTEURS = ["Oil Pressure", "Oil Temperature", "Oil System Vibration",
            "Viscosity at 40°C", "Kinematic Viscosity", "Dynamic Viscosity",
            "Density", "DC", "ISO 4", "Oil H2O ppm", "Oil H2O Saturation",
            "Oil Conductivity"]
CLEANED = os.path.join(ROOT, "isense_oil_data_cleaned.csv")


def main():
    print("=" * 100)
    print("DERIVE — mesures qui arrivent  vs  mesures d'entrainement")
    print("=" * 100)

    # ------------------------------------------------- 1. domaine d'entrainement
    if not os.path.exists(CLEANED):
        print("  Jeu nettoye introuvable :", CLEANED)
        return 1
    hist = pd.read_csv(CLEANED, parse_dates=["created_at"])
    print(f"\n  Entrainement : {len(hist)} lignes, "
          f"{hist['created_at'].min():%d/%m/%Y} -> {hist['created_at'].max():%d/%m/%Y}")

    # ------------------------------------------------------- 2. mesures en direct
    from live_feed import SourceAPI, nettoyer, pivoter
    print("  Relevé en cours ...")
    src = SourceAPI(jours_initial=3)
    brut = src.relever(None)
    if brut is None or brut.empty:
        print("  Aucune mesure renvoyée par l'API.")
        return 1
    live = nettoyer(pivoter(brut), {})
    print(f"  Direct       : {len(live)} lignes, "
          f"{live['created_at'].min():%d/%m/%Y %H:%M} -> "
          f"{live['created_at'].max():%d/%m/%Y %H:%M}")

    # -------------------------------------------------------------- 3. comparaison
    for machine in sorted(set(live["asset_name"]) & set(hist["asset_name"])):
        h = hist[hist["asset_name"] == machine]
        v = live[live["asset_name"] == machine]
        print(f"\n{'=' * 100}\n{machine}  —  {len(v)} relevés en direct\n{'-' * 100}")
        print(f"{'capteur':26} {'entrainement p05..p95':>28} {'médiane dir.':>14} "
              f"{'écart':>10}   verdict")
        for c in CAPTEURS:
            if c not in h.columns or c not in v.columns:
                continue
            hv = pd.to_numeric(h[c], errors="coerce").dropna()
            vv = pd.to_numeric(v[c], errors="coerce").dropna()
            if hv.empty or vv.empty:
                print(f"{c:26} {'—':>28} {'—':>14} {'':>10}   pas de mesure")
                continue
            p05, p95, med_h = hv.quantile(.05), hv.quantile(.95), hv.median()
            med_v = vv.median()
            dedans = p05 <= med_v <= p95
            # ecart en "largeurs de plage d'entrainement"
            largeur = max(p95 - p05, 1e-9)
            ecart = (med_v - med_h) / largeur
            verdict = "dans la plage" if dedans else "HORS PLAGE"
            print(f"{c:26} {f'{p05:10.3f} .. {p95:10.3f}':>28} {med_v:14.3f} "
                  f"{ecart:+9.1f}x   {verdict}")

    print("\n" + "=" * 100)
    print("  Un capteur hors plage sur LES DEUX machines oriente vers un changement")
    print("  de capteur, d'unité ou de configuration — pas vers une dégradation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
