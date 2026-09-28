"""Test de non-regression EN LIGNE vs HORS LIGNE.

Exigence de ARCHITECTURE_PIPELINE.md, point dur n° 1 : le chemin de service doit
produire exactement les memes valeurs que le calcul hors ligne, sans quoi le
modele voit a l'inference des variables differentes de celles vues a
l'entrainement.

Protocole :
  1. calculer la reference HORS LIGNE sur le lot complet (protocol.prepare) ;
  2. rejouer les memes lignes par le FLUX INCREMENTAL, qui ne voit qu'un tampon ;
  3. comparer indice huile et indice systeme, ligne a ligne.

Echec du test = blocage du deploiement.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "hi_forecast"))
sys.path.insert(0, "C:\\pylib")
warnings.filterwarnings("ignore")

import ingestion                                                    # noqa: E402
from protocol import ASSET_COLS, prepare, train_mask_from_cutoffs   # noqa: E402
from systeme_hi import build_systeme_hi                             # noqa: E402

SOURCE = os.path.join(ROOT, "isense_oil_data_sensors_filled.csv")
TOLERANCE = 1e-9
N_NOUVELLES = 40          # lignes rejouees une a une apres l'amorce


def main():
    print("=" * 88)
    print("NON-REGRESSION — chemin en ligne contre chemin hors ligne")
    print("=" * 88)

    # ---------------------------------------------------- reference hors ligne
    print("\n1. reference hors ligne (lot complet) ...")
    off, tm, cut, _ = prepare(verbose=False, mask_off_viscosity=False)
    off, _ = build_systeme_hi(off, train_mask_from_cutoffs(off, cut))
    off["machine"] = np.where(off[ASSET_COLS["Motosoufflante A"]] == 1,
                              "Motosoufflante A", "Motosoufflante B")
    ref = off.set_index(["machine", "created_at"])[
        ["health_index_lf", "indice_pression_lf", "indice_vibration_lf",
         "time_in_session_h", "measure_index_in_session"]]
    print(f"   {len(off)} lignes de reference")

    # ---------------------------------------------------- rejeu en ligne
    print("\n2. rejeu par le flux incremental ...")
    brut = pd.read_csv(SOURCE, parse_dates=["created_at"])
    brut = brut.sort_values("created_at").reset_index(drop=True)

    art = ingestion.charger_pretraitement()
    coupe = len(brut) - N_NOUVELLES
    flux = ingestion.FluxIncremental(art, taille_tampon=ingestion.TAMPON_DEFAUT)
    flux.amorcer(brut.iloc[:coupe])

    ecarts_hi, ecarts_pre, ecarts_vib, ecarts_sess, n_cmp = [], [], [], [], 0
    for i in range(coupe, len(brut)):
        traite = flux.ingerer(brut.iloc[[i]])
        if traite is None or not len(traite):
            continue
        ligne = traite.sort_values("created_at").iloc[-1]
        mach = ("Motosoufflante A"
                if ligne[ASSET_COLS["Motosoufflante A"]] == 1 else "Motosoufflante B")
        cle = (mach, ligne["created_at"])
        if cle not in ref.index:
            continue
        r = ref.loc[cle]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        n_cmp += 1

        def _d(a, b):
            if pd.isna(a) and pd.isna(b):
                return 0.0
            return abs(float(a) - float(b))

        ecarts_hi.append(_d(ligne["health_index_lf"], r["health_index_lf"]))
        ecarts_pre.append(_d(ligne["indice_pression_lf"], r["indice_pression_lf"]))
        ecarts_vib.append(_d(ligne["indice_vibration_lf"], r["indice_vibration_lf"]))
        ecarts_sess.append(max(_d(ligne["time_in_session_h"], r["time_in_session_h"]),
                               _d(ligne["measure_index_in_session"],
                                  r["measure_index_in_session"])))

    # ---------------------------------------------------- verdict
    print(f"   {n_cmp} lignes comparees\n")
    print("=" * 88)
    lignes = [("Health Index huile", ecarts_hi),
              ("Indicateur pression", ecarts_pre),
              ("Indicateur vibration", ecarts_vib),
              ("Variables de session", ecarts_sess)]
    ok = True
    for nom, e in lignes:
        if not e:
            print(f"  {nom:24} AUCUNE COMPARAISON")
            ok = False
            continue
        pire = max(e)
        verdict = "OK" if pire < TOLERANCE else "ECHEC"
        ok &= pire < TOLERANCE
        print(f"  {nom:24} ecart max {pire:.3e}   [{verdict}]")
    print("=" * 88)
    print(f"\n  {'CONFORME' if ok else 'NON CONFORME'} — tolerance {TOLERANCE:.0e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
