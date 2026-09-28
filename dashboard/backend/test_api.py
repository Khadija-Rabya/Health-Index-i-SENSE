"""Diagnostic du connecteur i-SENSE — a lancer avant de brancher le temps reel.

Quatre controles, du plus simple au plus engageant :
  1. les identifiants sont-ils dans l'environnement ?
  2. la connexion aboutit-elle ?
  3. un releve renvoie-t-il des mesures ?
  4. le pivot long -> large produit-il les colonnes attendues ?

Rien n'est ecrit, rien n'est ingere : ce script ne fait que lire et decrire.
Ni le mot de passe ni le jeton ne sont affiches.

    python test_api.py
"""
import os
import sys
import warnings

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, ROOT, os.path.join(ROOT, "hi_forecast"), "C:\\pylib"):
    if p not in sys.path:
        sys.path.insert(0, p)
warnings.filterwarnings("ignore")

OK, KO = "  [OK]   ", "  [ECHEC]"


def main():
    print("=" * 84)
    print("DIAGNOSTIC DU CONNECTEUR i-SENSE")
    print("=" * 84)

    # ------------------------------------------------------- 1. identifiants
    print("\n1. Identifiants")
    email = os.environ.get("ISENSE_EMAIL")
    mdp = os.environ.get("ISENSE_PASSWORD")
    print(f"{OK if email else KO} ISENSE_EMAIL    : {email or 'ABSENT'}")
    print(f"{OK if mdp else KO} ISENSE_PASSWORD : {'défini' if mdp else 'ABSENT'}")
    if not (email and mdp):
        print("\n  -> Positionner les deux variables, puis relancer.")
        return 1

    # ---------------------------------------------------------- 2. connexion
    print("\n2. Connexion")
    try:
        import Api_to_excel as api
        token = api.login()
        print(f"{OK} jeton obtenu ({len(token)} caractères, non affiché)")
    except Exception as e:
        print(f"{KO} {type(e).__name__} : {e}")
        print("\n  -> Identifiants refusés, ou serveur injoignable.")
        return 1

    # ------------------------------------------------------------ 3. releve
    print("\n3. Relevé d'une variable sur une machine")
    import requests
    session = requests.Session()
    nom, cfg = next(iter(api.ASSETS.items()))
    feature = api.FEATURES[0]
    libelle = feature.get("variable") or feature.get("api_name")
    # Fenetre COURTE : la startDate d'ASSETS remonte a decembre 2025 et fait
    # tomber le serveur en 504. Un flux ne demande que le recent.
    depuis = (pd.Timestamp.now() - pd.Timedelta(days=2)).date().isoformat()
    cfg = {**cfg, "startDate": depuis}
    print(f"           fenêtre demandée : depuis {depuis}")
    try:
        rows = api.fetch_trend(session, token, cfg, feature)
        print(f"{OK} {nom} / {libelle} : {len(rows)} mesures")
        if rows:
            print(f"           clés d'une mesure : {sorted(rows[0].keys())}")
            print(f"           exemple           : variable={rows[0].get('variable')!r} "
                  f"created_at={rows[0].get('created_at')!r} "
                  f"value={rows[0].get('value')!r}")
    except Exception as e:
        print(f"{KO} {type(e).__name__} : {e}")
        return 1

    # ------------------------------------------- 4. releve complet et pivot
    print("\n4. Relevé complet et pivot long -> large")
    try:
        from live_feed import SourceAPI, horodatage, pivoter
        src = SourceAPI()
        brut = src.relever(None)
        print(f"{OK} {len(brut)} mesures, {brut['variable'].nunique() if len(brut) else 0} "
              f"variables distinctes")
        if src._muettes:
            print(f"           variables en erreur ({len(src._muettes)}) : "
                  f"{', '.join(src._muettes[:5])}")
        large = pivoter(brut)
        print(f"{OK} après pivot : {len(large)} lignes x {len(large.columns)} colonnes")
        attendues = ["created_at", "asset_name", "Oil Pressure", "Oil Temperature",
                     "Oil System Vibration", "DC", "Density"]
        manquantes = [c for c in attendues if c not in large.columns]
        if manquantes:
            print(f"{KO} colonnes attendues manquantes : {manquantes}")
            print(f"           colonnes obtenues : {sorted(large.columns)[:20]}")
        else:
            print(f"{OK} toutes les colonnes clés sont présentes")
        if len(large):
            t = horodatage(large['created_at'])
            print(f"           période : {t.min()} -> {t.max()}  (UTC, naïf)")
            print(f"           machines : {sorted(large['asset_name'].unique())}")
            print(f"           variables : {sorted(c for c in large.columns if c not in ('created_at','asset_name'))}")
    except Exception as e:
        print(f"{KO} {type(e).__name__} : {e}")
        return 1

    print("\n" + "=" * 84)
    print("  Connecteur opérationnel. Le chemin d'ingestion peut être branché.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
