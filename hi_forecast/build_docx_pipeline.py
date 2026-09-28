"""Fiche d'une page : les etapes du pipeline, dans l'ordre correct."""
import os
import sys

from docx import Document
from docx.shared import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from build_docx import GREEN, GREY, NAVY, RED, h, para, rich, save_doc, setup, table

ETAPES = [
    ("1", "Extraction API", "Récupération des mesures i-SENSE, pivot long → large.",
     "aucun"),
    ("2", "Validation de schéma", "Colonnes, types, unités, plages attendues. "
     "Une ligne non conforme est rejetée.", "schema.json"),
    ("3", "Data quality", "Sentinelles, bornes physiques, règle viscosité. "
     "Statuts : valide / dégradée / rejetée.", "quality_rules.json"),
    ("4", "Nettoyage", "Doublons, conversion Oil Conductivity ×10 en nS/m, "
     "découpage en sessions (rupture > 60 min).", "cleaning_config.json"),
    ("5", "Imputation", "Vibration : 0 à l'arrêt, Random Forest en marche. "
     "Capteurs : forward-fill puis RF. Reste : médiane.",
     "vibration_imputer.joblib, fill_medians.json"),
    ("6", "Normalisation", "z-scores par machine et vi_proxy, avec des statistiques "
     "calculées sur la SEULE fenêtre d'entraînement.",
     "zscore_stats.json, vi_proxy_coefs.json"),
    ("7", "Ingénierie de variables", "Lags 1-3, diff1, EWMA(18), pente 3 h, indices "
     "physico-chimiques, calendaire, session.", "aucun"),
    ("8", "Construction du Health Index", "ACP (T²/SPE) + Isolation Forest sur les lignes "
     "saines du train. HI = 1 / (1 + sévérité).",
     "pca_{A,B}.joblib, isoforest_{A,B}.joblib, alarm_thresholds.json"),
    ("9", "Prédiction du HI", "Régresseur Lasso + porte LightGBM, un couple par machine et "
     "par horizon (20 min / 3 h / 24 h).", "modeles_multi_horizon.joblib"),
    ("10", "Restitution", "API FastAPI, tableau de bord React, journal des prédictions.",
     "aucun"),
]


def main():
    doc = setup(Document())
    for s in doc.sections:
        s.top_margin = s.bottom_margin = Inches(0.5)
        s.left_margin = s.right_margin = Inches(0.6)

    h(doc, "Pipeline de traitement — Health Index i-SENSE", 0)
    para(doc, "Les 10 étapes, de l'API à la prédiction   •   projet i-SENSE / OCP — UM6P",
         size=9.5, color=GREY, space=8)

    table(doc, ["N°", "Étape", "Ce qu'elle fait", "Artefacts ajustés"],
          [[n, e, d, a] for n, e, d, a in ETAPES],
          widths=[0.3, 1.55, 3.35, 2.0], size=8)

    para(doc, "", space=6)
    rich(doc, [("Deux corrections par rapport à l'ordre initialement envisagé.", True, NAVY)],
         size=9.5)

    table(doc, ["Point", "Ordre envisagé", "Ordre correct", "Pourquoi"],
          [["Imputation / normalisation", "normaliser puis imputer",
            "IMPUTER (5) puis NORMALISER (6)",
            "StandardScaler ne sait pas traiter les valeurs manquantes : il propagerait des "
            "NaN dans la moyenne et l'écart-type."],
           ["Place de la data quality", "après l'extraction, sans lien avec le HI",
            "AVANT la construction du HI (3 avant 8)",
            "Le HI est calculé à partir des variables capteur. Si les valeurs fausses ne sont "
            "pas écartées d'abord, elles corrompent l'étiquette elle-même, pas seulement les "
            "entrées du modèle."]],
          widths=[1.35, 1.6, 1.85, 2.4], size=8)

    para(doc, "", space=6)
    rich(doc, [("Point de vigilance en production. ", True, RED),
               ("Aujourd'hui l'application exécute l'étape 8 sur toutes les lignes, y compris "
                "les 16 839 viscosités physiquement impossibles (39,96 % du jeu, Motosoufflante "
                "B à l'arrêt). Le tableau de bord les signale mais ne les écarte pas. Activer "
                "le masquage fait passer la Motosoufflante A d'un skill de −0,119 à +0,086 et "
                "rend le bénéfice global statistiquement significatif.", False, None)],
         size=9)
    rich(doc, [("Règle transverse. ", True, GREEN),
               ("Tout artefact ajusté — imputeurs, z-scores, ACP, Isolation Forest, seuils, "
                "modèles — est calculé sur la SEULE fenêtre d'entraînement, versionné avec son "
                "empreinte SHA-256, et chargé depuis un répertoire figé. Le service refuse de "
                "démarrer si une empreinte diffère.", False, None)],
         size=9)

    para(doc, "Détail complet : ARCHITECTURE_PIPELINE.md (diagrammes, budget de latence, "
              "gestion des pannes, skew entraînement/service, déclencheurs de réentraînement).",
         size=8, italic=True, color=GREY)

    return save_doc(doc, os.path.join(ROOT, "Pipeline_Etapes_iSENSE.docx"))


if __name__ == "__main__":
    p = main()
    print(f"Document genere : {p}  ({os.path.getsize(p)/1024:.0f} Ko)")
