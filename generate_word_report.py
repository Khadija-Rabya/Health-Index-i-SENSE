"""
Assemble un rapport Word consolidant l'EDA, l'imputation de la vibration et
l'imputation des capteurs (trou synchrone), a partir des analyses et graphiques
deja produits dans ce dossier. Le feature engineering n'est PAS inclus.

Sortie : Rapport_EDA_Imputation_iSENSE.docx
"""

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

IMG_W = Inches(6.3)


def add_title_page(doc):
    title = doc.add_heading("Rapport — Traitement des données i-SENSE", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = sub.add_run("Analyse exploratoire (EDA), imputation de la vibration\net imputation des capteurs")
    run.italic = True
    run.font.size = Pt(14)

    doc.add_paragraph()
    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.add_run(
        "Surveillance de la qualité de l'huile de lubrification — Motosoufflante A & B\n"
        "PFE OCP–UM6P — Traitement de données\n"
        "Périmètre de ce document : EDA, imputation Oil System Vibration, "
        "imputation du trou capteur synchrone (feature engineering exclu)"
    )
    doc.add_page_break()


def add_intro(doc):
    doc.add_heading("1. Introduction", level=1)
    doc.add_paragraph(
        "Ce rapport documente les étapes de traitement de données réalisées sur le dataset "
        "i-SENSE, extrait de l'API de la plateforme de surveillance de la qualité d'huile de "
        "lubrification de deux machines industrielles (Motosoufflante A et B). Il couvre trois "
        "étapes du pipeline, après le nettoyage initial (sentinelles, sessions, flags ON/OFF) :"
    )
    for txt in [
        "L'analyse exploratoire des données (EDA) : statistiques descriptives, distributions, "
        "corrélations, vue temporelle, état des machines.",
        "L'imputation de Oil System Vibration (65,9 % de valeurs manquantes) : comparaison "
        "rigoureuse de plusieurs méthodes et remplissage final.",
        "L'imputation d'un trou synchrone affectant 7 autres variables santé simultanément "
        "(2,36 % du dataset) : comparaison de méthodes et remplissage final.",
    ]:
        doc.add_paragraph(txt, style="List Bullet")
    doc.add_paragraph(
        "L'étape suivante du pipeline (feature engineering) n'est volontairement pas incluse "
        "dans ce document."
    )


def add_picture_centered(doc, path, width=IMG_W, caption=None, description=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(path, width=width)
    if caption:
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cap.add_run(caption)
        r.italic = True
        r.font.size = Pt(9)
    if description:
        desc = doc.add_paragraph()
        r = desc.add_run(description)
        r.font.size = Pt(10)
        r.font.color.rgb = RGBColor(0x40, 0x40, 0x40)
        desc.paragraph_format.space_after = Pt(12)


def add_table(doc, headers, rows, caption=None, note=None):
    if caption:
        cap = doc.add_paragraph()
        r = cap.add_run(caption)
        r.italic = True
        r.font.size = Pt(9)
        cap.paragraph_format.space_after = Pt(4)

    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = str(h)
        for p in hdr_cells[i].paragraphs:
            for r in p.runs:
                r.font.bold = True
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)

    if note:
        n = doc.add_paragraph()
        r = n.add_run(note)
        r.font.size = Pt(10)
        r.font.color.rgb = RGBColor(0x40, 0x40, 0x40)
        n.paragraph_format.space_before = Pt(4)
        n.paragraph_format.space_after = Pt(12)
    else:
        doc.add_paragraph()
    return table


def add_eda_section(doc):
    doc.add_heading("2. Analyse exploratoire des données (EDA)", level=1)
    doc.add_paragraph(
        "Réalisée par eda_isense.py, cette étape est purement exploratoire : aucune colonne "
        "n'est créée ni modifiée. Elle porte sur le dataset nettoyé (isense_oil_data_cleaned.csv, "
        "42 141 lignes, période du 29/12/2025 au 06/08/2026)."
    )

    doc.add_heading("2.1 Statistiques descriptives par machine (post-nettoyage)", level=2)
    headers = ["Variable", "Unité", "Valides A", "Min A", "Max A", "Moy. A",
               "Valides B", "Min B", "Max B", "Moy. B"]
    rows = [
        ["DC", "-", 19418, 1.01, 2.26, 2.20, 21729, 0.89, 2.11, 1.03],
        ["Density", "kg/m³", 19418, 845.54, 861.43, 851.67, 21729, 840.70, 868.18, 851.66],
        ["Dynamic Viscosity", "cP", 19418, 23.71, 72.13, 38.01, 21729, 1.00, 64.40, 13.52],
        ["ISO 4", "-", 19418, 7.00, 25.00, 18.47, 21729, 12.00, 22.00, 18.90],
        ["ISO 6", "-", 19896, 1.00, 25.00, 13.38, 22245, 4.00, 21.00, 16.29],
        ["ISO 14", "-", 19896, 1.00, 22.00, 9.98, 22245, 1.00, 20.00, 14.09],
        ["Oil Conductivity", "nS/m", 19896, -0.10, 0.07, 0.06, 22245, -0.10, 0.07, 0.06],
        ["Oil H2O Saturation", "%", 19418, 10.00, 95.00, 21.02, 21729, 11.00, 42.00, 25.37],
        ["Oil H2O ppm", "ppm", 19418, 22.00, 123.00, 39.08, 21729, 19.00, 62.00, 45.85],
        ["Oil Pressure", "Bar", 19896, 0.00, 7.70, 5.06, 22245, 0.00, 4.70, 0.42],
        ["Oil Temperature", "°C", 19418, 27.78, 51.40, 42.36, 21729, 17.60, 58.68, 42.26],
        ["Viscosity at 40°C", "cSt", 19418, 42.72, 48.06, 46.32, 21729, 3.02, 43.06, 17.56],
    ]
    add_table(
        doc, headers, rows,
        caption="Tableau 2.1 — Statistiques descriptives par machine (n valides, min, max, moyenne)",
        note=(
            "Ce tableau sert de première vérification de plausibilité avant toute analyse "
            "plus poussée. On y repère déjà des écarts marqués entre machines : Machine B a "
            "une viscosité moyenne bien plus faible (17,56 cSt contre 46,32 pour Machine A) et "
            "une pression moyenne près de 10 fois plus basse (0,42 contre 5,06 Bar), cohérent "
            "avec une machine très majoritairement à l'arrêt (cf. section 2.7)."
        ),
    )

    doc.add_heading("2.2 Valeurs manquantes", level=2)
    doc.add_paragraph(
        "Oil System Vibration présente 65,89 % de NaN (module non actif en continu) ; 8 autres "
        "variables présentent 2,36 % de NaN chacune, dues à un trou synchrone unique (voir "
        "section 4) ; ISO 6, ISO 14, Oil Conductivity et Oil Pressure n'ont jamais de NaN."
    )
    add_picture_centered(
        doc, "eda_output/01_missing_data.png",
        caption="Figure 2.1 — Taux de valeurs manquantes par variable",
        description=(
            "Lecture : Oil System Vibration se détache nettement avec 65,9 % de NaN — le "
            "module de mesure n'est pas actif en continu. Les 8 variables suivantes affichent "
            "toutes exactement le même taux de 2,36 %, ce qui indique un unique épisode "
            "d'indisponibilité capteur partagé plutôt que des pannes indépendantes (confirmé "
            "et détaillé en section 4). ISO 6, ISO 14, Oil Conductivity et Oil Pressure "
            "n'ont jamais de valeur manquante."
        ),
    )

    doc.add_heading("2.3 Distributions par variable et par machine", level=2)
    add_picture_centered(
        doc, "eda_output/02_distributions.png",
        caption="Figure 2.2 — Distributions comparées Motosoufflante A / B",
        description=(
            "Les histogrammes superposés montrent des formes de distribution nettement "
            "différentes entre les deux machines pour la plupart des variables (Dynamic "
            "Viscosity, Viscosity at 40°C notamment), cohérent avec des huiles ou des grades "
            "différents entre A et B plutôt qu'un artefact de mesure."
        ),
    )

    doc.add_heading("2.4 Box plots comparatifs", level=2)
    add_picture_centered(
        doc, "eda_output/03_boxplots.png",
        caption="Figure 2.3 — Box plots comparatifs Machine A vs B",
        description=(
            "Confirme visuellement les écarts de médiane et de dispersion déjà présents dans "
            "le tableau de statistiques descriptives (section 2.1) — utile pour repérer "
            "d'éventuelles valeurs extrêmes résiduelles après nettoyage."
        ),
    )

    doc.add_heading("2.5 Matrice de corrélation", level=2)
    add_picture_centered(
        doc, "eda_output/04_correlation_heatmap.png",
        caption="Figure 2.4 — Corrélations de Pearson entre variables santé",
        description=(
            "Cette heatmap a permis d'identifier les redondances entre variables. Deux cas se "
            "distinguent : Density / Oil Temperature (r = -1.000, une relation déterministe "
            "analysée en 2.5.1) et ISO 6 / ISO 14 (r = 0.966, cohérent — ces deux codes ISO "
            "4406 mesurent des tailles de particules voisines). Les corrélations plus modérées "
            "(DC avec Oil Pressure ou Viscosity at 40°C) reflètent des liens physiques "
            "attendus entre propriétés de l'huile."
        ),
    )
    corr_rows = [
        ["Density / Oil Temperature", "-1.000"],
        ["ISO 6 / ISO 14", "0.966"],
        ["DC / Oil Pressure", "0.942"],
        ["DC / Viscosity at 40°C", "0.889"],
        ["ISO 4 / ISO 6", "0.856"],
        ["Dynamic Viscosity / Viscosity at 40°C", "0.855"],
        ["Oil Pressure / Viscosity at 40°C", "0.832"],
        ["Oil H2O Saturation / Oil H2O ppm", "0.816"],
        ["ISO 4 / ISO 14", "0.772"],
        ["DC / Dynamic Viscosity", "0.706"],
        ["DC / ISO 14", "-0.700"],
        ["ISO 14 / Oil Pressure", "-0.656"],
        ["Dynamic Viscosity / Oil Pressure", "0.631"],
        ["ISO 14 / Viscosity at 40°C", "-0.609"],
        ["DC / ISO 6", "-0.606"],
    ]
    add_table(
        doc, ["Paire de variables", "Corrélation de Pearson"], corr_rows,
        caption="Tableau 2.2 — Paires de variables avec |r| > 0.6, triées par force décroissante",
        note=(
            "Liste complète des paires derrière la heatmap de la figure 2.4. La paire "
            "Density / Oil Temperature en tête (r = -1.000) est la seule relation "
            "véritablement déterministe du tableau ; les autres paires, bien que fortement "
            "corrélées, gardent chacune une variance propre et ne sont pas redondantes au "
            "même degré — elles ont donc été conservées dans les étapes suivantes."
        ),
    )

    doc.add_heading("2.5.1 Investigation — Density / Oil Temperature", level=3)
    doc.add_paragraph(
        "La corrélation quasi parfaite (r = -1.000, résidus < 0,09 kg/m³) indique que Density "
        "n'est pas mesurée indépendamment mais calculée à partir de Oil Temperature via une "
        "correction de dilatation thermique standard (Density ≈ -0,6734 × Température + 880,15). "
        "Density n'apporte donc aucune information supplémentaire et a été exclue des étapes "
        "suivantes du pipeline."
    )

    doc.add_heading("2.6 Vue temporelle globale", level=2)
    add_picture_centered(
        doc, "eda_output/05_timeseries_overview.png",
        caption="Figure 2.5 — Viscosité, température et indice de contamination dans le temps",
        description=(
            "Cette inspection visuelle directe du signal a révélé une chute brutale de "
            "viscosité sur Motosoufflante B fin février 2026, sans variation correspondante "
            "de pression — ce qui a déclenché l'investigation détaillée ci-dessous plutôt "
            "qu'une explication par un simple redémarrage machine."
        ),
    )
    doc.add_paragraph(
        "Investigation complémentaire : la chute de viscosité observée sur Motosoufflante B fin "
        "février 2026 ne correspond pas à un redémarrage (machine restée OFF). En creusant, "
        "Motosoufflante B n'a été en état ON qu'en janvier 2026 et reste arrêtée en continu "
        "depuis fin janvier 2026 jusqu'à la fin de la période couverte — les valeurs de capteurs "
        "collectées pendant les phases OFF de cette machine (variance nulle en ON, forte "
        "variance erratique en OFF) doivent être considérées comme du bruit plutôt que des "
        "mesures représentatives de l'état réel de l'huile."
    )

    doc.add_heading("2.7 État machine (ON / OFF)", level=2)
    add_picture_centered(
        doc, "eda_output/06_machine_state.png",
        caption="Figure 2.6 — Répartition du temps ON vs OFF par machine",
        description=(
            "Confirme numériquement l'anomalie détectée en 2.6 : Motosoufflante B est à l'état "
            "OFF l'immense majorité du temps couvert par le dataset, contre une Motosoufflante "
            "A très majoritairement active. Cet écart doit être pris en compte pour toute "
            "comparaison directe entre les deux machines."
        ),
    )

    doc.add_heading("2.8 Durée des sessions continues", level=2)
    add_picture_centered(
        doc, "eda_output/07_session_durations.png",
        caption="Figure 2.7 — Distribution de la durée des 352 sessions continues",
        description=(
            "Distribution fortement asymétrique : une majorité de sessions très courtes "
            "(médiane 0,5 h) coexiste avec quelques sessions exceptionnellement longues "
            "(jusqu'à 1339,7 h). Cette fragmentation limite la fenêtre maximale utilisable pour "
            "les features à fenêtre glissante calculées par session dans l'étape de feature "
            "engineering (non couverte par ce document)."
        ),
    )
    doc.add_paragraph(
        "352 sessions continues identifiées (séparées par des arrêts > 1h). Durée médiane : "
        "0,5 h ; durée maximale : 1339,7 h ; 190 sessions (54 %) durent moins d'1 heure."
    )


def add_vibration_section(doc):
    doc.add_page_break()
    doc.add_heading("3. Imputation de Oil System Vibration", level=1)

    doc.add_heading("3.1 Problématique", level=2)
    doc.add_paragraph(
        "Oil System Vibration est manquante à 65,9 % (module de mesure non actif en continu). "
        "L'analyse des séquences montre des rafales de valeurs connues (médiane 2 points "
        "consécutifs) séparées par des trous NaN pouvant atteindre ~2000 mesures d'affilée "
        "(plusieurs jours). L'évaluation des méthodes d'imputation doit donc couvrir à la fois "
        "les petits trous isolés et les très longs trous."
    )

    doc.add_heading("3.2 Méthodologie d'évaluation", level=2)
    doc.add_paragraph(
        "En l'absence de vraie valeur pour les points réellement manquants, l'évaluation masque "
        "artificiellement des valeurs connues puis compare la prédiction à la vraie valeur "
        "(MAE / RMSE / R²), selon deux scénarios : masquage aléatoire de 20 % des points connus "
        "(petits trous isolés) et masquage de segments entiers ≥ 20 mesures (simulation des "
        "vrais longs trous)."
    )

    doc.add_heading("3.3 Comparaison de 4 méthodes — masquage aléatoire", level=2)
    add_picture_centered(
        doc, "eda_output/vibration_imputation/01_metrics_random_masking.png",
        caption="Figure 3.1 — MAE / RMSE / R² par méthode (masquage aléatoire, 2 875 points)",
        description=(
            "Sur des trous courts et isolés, Random Forest obtient le meilleur score sur les "
            "trois métriques simultanément (MAE et RMSE les plus basses, R² le plus élevé), "
            "devant HistGradientBoosting, l'interpolation temporelle et KNN."
        ),
    )
    add_table(
        doc, ["Méthode", "MAE", "RMSE", "R²"], [
            ["Interpolation (mathématique)", 0.1569, 0.2390, 0.4268],
            ["Random Forest", 0.1366, 0.2190, 0.5186],
            ["HistGradientBoosting", 0.1413, 0.2243, 0.4951],
            ["K-Nearest Neighbors", 0.1457, 0.2410, 0.4171],
        ],
        caption="Tableau 3.1 — Métriques détaillées, masquage aléatoire (2 875 points)",
        note=(
            "Random Forest est meilleure sur les trois métriques à la fois (MAE, RMSE les plus "
            "basses, R² le plus élevé) — pas seulement en moyenne : c'est un gain cohérent, pas "
            "un compromis entre métriques. Meilleure méthode retenue pour ce régime : "
            "Random Forest (R² = 0.519)."
        ),
    )
    add_picture_centered(
        doc, "eda_output/vibration_imputation/02_scatter_random_masking.png",
        caption="Figure 3.2 — Valeurs prédites vs réelles, masquage aléatoire",
        description=(
            "Random Forest et HistGradientBoosting suivent la diagonale (identité prédit=réel) "
            "de façon relativement serrée pour les valeurs faibles à moyennes, alors que "
            "l'interpolation et surtout KNN dispersent davantage — confirmation visuelle du "
            "classement chiffré de la figure 3.1."
        ),
    )

    doc.add_heading("3.4 Test de robustesse — segments longs masqués", level=2)
    doc.add_paragraph(
        "690 points masqués via 15 segments de mesures connues ≥ 20 points, simulant les "
        "vrais grands trous du dataset."
    )
    add_picture_centered(
        doc, "eda_output/vibration_imputation/03_metrics_long_gaps.png",
        caption="Figure 3.3 — MAE / RMSE / R² par méthode (trous longs)",
        description=(
            "Les barres R² sont toutes négatives (axe passant sous zéro) : contrairement à la "
            "figure 3.1, aucune méthode ne bat une prédiction proche de la moyenne globale sur "
            "ce régime de trous longs. HistGradientBoosting reste la moins mauvaise des quatre."
        ),
    )
    add_table(
        doc, ["Méthode", "MAE", "RMSE", "R²"], [
            ["Interpolation", 0.3755, 0.5385, -0.1440],
            ["Random Forest", 0.3668, 0.5472, -0.1810],
            ["HistGradientBoosting", 0.3432, 0.5274, -0.0971],
            ["K-Nearest Neighbors", 0.3860, 0.5875, -0.3615],
        ],
        caption="Tableau 3.2 — Métriques détaillées, segments longs masqués (690 points)",
        note=(
            "Constat important : les 4 méthodes ont un R² négatif — aucune ne fait mieux qu'une "
            "prédiction proche de la moyenne globale quand le trou est long. Random Forest, "
            "pourtant meilleure sur le tableau 3.1, passe même dernière ex-æquo ici (R² = "
            "-0.181) : le classement d'une méthode dépend fortement du régime de trou visé. "
            "HistGradientBoosting est la « moins pire » (RMSE la plus basse) et sert de "
            "référence pour la suite."
        ),
    )
    add_picture_centered(
        doc, "eda_output/vibration_imputation/04_scatter_long_gaps.png",
        caption="Figure 3.4 — Valeurs prédites vs réelles, trous longs",
        description=(
            "Le nuage de points s'écarte nettement plus de la diagonale que sur la figure 3.2 "
            "pour les quatre méthodes, illustrant visuellement la perte de fiabilité générale "
            "sur ce régime de trous longs."
        ),
    )
    add_picture_centered(
        doc, "eda_output/vibration_imputation/05_gap_distance_vs_error.png",
        caption="Figure 3.5 — Dégradation de l'erreur selon la distance au point connu",
        description=(
            "Le panneau de droite (erreur moyenne par palier de distance) montre que l'erreur "
            "croît avec la distance au point connu le plus proche pour les quatre méthodes, y "
            "compris les modèles ML qui ne dépendent pourtant pas directement du temps — signe "
            "que les autres capteurs eux-mêmes perdent en pouvoir prédictif à mesure que le "
            "trou s'allonge, et pas seulement l'interpolation temporelle."
        ),
    )

    doc.add_heading("3.5 Proposition testée — médiane / moyenne conditionnelle", level=2)
    doc.add_paragraph(
        "Intuition testée : si aucune méthode ne bat une valeur « moyenne » sur les trous longs, "
        "autant utiliser directement une moyenne/médiane conditionnée par machine et état "
        "ON/OFF. Réévaluée sur exactement les mêmes 690 points masqués que le test précédent :"
    )
    add_picture_centered(
        doc, "eda_output/vibration_imputation/06_metrics_conditional_long_gaps.png",
        caption="Figure 3.6 — Médiane/moyenne conditionnelle vs référence ML",
        description=(
            "Réévalué sur exactement les mêmes points masqués que la figure 3.3, HistGradBoost "
            "(barre bleue, à droite) reste devant toutes les variantes de moyenne/médiane "
            "conditionnelle. La médiane fait systématiquement pire que la moyenne, et ajouter "
            "l'état ON/OFF au conditionnement par machine n'apporte quasiment rien."
        ),
    )
    add_table(
        doc, ["Méthode", "MAE", "RMSE", "R²"], [
            ["Médiane globale", 0.4633, 0.6775, -0.8106],
            ["Moyenne globale", 0.3996, 0.6178, -0.5054],
            ["Médiane cond. (machine)", 0.4589, 0.6722, -0.7824],
            ["Moyenne cond. (machine)", 0.3798, 0.5892, -0.3696],
            ["Médiane cond. (machine+état)", 0.4589, 0.6722, -0.7824],
            ["Moyenne cond. (machine+état)", 0.3797, 0.5891, -0.3691],
            ["HistGradBoost (référence)", 0.3432, 0.5274, -0.0971],
        ],
        caption="Tableau 3.3 — Médiane/moyenne (globale et conditionnelle) vs HistGradBoost, mêmes 690 points",
        note=(
            "Résultat : l'intuition ne se vérifie pas — la meilleure variante conditionnelle "
            "(moyenne, machine+état, R² = -0.369) reste nettement en retrait de HistGradBoost "
            "(R² = -0.097). Autre enseignement : la médiane fait systématiquement pire que la "
            "moyenne ici, et ajouter l'état ON/OFF au conditionnement par machine seule change "
            "à peine le résultat (-0.3696 → -0.3691). La proposition n'a pas été retenue."
        ),
    )

    doc.add_heading("3.6 Remplissage final retenu", level=2)
    doc.add_paragraph(
        "Décision finale, fondée sur les deux tests ci-dessus : utiliser Random Forest (meilleure "
        "méthode sur les trous courts) pour imputer uniquement les valeurs manquantes situées à "
        "moins de 20 mesures d'un point connu, et s'abstenir d'imputer au-delà (aucune méthode "
        "n'y est fiable)."
    )
    add_table(
        doc, ["Indicateur", "Valeur"], [
            ["Valeurs manquantes avant remplissage", "27 768 (65,9 %)"],
            ["Valeurs imputées (Random Forest, trous courts)", "14 737 (53,1 % des manquantes)"],
            ["Valeurs laissées NaN (trous ≥ 20 mesures)", "13 031 (46,9 % des manquantes)"],
            ["Valeurs manquantes après remplissage", "13 031 (30,9 % du dataset)"],
        ],
        caption="Tableau 3.4 — Bilan du remplissage final de Oil System Vibration",
        note=(
            "Un peu plus de la moitié des valeurs manquantes (53,1 %) sont comblées ; le reste "
            "(46,9 %) est laissé volontairement en NaN plutôt que rempli avec une méthode dont "
            "la fiabilité n'a pas pu être démontrée (tableau 3.2). Au global, 30,9 % du dataset "
            "reste donc en NaN sur cette variable — un compromis assumé entre complétude et "
            "fiabilité."
        ),
    )
    add_table(
        doc, ["Confiance", "Distance au point connu", "Nb. lignes", "% des valeurs imputées"], [
            ["Haute", "0-3 mesures", 9315, "63,2 %"],
            ["Moyenne", "3-12 mesures", 4304, "29,2 %"],
            ["Basse", "12-20 mesures", 1118, "7,6 %"],
        ],
        caption="Tableau 3.5 — Répartition de la confiance des valeurs imputées",
        note=(
            "Près des deux tiers (63,2 %) des valeurs imputées sont à moins de 3 mesures d'un "
            "point connu (confiance haute, erreur attendue ~0,22-0,30) — le remplissage est "
            "donc concentré sur les cas les plus favorables. Cette confiance est aussi exposée "
            "comme colonne (vibration_confidence) pour un usage direct en modélisation."
        ),
    )
    doc.add_paragraph(
        "Colonnes ajoutées au dataset : Oil System Vibration_filled, vibration_source, "
        "vibration_gap_distance, vibration_confidence. La colonne d'origine (avec ses NaN) est "
        "conservée pour traçabilité."
    )


def add_sensor_section(doc):
    doc.add_page_break()
    doc.add_heading("4. Imputation du trou capteur synchrone (7 variables)", level=1)

    doc.add_heading("4.1 Découverte du trou", level=2)
    doc.add_paragraph(
        "Les 2,36 % de NaN restants sur 8 variables (hors vibration) avaient d'abord été "
        "supposés être de petites désynchronisations ponctuelles (comme pour ISO 6 / ISO 14, "
        "déjà comblées par forward-fill lors du nettoyage). L'investigation a montré qu'il "
        "s'agit en réalité d'un trou contigu unique et bien plus long : 478 mesures consécutives "
        "pour Motosoufflante A (01→22 juillet 2026) et 516 pour Motosoufflante B "
        "(25→29 juillet 2026)."
    )
    doc.add_paragraph(
        "7 variables tombent en panne exactement ensemble sur ces lignes : DC, Dynamic "
        "Viscosity, ISO 4, Oil H2O Saturation, Oil H2O ppm, Oil Temperature, Viscosity at 40°C. "
        "En revanche, ISO 6, ISO 14, Oil Conductivity et Oil Pressure restent disponibles en "
        "continu pendant ce même trou — un modèle multivarié peut donc s'appuyer sur ces "
        "capteurs encore actifs comme prédicteurs."
    )

    doc.add_heading("4.2 Méthodologie d'évaluation", level=2)
    doc.add_paragraph(
        "Un segment synthétique de la même longueur que le vrai trou (478 / 516 mesures) est "
        "masqué dans la plus longue plage sans NaN du dataset (19 266 et 21 361 mesures "
        "disponibles), pour chaque machine, puis 7 méthodes sont comparées sur ce segment : "
        "forward-fill, interpolation, moyenne globale, médiane globale, moyenne conditionnelle "
        "(machine × état), médiane conditionnelle, et Random Forest (prédicteurs : ISO 6, "
        "ISO 14, Oil Conductivity, Oil Pressure, Oil System Vibration_filled, heure, machine, "
        "état)."
    )

    doc.add_heading("4.3 Résultats agrégés (moyenne sur les 7 variables)", level=2)
    add_picture_centered(
        doc, "eda_output/sensor_gap_imputation/01_metrics_aggregate.png",
        caption="Figure 4.1 — Comparaison agrégée des 7 méthodes",
        description=(
            "Résultat inverse de la section 3 : ici, forward-fill (barre grise, à gauche) "
            "domine largement, y compris devant Random Forest. La moyenne et la médiane "
            "(globales ou conditionnelles) restent nettement moins bonnes, avec des R² "
            "négatifs — ce trou n'est pas un cas où « prédire une valeur moyenne » suffit."
        ),
    )
    add_table(
        doc, ["Méthode", "R² moyen", "RMSE normalisée moyenne"], [
            ["Forward-fill", 0.783, 0.149],
            ["Interpolation", 0.685, 0.194],
            ["Moyenne globale", -1.008, 0.883],
            ["Médiane globale", -0.969, 0.960],
            ["Moyenne conditionnelle", -0.183, 0.402],
            ["Médiane conditionnelle", -0.312, 0.389],
            ["Random Forest", 0.622, 0.273],
        ],
        caption="Tableau 4.1 — R² moyen et RMSE normalisée (RMSE / écart-type), moyennés sur les 7 variables",
        note=(
            "La RMSE est normalisée par l'écart-type de chaque variable pour rendre les échelles "
            "comparables (ex. Oil H2O ppm ~20-120 vs DC ~1-2). Résultat contre-intuitif : le "
            "forward-fill l'emporte largement en moyenne, y compris devant Random Forest — la "
            "machine reste dans un régime suffisamment stable sur ce trou pour que « figer la "
            "dernière valeur connue » soit une bonne approximation. Voir toutefois le tableau "
            "4.2 pour l'exception notable."
        ),
    )

    doc.add_heading("4.4 Résultats détaillés par variable", level=2)
    add_picture_centered(
        doc, "eda_output/sensor_gap_imputation/02_r2_by_variable.png",
        caption="Figure 4.2 — R² par méthode et par variable",
        description=(
            "La moyenne agrégée de la figure 4.1 masque une exception : sur ISO 4 (3ᵉ groupe de "
            "barres), forward-fill et interpolation chutent près de zéro alors que Random "
            "Forest (barre orange) reste la seule méthode franchement positive. C'est ce "
            "constat variable-par-variable qui justifie la méthode hybride retenue en 4.5 "
            "plutôt qu'un choix unique pour les 7 variables."
        ),
    )
    add_table(
        doc, ["Variable", "Meilleure méthode", "R²"], [
            ["DC", "Forward-fill / Interpolation / Random Forest (ex-æquo)", "~1.0"],
            ["Dynamic Viscosity", "Forward-fill / Interpolation / Random Forest", "~1.0"],
            ["ISO 4", "Random Forest", "0.66"],
            ["Oil H2O Saturation", "Forward-fill", "0.96"],
            ["Oil H2O ppm", "Médiane conditionnelle", "0.89 (Forward-fill : 0.83)"],
            ["Oil Temperature", "Forward-fill", "0.76"],
            ["Viscosity at 40°C", "Forward-fill", "~1.0"],
        ],
        caption="Tableau 4.2 — Meilleure méthode et R² correspondant, variable par variable",
        note=(
            "ISO 4 est la seule variable où forward-fill et interpolation échouent nettement "
            "(R² proche de 0) : Random Forest, qui s'appuie sur ISO 6 / ISO 14 (fortement "
            "corrélées, cf. tableau 2.2) restées disponibles pendant le trou, comble ce point "
            "faible (R² = 0,66). À l'inverse, sur Oil Temperature, Random Forest est moins bon "
            "que forward-fill (R² négatif contre 0,76) — aucune méthode unique ne domine sur "
            "toutes les variables, d'où le choix d'une méthode hybride (section 4.5)."
        ),
    )

    doc.add_heading("4.5 Méthode hybride retenue et remplissage final", level=2)
    doc.add_paragraph(
        "Méthode hybride, par variable plutôt qu'un choix unique pour les 7 : forward-fill pour "
        "DC, Dynamic Viscosity, Oil H2O Saturation, Oil H2O ppm, Oil Temperature et Viscosity at "
        "40°C ; Random Forest uniquement pour ISO 4."
    )
    add_table(
        doc, ["Indicateur", "Avant", "Après"], [
            ["NaN total sur les 7 variables (994 lignes × 7)", "6 958", "0"],
            ["Lignes concernées (trou synchrone)", "994 (2,36 % du dataset)", "—"],
        ],
        caption="Tableau 4.3 — Bilan du remplissage final du trou capteur synchrone",
        note=(
            "Contrairement à Oil System Vibration (tableau 3.4, 30,9 % de NaN restants "
            "assumés), ce trou est comblé intégralement (0 NaN restant) car les tests montrent "
            "un R² fiable (0,65 à 1,0 selon la variable) pour l'ensemble des 7 variables — pas "
            "de raison de s'abstenir comme pour les longs trous de vibration."
        ),
    )
    doc.add_paragraph(
        "Colonnes ajoutées : DC_filled, Dynamic Viscosity_filled, ISO 4_filled, "
        "Oil H2O Saturation_filled, Oil H2O ppm_filled, Oil Temperature_filled, "
        "Viscosity at 40°C_filled, et un flag partagé sensor_gap_source "
        "(mesuré / comblé). Les colonnes d'origine sont conservées pour traçabilité."
    )


def add_conclusion(doc):
    doc.add_page_break()
    doc.add_heading("5. Synthèse", level=1)
    add_table(
        doc, ["Étape", "Résultat"], [
            ["EDA", "42 141 lignes analysées, 2 redondances identifiées (Density, Kinematic "
                    "Viscosity), Machine B arrêtée en continu depuis fin janvier 2026, 352 "
                    "sessions continues"],
            ["Imputation vibration", "53,1 % des NaN comblés (Random Forest, trous courts < 20 "
                                      "mesures) ; 46,9 % laissés NaN par choix méthodologique "
                                      "(aucune méthode fiable sur les longs trous)"],
            ["Imputation capteurs", "100 % des 6 958 NaN comblés (forward-fill pour 6 "
                                     "variables, Random Forest pour ISO 4)"],
        ],
        caption="Tableau 5.1 — Vue d'ensemble des trois étapes couvertes par ce rapport",
        note=(
            "Ce tableau récapitule en une ligne par étape les chiffres clés détaillés dans les "
            "sections 2 à 4. Le contraste entre les deux imputations (vibration : remplissage "
            "partiel assumé, capteurs : remplissage total) illustre le principe directeur du "
            "pipeline — la méthode et le niveau de remplissage sont choisis au cas par cas en "
            "fonction de ce que les tests démontrent, pas d'une règle unique appliquée "
            "partout."
        ),
    )


def main():
    doc = Document()

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    add_title_page(doc)
    add_intro(doc)
    add_eda_section(doc)
    add_vibration_section(doc)
    add_sensor_section(doc)
    add_conclusion(doc)

    doc.save("Rapport_EDA_Imputation_iSENSE.docx")
    print("Généré : Rapport_EDA_Imputation_iSENSE.docx")


if __name__ == "__main__":
    main()
