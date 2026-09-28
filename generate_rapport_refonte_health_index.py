"""
Rapport Word de synthese couvrant les 4 grands chantiers menes depuis la
refonte physico-chimique du pipeline :
  1. Plan de refonte du pipeline (approche physico-chimique)
  2. Data Quality (Phase 0)
  3. Feature engineering (imputation vibration/capteurs + features)
  4. Plan de construction du Health Index (Phases 1 a 5)

Reutilise les images deja generees par les scripts du projet (aucun calcul
recree ici, uniquement mise en forme et redaction).

Sortie : Rapport_Refonte_PhysicoChimique_HealthIndex_iSENSE.docx
"""

import os

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

IMG_W = Inches(6.3)
IMG_W_SMALL = Inches(4.6)


def add_picture_centered(doc, path, width=IMG_W, caption=None):
    if not os.path.exists(path):
        p = doc.add_paragraph()
        r = p.add_run(f"[Image manquante : {path}]")
        r.italic = True
        r.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
        return
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
        cap.paragraph_format.space_after = Pt(12)


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


def bullets(doc, items, style="List Bullet"):
    for it in items:
        doc.add_paragraph(it, style=style)


def paragraph(doc, text, size=11, italic=False, bold=False, color=None, space_after=8):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.italic = italic
    r.bold = bold
    if color:
        r.font.color.rgb = color
    p.paragraph_format.space_after = Pt(space_after)
    return p


def note(doc, text):
    paragraph(doc, text, size=10, italic=True, color=RGBColor(0x40, 0x40, 0x40))


# ===========================================================================
# TITLE PAGE
# ===========================================================================

def add_title_page(doc):
    title = doc.add_heading("Refonte physico-chimique du pipeline", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title2 = doc.add_heading("et construction du Health Index — i-SENSE", level=0)
    title2.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = sub.add_run(
        "De l'extraction API au dashboard : audit Data Quality physico-chimique, nettoyage "
        "révisé, imputation hybride, feature engineering, Health Index à seuils officiels OCP "
        "(règles, PCA, Isolation Forest), et prédiction à t+n (5 modèles comparés)."
    )
    run.italic = True
    run.font.size = Pt(13)
    doc.add_paragraph()

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.add_run(
        "Projet i-SENSE — Monitoring qualité huile de lubrification\n"
        "Motosoufflante A & B — OCP / UM6P"
    ).italic = True
    doc.add_page_break()


def add_toc_page(doc):
    doc.add_heading("Sommaire", level=1)
    parts = [
        "Partie 0 — Vue d'ensemble du pipeline (de l'API au dashboard)",
        "Partie 1 — Plan de refonte du pipeline (approche physico-chimique)",
        "Partie 2 — Data Quality physico-chimique (Phase 0)",
        "Partie 3 — Feature engineering (imputation vibration/capteurs + features)",
        "Partie 4 — Construction du Health Index (Phases 1 à 5)",
        "Partie 5 — Prédiction du Health Index à t+n",
        "Conclusion et points ouverts",
    ]
    bullets(doc, parts, style="List Number")
    doc.add_page_break()


# ===========================================================================
# PARTIE 0 — VUE D'ENSEMBLE DU PIPELINE (DE L'API AU DASHBOARD)
# ===========================================================================

def part0_vue_ensemble(doc):
    doc.add_heading("Partie 0 — Vue d'ensemble du pipeline (de l'API au dashboard)", level=1)
    paragraph(doc,
        "Chaîne complète du projet, de l'extraction brute des mesures i-SENSE jusqu'à "
        "l'affichage du Health Index — chaque étape est détaillée dans les parties suivantes."
    )
    add_table(doc,
        ["#", "Étape", "Entrée → Sortie", "Rôle"],
        [
            ["1", "Extraction API (`Api_to_excel.py`)", "API i-SENSE → `isense_oil_data_combined_wide.csv`",
             "Récupération des mesures brutes, mise en forme large (1 ligne = 1 timestamp × machine)"],
            ["2", "Data Quality (`data_quality_isense.py`)", "Brut → rapport de diagnostic",
             "Audit physico-chimique du brut, sans modification (Partie 2)"],
            ["3", "Nettoyage (`clean_isense_data.py`)", "Brut → `isense_oil_data_cleaned.csv`",
             "Sentinelles → NaN, machine_state, sessions (Partie 1)"],
            ["4", "Imputation vibration (`fill_vibration.py`)", "→ `isense_oil_data_vibration_filled.csv`",
             "Hybride OFF→0 / ON→Random Forest (Partie 3.1)"],
            ["5", "Imputation capteurs (`fill_sensor_gap.py`)", "→ `isense_oil_data_sensors_filled.csv`",
             "Trou synchrone, 11 variables (Partie 3.2)"],
            ["6", "EDA (`eda_isense.py`)", "→ `rapport_eda.md`", "Corrélations conditionnelles, indices physico-chimiques"],
            ["7", "Feature engineering (`feature_engineering.py`)", "→ `isense_oil_data_features.csv`",
             "150 colonnes, 12 familles de features (Partie 3.3)"],
            ["8", "Health Index (`health_index_*.py`)", "→ `isense_oil_data_health_index.csv`",
             "5 phases : règles, PCA, Isolation Forest, décision, combinaison (Partie 4)"],
            ["9", "Prédiction à t+n (`health_index_prediction_*.py`)", "→ rapports de comparaison",
             "5 modèles, horizons 10 min à 1 semaine (Partie 5)"],
            ["10", "Dashboard / Visualisation", "`health_index`, `health_state` → écran de pilotage",
             "Affichage temps réel de l'état Normal/Surveillance/Alarme par machine — compatible "
             "avec le type de plateforme déjà utilisé par i-SENSE (mesures, seuils, historique)"],
        ]
    )
    note(doc,
        "Le dashboard n'est pas développé dans ce projet : les colonnes `health_index` "
        "(continue, 0-1) et `health_state` (Normal/Surveillance/Alarme) sont conçues pour être "
        "branchées directement sur une plateforme de supervision (temps réel ou historique), à "
        "l'image de celle déjà opérée par i-SENSE pour les mesures brutes."
    )
    doc.add_page_break()


# ===========================================================================
# PARTIE 1 — PLAN DE REFONTE
# ===========================================================================

def part1_plan_refonte(doc):
    doc.add_heading("Partie 1 — Plan de refonte du pipeline (approche physico-chimique)", level=1)
    paragraph(doc,
        "Le pipeline initial excluait certaines variables (Kinematic Viscosity, Density) sur la "
        "seule base de corrélations élevées avec d'autres capteurs, et envisageait de remplacer "
        "toutes les valeurs manquantes de vibration par zéro. Cette partie résume la vérification "
        "de ces deux hypothèses sur les données réelles, et le plan de refonte qui en a découlé."
    )

    doc.add_heading("1.1 — Hypothèse « corrélation = redondance » : invalidée", level=2)
    paragraph(doc,
        "Le pipeline initial avait supprimé `Kinematic Viscosity` (corrélée à 0,86 avec "
        "`Viscosity at 40°C`) et exclu `Density` (corrélée à −1,00 avec `Oil Temperature`), en "
        "présumant que deux variables très corrélées sont redondantes. Raisonnement invalide dès "
        "lors que la corrélation provient d'une **loi physique connue** : la loi ne supprime pas "
        "l'information, elle la relie. Trois vérifications directes sur les données ont tranché "
        "la question :"
    )
    add_table(doc,
        ["Vérification", "Résultat", "Conclusion"],
        [
            ["μ = ν × ρ (viscosité dynamique = cinématique × densité, loi de mécanique des "
             "fluides)",
             "Erreur relative médiane 0,01 %, 0 violation sur 41 147 lignes",
             "Dynamic Viscosity est calculée en interne à partir de Kinematic Viscosity et "
             "Density, pas mesurée indépendamment — la relation devient un test de cohérence "
             "capteur réutilisable en continu"],
            ["Densité normalisée à 15 °C (ASTM D1298, retire l'effet de la température de "
             "mesure)",
             "Quasi constante : 869 kg/m³, écart-type 0,17, sur 8 mois et les 2 machines",
             "Même huile de base, aucune contamination/dilution majeure sur la période — fournit "
             "la ligne de base de density_deviation_pct"],
            ["Ratio Viscosity@40°C / Kinematic Viscosity vs Température (devrait être "
             "déterministe si redondant)",
             "Corrélation élevée mais imparfaite : 0,989 (A) vs 0,792 (B)",
             "Kinematic Viscosity porte une information non déductible de la seule température — "
             "le résidu du ratio devient vi_proxy, proxy de l'Indice de Viscosité (ASTM D2270)"],
        ],
        note="→ Conséquence sur le pipeline : Kinematic Viscosity et Density sont réintégrées et "
             "transformées en indices métier normalisés (density_15C, density_deviation_pct, "
             "vi_proxy) plutôt que supprimées ou laissées brutes — cf. Partie 3.3, famille "
             "« Indices physico-chimiques »."
    )
    add_picture_centered(doc, "eda_output/data_quality/01_viscosity_law_check.png",
        caption="Vérification de la loi μ = ν × ρ sur les données réelles.")
    add_picture_centered(doc, "eda_output/data_quality/03_density_15C_stability.png",
        caption="Stabilité de la densité normalisée à 15°C sur 8 mois, par machine.")

    doc.add_heading("1.2 — Hypothèse « vibration NaN = 0 » : résolution en 2 temps", level=2)
    paragraph(doc,
        "L'hypothèse de départ était simple : un `NaN` sur `Oil System Vibration` signifierait "
        "que le système ne vibre pas, et pourrait donc être remplacé directement par 0. Une "
        "première analyse en trois tests, comparant les `NaN` à l'état de la machine (ON/OFF, "
        "dérivé de `Oil Pressure`), a d'abord semblé la contredire :"
    )
    add_table(doc,
        ["Test", "Constat"],
        [
            ["A — Répartition des NaN par état", "63,3 % de NaN en état ON contre 68,4 % en état OFF — quasiment identique. Si les NaN signifiaient « pas de vibration », ils seraient concentrés sur l'état OFF, or 13 145 lignes ont la machine en marche avec une vibration NaN."],
            ["B — Transition observée en fonctionnement continu", "Le 29/12/2025, Motosoufflante A tourne à pression stable (~6 Bar) pendant 5h avec vibration NaN, puis le capteur se met soudainement à reporter des valeurs (0,28 à 1,16 mm/s²) sans aucun changement d'état machine détecté."],
            ["C — Le capteur ne mesure jamais exactement 0", "Sur l'ensemble de la colonne, aucune valeur mesurée n'est exactement 0 (minimum observé : 0,01 mm/s², y compris à l'arrêt, vibration ambiante transmise par la structure)."],
        ]
    )
    paragraph(doc,
        "Cette analyse utilisait cependant `machine_state` (dérivé d'un capteur différent, "
        "`Oil Pressure`) comme **preuve indirecte** du contenu du capteur de vibration — un "
        "raisonnement qui mélangeait deux mesures physiques distinctes. Le spécialiste métier a "
        "tranché directement, sans attendre une confirmation supplémentaire de l'équipe i-SENSE : "
        "à l'arrêt, l'absence de vibration mécanique est un fait physique (remplacement par 0 "
        "justifié indépendamment des tests ci-dessus, qui ne portaient que sur des cas machine en "
        "marche) ; en marche, le `NaN` reste une vraie valeur manquante à estimer, jamais à "
        "annuler. La décision finale (approche hybride OFF→0 / ON→Random Forest, validée par "
        "comparaison de 8 méthodes) est détaillée en Partie 3.1."
    )
    note(doc,
        "Point méthodologique retenu pour la suite du projet : `machine_state` peut servir à "
        "**router** une ligne vers le bon traitement (ici : remplacement direct vs prédiction), "
        "mais ne doit jamais servir de **preuve indirecte** du contenu d'un autre capteur — "
        "l'erreur identifiée dans les tests A/B/C ci-dessus."
    )
    add_picture_centered(doc, "eda_output/08_conditional_correlation.png",
        caption="Corrélation DC / Oil Pressure : globale (r=0,942) vs conditionnée par machine_state "
                "(r≈0,3) — preuve qu'un effet de confusion peut masquer la vraie relation physique.")

    doc.add_heading("1.3 — Vue d'ensemble du pipeline révisé", level=2)
    add_table(doc,
        ["Phase", "Script", "Rôle"],
        [
            ["Phase 0 (nouvelle)", "data_quality_isense.py", "Audit physico-chimique du dataset BRUT, aucune modification"],
            ["Phase 1", "clean_isense_data.py", "Nettoyage révisé : plus de suppression de variable par corrélation"],
            ["Phase 2", "fill_vibration.py / fill_sensor_gap.py", "Imputation hybride vibration + trou capteur synchrone"],
            ["Phase 3", "eda_isense.py", "EDA incluant corrélations conditionnelles et indices physico-chimiques"],
            ["Phase 4", "feature_engineering.py", "150 colonnes, 12 familles de features"],
            ["Validation", "eda_post_imputation.py", "Vérification finale : 0 NaN structurel, corrélations physiques stables"],
        ]
    )
    add_picture_centered(doc, "eda_output/09_physicochemical_indices.png",
        caption="Indices physico-chimiques dérivés (density_15C, vi_proxy) suivis dans le temps.")

    doc.add_heading("1.4 — Découverte du 2026-09-01 : sentinelle et unité de `Oil Conductivity`", level=2)
    paragraph(doc,
        "En croisant les seuils officiels OCP (`Cadrage des seuils_Système d'huile.pdf`) avec les "
        "données réelles, deux problèmes non détectés jusqu'ici ont été trouvés sur "
        "`Oil Conductivity` :"
    )
    bullets(doc, [
        "Une sentinelle non standard (-0,09999, différente du -99,99 générique des autres "
        "colonnes) touchait les 994 lignes du trou capteur synchrone déjà connu — non détectée "
        "par le test générique `< -9`, donc traitée comme une vraie mesure depuis le début du "
        "projet. Corrigée dans `clean_isense_data.py`, avec un bug de régression intermédiaire "
        "également corrigé (994 lignes légitimes presque supprimées par erreur avant la "
        "correction finale).",
        "Une unité incompatible avec les seuils officiels (nS/m) : les valeurs réelles se "
        "regroupent entre 0,062 et 0,072. Analyse : un facteur ×10 les ramène exactement dans "
        "la plage « Normal » officielle (0,4-2,0 nS/m) — nouvelle colonne `Oil Conductivity_nSm`, "
        "à confirmer avec l'équipe i-SENSE.",
    ])
    note(doc,
        "Conséquence : le trou capteur synchrone passe de 9 à 11 variables comblées "
        "(`Oil Conductivity` et `Oil Conductivity_nSm` ajoutées), cf. Partie 3.2."
    )


# ===========================================================================
# PARTIE 2 — DATA QUALITY
# ===========================================================================

def part2_data_quality(doc):
    doc.add_page_break()
    doc.add_heading("Partie 2 — Data Quality physico-chimique (Phase 0)", level=1)
    paragraph(doc,
        "Avant tout nettoyage, `data_quality_isense.py` audite le dataset BRUT à partir de lois "
        "physico-chimiques connues. Ce script ne modifie aucune donnée : il produit un rapport et "
        "des graphiques de diagnostic, utilisés pour orienter les décisions de nettoyage (Partie 1)."
    )

    add_table(doc,
        ["#", "Contrôle", "Loi / règle", "Résultat"],
        [
            ["1", "Cohérence μ = ν × ρ", "Viscosité dynamique = cinématique × densité", "✅ 0 violation, erreur 0,01 %"],
            ["2", "Monotonie ISO 4406", "ISO 4 ≥ ISO 6 ≥ ISO 14", "✅ 0 violation sur 39 988 lignes"],
            ["3", "Densité normalisée 15 °C", "ASTM D1298, α = 0,00075/°C", "✅ Quasi constante (869 kg/m³ ± 0,17)"],
            ["4", "Cohérence H₂O ppm ↔ saturation", "ppm_absolu ≈ %saturation × ppm_saturation", "✅ r = 0,816"],
            ["5", "Plages physiques admissibles", "Bornes constructeur par variable", "⚠️ Anomalie DC détectée"],
        ],
        caption="5 contrôles effectués sur le dataset brut (rapport_data_quality.md)."
    )

    doc.add_heading("Découverte majeure : anomalie de calibration DC sur Motosoufflante B", level=2)
    paragraph(doc,
        "Le contrôle 5 (plages physiques) a révélé que 88,4 % des mesures de constante "
        "diélectrique (DC) de Motosoufflante B sont physiquement impossibles (ε_r < 1, ce qui "
        "n'existe pour aucun matériau réel). Motosoufflante A ne présente jamais ce problème "
        "(minimum observé : 1,010). Ceci suggère un biais de calibration du capteur DC sur "
        "Machine B — non corrigé (décision volontaire, en attente de confirmation de l'équipe "
        "i-SENSE), documenté et exclu du Health Index par précaution."
    )
    add_picture_centered(doc, "eda_output/data_quality/05_dc_physical_bound.png",
        caption="Distribution de DC par machine — 88,4 % des valeurs de B sont physiquement impossibles.")

    doc.add_heading("Mise à jour du 2026-09-01 : bornes officielles OCP", level=2)
    paragraph(doc,
        "Les bornes physiques génériques initiales ont été remplacées par les bornes "
        "« Impossible » officielles (`Cadrage des seuils_Système d'huile.pdf`), plus strictes :"
    )
    add_table(doc,
        ["Variable", "Ancienne borne", "Nouvelle borne officielle"],
        [
            ["DC", "(1, 80)", "(1,8, 10)"],
            ["Density", "(600, 1300)", "(700, 1000)"],
            ["Oil Temperature", "(−40, 130)", "(0, 150)"],
            ["ISO 4 / 6 / 14", "(0, 30)", "(0, 28)"],
        ],
        note="Conséquence : l'anomalie DC de Motosoufflante B est confirmée et renforcée — avec "
             "la borne resserrée à 1,8, une part encore plus grande de ses mesures (moyenne "
             "~1,03) tombe sous le seuil « impossible »."
    )

    doc.add_heading("Création de `machine_state` sur les données brutes", level=2)
    paragraph(doc,
        "Le flag ON/OFF (dérivé de Oil Pressure > 0,1 Bar) est calculé dès la Phase 0 pour "
        "permettre l'analyse sur les données brutes, avec le même seuil que le nettoyage."
    )
    add_table(doc,
        ["Machine", "ON", "OFF", "% OFF"],
        [
            ["Motosoufflante A", "18 562", "1 335", "6,7 %"],
            ["Motosoufflante B", "2 204", "20 041", "90,1 %"],
        ]
    )
    add_picture_centered(doc, "eda_output/data_quality/00_machine_state.png",
        caption="Répartition ON/OFF par machine sur le dataset brut.")

    add_picture_centered(doc, "eda_output/data_quality/04_h2o_consistency.png",
        caption="Cohérence entre Oil H2O ppm et Oil H2O Saturation.")


# ===========================================================================
# PARTIE 3 — FEATURE ENGINEERING
# ===========================================================================

def part3_feature_engineering(doc):
    doc.add_page_break()
    doc.add_heading("Partie 3 — Feature engineering", level=1)
    paragraph(doc,
        "Avant de calculer les features, deux imputations restaient nécessaires : la vibration "
        "(65,9 % de valeurs manquantes) et un trou capteur synchrone (994 mesures, 11 variables "
        "depuis l'ajout de `Oil Conductivity`/`Oil Conductivity_nSm`, cf. Partie 1.4)."
    )

    doc.add_heading("3.1 — Imputation hybride de `Oil System Vibration`", level=2)
    paragraph(doc,
        "Décision finale : à l'arrêt (machine_state = OFF), remplacement direct par 0 (décision "
        "métier — la machine ne vibre pas). En marche (ON), la valeur est prédite, après "
        "comparaison chiffrée de 8 méthodes sur les 7 621 valeurs ON déjà connues (masquage de "
        "20 %, comparaison MAE/RMSE/R²)."
    )
    add_table(doc,
        ["Méthode", "RMSE", "R²"],
        [
            ["Random Forest (retenue)", "0,302", "0,438"],
            ["XGBoost", "0,307", "0,422"],
            ["KNN", "0,325", "0,351"],
            ["Interpolation temporelle", "0,331", "0,327"],
            ["Moyenne / médiane (globale ou conditionnelle)", "0,40 – 0,44", "≤ 0"],
        ],
        note="machine_state sert uniquement à router chaque NaN vers le bon traitement (0 ou "
             "prédiction), jamais à estimer la valeur elle-même."
    )
    add_picture_centered(doc, "eda_output/vibration_on_state/01_metrics_comparison.png",
        caption="Comparaison des 8 méthodes pour la vibration en état ON (MAE/RMSE/R²).")
    add_picture_centered(doc, "eda_output/vibration_on_state/02_scatter_grid.png",
        caption="Valeurs prédites vs réelles pour les 8 méthodes comparées.", width=IMG_W)

    add_table(doc,
        ["Sous-ensemble", "Traitement", "Volume"],
        [
            ["OFF & NaN", "Remplacement par 0", "14 623 (52,7 %)"],
            ["ON & NaN", "Prédiction Random Forest", "13 145 (47,3 %)"],
        ],
        caption="Résultat final : 0 NaN résiduel dans Oil System Vibration_filled."
    )

    doc.add_heading("3.2 — Trou capteur synchrone (994 mesures, 11 variables)", level=2)
    paragraph(doc,
        "10 variables comblées par forward-fill (DC, Density, Dynamic/Kinematic Viscosity, Oil H2O "
        "Saturation/ppm, Oil Temperature, Viscosity at 40°C, Oil Conductivity, "
        "Oil Conductivity_nSm), 1 variable (ISO 4) comblée par Random Forest à partir des autres "
        "codes ISO 4406 et capteurs disponibles."
    )
    add_picture_centered(doc, "eda_output/sensor_gap_imputation/01_metrics_aggregate.png",
        caption="Performance agrégée de l'imputation du trou capteur synchrone.")

    doc.add_heading("3.3 — Les 12 familles de features (`feature_engineering.py`)", level=2)
    paragraph(doc,
        "Le dataset suit une logique proche du dataset NASA C-MAPSS (mentionné dans l'audit "
        "documentaire) utilisé pour la prédiction de dégradation — les familles ci-dessous "
        "s'inspirent des pratiques standards en maintenance prédictive sur séries temporelles de "
        "capteurs. Chacune répond à un besoin précis, détaillé ci-dessous."
    )
    add_table(doc,
        ["#", "Famille", "Exemple de colonnes", "À quoi ça sert"],
        [
            ["1", "Calendaires", "hour_sin, hour_cos, day_of_week",
             "Encodage cyclique de l'heure (23h proche de 0h) ; capture un éventuel rythme journalier."],
            ["2", "Session", "time_in_session_h, measure_index_in_session",
             "Temps depuis le redémarrage — capture le transitoire thermique après un arrêt."],
            ["3", "Indices physico-chimiques", "density_15C, vi_proxy, density_deviation_pct",
             "Composition d'huile normalisée (densité hors effet température, proxy Indice de Viscosité)."],
            ["4", "Tendance", "..._diff1, ..._slope_3h",
             "Vitesse de dégradation — souvent plus informative que la valeur absolue."],
            ["5", "Lags", "..._lag1, ..._lag2, ..._lag3",
             "Mémoire du passé récent pour les modèles non récurrents (RF, XGBoost)."],
            ["6", "EWMA", "..._ewma",
             "Moyenne lissée réactive, plus pondérée sur les mesures récentes."],
            ["7", "Indices métier", "contamination_index, viscosity_grade_gap, temp_viscosity_interaction",
             "Connaissance métier encodée directement (synthèse ISO, écart au grade, interaction physique)."],
            ["8", "Z-score par machine", "..._zscore",
             "Anomalie relative à la ligne de base propre de chaque machine (A ≠ B)."],
            ["9", "Flags de seuils", "flag_high_contamination, flag_high_water, flag_high_temperature, flag_high_vibration",
             "Seuils d'alerte métier en variables binaires exploitables directement."],
            ["10", "Confiance vibration", "vibration_confidence_score, vibsource_*",
             "Distingue mesuré / imputé à 0 / prédit, sans filtrer les lignes."],
            ["11", "Confiance capteur", "sensor_gap_filled",
             "Distingue les 994 lignes reconstruites du trou synchrone des mesures réelles."],
            ["12", "Encodage catégoriel", "asset_*, state_*",
             "Entrées numériques requises par la plupart des modèles ML."],
        ],
        note="36 → 150 colonnes (114 features ajoutées). Une 13e famille (fenêtres glissantes, "
             "78 colonnes) a été retirée après un test d'importance de features l'ayant jugée non "
             "prioritaire — confirmé empiriquement en Partie 5 (nuit à la prédiction)."
    )
    add_picture_centered(doc, "eda_output/report_charts/calendar_cyclic_encoding.png",
        caption="Encodage cyclique de l'heure (hour_sin / hour_cos).", width=IMG_W_SMALL)
    add_picture_centered(doc, "eda_output/report_charts/session_features_distribution.png",
        caption="Distribution des features de session.")

    doc.add_heading("3.4 — Importance des features (exploratoire)", level=2)
    paragraph(doc,
        "Classement par Random Forest, en utilisant `contamination_index` comme proxy de "
        "dégradation (aucun Health Index n'existait encore à ce stade) — à titre indicatif "
        "uniquement, un vrai classement se fera une fois le Health Index final disponible "
        "(Partie 4)."
    )
    add_picture_centered(doc, "eda_output/feature_importance/01_top_features.png",
        caption="Top 25 features par importance (proxy contamination_index).")


# ===========================================================================
# PARTIE 4 — HEALTH INDEX
# ===========================================================================

def part4_health_index(doc):
    doc.add_page_break()
    doc.add_heading("Partie 4 — Construction du Health Index (Phases 1 à 5)", level=1)
    paragraph(doc,
        "Objectif : construire un score continu de santé de l'huile (0-1) et un état catégoriel "
        "(Normal / Surveillance / Alarme), en l'absence de tout label de panne connu. Approche en "
        "couches, de la plus simple à la plus complexe, avec un critère explicite de décision "
        "avant d'engager une méthode plus coûteuse."
    )

    doc.add_heading("Phase 1 — Règles métier à seuils officiels (`health_index_baseline.py`)", level=2)
    paragraph(doc,
        "Révisée le 2026-09-01 : utilise désormais les seuils officiels OCP "
        "(`Cadrage des seuils_Système d'huile.pdf`) sur **12 variables** plutôt que 4 seuils "
        "empiriques par percentile — la justification initiale de la Phase 1 (« s'appuyer sur des "
        "seuils déjà établis ») s'applique désormais à la quasi-totalité des variables santé : "
        "température, saturation H₂O, DC, conductivité (nS/m), ISO 4/6/14, H₂O ppm, densité, "
        "viscosités cinématique/dynamique/à 40°C. Chaque variable est convertie en une distance "
        "0 (dans la plage « Normal ») à 1 (seuil « Surveillance » atteint), combinées à poids "
        "égal (1/12 chacune)."
    )
    paragraph(doc,
        "Trois décisions méthodologiques tranchées explicitement avant l'implémentation :"
    )
    bullets(doc, [
        "`Oil Conductivity` : seuils appliqués sur la colonne convertie ×10 (`Oil "
        "Conductivity_nSm_filled`, cf. Partie 1.4), pas la colonne brute.",
        "`DC` : sens « élevée = critique » retenu tel quel (tableau de seuils), malgré une "
        "contradiction avec les diapositives i-SENSE (DC élevée = bon état) — point ouvert, "
        "cf. Conclusion.",
        "Référence d'huile TD46 (Shell Turbo T46, ISO VG 46) appliquée telle quelle aux deux "
        "machines, y compris B qui en sort massivement sur la viscosité — une sortie de plage "
        "traitée comme un signal réel plutôt que masquée par un recalibrage par machine.",
    ])
    add_table(doc,
        ["Machine", "HI moyen", "HI min", "Normal", "Surveillance", "Alarme"],
        [
            ["A", "0,814", "0,482", "—", "3,9 %", "1,5 %"],
            ["B", "0,606", "0,450", "—", "6,0 %", "1,3 %"],
            ["Total", "—", "—", "39 421 (93,6 %)", "2 123 (5,0 %)", "597 (1,4 %)"],
        ],
        caption="Résultat de la Phase 1, avant combinaison avec la PCA/Isolation Forest (Phase 5)."
    )
    note(doc,
        "Deux limites découvertes et documentées : (1) la formule unidirectionnelle "
        "« DC élevée = critique » ne pénalise pas une DC anormalement basse (cas de B, ~1,03 en "
        "moyenne) — distance DC = 0,000 pour les deux machines, l'anomalie de B reste invisible "
        "ici, visible seulement en Phase 0. (2) `Kinematic`/`Dynamic Viscosity` n'ont pas de "
        "référence absolue officielle (règle relative seulement) — approximée par la médiane par "
        "machine, ce qui capture surtout du bruit de mesure (distances élevées, 0,71-0,86, pour "
        "les deux machines symétriquement) plutôt qu'une vraie dégradation. `Viscosity at 40°C` "
        "(référence absolue 46,0 cSt) reste beaucoup plus discriminante (0,040 pour A vs 0,945 "
        "pour B — confirme que B sort largement du grade TD46)."
    )
    add_picture_centered(doc, "eda_output/health_index/01_hi_regles_timeline.png",
        caption="Health Index à base de règles métier (seuils officiels) — trajectoire par machine.")
    add_picture_centered(doc, "eda_output/health_index/00_distances_par_variable.png",
        caption="Contribution moyenne de chaque variable au Health Index, par machine.")

    doc.add_heading("Phase 2 — PCA par machine (T² / SPE)", level=2)
    paragraph(doc,
        "Alors que la Phase 1 examine 4 variables de manière isolée, la PCA surveille 13 variables "
        "santé simultanément, pour détecter si plusieurs variables dérivent de concert de façon "
        "inhabituelle — même si chacune reste individuellement sous ses seuils d'alerte physiques "
        "(ex. une combinaison anormale de température et de viscosité)."
    )
    paragraph(doc,
        "La constante diélectrique (DC) est explicitement exclue de ces 13 paramètres : l'audit "
        "Data Quality a révélé que 88,4 % des mesures de DC sur Motosoufflante B sont "
        "physiquement impossibles (ε_r < 1), signe d'un capteur mal calibré — l'inclure aurait "
        "injecté un artefact de mesure dans l'indice de santé."
    )
    paragraph(doc,
        "Étape 1, définir la période de référence « saine » : en l'absence d'historique de pannes "
        "validé, le système approxime cet état en sélectionnant les lignes sans aucun flag "
        "d'alerte actif (`flag_high_*`) — une hypothèse qui conserve 97,6 à 98,5 % du dataset. "
        "Étape 2, entraîner la PCA séparément par machine : le nombre de composantes (k) est "
        "choisi pour expliquer au moins 90 % de la variance ; 5 composantes suffisent pour les "
        "deux machines (94,3 % pour A, 94,4 % pour B), preuve que les 13 variables de départ sont "
        "très redondantes entre elles, reliées par des lois physico-chimiques."
    )
    add_table(doc,
        ["Machine", "Composantes (k)", "Variance expliquée", "Seuil Alarme T²", "Seuil Alarme SPE"],
        [
            ["A", "5", "94,3 %", "29,72", "8,46"],
            ["B", "5", "94,4 %", "23,57", "5,13"],
        ]
    )
    paragraph(doc,
        "Étape 3, calculer deux indicateurs de dérive : le T² d'Hotelling mesure si la donnée est "
        "extrême à l'intérieur de l'espace réduit des 5 dimensions de variation normale (distance "
        "statistique de Mahalanobis) ; le SPE (Squared Prediction Error, ou erreur de "
        "reconstruction) mesure à quel point la donnée ne correspond pas du tout à la structure "
        "apprise (‖x − x̂‖²). Le SPE est l'équivalent linéaire exact de l'erreur de reconstruction "
        "d'un autoencodeur — ce qui a rendu le développement de ce dernier non prioritaire dans la "
        "suite du plan. Étape 4, déterminer les seuils : conformément aux standards du contrôle "
        "statistique de procédés (SPC), les seuils de Surveillance (95ᵉ percentile) et d'Alarme "
        "(99ᵉ percentile) sont calculés uniquement sur les lignes saines d'entraînement, jamais sur "
        "l'ensemble du dataset, pour garantir que les limites de contrôle proviennent strictement "
        "de la période de référence."
    )
    add_picture_centered(doc, "eda_output/health_index/02_pca_timeline.png",
        caption="T² et SPE dans le temps, par machine, avec seuils Surveillance/Alarme.")

    doc.add_heading("Phase 3 — Isolation Forest (vérification)", level=2)
    paragraph(doc,
        "La PCA suppose implicitement une évolution linéaire des variables, une structure de "
        "covariance stable, et des écarts à peu près symétriques. Or l'EDA a montré que certaines "
        "variables critiques, comme la teneur en eau (`Oil H2O ppm`), ont une distribution très "
        "asymétrique (non gaussienne). L'Isolation Forest ne fait aucune hypothèse sur la forme de "
        "la distribution ou la linéarité : elle isole les points anormaux par des coupures "
        "aléatoires successives — un point normal (au cœur de la masse des données) nécessite de "
        "nombreux découpages pour être isolé, un point anormal (en périphérie) est isolé très "
        "rapidement."
    )
    paragraph(doc,
        "Pour rester rigoureusement comparable à la PCA, la Phase 3 conserve exactement le même "
        "cadre : les mêmes 13 variables, le même sous-ensemble sain de référence, les mêmes "
        "conventions de seuils (95ᵉ/99ᵉ percentile). Résultat : 39 659 lignes classées Normal, en "
        "**accord parfait avec la PCA** sur la zone normale (0 désaccord vers Alarme), 2 001 en "
        "Surveillance et 481 en Alarme. Les légères variations aux frontières "
        "Surveillance/Alarme sont normales : les deux méthodes ne mesurent pas le même type "
        "d'écart (dérive linéaire par rapport à un espace de projection pour la PCA, isolement "
        "récursif dans l'espace global pour l'Isolation Forest). Elle sert donc de **vérification "
        "robuste**, qui valide ou complète les alertes de la PCA, en particulier pour les "
        "comportements non linéaires et les distributions non gaussiennes."
    )
    add_picture_centered(doc, "eda_output/health_index/03_isoforest_timeline.png",
        caption="Score d'isolement dans le temps, par machine.")

    doc.add_heading("Décision — pourquoi l'autoencodeur (Phase 4) n'a pas été construit", level=2)
    paragraph(doc,
        "3 tests de validation ont été appliqués aux 3 méthodes déjà calculées, selon le critère "
        "explicite du plan : construire l'autoencodeur seulement si PCA/Isolation Forest échouent."
    )
    add_table(doc,
        ["Test", "Résultat"],
        [
            ["1. Cohérence avec les flags métier", "✅ p < 10⁻⁵³ partout ; T² ×5,7, SPE ×3,3 sur lignes flaggées"],
            ["2. Injection de défauts synthétiques", "✅ ≥ 94,8 % des lignes dégradées voient leur score augmenter"],
            ["3. Stabilité en session stable", "✅ Ratio écart-type intra-session/global < 0,7 partout"],
        ],
        note="Conclusion : la PCA linéaire, complétée par l'Isolation Forest, suffit à détecter "
             "les écarts connus — l'autoencodeur n'apporterait pas de valeur ajoutée démontrée."
    )

    doc.add_heading("Phase 5 — Health Index final", level=2)
    paragraph(doc,
        "Pour combiner les deux modèles statistiques, le système applique une logique "
        "conservatrice de type « OU », le pire des deux l'emporte : si l'une des deux méthodes "
        "(PCA ou Isolation Forest) détecte une anomalie, le score final reflète immédiatement "
        "cette gravité — le système ne se laisse pas « rassurer » par un modèle qui n'aurait rien "
        "vu. Justification industrielle : rater une véritable dégradation mécanique (faux négatif) "
        "coûte extrêmement cher en réparations et en arrêts de production, bien plus que de gérer "
        "une fausse alerte occasionnelle (faux positif)."
    )
    paragraph(doc,
        "Concrètement, le système calcule d'abord une « sévérité » de dérive par rapport aux "
        "seuils de chaque modèle, puis applique la formule "
        "health_index = 1 / (1 + max(sévérité_PCA, sévérité_IsoForest)). Cette formule a trois "
        "propriétés utiles : elle reste toujours bornée entre 0 et 1 (1 = état le plus sain) ; "
        "elle vaut exactement 0,5 au seuil d'alarme, donc dès que le score descend sous 0,5, "
        "l'opérateur sait qu'au moins un seuil critique a été franchi ; et elle décroît "
        "progressivement plutôt que de chuter brutalement à 0, ce qui permet de mesurer à quel "
        "point la machine dépasse les limites."
    )
    paragraph(doc,
        "Les règles physiques de la Phase 1 ne participent pas au calcul de ce score final "
        "consolidé — elles conservent uniquement un rôle de garde-fou interprétatif, disponibles "
        "dans les données comme double vérification de bon sens pour aider les équipes à "
        "comprendre rapidement l'origine physique d'une alerte statistique."
    )
    add_table(doc,
        ["Machine", "HI moyen", "HI min", "Normal", "Surveillance", "Alarme"],
        [
            ["A", "0,575", "0,188", "17 658", "1 421", "817"],
            ["B", "0,583", "0,043", "19 928", "1 698", "619"],
            ["Total", "—", "—", "37 586 (89,2 %)", "3 119 (7,4 %)", "1 436 (3,4 %)"],
        ]
    )
    paragraph(doc,
        "Dans le fichier de sortie (`isense_oil_data_health_index.csv`), deux colonnes sont "
        "recommandées pour le pilotage opérationnel : `health_index` (score continu 0-1, 1 = "
        "sain) et `health_state` (état catégoriel final : Normal, Surveillance ou Alarme)."
    )
    add_picture_centered(doc, "eda_output/health_index/04_final_timeline.png",
        caption="Health Index final dans le temps, coloré par état (Normal/Surveillance/Alarme).")
    add_picture_centered(doc, "eda_output/health_index/05_final_distribution.png",
        caption="Répartition finale des 3 états sur l'ensemble du dataset.", width=IMG_W_SMALL)


# ===========================================================================
# PARTIE 5 — PREDICTION DU HEALTH INDEX A T+N
# ===========================================================================

def part5_prediction(doc):
    doc.add_page_break()
    doc.add_heading("Partie 5 — Prédiction du Health Index à t+n", level=1)
    paragraph(doc,
        "Objectif : prédire `health_index` à un horizon futur (t+n), de 10 minutes à 1 semaine, "
        "avec 2 modèles baseline (Random Forest, XGBoost) et 3 modèles à base d'autoencodeur. "
        "Découpage train/test **temporel** par machine (jamais aléatoire, pour ne pas fuiter "
        "l'information entre lignes consécutives très corrélées)."
    )
    note(doc,
        "Découverte méthodologique clé : `health_index` est fortement autocorrélé (une simple "
        "persistance — « pas de changement » — obtient déjà R²=0,94 à 10 min). Prédire "
        "directement la valeur absolue force les modèles à réapprendre cette persistance, ce qui "
        "ajoute du bruit. Tous les résultats ci-dessous utilisent donc une formulation en "
        "**delta** (`health_index[t+n] − health_index[t]`), systématiquement meilleure ou égale."
    )

    doc.add_heading("Les 5 modèles comparés", level=2)
    add_table(doc,
        ["Modèle", "Définition courte"],
        [
            ["Random Forest", "Ensemble d'arbres de décision entraînés sur des sous-échantillons aléatoires ; moyenne des prédictions."],
            ["XGBoost", "Arbres de décision construits séquentiellement, chacun corrigeant les erreurs du précédent (gradient boosting)."],
            ["A. Autoencodeur + RF (latent)", "Un autoencodeur compresse les features en un espace latent réduit ; un Random Forest prédit à partir de cet espace compressé."],
            ["B. LSTM Encoder-Decoder", "Réseau récurrent qui encode une fenêtre de 12 pas passés (~2h) en un vecteur, puis prédit le futur à partir de ce résumé temporel."],
            ["C. Débruiteur + fine-tuning", "Autoencodeur pré-entraîné à reconstruire les features à partir d'une version bruitée, puis une tête de prédiction est branchée sur l'espace latent (encodeur figé)."],
        ],
        note="Le modèle C a nécessité 3 corrections successives pour stabiliser son entraînement "
             "(gradient qui divergeait, R² jusqu'à −24 413) : limitation de la norme du gradient, "
             "gel de l'encodeur après pré-entraînement, puis normalisation de l'espace latent "
             "(LayerNorm) — la cause racine."
    )

    doc.add_heading("Meilleur modèle par horizon et par machine", level=2)
    add_table(doc,
        ["Horizon", "Machine", "Meilleur modèle", "R²"],
        [
            ["10 min", "Motosoufflante A", "Random Forest", "0,945"],
            ["10 min", "Motosoufflante B", "Random Forest", "0,705"],
            ["3h", "Motosoufflante A", "A. Autoencodeur+RF (latent)", "0,617"],
            ["3h", "Motosoufflante B", "Random Forest", "0,557"],
            ["24h", "Motosoufflante A", "C. Débruiteur + fine-tuning", "0,172"],
            ["24h", "Motosoufflante B", "A. Autoencodeur+RF (latent)", "−0,083"],
        ],
        note="Comparaison entre les 5 modèles entraînés uniquement — la persistance (référence "
             "méthodologique, cf. ci-dessous) n'est pas incluse dans ce classement."
    )

    doc.add_heading("Classement global (rang moyen de R², 6 combinaisons horizon×machine)", level=2)
    add_table(doc,
        ["Rang", "Modèle", "Rang moyen", "Meilleur (/6)"],
        [
            ["1", "Random Forest", "2,00", "3"],
            ["2", "A. Autoencodeur+RF (latent)", "2,17", "2"],
            ["3", "C. Débruiteur + fine-tuning", "3,17", "1"],
            ["4", "XGBoost", "3,50", "0"],
            ["5", "B. LSTM Encoder-Decoder", "4,17", "0"],
        ]
    )
    note(doc,
        "Random Forest reste le meilleur modèle en moyenne, mais la course est serrée avec le "
        "modèle A (Autoencodeur+RF), qui le devance sur les 2 horizons les plus longs testés — "
        "seul le modèle B (LSTM) n'apporte jamais de valeur. Rappel méthodologique : une simple "
        "persistance (« pas de changement ») a d'abord servi de garde-fou pour découvrir que la "
        "prédiction directe de la valeur absolue perdait face à elle — d'où la formulation en "
        "delta utilisée pour tous les résultats ci-dessus. Elle n'est pas retenue comme "
        "concurrente dans ce classement final."
    )
    add_picture_centered(doc, "eda_output/health_index_prediction/01_r2_vs_horizon.png",
        caption="R² des baselines (RF, XGBoost) vs persistance, par horizon et par machine.")
    add_picture_centered(doc, "eda_output/health_index_prediction/02_autoencoder_comparison.png",
        caption="R² des 3 modèles autoencodeur vs persistance, par horizon et par machine.")
    add_picture_centered(doc, "eda_output/health_index_prediction/03_comparaison_finale.png",
        caption="Comparaison en barres des 6 méthodes (5 modèles + persistance).")
    add_picture_centered(doc, "eda_output/health_index_prediction/04_meilleur_modele.png",
        caption="Classement global par rang moyen.", width=IMG_W_SMALL)

    doc.add_heading("Recommandation par cas d'usage", level=2)
    add_table(doc,
        ["Cas", "Horizon", "Modèle recommandé", "Justification"],
        [
            ["Court terme", "10 min – 1h", "Random Forest",
             "Meilleur modèle sur les 2 machines à cet horizon ; R² > 0,70 partout."],
            ["Moyen terme", "3h", "Random Forest ou A. Autoencodeur+RF",
             "Résultats proches (R² 0,50-0,62) ; le modèle A devance légèrement sur Motosoufflante A."],
            ["Long terme", "24h et au-delà", "Aucun modèle fiable en absolu",
             "Même les meilleurs modèles (A, C) restent proches de 0 ou négatifs — reflet de "
             "l'absence d'historique de pannes/maintenance plutôt qu'un défaut de modèle."],
        ],
        note="Un test complémentaire (ajout d'un écart-type glissant, la seule information de la "
             "famille « rolling » non redondante avec les features déjà présentes) a confirmé "
             "que plus de features n'aide pas ici : sur 6 cas testés, 5 empirent (jusqu'à −6,69 "
             "de R²), aucun ne s'améliore significativement."
    )


# ===========================================================================
# CONCLUSION
# ===========================================================================

def conclusion(doc):
    doc.add_page_break()
    doc.add_heading("Conclusion et points ouverts", level=1)
    paragraph(doc,
        "Le pipeline complet — audit physico-chimique, nettoyage révisé, imputation hybride, "
        "feature engineering, Health Index — est opérationnel et documenté à chaque étape. "
        "Les limites suivantes restent ouvertes, en attente de retour de l'équipe i-SENSE "
        "(détail complet dans Questions_Reunion_Equipe_iSENSE.docx) :"
    )
    bullets(doc, [
        "Anomalie de calibration DC sur Motosoufflante B (88,4 % de valeurs physiquement "
        "impossibles, renforcé par la borne officielle resserrée à 1,8) — non corrigée, exclue "
        "de la PCA/Isolation Forest par précaution.",
        "Sens de dégradation de `DC` contradictoire entre le tableau de seuils officiel (élevée "
        "= critique, retenu dans la Phase 1) et les diapositives i-SENSE (élevée = bon état) — "
        "non tranché.",
        "Facteur de conversion `Oil Conductivity` (×10 vers nS/m) déterminé par analyse des "
        "données, non confirmé par l'équipe i-SENSE.",
        "Référence d'huile TD46 appliquée aux deux machines malgré l'écart massif de B "
        "(viscosité, densité) — à confirmer si B utilise réellement une huile différente.",
        "Période « saine » de référence (PCA/Isolation Forest) approximée par l'absence de "
        "flags, non confirmée par l'équipe métier.",
        "Aucun événement de panne réel connu — le Health Index détecte des écarts statistiques, "
        "pas une dégradation validée par un historique de pannes ; c'est aussi la limite "
        "principale de la prédiction à long terme (Partie 5).",
        "Seuils Normal/Surveillance/Alarme (PCA/Isolation Forest) calibrés par percentiles, à "
        "ajuster si des seuils métier spécifiques sont fournis.",
        "Kinematic/Dynamic Viscosity sans référence absolue officielle — approximée par la "
        "médiane par machine, signal plus bruité que souhaité (Partie 4, Phase 1).",
    ])


def main():
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    add_title_page(doc)
    add_toc_page(doc)
    part0_vue_ensemble(doc)
    part1_plan_refonte(doc)
    part2_data_quality(doc)
    part3_feature_engineering(doc)
    part4_health_index(doc)
    part5_prediction(doc)
    conclusion(doc)

    doc.save("Rapport_Refonte_PhysicoChimique_HealthIndex_iSENSE.docx")
    print("Généré : Rapport_Refonte_PhysicoChimique_HealthIndex_iSENSE.docx")


if __name__ == "__main__":
    main()
