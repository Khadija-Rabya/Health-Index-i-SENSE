"""Genere les deux documents Word :
  1. Note_Viscosite_iSENSE.docx            — note autonome pour l'equipe i-SENSE
  2. Rapport_Audit_HealthIndex_Complet.docx — entree / processus / sortie, avec figures
"""
import json
import os
import sys

import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figures")
sys.path.insert(0, HERE)

NAVY, RED, GREEN, GREY = RGBColor(0x1A, 0x36, 0x5D), RGBColor(0xC5, 0x30, 0x30), \
    RGBColor(0x2F, 0x85, 0x5A), RGBColor(0x55, 0x5F, 0x6D)

with open(os.path.join(HERE, "verif_resultats.json"), encoding="utf-8") as f:
    V = json.load(f)
with open(os.path.join(HERE, "artifacts", "metriques_finales.json"), encoding="utf-8") as f:
    M = json.load(f)
with open(os.path.join(HERE, "verif_decisive.json"), encoding="utf-8") as f:
    DEC = json.load(f)


def fnum(x, n=4):
    return f"{x:+.{n}f}".replace(".", ",")


def fci(ci, n=4):
    return f"[{ci[0]:+.{n}f} ; {ci[1]:+.{n}f}]".replace(".", ",")


# ------------------------------------------------------------------ utilitaires
def setup(doc, landscape=False):
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    for s in doc.sections:
        if landscape:
            s.orientation = WD_ORIENT.LANDSCAPE
            s.page_width, s.page_height = s.page_height, s.page_width
        s.left_margin = s.right_margin = Inches(0.75)
        s.top_margin = s.bottom_margin = Inches(0.7)
    return doc


def h(doc, text, level=1, color=NAVY):
    p = doc.add_heading(text, level=level)
    for r in p.runs:
        r.font.color.rgb = color
        r.font.name = "Calibri"
    return p


def para(doc, text, bold=False, italic=False, size=10, color=None, align=None, space=4):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space)
    if align:
        p.alignment = align
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    r.font.size = Pt(size)
    if color:
        r.font.color.rgb = color
    return p


def rich(doc, parts, size=10, space=4):
    """parts : liste de (texte, gras, couleur|None)."""
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space)
    for txt, bold, col in parts:
        r = p.add_run(txt)
        r.bold = bold
        r.font.size = Pt(size)
        if col:
            r.font.color.rgb = col
    return p


def bullets(doc, items, size=10):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(it)
        r.font.size = Pt(size)


def table(doc, headers, rows, widths=None, size=8.5, highlight=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, htxt in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(str(htxt))
        r.bold = True
        r.font.size = Pt(size)
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(val))
            r.font.size = Pt(size)
            if highlight and ri in highlight:
                r.bold = True
    if widths:
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Inches(w)
    return t


def save_doc(doc, path):
    """Enregistre ; si le fichier est ouvert dans Word (verrou Windows), ecrit une
    version suffixee plutot que d'echouer."""
    try:
        doc.save(path)
        return path
    except PermissionError:
        base, ext = os.path.splitext(path)
        alt = f"{base}_v2{ext}"
        doc.save(alt)
        print(f"  ! {os.path.basename(path)} est ouvert dans Word — ecrit dans "
              f"{os.path.basename(alt)}")
        return alt


def figure(doc, name, caption, width=6.6):
    path = os.path.join(FIG, name)
    if not os.path.exists(path):
        return
    doc.add_picture(path, width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    para(doc, caption, italic=True, size=8.5, color=GREY,
         align=WD_ALIGN_PARAGRAPH.CENTER, space=10)


# =========================================================== 1. NOTE VISCOSITE
def note_viscosite():
    doc = setup(Document())
    h(doc, "Note technique — Viscosité hors plage physique (Motosoufflante B)", 0)
    para(doc, "Destinataires : équipe i-SENSE / OCP Maintenance Solutions   •   "
              "Date : 6 septembre 2026", size=9, color=GREY)
    rich(doc, [("Objet : ", True, None),
               ("40 % des mesures de viscosité du jeu de données sont physiquement "
                "impossibles, et ces valeurs entrent dans le calcul du Health Index.",
                False, RED)])
    para(doc, "Source : isense_oil_data_health_index.csv — 42 141 lignes, "
              "29/12/2025 → 06/08/2026.", size=9, color=GREY, space=10)

    h(doc, "1. Le constat", 1)
    para(doc, "Le capteur « Viscosity at 40 °C » renvoie 16 839 valeurs comprises entre "
              "3,0 et 19,9 cSt (médiane 8,9 cSt) pour une huile Shell Turbo T46 / ISO VG 46, "
              "dont la viscosité nominale de référence est 46,0 cSt.")
    para(doc, "Une huile de turbine à 8,9 cSt à 40 °C ne correspond à aucun état de "
              "dégradation connu : même une dilution sévère par du carburant ou une oxydation "
              "avancée ne divise pas la viscosité par cinq. L'ordre de grandeur est celui d'un "
              "solvant léger, pas d'une ISO VG 46 dégradée. Ces valeurs représentent "
              "39,96 % de l'ensemble du jeu de données.")

    h(doc, "2. Elles ne sont pas réparties au hasard", 1)
    ev = pd.DataFrame(V["E"]["par_groupe"])
    rows = [[r["machine"], r["etat"], f"{int(r['n']):,}".replace(",", " "),
             f"{int(r['n_basse']):,}".replace(",", " "), f"{r['pct']:.1%}"]
            for _, r in ev.iterrows()]
    table(doc, ["Machine", "État", "Lignes", "Dont viscosité < 20 cSt", "Part"], rows,
          widths=[2.0, 0.8, 1.2, 1.9, 0.9], highlight={3})
    para(doc, "")
    rich(doc, [("100 % des valeurs aberrantes proviennent de la Motosoufflante B à l'arrêt", True, RED),
               (" — aucune de la Motosoufflante A, aucune de la Motosoufflante B en marche.",
                False, None)])
    table(doc, ["Machine", "Viscosité médiane — ON", "Viscosité médiane — OFF"],
          [["Motosoufflante A", "46,61 cSt", "45,59 cSt"],
           ["Motosoufflante B", "41,61 cSt", "11,92 cSt"]],
          widths=[2.2, 2.2, 2.2], highlight={1})
    para(doc, "")
    para(doc, "La Motosoufflante B mesure correctement lorsqu'elle tourne et décroche "
              "complètement à l'arrêt. L'hypothèse la plus simple est que le viscosimètre ne "
              "fournit pas de mesure valide lorsque l'huile ne circule pas, et renvoie une "
              "valeur par défaut ou une lecture à vide au lieu d'un code d'erreur.")
    para(doc, "La Motosoufflante A ne présente pas le problème, mais elle n'est à l'arrêt que "
              "6,7 % du temps (1 334 lignes) contre 90,1 % pour la Motosoufflante B. Le défaut "
              "peut donc exister sur les deux machines sans être visible sur A, faute "
              "d'occasions de l'observer — point à vérifier plutôt qu'à écarter.")

    figure(doc, "07_viscosite.png",
           "Distribution de la viscosité par machine et par état ; part des valeurs "
           "impossibles ; série temporelle de la Motosoufflante B.", width=6.8)

    h(doc, "3. Pourquoi c'est bloquant : ces valeurs entrent dans le Health Index", 1)
    para(doc, "« Viscosity at 40°C_filled » fait partie de PCA_VARS, la liste des 13 variables "
              "sur lesquelles sont ajustées l'ACP (statistiques T² et SPE) et l'Isolation Forest "
              "qui construisent le Health Index (health_index_pca.py, "
              "health_index_isolation_forest.py). La même colonne alimente aussi "
              "dist_Viscosity at 40°C dans le Health Index à seuils officiels.")
    rich(doc, [("Conséquence : le problème ne touche pas seulement les variables d'entrée d'un "
                "modèle, il corrompt l'étiquette elle-même.", True, RED),
               (" Tout modèle entraîné sur ce Health Index apprend en partie à reproduire un "
                "artefact de capteur.", False, None)])
    table(doc, ["Motosoufflante B", "À l'arrêt (OFF)", "En marche (ON)"],
          [["Health Index moyen", "0,5856", "0,5590"],
           ["Part d'états Surveillance ou Alarme", "10,8 %", "7,2 %"]],
          widths=[2.6, 1.9, 1.9], highlight={1})
    para(doc, "")
    para(doc, "La machine est classée en état dégradé plus souvent à l'arrêt qu'en "
              "fonctionnement — l'inverse de ce qu'on attend physiquement, puisqu'à l'arrêt "
              "l'huile ne subit ni cisaillement, ni échauffement, ni contamination active.")

    h(doc, "4. Ce que nous avons besoin de savoir", 1)
    bullets(doc, [
        "Le viscosimètre de la Motosoufflante B fournit-il une mesure valide lorsque la "
        "machine est à l'arrêt ? Si non, quelle valeur renvoie-t-il (valeur par défaut, "
        "dernière mesure figée, lecture à vide) ?",
        "Existe-t-il un code d'erreur ou un indicateur de validité associé à cette mesure, "
        "disponible dans le système source mais absent de l'export utilisé ici ?",
        "La Motosoufflante A est-elle équipée du même modèle de capteur ? Si oui, le même "
        "comportement est probablement présent mais masqué par le très faible nombre d'arrêts.",
        "Confirmez-vous que la référence TD46 (ISO VG 46, 46,0 cSt) s'applique bien aux deux "
        "machines ? La Motosoufflante B mesure 41,6 cSt en marche, soit −9,5 % par rapport à "
        "la référence, ce qui reste hors de la tolérance ±5 % du tableau de cadrage.",
    ])

    h(doc, "5. La correction a été testée, et son effet est mesuré", 1)
    para(doc, "Nous n'avons pas seulement supposé que ces valeurs nuisaient : l'étiquette a "
              "été reconstruite en traitant la viscosité comme manquante sur toutes les lignes "
              "à l'arrêt (les trois colonnes de viscosité, imputées par la médiane en marche de "
              "la même machine), puis le protocole d'évaluation complet a été rejoué à "
              "l'identique — même découpage, mêmes 7 795 lignes de test.")
    av, ap = DEC["decisive"]["avant"], DEC["decisive"]["apres"]
    table(doc, ["Indicateur (test gelé)", "Avant", "Après"],
          [["Motosoufflante A — skill", fnum(av["Motosoufflante A"]["skill"]),
            fnum(ap["Motosoufflante A"]["skill"])],
           ["Motosoufflante A — écart d'accuracy", fnum(av["Motosoufflante A"]["delta_acc"]),
            fnum(ap["Motosoufflante A"]["delta_acc"])],
           ["Motosoufflante B — skill", fnum(av["Motosoufflante B"]["skill"]),
            fnum(ap["Motosoufflante B"]["skill"])],
           ["Global — écart d'accuracy", fnum(av["global"]["delta_acc"]),
            fnum(ap["global"]["delta_acc"])],
           ["Global — IC 95 %", fci(av["global"]["ci_acc"]) + "  ✗",
            fci(ap["global"]["ci_acc"]) + "  ✓"],
           ["Global — n_eff", f"{av['global']['n_eff']:.0f}", f"{ap['global']['n_eff']:.0f}"]],
          widths=[2.7, 2.1, 2.1], highlight={0, 4, 5})
    para(doc, "")
    bullets(doc, [
        f"L'artefact détruisait la Motosoufflante A : son skill passe de "
        f"{fnum(av['Motosoufflante A']['skill'])} (le modèle faisait moins bien que de ne rien "
        f"prédire) à {fnum(ap['Motosoufflante A']['skill'])}. La machine qui n'était PAS "
        f"affectée par le capteur défaillant était en réalité la plus pénalisée, parce que "
        f"l'étiquette des deux machines est construite avec les mêmes variables.",
        "Le bénéfice du modèle devient statistiquement significatif pour la première fois : "
        "l'intervalle de confiance global cesse de contenir zéro, et la taille d'échantillon "
        f"effective passe de {av['global']['n_eff']:.0f} à {ap['global']['n_eff']:.0f}.",
        "L'étiquette corrigée est plus difficile (la persistance passe de 0,791 à 0,762 en "
        "validation croisée) : les valeurs fausses maintenaient l'indice artificiellement "
        "stable pendant les 50 % du temps où la machine est à l'arrêt.",
    ])
    figure(doc, "09_experience_decisive.png",
           "Intervalles de confiance et taille d'échantillon effective, avant et après "
           "masquage de la viscosité à l'arrêt.", width=6.8)

    h(doc, "6. Recommandation", 1)
    para(doc, "Traiter la viscosité comme manquante (et non comme mesurée) sur toutes les "
              "lignes en état OFF, puis reconstruire le Health Index. Cela retire 16 839 "
              "valeurs fausses de la définition de l'étiquette, au prix d'une imputation "
              "explicite et traçable — nettement préférable à une valeur fausse traitée comme "
              "une mesure. Cette correction est indépendante de tout choix de modélisation, "
              "elle est déjà validée expérimentalement, et devrait être appliquée avant toute "
              "nouvelle campagne d'entraînement.")
    b = DEC["b_off"]
    rich(doc, [("Un point reste ouvert. ", True, RED),
               (f"Nous avons vérifié si les sauts du Health Index de la Motosoufflante B à "
                f"l'arrêt ({b['pct_excursion']:.1%} des lignes voient l'indice bouger de plus "
                f"de 0,01 en 3 h, machine pourtant stoppée) s'expliquaient par une bascule du "
                f"viscosimètre entre ses deux modes (~12 et ~42 cSt). Ce n'est pas le cas : "
                f"seules {b['pct_bascule']:.2%} des lignes présentent une telle bascule, et "
                f"l'association joue même en sens inverse ({b['p_exc_si_bascule']:.1%} "
                f"d'excursions quand il y a bascule contre {b['p_exc_si_pas']:.1%} sinon). Les "
                f"deux modes sont stables, pas oscillants. L'origine de ces excursions sur une "
                f"machine à l'arrêt reste donc à expliquer — c'est la cinquième question que "
                f"nous vous adressons.", False, None)])
    para(doc, "Éléments chiffrés reproductibles : hi_forecast/run_13_verify.py (section E), "
              "hi_forecast/verif_E_viscosite.csv, figure hi_forecast/figures/07_viscosite.png.",
         italic=True, size=8.5, color=GREY)

    out = os.path.join(ROOT, "Note_Viscosite_iSENSE.docx")
    return save_doc(doc, out)


# ================================================= 2. RAPPORT COMPLET (E/P/S)
def rapport_complet():
    doc = setup(Document())
    h(doc, "Audit et optimisation du modèle de prédiction du Health Index", 0)
    para(doc, "Entrée · Processus · Sortie — compte rendu détaillé", size=12, color=GREY)
    para(doc, "Projet i-SENSE / OCP Maintenance Solutions — UM6P   •   6 septembre 2026",
         size=9, color=GREY, space=12)

    # ---- resume
    h(doc, "Résumé exécutif", 1)
    rich(doc, [("Le R² de 0,9453 rapporté par le pipeline d'origine mesurait "
                "l'autocorrélation, pas la prédiction.", True, RED),
               (" À l'horizon 10 minutes, la simple persistance (« rien ne change ») "
                "atteint R² = 0,9427 ; le Random Forest ajoutait +0,0025. Le fichier "
                "realtime/artifacts/manifest.json le concédait déjà : 8 de ses 12 entrées "
                "basculent sur « model: persistence ».", False, None)])
    para(doc, "Six fuites de données ont été identifiées et corrigées, dont la construction de "
              "l'étiquette elle-même. Un protocole d'évaluation gelé a été mis en place, puis "
              "33 configurations ont été évaluées. Le modèle final utilise une architecture à "
              "porte : il ne corrige la persistance que lorsqu'un second modèle prédit un "
              "mouvement réel du Health Index.")
    table(doc, ["Configuration", "CV acc @ ±0,01", "Test acc", "Test R²", "Skill"],
          [["Persistance (référence)", "0,7912 ± 0,0391", "0,8164", "0,1142", "0,0000"],
           ["RF d'origine, sans fuite", "0,5202 ± 0,2717", "0,8310", "0,2849", "+0,1927"],
           ["Modèle final à porte", "0,8094 ± 0,0370", "0,8259", "0,1975", "+0,0941"]],
          widths=[2.2, 1.5, 1.0, 1.0, 0.9], highlight={2})
    para(doc, "")
    rich(doc, [("Nuance essentielle : ", True, None),
               ("l'écart d'accuracy global (+0,0095) a un intervalle de confiance à 95 % de "
                "[−0,057 ; +0,061] — il contient zéro. Seul le gain sur la Motosoufflante B "
                "est statistiquement établi.", False, RED)])

    figure(doc, "08_pipeline.png", "Figure 1 — Chaîne de traitement complète, de la donnée "
                                   "brute au modèle final.", width=7.0)
    doc.add_page_break()

    # ================================================================= ENTREE
    h(doc, "PARTIE I — ENTRÉE (Input)", 1)
    h(doc, "I.1 Les données", 2)
    table(doc, ["Caractéristique", "Valeur"],
          [["Fichier source", "isense_oil_data_health_index.csv"],
           ["Dimensions", "42 141 lignes × 171 colonnes"],
           ["Période couverte", "29/12/2025 → 06/08/2026 (≈ 7 mois)"],
           ["Machines", "Motosoufflante A (19 896 lignes), Motosoufflante B (22 245)"],
           ["Sessions", "352 — médiane 3 lignes, 75ᵉ centile 38, maximum 7 904"],
           ["Fréquence d'échantillonnage", "≈ 10 minutes (médiane 10, maximum 60)"],
           ["Lignes dupliquées", "0"],
           ["Colonnes avec valeurs manquantes", "90 sur 171"],
           ["Taux de manquants maximal", "65,9 % (Oil System Vibration brute)"],
           ["État machine", "ON 50 % / OFF 50 % des lignes"]],
          widths=[2.6, 4.3])

    h(doc, "I.2 La cible : ce qu'est réellement le « Health Index »", 2)
    rich(doc, [("Ce n'est pas une mesure, c'est une étiquette construite par le projet "
                "lui-même.", True, RED)])
    para(doc, "Formule (health_index_comparison.py, phase 5) :")
    p = doc.add_paragraph()
    r = p.add_run("sévérité = max( T²_ACP / seuil_alarme ,  SPE_ACP / seuil_alarme ,  "
                  "IsolationForest / seuil_alarme )\nhealth_index = 1 / (1 + sévérité)")
    r.font.name, r.font.size, r.bold = "Consolas", Pt(9), True
    para(doc, "L'ACP et l'Isolation Forest sont ajustées par machine sur les lignes sans "
              "aucun indicateur flag_high_* actif ; les seuils d'alarme sont les 99ᵉ centiles "
              "de ces lignes saines.")
    table(doc, ["Statistique de la cible", "Valeur"],
          [["Moyenne / écart-type", "0,5793 / 0,0447"],
           ["Étendue effective", "0,0426 → 0,6244"],
           ["Intervalle interquartile", "0,5619 → 0,6111 (largeur 0,049)"],
           ["États dérivés", "Normal 89,2 % / Surveillance 7,4 % / Alarme 3,4 %"]],
          widths=[3.0, 3.9])
    para(doc, "")
    para(doc, "Conséquence directe : il n'existe aucune vérité terrain de dégradation dans ce "
              "jeu de données — ni panne, ni intervention de maintenance, ni vidange, ni analyse "
              "d'huile en laboratoire. Le modèle prédit le score d'anomalie du projet, et son "
              "plafond est le bruit de ce score, pas la santé réelle de la machine.")

    h(doc, "I.3 Le modèle existant, avant audit", 2)
    table(doc, ["Élément", "Configuration d'origine"],
          [["Tâche", "Régression, un modèle par (horizon × machine) = 12 cellules"],
           ["Horizons", "t+1, t+6, t+18, t+144, t+432, t+1008 (10 min → 1 semaine)"],
           ["Modèles", "RandomForestRegressor(300, profondeur 14) ; XGBRegressor(300, 6, 0,1)"],
           ["Formulation", "Prédiction du delta y[t+n] − y[t], puis reconstruction"],
           ["Prétraitement", "SimpleImputer(médiane) uniquement"],
           ["Évaluation", "Un seul découpage temporel 80/20 — aucune validation croisée"],
           ["Score rapporté", "R² = 0,9453 (RF, 10 min, Motosoufflante A)"]],
          widths=[1.8, 5.1])

    h(doc, "I.4 Décisions de cadrage validées avec le porteur du projet", 2)
    bullets(doc, [
        "Cible primaire : régression du health_index à l'horizon t+18 (≈ 3 h), les deux machines.",
        "Définition de l'« accuracy » : part des prédictions du test avec |ŷ − y| ≤ 0,01 "
        "(≈ 22 % de l'écart-type de la cible). R² rapporté systématiquement à côté.",
        "Étiquette : reconstruite sans fuite (ACP / Isolation Forest / seuils réajustés sur la "
        "seule fenêtre d'entraînement), au prix de la comparabilité avec les rapports existants.",
    ])
    doc.add_page_break()

    # ============================================================== PROCESSUS
    h(doc, "PARTIE II — PROCESSUS (Process)", 1)
    h(doc, "II.1 Phase 0 — Audit : six fuites de données identifiées", 2)
    para(doc, "Le document de plan (plan_prediction_health_index.md, §1.3) affirmait que "
              "toutes les variables étaient causales, « vérifié ». Trois ne l'étaient pas.")
    table(doc, ["#", "Fuite", "Localisation", "Correction"],
          [["F1", "L'étiquette elle-même : ACP, Isolation Forest et seuils d'alarme ajustés "
                  "sur tout l'historique, période de test comprise",
            "health_index_comparison.py:158-204", "Réajustés sur la fenêtre d'entraînement seule"],
           ["F2", "vi_proxy : np.polyfit sur toutes les lignes de chaque machine ; se propage "
                  "dans 7 colonnes dérivées", "feature_engineering.py:109-116",
            "Réajusté train-only, dérivées recalculées"],
           ["F3", "13 colonnes *_zscore standardisées avec moyenne/écart-type globaux",
            "feature_engineering.py:176-181", "Statistiques de la fenêtre d'entraînement"],
           ["F4", "HI_regles : référence de viscosité = médiane de tout l'historique machine",
            "health_index_baseline.py:107-111", "Médiane de la fenêtre d'entraînement"],
           ["F5", "SimpleImputer ajusté sur l'ensemble X avant le découpage",
            "health_index_prediction_baseline.py:71-72", "Déplacé dans le pipeline, ajusté par pli"],
           ["F6", "Sélection du modèle effectuée sur le jeu de test (marqueurs 🏆, choix "
                  "persistance/RF par cellule)", "health_index_prediction_comparison.py",
            "Toute sélection sur CV ; test scoré une seule fois"]],
          widths=[0.3, 2.5, 2.0, 2.1], size=8)
    para(doc, "")
    para(doc, "Points vérifiés et sains : aucune ligne dupliquée (0 sur 42 141), le découpage "
              "était déjà temporel et non aléatoire, et utiliser health_index[t] pour prédire "
              "health_index[t+n] est de la prévision légitime, pas une fuite.", size=9)

    h(doc, "II.2 Phase 1 — Le protocole d'évaluation, gelé une fois pour toutes", 2)
    bullets(doc, [
        "Test gelé : les 20 % de lignes les plus récentes de chaque machine (coupures "
        "24/05/2026 pour A, 28/06/2026 pour B) — 7 795 lignes, scorées une seule fois par "
        "configuration.",
        "Validation croisée : 5 blocs temporels expansifs, avec purge et embargo de 3 h de "
        "chaque côté de la frontière de validation. Sans cet embargo, les dernières lignes "
        "d'entraînement partagent leur cible avec le début de la validation et le score gonfle.",
        "Graine aléatoire fixée à 42 partout.",
        "Règle absolue : toute configuration est choisie sur cv_acc_tol_mean. La configuration "
        "la meilleure au test (itération 29, 0,8476) n'a pas été retenue, car la CV la classait "
        "derrière l'itération 32.",
    ])

    h(doc, "II.3 Phase 2 — Référence de base", 2)
    para(doc, "La persistance (« le health index dans 3 h sera celui d'aujourd'hui ») atteint "
              "déjà 0,8164 sur le test gelé. L'objectif de 80 % était donc franchi avant toute "
              "modélisation. La tolérance n'a été ni assouplie ni resserrée après coup ; le "
              "chiffre est rapporté tel que défini, et c'est le gain sur la persistance qui "
              "constitue la vraie question.")

    h(doc, "II.4 Phase 3 — 33 itérations, dix leviers", 2)
    table(doc, ["Levier", "Ce qui a été testé", "Résultat en CV"],
          [["1. Qualité des données", "Constantes, doublons structurels, élagage |r| > 0,995",
            "163 → 70 variables, +0,016"],
           ["2. Valeurs manquantes", "Médiane, moyenne, indicateur, MICE, KNN",
            "Négligeable (0,7908 → 0,7927)"],
           ["3-4. Variables / encodage", "Régime ON-OFF, dynamique, interactions", "Marginal"],
           ["5. Mise à l'échelle", "Standard, robuste, min-max, quantile, Yeo-Johnson, aucune",
            "Négligeable (0,7905 → 0,7919)"],
           ["6. Sélection de variables", "Corrélation, information mutuelle, permutation, SHAP",
            "SHAP top-20 : +0,004, 70 → 20 variables"],
           ["8. Familles de modèles", "14 familles évaluées", "Voir livrable 2"],
           ["8. Ensembles", "Vote, vote pondéré, empilement Ridge", "Perdant (0,7960)"],
           ["9. Hyperparamètres", "10 essais aléatoires + 12 essais Optuna (TPE)",
            "0,7956 → 0,8057"],
           ["10. Calibration", "Platt, isotonique", "Brier 0,1795 → 0,1566, mais AUC en baisse"],
           ["Combinaison finale", "Optuna × SHAP top-20", "0,8094 ± 0,0413"]],
          widths=[1.7, 3.0, 2.2], size=8.5, highlight={9})

    h(doc, "II.5 La découverte méthodologique centrale", 2)
    rich(doc, [("Toute correction apprise perdait d'abord contre la persistance.", True, RED)])
    para(doc, "Un balayage du facteur de rétrécissement α dans ŷ = hi[t] + α·δ̂, sur cinq "
              "familles de modèles, donne un α optimal en CV égal à zéro pour chacune d'elles — "
              "la courbe décroît de façon monotone. Le mécanisme est arithmétique : la "
              "persistance est déjà dans la bande sur environ 79 % des lignes, et une correction "
              "bruitée casse celles-là plus vite qu'elle n'en rattrape parmi les 21 % restantes.")
    rich(doc, [("La solution est une porte.", True, GREEN),
               (" On entraîne un second modèle à prédire si l'indice va bouger de plus que la "
                "tolérance, et on n'applique la correction que lorsqu'il se déclenche. C'est la "
                "première configuration à battre la persistance en CV, et l'écart CV/test passe "
                "de +0,31 à +0,017.", False, None)])
    figure(doc, "00_iterations.png", "Figure 2 — Toutes les itérations : CV (sélection) contre "
                                     "test gelé, et écart entre les deux.", width=7.0)
    doc.add_page_break()

    # ================================================================= SORTIE
    h(doc, "PARTIE III — SORTIE (Output)", 1)
    h(doc, "III.1 Le modèle retenu", 2)
    p = doc.add_paragraph()
    r = p.add_run("prédiction = health_index[t] + α · δ̂ · 1[ P(mouvement > 0,01) > 0,35 ]"
                  "     avec α = 1,0")
    r.font.name, r.font.size, r.bold = "Consolas", Pt(9.5), True
    r.font.color.rgb = NAVY
    table(doc, ["Composant", "Configuration"],
          [["Régresseur du delta", "LightGBM, perte Huber (α = 0,0059), 400 arbres, "
                                   "lr 0,0249, 31 feuilles, min_child 192"],
           ["Classifieur de porte", "LightGBM, 300 arbres, lr 0,0176, 63 feuilles, "
                                    "min_child 253"],
           ["Variables", "20, sélectionnées par SHAP"],
           ["Qualité de la porte (test)", "AUC 0,7056 · PR-AUC 0,459 · Brier 0,1486 · "
                                          "se déclenche sur 32,2 % des lignes"],
           ["Artefact", "artifacts/health_index_t18_pipeline.joblib (1,4 Mo)"],
           ["Reproduction", "python hi_forecast/train_best.py — 24 secondes"]],
          widths=[2.0, 4.9])

    h(doc, "III.2 Toutes les métriques sur le test gelé", 2)
    t = M["test"]
    table(doc, ["Métrique", "Modèle final", "Persistance"],
          [["accuracy @ ±0,01", f"{t['acc_tol']:.4f}", f"{t['acc_tol_persist']:.4f}"],
           ["R²", f"{t['r2']:.4f}", f"{t['r2_persist']:.4f}"],
           ["R² ajusté", f"{t['adj_r2']:.4f}", "—"],
           ["RMSE", f"{t['rmse']:.5f}", "0,02737"],
           ["MAE", f"{t['mae']:.5f}", "0,00877"],
           ["MedAE", f"{t['medae']:.5f}", "—"],
           ["MAPE", f"{t['mape']:.2f} %", "—"],
           ["Variance expliquée", f"{t['explained_var']:.4f}", "—"],
           ["Erreur maximale", f"{t['max_error']:.4f}", "—"],
           ["Biais", f"{t['bias']:+.5f}", "—"],
           ["Skill vs persistance", f"{t['skill_vs_persist']:+.4f}", "0,0000"]],
          widths=[2.4, 2.2, 2.3], highlight={0, 10})
    para(doc, "")
    para(doc, f"Validation croisée : {M['cv']['acc_tol_mean']:.4f} ± "
              f"{M['cv']['acc_tol_std']:.4f} (plis : "
              + ", ".join(f"{v:.3f}" for v in M["cv"]["acc_par_pli"]) +
              f"). Écart test − CV : +{t['acc_tol'] - M['cv']['acc_tol_mean']:.4f} — faible, "
              "donc l'estimation par CV est honnête.", size=9)

    h(doc, "III.3 Pourquoi la porte est retenue malgré un test apparemment moins bon", 2)
    para(doc, "Sur des plis identiques et le même protocole, le modèle à porte bat le Random "
              "Forest d'origine sur 5 plis sur 5, avec un écart moyen de +0,289. L'écart-type "
              "en CV du Random Forest est de 0,2717 — sept fois celui de la porte — avec une "
              "accuracy par pli oscillant entre 0,185 et 0,788 et un R² descendant à −11,3. Son "
              "score de test favorable est un tirage unique dans une distribution très large.")
    figure(doc, "04_cv_vs_test_rf_porte.png",
           "Figure 3 — Accuracy par pli de validation croisée, et inversion du classement "
           "lorsqu'on ne regarde que le test.", width=7.0)

    h(doc, "III.4 Significativité statistique du gain", 2)
    rich(doc, [("Le gain global n'est pas distinguable de zéro.", True, RED)])
    para(doc, "Le jeu de test contient 7 795 lignes mais seulement 11 sessions. Un bootstrap "
              "par blocs (rééchantillonnage des sessions, pas des lignes) donne un effet de "
              "plan de 79,6 : la taille d'échantillon effective est de 98, pas de 7 795.")
    B = V["B"]["bootstrap"]
    para(doc, "⚠️ Le ✓ de la Motosoufflante B repose sur 5 BLOCS de bootstrap seulement. Un "
              "intervalle par percentiles construit sur 5 unités de rééchantillonnage n'est pas "
              "fiable, aussi étroit soit-il : 5 blocs ne produisent que 126 tirages distincts, "
              "et les percentiles 2,5 % et 97,5 % sont déterminés par le bloc le plus extrême. "
              "Il faut lire cet intervalle comme « les cinq sessions s'accordent en signe et en "
              "ordre de grandeur », non comme une garantie de couverture à 95 %. La même "
              "réserve vaut a fortiori pour les 6 blocs de la machine A et les 11 blocs du "
              "global.", size=9, color=RED)
    table(doc, ["Périmètre", "Blocs", "Δ accuracy", "IC 95 %", "Skill", "IC 95 %", "n_eff"],
          [["Global", B["global"]["n_sessions"], f"{B['global']['delta_acc']:+.4f}",
            f"[{B['global']['ci_acc'][0]:+.4f} ; {B['global']['ci_acc'][1]:+.4f}] ⚠",
            f"{B['global']['skill']:+.4f}",
            f"[{B['global']['ci_skill'][0]:+.4f} ; {B['global']['ci_skill'][1]:+.4f}] ⚠",
            f"{B['global']['n_eff']:.0f}"],
           ["Motosoufflante A", B["Motosoufflante A"]["n_sessions"],
            f"{B['Motosoufflante A']['delta_acc']:+.4f}",
            f"[{B['Motosoufflante A']['ci_acc'][0]:+.4f} ; {B['Motosoufflante A']['ci_acc'][1]:+.4f}] ⚠",
            f"{B['Motosoufflante A']['skill']:+.4f}",
            f"[{B['Motosoufflante A']['ci_skill'][0]:+.4f} ; {B['Motosoufflante A']['ci_skill'][1]:+.4f}] ⚠",
            f"{B['Motosoufflante A']['n_eff']:.0f}"],
           ["Motosoufflante B", B["Motosoufflante B"]["n_sessions"],
            f"{B['Motosoufflante B']['delta_acc']:+.4f}",
            f"[{B['Motosoufflante B']['ci_acc'][0]:+.4f} ; {B['Motosoufflante B']['ci_acc'][1]:+.4f}] ✔",
            f"{B['Motosoufflante B']['skill']:+.4f}",
            f"[{B['Motosoufflante B']['ci_skill'][0]:+.4f} ; {B['Motosoufflante B']['ci_skill'][1]:+.4f}] ✔",
            f"{B['Motosoufflante B']['n_eff']:.0f}"]],
          widths=[1.5, 0.7, 1.0, 1.5, 0.8, 1.4, 0.6], size=8, highlight={2})
    para(doc, "")
    para(doc, "Le test de McNemar donne p = 0,0135 en global, mais il suppose 7 795 "
              "observations appariées indépendantes — hypothèse fausse ici, puisque des lignes "
              "espacées de 10 minutes dans une même session ne sont pas indépendantes. Le "
              "bootstrap par blocs est l'inférence correcte, et son intervalle contient zéro.",
         size=9)
    h(doc, "III.4a Pourquoi n_eff global (98) est très inférieur à celui de B seule (2 255)", 3)
    para(doc, "Ce n'est PAS un effet de taille d'échantillon, et ce n'est pas l'autocorrélation "
              "intra-session : l'effet de plan de la Motosoufflante B vaut 1,9, soit "
              "pratiquement 1. L'inflation vient d'ailleurs.")
    var = DEC["variance"]
    table(doc, ["Machine", "Sessions", "Δ accuracy moyen", "Écart-type"],
          [["Motosoufflante A", var["n_sessions_A"], fnum(var["delta_A"]), "0,0557"],
           ["Motosoufflante B", var["n_sessions_B"], fnum(var["delta_B"]), "0,0195"]],
          widths=[2.2, 1.2, 1.8, 1.5], highlight={0, 1})
    para(doc, "")
    para(doc, f"Les deux machines ont des écarts de SIGNES OPPOSÉS. Chaque tirage de bootstrap "
              f"prélève 11 sessions avec remise : le mélange A/B varie d'un tirage à l'autre et "
              f"la statistique globale suit ce mélange plutôt qu'un bruit d'échantillonnage "
              f"interne. Décomposition de la variance des écarts par session : inter-machine "
              f"SSB = {var['ssb']:.4f} ({var['part_inter']:.1%} du total), intra-machine "
              f"SSW = {var['ssw']:.4f}.")
    bullets(doc, [
        f"La variance inter-machine est importante mais MINORITAIRE ({var['part_inter']:.1%}) — "
        "il serait inexact de dire qu'elle domine. Les 60 % restants sont concentrés presque "
        "entièrement sur une session anormale de la machine A (Motosoufflante A_S43, 1 868 "
        "lignes, Δ = −0,1178).",
        "L'implication n'est donc pas « collecter plus de sessions » mais « cesser d'agréger "
        "deux machines qui se comportent en sens opposé » : une moyenne sur des populations de "
        "signes opposés n'est pas un estimand interprétable, quelle que soit la taille de "
        "l'échantillon.",
        "La section III.4bis montre POURQUOI elles se comportaient en sens opposé : l'artefact "
        "du viscosimètre. Une fois masqué, les signes s'alignent (A +0,0088, B +0,0304) et "
        "n_eff global passe de 98 à 1 707 sans une seule observation supplémentaire.",
    ])
    rich(doc, [("La seule affirmation soutenue à 95 % de confiance : le modèle améliore la "
                "Motosoufflante B (+0,059 d'accuracy, +0,102 de skill).", True, GREEN)])
    figure(doc, "05_intervalles_confiance.png",
           "Figure 4 — Intervalles de confiance à 95 % par bootstrap par blocs. "
           "Rouge : l'intervalle croise zéro.", width=7.0)

    h(doc, "III.4bis EXPÉRIENCE DÉCISIVE — masquage de la viscosité à l'arrêt", 2)
    para(doc, "Hypothèse mise à l'épreuve : le gain de +0,0589 sur la Motosoufflante B "
              "viendrait du modèle qui suit l'artefact du viscosimètre plutôt que la machine. "
              "L'étiquette a été reconstruite avec la viscosité traitée comme manquante sur "
              "toutes les lignes à l'arrêt, puis le protocole gelé rejoué de bout en bout — "
              "persistance, resélection de la porte en CV, test gelé, bootstrap. "
              "Hyperparamètres inchangés, découpage identique.")
    av, ap = DEC["decisive"]["avant"], DEC["decisive"]["apres"]
    table(doc, ["Indicateur", "Avant", "Après"],
          [["Persistance (CV)", "0,7912", "0,7616"],
           ["Modèle à porte (CV)", "0,8094 ± 0,0370", "0,7817 ± 0,0433"],
           ["Écart-type de la cible (test)", "0,0291", "0,0372"],
           ["B — écart d'accuracy", fnum(av["Motosoufflante B"]["delta_acc"]),
            fnum(ap["Motosoufflante B"]["delta_acc"])],
           ["B — skill", fnum(av["Motosoufflante B"]["skill"]),
            fnum(ap["Motosoufflante B"]["skill"])],
           ["A — écart d'accuracy", fnum(av["Motosoufflante A"]["delta_acc"]),
            fnum(ap["Motosoufflante A"]["delta_acc"])],
           ["A — skill", fnum(av["Motosoufflante A"]["skill"]),
            fnum(ap["Motosoufflante A"]["skill"])],
           ["Global — écart d'accuracy", fnum(av["global"]["delta_acc"]),
            fnum(ap["global"]["delta_acc"])],
           ["Global — IC 95 %", fci(av["global"]["ci_acc"]) + " ✗",
            fci(ap["global"]["ci_acc"]) + " ✓"],
           ["Global — n_eff", f"{av['global']['n_eff']:.0f}", f"{ap['global']['n_eff']:.0f}"]],
          widths=[2.5, 2.2, 2.2], highlight={3, 6, 8, 9})
    para(doc, "")
    rich(doc, [("Verdict : hypothèse à moitié confirmée, et la moitié qui échoue est la plus "
                "instructive.", True, NAVY)])
    bullets(doc, [
        "Le gain d'accuracy de B diminue de moitié (−48 %, de +0,0589 à +0,0304) : une part "
        "réelle du gain suivait bien l'artefact. La question méritait d'être posée.",
        "Mais il ne disparaît pas, et le skill de B est inchangé (+0,1017 → +0,1075). La "
        "composante « variance expliquée » de l'avantage de B n'a jamais dépendu de l'artefact.",
        "L'artefact détruisait la Motosoufflante A : son écart d'accuracy passe de −0,0506 à "
        "+0,0088 et son skill de −0,1191 à +0,0857. La contradiction CV/test de la section "
        "IV.1 était l'artefact, pas un hasard de la fenêtre de test.",
        "Le résultat global devient significatif pour la première fois, et n_eff passe de 98 "
        "à 1 707 — parce que les deux machines cessent d'avoir des écarts de signes opposés.",
    ])
    para(doc, "L'étiquette corrigée est plus DIFFICILE : la persistance recule de 0,7912 à "
              "0,7616 et l'écart-type de la cible augmente. L'artefact maintenait l'indice "
              "artificiellement stable pendant les 50 % du temps où la machine est à l'arrêt. "
              "Les valeurs absolues d'accuracy se dégradent et la qualité de la preuve "
              "s'améliore — c'est précisément l'allure d'une correction de facteur de confusion.",
         size=9.5)
    figure(doc, "09_experience_decisive.png",
           "Figure 5 — Expérience décisive : intervalles de confiance et taille d'échantillon "
           "effective, avant et après masquage.", width=7.0)

    h(doc, "III.4ter Découpage {machine} × {état} et excursions à l'arrêt", 2)
    t2 = pd.read_csv(os.path.join(HERE, "verif_2x2_machine_etat.csv"))
    table(doc, ["Étiquette", "Cellule", "n", "acc", "acc pers.", "R²", "R² pers.", "skill"],
          [[r["etiquette"], f"{r['machine']}/{r['etat']}", int(r["n"]),
            f"{r['acc']:.4f}".replace(".", ","), f"{r['acc_pers']:.4f}".replace(".", ","),
            f"{r['r2']:.4f}".replace(".", ","), f"{r['r2_pers']:.4f}".replace(".", ","),
            fnum(r["skill"])] for _, r in t2.iterrows()],
          widths=[0.9, 0.8, 0.6, 0.8, 0.9, 0.9, 0.9, 0.8], size=8)
    para(doc, "")
    bullets(doc, [
        "Le jeu de test ne contient quasiment AUCUNE ligne de la Motosoufflante B en marche "
        "(moins de 20, cellule omise). B est à l'arrêt 90 % du temps et la fenêtre de test est "
        "tardive : tout ce qui est affirmé sur la Motosoufflante B dans ce rapport est affirmé "
        "sur la Motosoufflante B À L'ARRÊT. Le modèle n'a jamais été évalué sur B en charge.",
        "A/OFF ne compte que 25 lignes de test : ses chiffres sont du bruit et ne doivent pas "
        "être cités.",
        "Le dommage causé par l'artefact à la machine A était concentré sur A/ON : skill "
        "−0,1226 avant, +0,0564 après.",
    ])
    b = DEC["b_off"]
    rich(doc, [("Une machine à l'arrêt dont l'indice bouge demande une explication. ", True, None),
               (f"Sur {b['n']:,} lignes B/OFF, {b['pct_excursion']:.2%} voient le health index "
                f"varier de plus de 0,01 en 3 heures. Le mécanisme proposé — le viscosimètre "
                f"basculant entre ses deux modes — n'est PAS la cause : seules "
                f"{b['pct_bascule']:.2%} des lignes basculent, et l'association joue en sens "
                f"inverse (P(excursion | bascule) = {b['p_exc_si_bascule']:.1%} contre "
                f"{b['p_exc_si_pas']:.1%} sinon, soit un rapport de "
                f"{b['p_exc_si_bascule']/b['p_exc_si_pas']:.2f}×). Les deux modes sont stables, "
                f"pas oscillants : l'artefact biaise le NIVEAU de l'étiquette, pas ses "
                f"excursions. Leur origine reste ouverte.", False, None)]
         )

    h(doc, "III.5 Le choix de la métrique décide du gagnant", 2)
    C = pd.read_csv(os.path.join(HERE, "verif_C_arbitrage.csv"))
    table(doc, ["Configuration", "acc @ ±0,01", "R²", "RMSE", "MAE", "Skill"],
          [[r["configuration"], f"{r['acc']:.4f}", f"{r['r2']:.4f}", f"{r['rmse']:.5f}",
            f"{r['mae']:.5f}", f"{r['skill']:+.4f}"] for _, r in C.iterrows()],
          widths=[2.1, 1.1, 0.9, 1.0, 1.0, 0.9], highlight={2})
    para(doc, "")
    para(doc, "Optimiser la bande ±0,01 coûte environ 0,216 de R² (0,4136 → 0,1975). Sur le "
              "test gelé, ElasticNet gagne sur les quatre métriques — mais son accuracy en CV "
              "est de 0,4435 ± 0,3018, c'est-à-dire catastrophiquement instable d'un bloc "
              "temporel à l'autre, ce qui explique que la CV ne l'ait pas retenue.")
    rich(doc, [("Quelle métrique correspond à l'usage réel ? ", True, None),
               ("Le rôle du système est de déclencher des états Surveillance/Alarme au bon "
                "moment. Cette décision dépend du franchissement d'un seuil : ce qui compte est "
                "donc de bien prévoir les grandes excursions, ce que mesurent RMSE, R² et le "
                "skill — et ce qu'une bande ±0,01 ignore explicitement, en traitant une erreur "
                "de 0,009 et une erreur nulle comme équivalentes. Pour l'exploitation en alarme, "
                "R² / skill est donc le meilleur objectif.", False, None)])
    rich(doc, [("Recommandation ElasticNet : RETIRÉE. ", True, RED),
               ("Une version antérieure de cette section recommandait ElasticNet pour "
                "l'exploitation en alarme. Cette recommandation était lue dans le tableau du "
                "TEST alors que la CV l'avait rejetée — exactement la faute F6 que cet audit "
                "corrige. La sélection des familles a donc été rejouée en CV avec R²/skill "
                "comme critère, mêmes plis, même protocole.", False, None)])
    lb = pd.read_csv(os.path.join(HERE, "leaderboard_familles.csv"))
    lb = lb.sort_values("cv_r2_mean", ascending=False).head(6)
    table(doc, ["Rang (CV R²)", "Famille", "CV R²", "CV écart-type", "CV skill"],
          [[i + 1, r["name"], f"{r['cv_r2_mean']:.4f}".replace(".", ","),
            f"{r['cv_r2_std']:.4f}".replace(".", ","), fnum(r["cv_skill_mean"])]
           for i, (_, r) in enumerate(lb.iterrows())],
          widths=[1.0, 2.2, 1.2, 1.3, 1.2], highlight={0, 3})
    para(doc, "")
    para(doc, "Sous un objectif R²/skill, la CV ne choisit PAS ElasticNet : elle le classe 4ᵉ "
              "avec un R² de −0,6832 ± 1,5775 et un skill de −1,3695. Son R² de 0,4136 sur le "
              "test est un tirage dans une distribution d'écart-type 1,58. La CV préfère "
              "faiblement Extra Trees (+0,2671 ± 0,1452), mais cette famille n'est pas "
              "séparable du modèle retenu (porte : R² CV +0,2554 ± 0,2475 — écart de 0,0117 "
              "pour des écarts-types de 0,25 et 0,15), ni de la suivante (marge 0,2993 contre "
              "un écart-type de 0,3402).")
    rich(doc, [("Conclusion : le changement d'objectif ne désigne pas de gagnant. Le modèle à "
                "porte est conservé.", True, GREEN)])

    doc.add_page_break()
    h(doc, "III.6 CLASSEMENT COMPLET DES 14 FAMILLES DE MODÈLES", 2)
    para(doc, "Toutes les familles sont évaluées à alpha = 1 (correction brute, sans porte) "
              "afin d'être comparables entre elles : le facteur de rétrécissement optimal en "
              "CV vaut 0 pour presque toutes, ce qui les ramènerait toutes à la persistance et "
              "produirait un classement plat. La colonne « alpha* » donne ce facteur optimal "
              "par famille. Tableau trié par R² sur le test gelé.")
    para(doc, "Naive Bayes ne figure pas dans ce classement : c'est un classifieur, sans "
              "équivalent de régression applicable à une cible continue.", italic=True,
         size=9, color=GREY)
    lb_all = pd.read_csv(os.path.join(HERE, "leaderboard_familles.csv")).sort_values(
        "test_r2", ascending=False)

    def d4(v):
        return f"{v:.4f}".replace(".", ",")

    table(doc, ["Famille", "CV acc", "CV sd", "CV R²", "TEST acc", "TEST R²", "RMSE",
                "MAE", "MedAE", "skill", "alpha*"],
          [[r["name"], d4(r["cv_acc_tol_mean"]), d4(r["cv_acc_tol_std"]),
            d4(r["cv_r2_mean"]), d4(r["test_acc_tol"]), d4(r["test_r2"]),
            d4(r["test_rmse"]), d4(r["test_mae"]), d4(r["test_medae"]),
            fnum(r["test_skill_vs_persist"]), d4(r["alpha_cv_best"])]
           for _, r in lb_all.iterrows()],
          widths=[1.35, 0.62, 0.55, 0.72, 0.65, 0.65, 0.62, 0.6, 0.62, 0.62, 0.6], size=7.5,
          highlight={0, 1})
    para(doc, "")
    rich(doc, [("Comment lire ce tableau. ", True, None),
               ("La colonne « CV acc » a un écart-type de 0,12 à 0,34 (médiane 0,28) pour une "
                "étendue des moyennes de seulement 0,53 : elle NE SÉPARE PAS les familles, "
                "presque tous les écarts deux à deux tiennent dans un écart-type. Le classement "
                "relatif se lit sur les colonnes de test ; la colonne CV ne prouve qu'une chose, "
                "à savoir que toutes les familles sans porte sont instables d'un bloc temporel "
                "à l'autre. Seule une comparaison avec l'architecture à porte (écart-type 0,037) "
                "est réellement concluante.", False, None)])
    bullets(doc, [
        "Meilleurs R² de test : ElasticNet 0,4136, Lasso 0,4082, Random Forest 0,4082 — les "
        "modèles linéaires régularisés tiennent tête aux ensembles d'arbres, ce qui est cohérent "
        "avec un delta de faible amplitude et approximativement linéaire.",
        "Pires : SVM (RBF exact) −0,7968 et MLP −0,3091, tous deux SOUS la persistance : sur ce "
        "problème, la puissance d'un modèle non régularisé se retourne contre lui.",
        "Extra Trees est la seule famille dont le R² en CV est positif (+0,2671) et la seule "
        "dont le skill en CV est positif (+0,0190) — c'est ce qui en fait le choix de la CV "
        "sous un objectif R², cf. section III.5.",
        "alpha* = 0 pour 13 familles sur 14 : sans porte, la CV demande de ne PAS corriger la "
        "persistance du tout. C'est le constat qui a conduit à l'architecture à porte.",
    ])
    figure(doc, "06_arbitrage_metriques.png",
           "Figure 5 — Trois configurations, mêmes lignes de test : le classement dépend "
           "entièrement de la métrique choisie.", width=7.0)
    doc.add_page_break()

    h(doc, "III.7 Diagnostic des résidus et importance des variables", 2)
    figure(doc, "01_residus_final.png",
           "Figure 6 — Résidus contre valeurs ajustées, distribution, QQ-plot et fonction de "
           "répartition de l'erreur absolue.", width=6.6)
    figure(doc, "02_calibration.png",
           "Figure 7 — Courbes de fiabilité du classifieur de porte, avant et après "
           "calibration (Platt, isotonique).", width=6.6)
    figure(doc, "03_shap.png", "Figure 8 — Importance SHAP du modèle retenu.", width=5.6)
    imp = pd.read_csv(os.path.join(HERE, "importance_shap_final.csv")).head(10)
    table(doc, ["Rang", "Variable", "SHAP moyen |valeur|"],
          [[i + 1, r["feature"], f"{r['mean_abs_shap']:.6f}"] for i, r in imp.iterrows()],
          widths=[0.6, 4.5, 1.8])
    doc.add_page_break()

    # ============================================================== LIMITES
    h(doc, "PARTIE IV — LIMITES ET RECOMMANDATIONS", 1)
    h(doc, "IV.1 Faiblesses connues", 2)
    bullets(doc, [
        "Le gain global n'est pas statistiquement significatif (11 sessions seulement dans le "
        "test, n_eff = 98). Seule la Motosoufflante B présente un gain établi.",
        "Sur le test, le modèle est moins bon que la persistance sur la Motosoufflante A "
        "(0,8752 contre 0,9258) — mais la CV dit l'inverse sur les deux machines (A : 0,8151 "
        "contre 0,8027). Choisir une politique par machine d'après le test serait exactement la "
        "faute F6 auditée ; le point reste ouvert, à trancher sur de nouvelles données.",
        "39,96 % des lignes portent une viscosité physiquement impossible, toutes sur la "
        "Motosoufflante B à l'arrêt, et ces valeurs entrent dans PCA_VARS donc corrompent "
        "l'étiquette (voir la note technique dédiée).",
        "La cible est un construit sans vérité terrain : aucune panne, aucune intervention, "
        "aucune vidange n'est enregistrée dans le jeu de données.",
        "R² et accuracy en bande divergent fortement : Motosoufflante A R² 0,87 mais accuracy "
        "sous la persistance ; Motosoufflante B R² −0,14 mais accuracy au-dessus.",
        "Deux machines seulement ; la médiane des sessions est de 3 lignes ; la machine est à "
        "l'arrêt 50 % du temps, et à l'arrêt l'indice ne bouge presque pas (|Δ| médian 0,00037 "
        "contre 0,00439 en marche).",
    ])

    h(doc, "IV.2 Les quatre actions qui relèveraient réellement le plafond", 2)
    bullets(doc, [
        "Obtenir une vérité terrain : dates de vidange, remplacements de filtres, ordres de "
        "travail, analyses d'huile en laboratoire. Cela permettrait de prédire une durée de vie "
        "résiduelle, ce dont OCP a réellement besoin, et qu'aucun réglage sur l'étiquette "
        "actuelle ne peut produire.",
        "Corriger le viscosimètre de la Motosoufflante B, ou exclure les lectures en état OFF "
        "de l'étiquette. C'est la plus grande source de bruit d'étiquette, et c'est une "
        "correction de collecte, pas de modélisation.",
        "Prédire la transition d'état plutôt que la valeur de l'indice : une prévision "
        "ordinale à 3 classes du health_state à t+3 h, avec un seuil de décision réglé sur le "
        "coût et non sur 0,5, correspond mieux à l'usage en alarme.",
        "Collecter un jeu de test contenant plus de 11 sessions. Le résultat global est "
        "actuellement non concluant uniquement à cause de n_eff = 98 ; aucun changement de "
        "modèle ne corrige cela, seules des sessions indépendantes supplémentaires le peuvent.",
    ])

    h(doc, "IV.3 Niveau de confiance", 2)
    para(doc, "Confiant sur le chiffre, pas sur son origine. La persistance seule atteint "
              "81,6 %, et l'amélioration globale a un intervalle de confiance qui contient "
              "zéro. J'attends donc que le chiffre se situe entre 78 % et 84 % sur de nouvelles "
              "données, centré autour de 81 %, mais je ne peux pas affirmer à 95 % de confiance "
              "que le modèle en est responsable, sauf sur la Motosoufflante B.")

    h(doc, "IV.4 Livrables produits", 2)
    table(doc, ["#", "Livrable", "Fichier"],
          [["1", "Tableau de toutes les itérations, trié par performance de test",
            "hi_forecast/livrable1_iterations.md / .csv"],
           ["2", "Classement de toutes les familles de modèles, toutes métriques",
            "hi_forecast/livrable2_leaderboard.md / .csv"],
           ["3", "Pipeline final en un artefact réutilisable",
            "hi_forecast/artifacts/health_index_t18_pipeline.joblib"],
           ["4", "Script d'entraînement reproductible de bout en bout",
            "hi_forecast/train_best.py"],
           ["5", "Importance des variables / résumé SHAP",
            "hi_forecast/importance_shap_final.csv + figures/03_shap.png"],
           ["6", "Rapport de synthèse", "hi_forecast/REPORT.md"],
           ["+", "Note technique viscosité (autonome, équipe i-SENSE)",
            "Note_Viscosite_iSENSE.docx"],
           ["+", "Le présent document", "Rapport_Audit_HealthIndex_Complet.docx"]],
          widths=[0.3, 3.1, 3.5], size=8.5)

    out = os.path.join(ROOT, "Rapport_Audit_HealthIndex_Complet.docx")
    return save_doc(doc, out)


if __name__ == "__main__":
    a = note_viscosite()
    b = rapport_complet()
    print("Documents Word generes :")
    for p in (a, b):
        print(f"  {p}   ({os.path.getsize(p)/1e6:.2f} Mo)")
