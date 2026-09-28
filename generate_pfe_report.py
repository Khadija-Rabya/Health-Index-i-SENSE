"""
Rapport PFE complet : Monitoring et Prediction de la Qualite de l'Huile de
Lubrification (i-SENSE).

Organisation par CONTENU (pas par nom de fichier/etape) :
  1. Etat de l'art       -> tout ce qui vient de la litterature : parametres,
                             algorithmes, datasets publics, etude de cas NASA
                             C-MAPSS, methodes de Deep Learning (article de
                             reference).
  2. Contexte            -> tout ce qui concerne les capteurs et le dataset
                             i-SENSE du projet (le "terrain").
  3. Traitement des donnees -> nettoyage + imputations (vibration, capteurs).
  4. EDA
  5. Feature engineering -> explique en detail les 2 taches terminees
                             (calendaire, session) avec leurs resultats reels ;
                             les 9 autres familles restent a faire.
  6. Prediction du Health Index par autoencodeur (a venir)
  7. Visualisation (a venir)

Sortie : Rapport_PFE_Monitoring_Huile_iSENSE.docx
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

IMG_W = Inches(6.3)
CHART_DIR = "eda_output/report_charts"


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


def bullets(doc, items, style="List Bullet"):
    for it in items:
        doc.add_paragraph(it, style=style)


# ===========================================================================
# GENERATION DES GRAPHIQUES AUXILIAIRES (tableaux -> visuels)
# ===========================================================================

def generate_auxiliary_charts():
    os.makedirs(CHART_DIR, exist_ok=True)

    # --- Etat de l'art / NASA : correlation capteurs - RUL ---
    sensors = ["Ps30", "T50", "BPR", "htBleed", "T24", "T30", "Nf", "NRf", "W31", "W32", "P30", "phi"]
    corr = [-0.78, -0.76, -0.72, -0.68, -0.68, -0.66, -0.62, -0.62, 0.70, 0.71, 0.73, 0.75]
    order = np.argsort(corr)
    sensors = [sensors[i] for i in order]
    corr = [corr[i] for i in order]
    colors = ["#c53030" if c < 0 else "#2f855a" for c in corr]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(sensors, corr, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Corrélation de Pearson avec la RUL")
    ax.set_title("Capteurs NASA C-MAPSS les plus corrélés à la RUL (FD001)")
    for i, v in enumerate(corr):
        ax.text(v, i, f" {v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{CHART_DIR}/nasa_sensor_correlation.png", dpi=120)
    plt.close(fig)

    # --- Etat de l'art / NASA : benchmark 4 modeles ---
    models = ["Régression\nLinéaire", "SVR", "Random\nForest", "XGBoost"]
    rmse = [21.90, 19.84, 18.16, 17.69]
    r2 = [0.722, 0.772, 0.809, 0.819]
    mcolors = ["#a0aec0", "#63b3ed", "#3182ce", "#c05621"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].bar(models, rmse, color=mcolors)
    axes[0].set_title("RMSE en cycles (plus bas = meilleur)")
    for i, v in enumerate(rmse):
        axes[0].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    axes[1].bar(models, r2, color=mcolors)
    axes[1].set_title("R² (plus haut = meilleur)")
    axes[1].set_ylim(0, 1)
    for i, v in enumerate(r2):
        axes[1].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Benchmark des 4 modèles sur NASA C-MAPSS FD001")
    fig.tight_layout()
    fig.savefig(f"{CHART_DIR}/nasa_benchmark.png", dpi=120)
    plt.close(fig)

    # --- Etat de l'art / Deep Learning : monotonicite / prognosabilite FD001 ---
    indicators = ["ε_NAP_LS\n(proposé)", "ε_SAP_LS\n(proposé)", "ε_NAP\n(RaPP)", "ε_SAP\n(RaPP)", "ε_REC\n(classique)"]
    mono = [0.455, 0.416, 0.156, 0.265, 0.199]
    prog = [0.991, 0.991, 0.979, 0.909, 0.893]
    y = np.arange(len(indicators))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(y - width / 2, mono, height=width, label="Monotonicité", color="#2b6cb0")
    ax.barh(y + width / 2, prog, height=width, label="Prognosabilité", color="#2f855a")
    ax.set_yticks(y)
    ax.set_yticklabels(indicators)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Score (0 à 1, plus élevé = meilleur)")
    ax.set_title("Monotonicité et prognosabilité des 5 indicateurs — FD001 (autoencodeur profond)")
    ax.legend(loc="lower right")
    for i, (m, p) in enumerate(zip(mono, prog)):
        ax.text(m + 0.01, i - width / 2, f"{m:.3f}", va="center", fontsize=8)
        ax.text(p + 0.01, i + width / 2, f"{p:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{CHART_DIR}/fd001_hi_comparison.png", dpi=120)
    plt.close(fig)

    # --- Etat de l'art / datasets : couverture des 11 parametres ---
    codes = [f"D{i}" for i in range(1, 15)]
    scores = [3, 2, 2, 2, 1, 1, 2, 3, 3, 0, 3, 3, 1, 3]
    dcolors = ["#c05621" if s >= 3 else "#63b3ed" for s in scores]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(codes, scores, color=dcolors)
    ax.set_ylabel("Score / 11 paramètres couverts")
    ax.set_title("Couverture des 11 paramètres cibles par dataset public identifié")
    ax.set_ylim(0, 11.5)
    ax.axhline(11, color="red", linestyle="--", linewidth=0.8, label="Couverture cible (11/11)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{CHART_DIR}/datasets_score.png", dpi=120)
    plt.close(fig)

    # --- Feature engineering : avancement ---
    fig, ax = plt.subplots(figsize=(6, 5.5))
    values = [2, 9]
    labels = [f"Fait ({values[0]})", f"À faire ({values[1]})"]
    ax.pie(values, labels=labels, colors=["#2f855a", "#cbd5e0"], autopct="%1.0f%%", startangle=90,
           textprops={"fontsize": 11})
    ax.set_title("Avancement du feature engineering\n(11 familles de features prévues)")
    fig.tight_layout()
    fig.savefig(f"{CHART_DIR}/feature_engineering_status.png", dpi=120)
    plt.close(fig)

    # --- Feature engineering : encodage cyclique de l'heure (resultat reel) ---
    hours = np.linspace(0, 23.99, 500)
    hs = np.sin(2 * np.pi * hours / 24)
    hc = np.cos(2 * np.pi * hours / 24)
    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(hc, hs, c=hours, cmap="twilight", s=14)
    for h in [0, 6, 12, 18]:
        x, y_ = np.cos(2 * np.pi * h / 24), np.sin(2 * np.pi * h / 24)
        ax.annotate(f"{h}h", (x, y_), fontsize=12, fontweight="bold",
                    xytext=(x * 1.32, y_ * 1.32), ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.9))
    ax.set_xlabel("hour_cos")
    ax.set_ylabel("hour_sin")
    ax.set_title("Encodage cyclique de l'heure — hour_sin / hour_cos", pad=16)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, label="Heure (0-23)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(f"{CHART_DIR}/calendar_cyclic_encoding.png", dpi=120)
    plt.close(fig)

    # --- Feature engineering : distribution de time_in_session_h (resultat reel) ---
    try:
        df = pd.read_csv("isense_oil_data_features.csv", usecols=["time_in_session_h", "measure_index_in_session"])
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        vals = df["time_in_session_h"].replace(0, 0.01)
        axes[0].hist(np.log10(vals), bins=50, color="#6b46c1")
        axes[0].set_xlabel("log10(time_in_session_h + ε)")
        axes[0].set_ylabel("Nombre de mesures")
        axes[0].set_title("Distribution de time_in_session_h\n(temps écoulé depuis le début de session)")
        axes[1].hist(df["measure_index_in_session"], bins=50, color="#3182ce")
        axes[1].set_xlabel("measure_index_in_session")
        axes[1].set_title("Distribution de measure_index_in_session\n(rang de la mesure dans sa session)")
        fig.tight_layout()
        fig.savefig(f"{CHART_DIR}/session_features_distribution.png", dpi=120)
        plt.close(fig)
    except FileNotFoundError:
        pass


# ===========================================================================
# TITLE PAGE
# ===========================================================================

def add_title_page(doc):
    title = doc.add_heading("Monitoring et Prédiction de la Qualité de", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title2 = doc.add_heading("l'Huile de Lubrification", level=0)
    title2.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = sub.add_run("Rapport de synthèse — État de l'art, méthodologie et travaux réalisés")
    run.italic = True
    run.font.size = Pt(14)

    doc.add_paragraph()
    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.add_run(
        "PFE OCP–UM6P — Dataset i-SENSE (Motosoufflante A & B)\n"
        "Capteur IoT 11 paramètres — Objectif : classification d'état et prédiction de la RUL\n\n"
        "Organisation de ce document : (1) l'état de l'art — paramètres, algorithmes, datasets "
        "publics, étude de cas NASA C-MAPSS et méthodes de Deep Learning de référence ; "
        "(2) le contexte du projet — le capteur i-SENSE et son dataset ; (3) le traitement des "
        "données — nettoyage et imputations ; (4) l'analyse exploratoire ; (5) le feature "
        "engineering ; (6) la prédiction du Health Index par autoencodeur (à venir) ; "
        "(7) la visualisation (à venir)."
    )
    doc.add_page_break()


def add_toc_note(doc):
    doc.add_heading("Sommaire", level=1)
    items = [
        "1. État de l'art",
        "2. Contexte — le capteur i-SENSE et son dataset",
        "3. Traitement des données — nettoyage et imputations",
        "4. Analyse exploratoire des données (EDA)",
        "5. Feature engineering",
        "6. Prédiction du Health Index par autoencodeur (à venir)",
        "7. Visualisation (à venir)",
    ]
    bullets(doc, items)
    doc.add_page_break()


# ===========================================================================
# 1. ETAT DE L'ART (litterature : parametres, algos, datasets, NASA, DL)
# ===========================================================================

def add_section1_etat_art(doc):
    doc.add_heading("1. État de l'art", level=1)
    doc.add_paragraph(
        "Cette section rassemble tout ce qui provient de la littérature scientifique : "
        "l'importance de la lubrification pour la fiabilité industrielle, les paramètres et "
        "indices de qualité normalisés, les algorithmes de Machine Learning déjà appliqués à ce "
        "type de problème, les datasets publics existants, une étude de cas approfondie sur le "
        "benchmark de référence NASA C-MAPSS, et la méthodologie de Deep Learning retenue comme "
        "référence pour la construction d'un Health Index."
    )

    # --- 1.1 Importance ---
    doc.add_heading("1.1 Importance de l'huile de lubrification dans la fiabilité industrielle", level=2)
    doc.add_paragraph(
        "La lubrification constitue l'un des piliers fondamentaux du bon fonctionnement des "
        "machines industrielles. Le marché mondial des lubrifiants, estimé à 201,56 milliards de "
        "dollars USD en 2023, devrait atteindre 253,85 milliards d'ici 2030 (CAGR de 3,35 %). "
        "Un lubrifiant remplit plusieurs fonctions simultanées : réduction de la friction entre "
        "surfaces mobiles, refroidissement des zones de contact, protection contre la corrosion, "
        "évacuation des particules d'usure, étanchéification des systèmes mécaniques. Sa "
        "dégradation progressive (température, charge, contamination) réduit son efficacité "
        "protectrice, jusqu'à la défaillance catastrophique des composants mécaniques si rien "
        "n'est fait."
    )
    doc.add_paragraph(
        "La notion centrale de Durée de Vie Résiduelle (Remaining Useful Life, RUL) du lubrifiant "
        "— la durée pendant laquelle il peut encore assurer efficacement sa fonction — est au "
        "cœur de toute stratégie de maintenance conditionnelle basée sur l'état de l'huile. "
        "Luong et al. (2024, Heliyon) proposent de consolider l'ensemble des paramètres d'analyse "
        "en un indice unique, le Performance Rating Index (PRI) : "
        "PRI = Σᵢ [wᵢ × (Pᵢ − Pᵢ,min) / (Pᵢ,max − Pᵢ,min)] × 100. "
        "Un lubrifiant bien entretenu peut prolonger la durée de vie des machines jusqu'à 50 %."
    )

    # --- 1.2 Parametres qualite ---
    doc.add_heading("1.2 Paramètres et indices de qualité de l'huile lubrifiante", level=2)
    add_table(
        doc, ["Paramètre", "Signification", "Norme"], [
            ["Viscosité cinématique", "Résistance à l'écoulement en cSt à 40°C/100°C — paramètre protecteur clé", "ASTM D445"],
            ["Indice de Viscosité (VI)", "Stabilité de la viscosité selon la température", "ASTM D2270"],
        ],
        caption="Tableau 1.1 — Paramètres de viscosité identifiés par Al-Ghouti et al. (2025, Sci. Reports)",
        note=(
            "La viscosité cinématique est le paramètre le plus critique : une hausse signale "
            "usure/contamination/suies, une baisse signale dilution (carburant, eau) — dans les "
            "deux cas, la capacité du film protecteur est compromise. L'Indice de Viscosité "
            "(VI = [(L − U) / (L − H)] × 100) quantifie la stabilité thermique de l'huile."
        ),
    )

    # --- 1.3 Algorithmes ML (fusion ancien 1.3 systemes monitoring + ancien 2.3 algos etudies) ---
    doc.add_heading("1.3 Algorithmes de Machine Learning pour la qualité de l'huile", level=2)
    doc.add_paragraph(
        "La littérature distingue le monitoring hors ligne (laboratoire, précis mais différé : "
        "spectrométrie, FTIR, ferrographie) et le monitoring en ligne (capteurs embarqués en "
        "continu : viscosité, permittivité diélectrique, particules, température — détection "
        "immédiate). Les capteurs triboélectriques nanogénérateurs (TENG), diélectriques et "
        "microfluidiques représentent les avancées technologiques les plus prometteuses pour la "
        "surveillance temps réel — c'est cette approche que suit le capteur i-SENSE (section 2)."
    )
    doc.add_paragraph(
        "Objectifs ML rapportés dans la littérature : classification multi-classes (Bonne / "
        "Dégradée / À changer) et régression de la RUL. Cinq familles d'algorithmes reviennent "
        "systématiquement : Random Forest (bagging), XGBoost (boosting séquentiel, "
        "régularisation L1/L2), SVM/SVR (marge maximale, noyaux RBF/polynomial), LSTM "
        "(dépendances temporelles longues), et Autoencodeurs (AE/VAE/LSTM-AE/CAELSTM) pour la "
        "détection d'anomalies non supervisée."
    )
    add_table(
        doc, ["Algorithme", "Application", "Performance rapportée"], [
            ["Régression Linéaire Multiple (MLR)", "Prédiction RUL à partir de VI/viscosité", "R² > 0,90"],
            ["Random Forest / XGBoost", "Classification état huile + score dégradation", "Précision > 92 %"],
            ["SVM", "Détection d'anomalies et classification d'état", "Robuste aux données déséquilibrées"],
            ["LSTM", "Prédiction temporelle de la dégradation (séries)", "Capture tendances longue durée"],
            ["Autoencodeurs", "Détection non supervisée d'anomalies", "Efficace sans données labellisées"],
        ],
        caption="Tableau 1.2 — Approches ML pour la prédiction de la qualité de l'huile (vue d'ensemble)",
    )
    add_table(
        doc, ["Algorithme", "Étude de référence", "Performance"], [
            ["Random Forest", "Wakiru 2020, Ye 2025", "~95 % accuracy"],
            ["XGBoost", "Eng. Proc. 2025 (pipeline flow)", "RMSE 0,00227 / R² 0,997"],
            ["SVM (noyau polynomial)", "Machines 2025 (TU5+E-nose)", "95,44 % (RBF 85,86 %, linéaire 68,22 %)"],
            ["LSTM", "arXiv 2024 (C-MAPSS FD001/3)", "RMSE 14,93 (FD001)"],
            ["CAELSTM (autoencodeur)", "Elsherif 2025 (C-MAPSS)", "RMSE 13,40 (FD003) — SOTA"],
            ["MLR", "Nguyen 2025 (221 huiles FTIR)", "R² = 0,808"],
        ],
        caption="Tableau 1.3 — Performances rapportées par algorithme, études spécifiques",
        note=(
            "Constat récurrent : les méthodes d'ensemble (Random Forest, XGBoost) et les "
            "architectures profondes spécialisées (CAELSTM) dominent systématiquement les "
            "benchmarks — ce constat oriente directement le choix méthodologique de ce projet "
            "(section 6)."
        ),
    )

    # --- 1.4 Datasets publics ---
    doc.add_heading("1.4 Datasets publics de référence", level=2)
    doc.add_paragraph(
        "14 datasets publics ont été recensés et notés selon le nombre des 11 paramètres cibles "
        "du capteur i-SENSE (section 2.1) qu'ils couvrent effectivement."
    )
    add_table(
        doc, ["Code", "Dataset", "Source", "Taille", "Score /11"], [
            ["D1", "NASA C-MAPSS", "Kaggle / NASA", "100–248 moteurs", "3"],
            ["D2", "Automotive Vehicles Engine Health", "Kaggle", "~19 000 lignes", "2"],
            ["D3", "555 Engine Oil Reports", "ACS Omega 2024", "555 échantillons", "2"],
            ["D4", "221 Used Motor Oils FTIR", "Nature 2025", "221 échantillons", "2"],
            ["D5", "1948 Spectral Diesel", "Pourramezan 2022", "1948 datasets", "1"],
            ["D6", "AES Diesel 21 ech.", "Liu 2023", "21 ech. × 17 éléments", "1"],
            ["D7", "TU5 + E-nose", "Machines 2025", "6 niveaux × N rep.", "2"],
            ["D8", "Brake Industry", "PMC 2025", "Multi-paramètres", "3"],
            ["D9", "Marine Diesel Lubrication", "Ye 2025", "15 paramètres", "3"],
            ["D10", "Predictive Maintenance Vehicles", "Kaggle", "Variable", "0"],
            ["D11", "UCI Hydraulic Systems", "UCI ML Repo", "2205 cycles × 17 capteurs", "3"],
            ["D12", "Transformer Oil Moisture DFDS", "Mahanta 2023", "14 samples × aging 0-1920h", "3"],
            ["D13", "Microwave Oil Dielectric", "arXiv 2506.09867", "4 types huile", "1"],
            ["D14", "Oil-Immersed Cellulose FDS", "Zhang, Polymers 2020", "Lab samples", "3"],
        ],
        caption="Tableau 1.4 — Les 14 datasets publics identifiés et leur couverture des 11 paramètres cibles",
        note=(
            "Constat central : aucun dataset public ne couvre simultanément les 11 paramètres "
            "cibles — les mieux notés (D8, D9, D11, D12, D14) n'en couvrent que 3 sur 11 au "
            "maximum. Ce constat justifie la valeur propre du dataset i-SENSE collecté pour ce "
            "projet (section 2)."
        ),
    )
    add_picture_centered(
        doc, f"{CHART_DIR}/datasets_score.png",
        caption="Figure 1.1 — Couverture des 11 paramètres cibles par dataset public",
        description=(
            "Aucune barre n'atteint la ligne rouge (11/11) : même les datasets les mieux notés "
            "(en orange, score ≥ 3) restent très loin de couvrir l'ensemble des 11 paramètres du "
            "capteur IoT visé par ce projet."
        ),
    )

    # --- 1.5 Etudes comparatives ---
    doc.add_heading("1.5 Études comparatives multi-algorithmes", level=2)
    add_table(
        doc, ["Étude", "Dataset", "Gagnant", "Écart"], [
            ["Pourramezan & Asgari 2024", "555 huiles moteur", "RBF (RMSE 0,11)", "+154 % vs MLR"],
            ["Brake PMC 2025", "Brake multi-capteurs", "Random Forest (95,85 %)", "+6,6 % vs SVM"],
            ["arXiv 2024 RUL", "C-MAPSS FD001/3", "LSTM (RMSE 14,93)", "−40 % vs Ridge"],
            ["Elsherif 2025 Nature", "C-MAPSS FD001-4", "CAELSTM (RMSE 13,40)", "−6 % vs LSTM"],
            ["Hydraulic IEEE 2024", "UCI Hydraulic", "XGBoost/RF (F1 0,9988)", "+1,6 % vs Stacking"],
        ],
        caption="Tableau 1.5 — Sélection des 8 études comparatives multi-algorithmes (E1–E8)",
    )

    # --- 1.6 Etude de cas NASA ---
    doc.add_heading("1.6 Étude de cas approfondie — Dataset NASA C-MAPSS (Turbofan)", level=2)
    doc.add_paragraph(
        "Le dataset FD001 du C-MAPSS (Commercial Modular Aero-Propulsion System Simulation, NASA "
        "Ames Research Center) est le benchmark de référence mondial pour la maintenance "
        "prédictive et l'estimation de la RUL. Il simule des moteurs turbofan fonctionnant "
        "normalement puis se dégradant progressivement jusqu'à défaillance, avec un niveau "
        "d'usure initial inconnu et variable par moteur — un défi réaliste pour la modélisation. "
        "Cette étude de cas est traitée en détail ci-dessous car sa méthodologie (audit, "
        "nettoyage, EDA, benchmark de modèles) sert de modèle direct à la méthodologie appliquée "
        "au dataset i-SENSE (sections 2 à 4)."
    )

    doc.add_heading("1.6.1 Structure des données", level=3)
    doc.add_paragraph(
        "Chaque ligne = un cycle de fonctionnement d'un moteur : 2 identifiants (unit_number, "
        "time_cycles), 3 paramètres opérationnels (altitude, Mach, angle manette), 21 capteurs "
        "physiques (températures T2/T24/T30/T50, pressions P2/P15/P30/Ps30, vitesses de "
        "rotation Nf/Nc/NRf/NRc, ratios epr/phi/BPR, flux de refroidissement W31/W32/htBleed, "
        "demandes actionneurs Nf_dmd/PCNfR_dmd)."
    )
    add_table(
        doc, ["Fichier", "Contenu"], [
            ["train_FD001.txt", "Historique complet run-to-failure, 100 moteurs, 20 631 lignes"],
            ["test_FD001.txt", "Séquences tronquées avant défaillance, 100 moteurs de test"],
            ["RUL_FD001.txt", "Vérité terrain — RUL réelle au dernier cycle observé (100 valeurs)"],
        ],
        caption="Tableau 1.6 — Fichiers du dataset FD001",
    )

    doc.add_heading("1.6.2 Audit de qualité et nettoyage", level=3)
    doc.add_paragraph(
        "L'audit confirme l'absence totale de valeurs manquantes (caractéristique d'un dataset de "
        "simulation). Le nettoyage porte donc sur les capteurs constants (« morts », sans "
        "information de dégradation). Un seuillage par variance s'avère insuffisant (le « machine "
        "epsilon » en virgule flottante donne un écart-type non nul mais infime, ~10⁻¹⁵, pour un "
        "capteur réellement constant). La méthode retenue — comptage des valeurs uniques — "
        "détecte tous les capteurs constants sans ambiguïté : "
        "7 capteurs sont supprimés (T2, P2, epr, farB, Nf_dmd, PCNfR_dmd + 1 autre), réduisant "
        "le dataset de 26 à 19 colonnes actives. Une validation visuelle (P2 parfaitement "
        "horizontal vs T50 en tendance nette) confirme la pertinence du nettoyage."
    )

    doc.add_heading("1.6.3 Construction de la variable cible (RUL)", level=3)
    doc.add_paragraph(
        "RUL(t) = cycle_max(moteur) − t, calculée pour chaque ligne d'entraînement. La "
        "distribution brute atteint plusieurs centaines de cycles en début de vie — non "
        "informatif pour l'apprentissage (le moteur est alors en parfaite santé). La convention "
        "standard (Heimes, 2008) plafonne la RUL à 125 cycles (RUL clipping), évitant de "
        "demander au modèle de distinguer une RUL de 200 d'une RUL de 300."
    )

    doc.add_heading("1.6.4 Analyse exploratoire du dataset NASA", level=3)
    add_table(
        doc, ["Statistique", "Valeur"], [
            ["Durée de vie moyenne", "206 cycles"],
            ["Capteurs les plus corrélés à la RUL", "Ps30 (r = −0,78), T50 (r = −0,76)"],
            ["Capteurs positivement corrélés", "phi (0,75), P30 (0,73), W32 (0,71), W31 (0,70)"],
        ],
        caption="Tableau 1.7 — Résultats clés de l'EDA NASA C-MAPSS",
        note=(
            "T50 (température sortie turbine BP) et Ps30 (pression statique sortie compresseur "
            "HP) ressortent comme les deux capteurs les plus informatifs — cohérent avec la "
            "physique de dégradation d'un turbofan. La séparation Healthy (20 premiers cycles) "
            "vs Failing (20 derniers cycles) par KDE confirme que ces capteurs sont discriminants. "
            "Une projection PCA 2D révèle un gradient visuel clair de la trajectoire de "
            "dégradation, du moteur sain (RUL élevée) au moteur défaillant (RUL proche de 0)."
        ),
    )
    add_picture_centered(
        doc, f"{CHART_DIR}/nasa_sensor_correlation.png",
        caption="Figure 1.2 — Corrélation des capteurs NASA C-MAPSS avec la RUL",
        description=(
            "Ps30 et T50 (barres rouges, corrélation négative la plus forte) sont les 2 capteurs "
            "les plus informatifs pour la prédiction : leur valeur augmente avec la dégradation, "
            "donc diminue quand la RUL restante diminue — d'où une corrélation négative."
        ),
    )

    doc.add_heading("1.6.5 Prétraitement et modélisation", level=3)
    doc.add_paragraph(
        "Normalisation Min-Max [0,1], scaler ajusté uniquement sur l'entraînement puis appliqué "
        "au test (prévention du data leakage). Spécificité du protocole C-MAPSS : le jeu de test "
        "ne contient que le dernier cycle observé de chaque moteur (100 observations comparées "
        "aux 100 RUL de référence)."
    )
    add_table(
        doc, ["Rang", "Modèle", "RMSE (cycles)", "R²"], [
            ["1er", "XGBoost", "17,69", "0,819"],
            ["2ème", "Random Forest", "18,16", "0,809"],
            ["3ème", "SVR", "19,84", "0,772"],
            ["4ème", "Régression Linéaire", "21,90", "0,722"],
        ],
        caption="Tableau 1.8 — Benchmark comparatif des 4 modèles sur NASA C-MAPSS FD001",
        note=(
            "XGBoost est retenu comme modèle final : meilleur RMSE, gestion native des "
            "non-linéarités, régularisation intégrée, inférence rapide. Une simulation "
            "cycle-par-cycle (« jumeau numérique ») sur le moteur de test n°24 confirme "
            "visuellement que XGBoost suit fidèlement la décroissance réelle de la RUL, avec une "
            "zone critique surlignée sur les 20 derniers cycles avant panne."
        ),
    )
    add_picture_centered(
        doc, f"{CHART_DIR}/nasa_benchmark.png",
        caption="Figure 1.3 — Benchmark RMSE et R² des 4 modèles sur NASA C-MAPSS FD001",
        description=(
            "Progression régulière de la régression linéaire (baseline) vers XGBoost : chaque "
            "modèle plus complexe apporte un gain mesurable, justifiant le surcoût computationnel "
            "de XGBoost par rapport à une simple régression linéaire."
        ),
    )

    # --- 1.7 Deep learning methods ---
    doc.add_heading("1.7 Méthodes de Deep Learning pour la construction d'un Health Index", level=2)
    doc.add_paragraph(
        "Référence principale : González-Muñiz et al. (2022), « Health indicator for machine "
        "condition monitoring built in the latent space of a deep autoencoder », Reliability "
        "Engineering and System Safety, vol. 224. Contrairement aux approches supervisées (RUL "
        "connue), cette méthode construit un Health Index (HI) de façon non supervisée : un "
        "autoencodeur est entraîné uniquement sur des données saines, et l'erreur de "
        "reconstruction sur de nouvelles données sert d'indicateur de dégradation. C'est la "
        "méthodologie retenue pour la section 6 (à venir) de ce projet."
    )

    doc.add_heading("1.7.1 Principe de l'autoencodeur profond (DAE)", level=3)
    doc.add_paragraph(
        "Un autoencodeur comprime l'entrée X vers un espace latent Z (goulot d'étranglement) via "
        "un encodeur, puis reconstruit X' depuis Z via un décodeur symétrique. Entraîné "
        "uniquement sur du comportement normal, il reconstruit bien les échantillons sains mais "
        "mal les échantillons dégradés — d'où une erreur de reconstruction élevée en cas "
        "d'anomalie. Loss : L_DAE = (1/N) × Σ ||Xᵢ − Xᵢ'||²."
    )

    doc.add_heading("1.7.2 Trois familles de Health Index comparées", level=3)
    add_table(
        doc, ["Approche", "Formule", "Principe"], [
            ["a) Erreur de reconstruction classique", "ε_REC(x) = ‖x − x̂‖₂", "Compare x et x̂ uniquement dans l'espace d'entrée"],
            ["b) RaPP (couches cachées)", "ε_SAP, ε_NAP", "Compare les activations à CHAQUE couche cachée de l'encodeur"],
            ["c) Proposition — espace latent seul", "ε_SAP_LS, ε_NAP_LS", "Compare uniquement au niveau du bottleneck (représentation la plus compacte)"],
        ],
        caption="Tableau 1.9 — Les trois familles de Health Index comparées dans l'article",
        note=(
            "Hypothèse centrale de l'article : l'espace latent, plus robuste au bruit que "
            "l'espace d'entrée complet, produit un HI plus fiable pour les tâches de pronostic. "
            "ε_NAP_LS ajoute une normalisation de type distance de Mahalanobis pour atténuer la "
            "dépendance entre dimensions latentes."
        ),
    )

    doc.add_heading("1.7.3 Protocole expérimental de référence", level=3)
    add_table(
        doc, ["Dataset", "Capteurs", "Cycles", "Origine"], [
            ["FD001", "21", "33 727", "NASA C-MAPSS (turbofan)"],
            ["FD003", "21", "41 316", "NASA C-MAPSS (turbofan)"],
            ["Mill", "6", "8 350 (fenêtré)", "BEST Lab UC Berkeley (usure outil de fraisage)"],
        ],
        caption="Tableau 1.10 — Les 3 datasets utilisés pour valider le Health Index",
    )
    doc.add_paragraph(
        "Deux architectures sont testées par dataset : autoencodeur profond classique (ex. FD001 : "
        "encodeur 21→10→20→10, bottleneck (2), décodeur symétrique, 200 époques, Adam) et "
        "autoencodeur variationnel — VAE (encodeur produisant une distribution μ,σ plutôt qu'un "
        "point fixe, perte = reconstruction + β × divergence KL vers une gaussienne standard, "
        "400 époques, rmsprop)."
    )

    doc.add_heading("1.7.4 Résultats — le Health Index latent l'emporte", level=3)
    add_table(
        doc, ["Indicateur", "Monotonicité (FD001)", "Prognosabilité (FD001)"], [
            ["ε_NAP_LS (proposé)", "0,455", "0,991"],
            ["ε_SAP_LS (proposé)", "0,416", "0,991"],
            ["ε_NAP (RaPP complet)", "0,156", "0,979"],
            ["ε_SAP (RaPP complet)", "0,265", "0,909"],
            ["ε_REC (classique)", "0,199", "0,893"],
        ],
        caption="Tableau 1.11 — Comparaison des 5 indicateurs sur FD001 (autoencodeur profond)",
        note=(
            "Sur les 3 datasets et les 2 architectures (sauf prognosabilité sur FD003), les "
            "indicateurs latents dominent l'erreur classique ET le RaPP complet. Défaut notable "
            "de ε_NAP (RaPP complet) : une hausse brutale juste avant la panne plutôt qu'une "
            "dérive progressive, nuisant à la détection précoce. Comparés à 5 descripteurs "
            "statistiques manuels (skewness, kurtosis, RMS, crest factor, variance), les "
            "indicateurs profonds restent globalement supérieurs et ne nécessitent pas "
            "d'expertise préalable pour choisir la bonne caractéristique."
        ),
    )
    add_picture_centered(
        doc, f"{CHART_DIR}/fd001_hi_comparison.png",
        caption="Figure 1.4 — Monotonicité et prognosabilité des 5 indicateurs sur FD001",
        description=(
            "Les deux indicateurs proposés par l'article (ε_NAP_LS, ε_SAP_LS, en haut) dominent "
            "nettement sur la monotonicité (barres bleues) et restent au niveau des meilleurs sur "
            "la prognosabilité (barres vertes), confirmant visuellement la supériorité de "
            "l'approche « espace latent seul » sur l'erreur classique et le RaPP complet."
        ),
    )
    doc.add_paragraph(
        "Interprétation géométrique : la projection de l'espace latent 2D en carte de dégradation "
        "montre que les indicateurs latents produisent une géométrie plus lisse et mieux alignée "
        "avec la densité des données saines (KDE) que les approches classiques — expliquant leur "
        "robustesse supérieure."
    )

    doc.add_heading("1.7.5 Autres architectures de Deep Learning pertinentes", level=3)
    add_table(
        doc, ["Méthode", "HI utilisé", "Données temporelles", "Entraînement"], [
            ["DAE classique", "‖X − X'‖² (entrée)", "Non", "Rapide"],
            ["DAE latent", "‖Z − Ẑ‖² (latent)", "Non", "Rapide"],
            ["VAE latent", "‖Z − Ẑ‖² + KL", "Non", "Modéré"],
            ["RaPP latent", "‖Z − Ẑ‖² (latent)", "Non", "Rapide"],
            ["LSTM-AE", "‖seq − seq'‖² latent", "Oui (T cycles)", "Lent"],
            ["CNN-AE", "‖X − X'‖² (filtres conv.)", "Oui (convolution)", "Modéré"],
        ],
        caption="Tableau 1.12 — Comparaison des 6 architectures envisageables",
        note=(
            "Le LSTM-AE capture les dépendances temporelles longues via ses portes "
            "(forget/input/output) mais est coûteux à entraîner ; le CNN-AE extrait des motifs "
            "locaux (oscillations, pics) plus rapidement mais avec une portée temporelle plus "
            "courte. Le DAE/VAE classique, appliqué instant par instant, reste le point de départ "
            "recommandé pour ce projet (section 6)."
        ),
    )
    doc.add_paragraph(
        "Conclusion de l'état de l'art : la chaîne complète capteurs → calcul d'indices → score "
        "ML → alerte de maintenance reste rarement déployée de bout en bout en environnement "
        "industriel réel — c'est précisément le verrou que ce projet cherche à lever."
    )


# ===========================================================================
# 2. CONTEXTE (capteur i-SENSE et son dataset)
# ===========================================================================

def add_section2_contexte(doc):
    doc.add_page_break()
    doc.add_heading("2. Contexte — le capteur i-SENSE et son dataset", level=1)
    doc.add_paragraph(
        "Cette section rassemble tout ce qui concerne le terrain réel du projet : le capteur IoT "
        "visé, ses 11 paramètres et leurs seuils normatifs, puis le dataset i-SENSE effectivement "
        "collecté (2 machines industrielles, données réelles) sur lequel portent les traitements "
        "des sections 3 et suivantes."
    )

    doc.add_heading("2.1 Le capteur IoT cible — 11 paramètres", level=2)
    add_table(
        doc, ["#", "Paramètre", "Plage"], [
            ["1", "Viscosité dynamique", "10–1500 cP"],
            ["2", "Viscosité cinématique 40°C", "2–800 cSt"],
            ["3", "Densité", "600–1300 kg/m³"],
            ["4", "Température", "−40°C à +125°C"],
            ["5", "Constante diélectrique", "1–80"],
            ["6", "Humidité", "0–100 % / 0–100 000 ppm"],
            ["7", "Conductivité électrique", "1,0 pS/cm – 100 µS/cm"],
            ["8", "Niveau de contamination", "ISO 4406 / SAE AS4059"],
            ["9", "Particules d'usure", "30–1600+ µm"],
            ["10", "Vibration", "0–9,8 m/s²"],
            ["11", "Pression", "0–10 MPa"],
        ],
        caption="Tableau 2.1 — 11 paramètres mesurés par le capteur IoT cible du projet",
        note=(
            "Le dataset i-SENSE actuellement disponible (section 2.2) couvre une partie de ces "
            "paramètres (viscosité, densité, température, constante diélectrique, humidité, "
            "conductivité, contamination ISO 4406, vibration, pression) — les particules d'usure "
            "ne sont pas mesurées dans le jeu de données actuel."
        ),
    )
    doc.add_paragraph(
        "Objectifs ML du projet : classification multi-classes (Bonne / Dégradée / À changer) et "
        "régression de la Durée de Vie Résiduelle (RUL) en heures."
    )
    add_table(
        doc, ["Paramètre", "Norme", "Seuils"], [
            ["Vibration", "ISO 20816-3:2022", "Zone A < 1,4–3,5 m/s² selon puissance ; Zone D > 7,1–18,0 m/s²"],
            ["Contamination ISO 4406", "ISO 4406:2021", "Ex. hydraulique industrielle : code cible 18/16/13"],
            ["Humidité", "ASTM D6304 (Karl Fischer)", "Ex. hydraulique : bon < 200 ppm, alarme > 500 ppm"],
            ["Constante diélectrique", "ASTM D924 / IEC 60247", "Bon Δε < 0,3 ; imminente ≥ 1,5"],
            ["Pression", "Spécifications OEM", "Ex. hydraulique légère : nominale 5–10 MPa"],
        ],
        caption="Tableau 2.2 — Exemples de seuils d'alerte normalisés par paramètre",
    )

    doc.add_heading("2.2 Le dataset i-SENSE", level=2)
    doc.add_paragraph(
        "Les données sont extraites depuis l'API REST de la plateforme i-SENSE "
        "(v3back-demo.i-sense.io), qui surveille en temps réel la qualité de l'huile de "
        "lubrification de deux machines industrielles : Motosoufflante A (Oil-AA0005) et "
        "Motosoufflante B (Oil-AA0004). Format d'export : CSV wide — une ligne par instant de "
        "mesure, une colonne par variable physique."
    )
    add_table(
        doc, ["Indicateur", "Valeur (audit initial)"], [
            ["Dimensions", "39 159 lignes × 16 colonnes (14 capteurs + timestamp + ID machine)"],
            ["Fréquence d'acquisition", "10 minutes (nominale)"],
            ["Période Machine A", "29 déc. 2025 → 13 juil. 2026 (196 jours)"],
            ["Période Machine B", "02 jan. 2026 → 13 juil. 2026 (192 jours)"],
        ],
        caption="Tableau 2.3 — Structure générale du dataset i-SENSE (audit initial)",
        note=(
            "Le dataset a continué de croître depuis cet audit initial ; l'extraction utilisée "
            "pour les traitements de ce projet (sections 3-5) compte 42 141 lignes, la période "
            "s'étendant jusqu'au 06/08/2026."
        ),
    )
    add_table(
        doc, ["Variable", "Unité"], [
            ["DC (constante diélectrique)", "adimensionnel"],
            ["Density", "kg/m³"],
            ["Dynamic Viscosity", "cP"],
            ["ISO 4 / ISO 6 / ISO 14", "codes ISO 4406 (contamination)"],
            ["Kinematic Viscosity", "cSt"],
            ["Oil Conductivity", "nS/m"],
            ["Oil H2O Saturation", "%"],
            ["Oil H2O ppm", "ppm"],
            ["Oil Pressure", "Bar"],
            ["Oil System Vibration", "mm/s²"],
            ["Oil Temperature", "°C"],
            ["Viscosity at 40°C", "cSt"],
        ],
        caption="Tableau 2.4 — Les 14 variables physiques du dataset i-SENSE",
    )
    doc.add_paragraph(
        "Contrairement au dataset NASA (0 % de NaN, section 1.6.2), le dataset i-SENSE, issu de "
        "capteurs réels, présente plusieurs classes de problèmes de qualité identifiés dès "
        "l'audit initial : valeurs sentinelles (−99.99 / −9999 signifiant « capteur "
        "indisponible », par analogie aux capteurs constants de NASA), valeurs manquantes "
        "structurelles concentrées sur `Oil System Vibration` (module non actif en continu), un "
        "état opérationnel « machine arrêtée » signalé par une pression quasi nulle "
        "(1×10⁻⁷ Bar), des gaps temporels (arrêts machine, jusqu'à 13,8 jours consécutifs), et "
        "une redondance entre `Kinematic Viscosity` et `Viscosity at 40°C` (corrélation de "
        "Pearson 0,85)."
    )
    add_table(
        doc, ["Problème", "Ampleur (audit initial)", "Analogie NASA"], [
            ["Valeurs sentinelles −99.99/−9999", "153 lignes (0,39 %)", "Capteurs constants (P2, T2...)"],
            ["Pression 1e-7 Bar (machine OFF)", "Jusqu'à 47 % des lignes (Machine B)", "Conditions opérationnelles (settings)"],
            ["Vibration structurellement incomplète", "67 % de NaN", "Aucun analogue direct"],
            ["Gaps temporels", "103 gaps A / 90 gaps B", "Aucun (NASA continu)"],
            ["Redondance viscosité", "r = 0,85", "Capteurs redondants"],
        ],
        caption="Tableau 2.5 — Synthèse de l'audit de qualité initial du dataset i-SENSE",
        note=(
            "Ce plan d'action initial a directement guidé le traitement effectif du dataset "
            "(section 3), avec des ajustements importants découverts lors du traitement réel "
            "(l'ampleur de certains problèmes s'est révélée différente de l'estimation initiale, "
            "notamment pour le trou capteur synchrone — voir section 3.3)."
        ),
    )


# ===========================================================================
# 3. TRAITEMENT DES DONNEES (nettoyage + imputations)
# ===========================================================================

def add_section3_traitement(doc):
    doc.add_page_break()
    doc.add_heading("3. Traitement des données — nettoyage et imputations", level=1)
    doc.add_paragraph(
        "Cette section documente le travail effectivement réalisé sur le dataset i-SENSE décrit "
        "en section 2 : nettoyage initial, puis deux campagnes d'imputation méthodiques — "
        "chacune comparant systématiquement des méthodes mathématiques, des méthodes de Machine "
        "Learning, et des méthodes statistiques simples (moyenne, médiane) avant de choisir la "
        "méthode retenue."
    )

    doc.add_heading("3.1 Nettoyage initial du dataset", level=2)
    add_table(
        doc, ["Étape", "Action"], [
            ["1", "Remplacement des valeurs sentinelles (−99.99 / −9999) par NaN sur 12 variables (994 lignes concernées, épisode synchrone unique)"],
            ["2", "Suppression des lignes totalement vides (artefact de jonction API)"],
            ["3", "Création du flag machine_state (ON si pression > 0,1 Bar, sinon OFF)"],
            ["4", "Forward-fill de ISO 6 / ISO 14 (désynchronisation export API, ~2 %)"],
            ["5", "Conservation de Oil System Vibration + flag vibration_available"],
            ["6", "Découpage en 352 sessions continues (nouvelle session si écart > 1h)"],
            ["7", "Suppression de Kinematic Viscosity (redondante avec Viscosity at 40°C, r=0,86) — variable normalisée conservée"],
        ],
        caption="Tableau 3.1 — Les 7 étapes du nettoyage initial",
    )
    doc.add_paragraph(
        "Investigation complémentaire : Density présente une corrélation quasi parfaite avec "
        "Oil Temperature (r = −1,000, résidus < 0,09 kg/m³) — relation déterministe "
        "(Density ≈ −0,6734 × Température + 880,15, cohérente avec une correction de dilatation "
        "thermique standard), pas une redondance physique mesurée. Density est donc exclue des "
        "étapes suivantes. Autre découverte : Motosoufflante B n'a été en état ON qu'en janvier "
        "2026 et reste arrêtée en continu depuis — ses données OFF (variance erratique) ne "
        "reflètent pas l'état réel de l'huile."
    )

    doc.add_heading("3.2 Imputation de la vibration (Oil System Vibration)", level=2)
    doc.add_paragraph(
        "Oil System Vibration est manquante à 65,9 % de ses valeurs. La structure des NaN "
        "alterne rafales courtes (médiane 2 points) et trous pouvant atteindre ~2000 mesures "
        "consécutives. Quatre méthodes sont comparées, couvrant les approches mathématique et "
        "ML, par masquage de valeurs connues (MAE/RMSE/R²) :"
    )
    add_table(
        doc, ["Méthode", "Type", "R² (trous courts)", "R² (trous longs)"], [
            ["Interpolation temporelle (spline)", "Mathématique", "0,427", "-0,144"],
            ["Random Forest", "ML", "0,519", "-0,181"],
            ["HistGradientBoosting", "ML", "0,495", "-0,097"],
            ["K-Nearest Neighbors", "ML", "0,417", "-0,362"],
        ],
        caption="Tableau 3.2 — Comparaison de 4 méthodes pour Oil System Vibration",
        note=(
            "Une 5ᵉ proposition, la moyenne/médiane conditionnelle (par machine × état ON/OFF), "
            "a également été testée sur le régime « trous longs » (mêmes points masqués) : "
            "moyenne conditionnelle R² = -0,369 au mieux, médiane conditionnelle encore moins "
            "bonne — toutes deux restent en retrait de HistGradBoost (R² = -0,097). La "
            "proposition n'a pas été retenue. Constat clé : sur les trous longs, AUCUNE méthode "
            "ne fait mieux qu'une prédiction proche de la moyenne globale (R² négatif partout)."
        ),
    )
    add_picture_centered(
        doc, "eda_output/vibration_imputation/01_metrics_random_masking.png",
        caption="Figure 3.1 — MAE / RMSE / R² des 4 méthodes, trous courts (masquage aléatoire)",
        description="Random Forest (ML) l'emporte sur les trois métriques simultanément pour ce régime de trous courts.",
    )
    add_picture_centered(
        doc, "eda_output/vibration_imputation/03_metrics_long_gaps.png",
        caption="Figure 3.2 — MAE / RMSE / R² des 4 méthodes, trous longs (≥ 20 mesures)",
        description=(
            "Les 4 R² sont négatifs (barres sous zéro) : le classement s'inverse par rapport à la "
            "figure 3.1 — HistGradientBoosting devient la méthode la moins mauvaise, Random Forest "
            "passe dernière ex-æquo."
        ),
    )
    add_picture_centered(
        doc, "eda_output/vibration_imputation/06_metrics_conditional_long_gaps.png",
        caption="Figure 3.3 — Moyenne/médiane conditionnelle vs référence ML, sur les mêmes trous longs",
        description=(
            "La proposition « moyenne/médiane conditionnelle par machine × état » (méthode "
            "statistique simple, 4 variantes à gauche) reste en retrait de HistGradBoost "
            "(référence ML, à droite) — elle n'a donc pas été retenue."
        ),
    )
    doc.add_paragraph(
        "Décision finale (méthode hybride) : Random Forest pour les trous courts (< 20 mesures "
        "du point connu), abstention (NaN conservé) au-delà. Résultat : 53,1 % des valeurs "
        "manquantes comblées (14 737), 46,9 % (13 031) laissées NaN par choix méthodologique "
        "assumé — un compromis complétude/fiabilité plutôt qu'un remplissage forcé."
    )

    doc.add_heading("3.3 Imputation du trou capteur synchrone (7 variables)", level=2)
    doc.add_paragraph(
        "Investigation : les 2,36 % de NaN restants sur 8 variables ont d'abord été supposés "
        "être de petites désynchronisations (comme ISO 6/14). En réalité, il s'agit d'un trou "
        "contigu unique de 478 mesures (Motosoufflante A, juillet 2026) et 516 mesures "
        "(Motosoufflante B, juillet 2026), affectant simultanément DC, Dynamic Viscosity, ISO 4, "
        "Oil H2O Saturation, Oil H2O ppm, Oil Temperature et Viscosity at 40°C — mais laissant "
        "ISO 6, ISO 14, Oil Conductivity et Oil Pressure disponibles en continu."
    )
    add_table(
        doc, ["Méthode", "Type", "R² moyen (7 variables)"], [
            ["Forward-fill", "Mathématique", "0,783"],
            ["Interpolation", "Mathématique", "0,685"],
            ["Moyenne globale", "Statistique simple", "-1,008"],
            ["Médiane globale", "Statistique simple", "-0,969"],
            ["Moyenne conditionnelle (machine×état)", "Statistique simple", "-0,183"],
            ["Médiane conditionnelle (machine×état)", "Statistique simple", "-0,312"],
            ["Random Forest", "ML", "0,622"],
        ],
        caption="Tableau 3.3 — Comparaison de 7 méthodes pour le trou capteur synchrone",
        note=(
            "Résultat contre-intuitif : forward-fill (mathématique) domine en moyenne — la "
            "machine reste dans un régime stable sur ce trou. Mais l'analyse variable par "
            "variable révèle une exception : sur ISO 4 seul, forward-fill échoue (R² ≈ 0) et "
            "Random Forest prend l'avantage (R² = 0,66), en s'appuyant sur ISO 6/ISO 14 restées "
            "disponibles. Sur Oil Temperature à l'inverse, Random Forest est moins bon que "
            "forward-fill (R² négatif contre 0,76)."
        ),
    )
    add_picture_centered(
        doc, "eda_output/sensor_gap_imputation/01_metrics_aggregate.png",
        caption="Figure 3.4 — Comparaison agrégée des 7 méthodes (moyenne sur les 7 variables)",
        description=(
            "Résultat inverse de la vibration (figures 3.1-3.2) : ici forward-fill, une méthode "
            "mathématique simple, domine largement en moyenne — y compris devant Random Forest."
        ),
    )
    add_picture_centered(
        doc, "eda_output/sensor_gap_imputation/02_r2_by_variable.png",
        caption="Figure 3.5 — R² par méthode et par variable",
        description=(
            "Détail justifiant la méthode hybride : sur ISO 4 (3ᵉ groupe de barres), forward-fill "
            "et interpolation chutent près de zéro alors que Random Forest (orange) reste la "
            "seule méthode franchement positive."
        ),
    )
    doc.add_paragraph(
        "Décision finale (méthode hybride, par variable) : forward-fill pour 6 variables "
        "(DC, Dynamic Viscosity, Oil H2O Saturation, Oil H2O ppm, Oil Temperature, "
        "Viscosity at 40°C), Random Forest uniquement pour ISO 4. Résultat : 100 % des 6 958 "
        "valeurs manquantes comblées (contrairement à la vibration, aucune abstention ici, les "
        "tests montrant un R² fiable pour les 7 variables)."
    )

    doc.add_heading("3.4 Synthèse méthodologique — mathématique, ML, moyenne, médiane", level=2)
    add_table(
        doc, ["Famille de méthode", "Exemples testés", "Cas où elle gagne"], [
            ["Mathématique", "Interpolation (spline/linéaire), forward-fill", "Trous courts à moyens sur signal stable (forward-fill : trou capteur synchrone)"],
            ["Machine Learning", "Random Forest, HistGradientBoosting, KNN", "Trous courts avec forte corrélation inter-capteurs (vibration ; ISO 4)"],
            ["Moyenne (globale/conditionnelle)", "Moyenne simple, moyenne par machine×état", "Jamais gagnante dans ce projet — toujours dominée par ML ou mathématique"],
            ["Médiane (globale/conditionnelle)", "Médiane simple, médiane par machine×état", "Jamais gagnante — systématiquement pire que la moyenne équivalente"],
        ],
        caption="Tableau 3.4 — Bilan comparatif des 4 familles de méthodes testées sur l'ensemble du projet",
        note=(
            "Enseignement transversal : le choix de la méthode dépend fortement du régime de "
            "trou (court vs long) et de la disponibilité de capteurs corrélés pendant le trou — "
            "aucune méthode unique ne domine dans tous les cas, d'où l'approche hybride retenue "
            "systématiquement (par distance au point connu pour la vibration, par variable pour "
            "le trou capteur synchrone)."
        ),
    )


# ===========================================================================
# 4. EDA
# ===========================================================================

def add_section4_eda(doc):
    doc.add_page_break()
    doc.add_heading("4. Analyse exploratoire des données (EDA)", level=1)
    doc.add_paragraph(
        "L'EDA a été menée en deux temps : avant imputation (sur le dataset nettoyé, section 3) "
        "pour guider les décisions de traitement, puis après imputation pour valider que le "
        "remplissage n'a pas dénaturé les données."
    )

    doc.add_heading("4.1 EDA avant imputation", level=2)
    add_picture_centered(
        doc, "eda_output/04_correlation_heatmap.png",
        caption="Figure 4.1 — Matrice de corrélation des variables santé (avant imputation)",
        description=(
            "A permis d'identifier deux redondances majeures : Density/Oil Temperature "
            "(r = -1,000, relation déterministe) et ISO 6/ISO 14 (r = 0,966, cohérent — tailles "
            "de particules voisines)."
        ),
    )
    add_picture_centered(
        doc, "eda_output/05_timeseries_overview.png",
        caption="Figure 4.2 — Viscosité, température et indice de contamination dans le temps",
        description=(
            "A révélé la chute de viscosité de Motosoufflante B (février 2026), qui s'est avérée "
            "liée à un arrêt prolongé de la machine (ON uniquement en janvier 2026) plutôt qu'à "
            "un événement de dégradation réelle de l'huile."
        ),
    )
    add_picture_centered(
        doc, "eda_output/06_machine_state.png",
        caption="Figure 4.3 — Répartition du temps ON vs OFF par machine",
        description="Confirme numériquement que Motosoufflante B est arrêtée l'immense majorité du temps couvert.",
    )

    doc.add_heading("4.2 EDA post-imputation (validation)", level=2)
    add_picture_centered(
        doc, "eda_output/post_imputation/01_missing_before_after.png",
        caption="Figure 4.4 — Taux de NaN avant vs après remplissage",
        description=(
            "Les 7 variables du trou capteur synchrone passent à 0 % de NaN ; Oil System "
            "Vibration passe de 65,9 % à 30,9 % (remplissage partiel assumé)."
        ),
    )
    add_picture_centered(
        doc, "eda_output/post_imputation/02_distribution_overlay.png",
        caption="Figure 4.5 — Distributions avant (mesuré) vs après remplissage",
        description=(
            "Test de sanité réussi : les distributions se superposent presque parfaitement pour "
            "les 8 variables imputées — aucun pic artificiel, aucune déformation détectée."
        ),
    )
    add_picture_centered(
        doc, "eda_output/post_imputation/04_gap_zoom_sensor.png",
        caption="Figure 4.6 — Zoom sur le trou capteur synchrone, avant/après remplissage",
        description=(
            "Vérification visuelle directe sur le vrai trou (pas un trou synthétique de test) : "
            "le forward-fill comble correctement le vide de juillet 2026 sur DC."
        ),
    )


# ===========================================================================
# 5. FEATURE ENGINEERING
# ===========================================================================

def add_section5_feature_engineering(doc):
    doc.add_page_break()
    doc.add_heading("5. Feature engineering", level=1)
    doc.add_paragraph(
        "Sur les 11 familles de features prévues pour le dataset final (issu de la section 3), "
        "2 sont terminées et détaillées ci-dessous avec leurs résultats effectifs ; les 9 autres "
        "restent à réaliser (tableau 5.1)."
    )

    # --- 5.1 Calendaires ---
    doc.add_heading("5.1 Features calendaires — réalisées", level=2)
    doc.add_paragraph(
        "Deux colonnes brutes sont extraites de `created_at` : `hour` (0-23) et `day_of_week` "
        "(0-6). Comme un modèle ne « voit » pas que 23h et 0h sont des heures proches si on lui "
        "donne l'entier brut (discontinuité 23→0), l'heure est en plus encodée de façon "
        "cyclique : `hour_sin = sin(2π×heure/24)` et `hour_cos = cos(2π×heure/24)`. Ce codage "
        "place chaque heure sur un cercle plutôt que sur une droite, de sorte que 23h59 et 0h01 "
        "restent des points géométriquement voisins pour le modèle."
    )
    add_picture_centered(
        doc, f"{CHART_DIR}/calendar_cyclic_encoding.png",
        caption="Figure 5.1 — Encodage cyclique de l'heure : hour_sin vs hour_cos",
        description=(
            "Chaque heure de la journée (0 à 23) est projetée sur un point du cercle unité "
            "(couleur = heure). 23h et 0h se retrouvent bien adjacents sur le cercle, contrairement "
            "à un encodage entier brut où ils seraient aux deux extrémités opposées de l'échelle."
        ),
    )
    add_table(
        doc, ["Résultat mesuré", "Valeur"], [
            ["Colonnes ajoutées", "hour, day_of_week, hour_sin, hour_cos"],
            ["Lignes traitées", "42 141 (100 % — created_at toujours renseigné, 0 NaN introduit)"],
            ["Plage de hour_sin / hour_cos", "[-1,000 ; 1,000] (bornes théoriques respectées)"],
        ],
        caption="Tableau 5.1 — Résultat effectif des features calendaires sur le dataset i-SENSE",
        note=(
            "Ces features seront exploitées si l'activité de la machine ou la dégradation de "
            "l'huile suivent un rythme journalier (équipes, cycles de production) — hypothèse à "
            "vérifier lors de la modélisation."
        ),
    )

    # --- 5.2 Session ---
    doc.add_heading("5.2 Features de session — réalisées", level=2)
    doc.add_paragraph(
        "Deux colonnes sont calculées à partir de `session_id` (352 sessions continues déjà "
        "identifiées lors du nettoyage, section 3.1) : `time_in_session_h`, le temps écoulé "
        "(en heures) depuis le premier point de la session en cours, et "
        "`measure_index_in_session`, le rang (0, 1, 2…) de la mesure dans sa session. Ces deux "
        "variables jouent le rôle du compteur `time_cycles` de NASA C-MAPSS (section 1.6.1) : "
        "après un arrêt/redémarrage, l'huile peut se comporter différemment (transitoire "
        "thermique), et un « temps depuis le début de session » est plus pertinent que la date "
        "calendaire brute pour modéliser une dégradation progressive à l'intérieur d'un cycle de "
        "fonctionnement continu."
    )
    add_picture_centered(
        doc, f"{CHART_DIR}/session_features_distribution.png",
        caption="Figure 5.2 — Distributions de time_in_session_h et measure_index_in_session",
        description=(
            "La distribution très étalée (échelle log à gauche) reflète l'hétérogénéité déjà "
            "observée à l'EDA (352 sessions de durées très variables, de quelques minutes à "
            "1339,7 heures) — cohérent avec le tableau 5.2 ci-dessous."
        ),
    )
    add_table(
        doc, ["Statistique", "time_in_session_h", "measure_index_in_session"], [
            ["Minimum", "0,00 h (début de session)", "0 (1ère mesure de la session)"],
            ["Médiane", "112,17 h", "610"],
            ["Moyenne", "272,50 h", "1 546,8"],
            ["Maximum", "1 339,68 h (≈ 55,8 jours)", "7 903"],
        ],
        caption="Tableau 5.2 — Résultat effectif des features de session sur le dataset i-SENSE (42 141 lignes, 0 NaN)",
        note=(
            "Le maximum de time_in_session_h (1 339,68 h) correspond exactement à la session "
            "continue la plus longue déjà identifiée à l'EDA (section 4) — cohérence vérifiée "
            "entre le nettoyage et le feature engineering."
        ),
    )

    # --- 5.3 A faire ---
    doc.add_heading("5.3 Features restantes à réaliser", level=2)
    add_table(
        doc, ["#", "Famille de features", "Description", "Statut"], [
            ["3", "Statistiques en fenêtre glissante", "Moyenne et écart-type sur 1h / 3h / 24h", "⏳ À faire"],
            ["4", "Tendance / vitesse de dégradation", "Dérivée discrète, pente sur fenêtre glissante", "⏳ À faire"],
            ["5", "Features décalées (lags)", "Valeurs à t-1, t-2, t-3", "⏳ À faire"],
            ["6", "Moyenne mobile exponentielle (EWMA)", "Lissage pondéré donnant plus de poids aux mesures récentes", "⏳ À faire"],
            ["7", "Indices métier", "Indice de contamination ISO, écart au grade de viscosité, interaction température×viscosité", "⏳ À faire"],
            ["8", "Standardisation par machine", "Z-score calculé séparément pour Motosoufflante A et B", "⏳ À faire"],
            ["9", "Flags de seuils d'alerte", "Contamination, eau, température, vibration élevées", "⏳ À faire"],
            ["10", "Features de confiance des données imputées", "Score de confiance pour la vibration, flag du trou capteur comblé", "⏳ À faire"],
            ["11", "Encodage catégoriel", "One-hot encoding de la machine et de l'état ON/OFF", "⏳ À faire"],
        ],
        caption="Tableau 5.3 — Familles de features restant à implémenter",
        note=(
            "Ces familles sont conceptuellement définies (méthodologie déjà éprouvée sur le "
            "dataset NASA, section 1.6) mais pas encore implémentées sur le dataset i-SENSE "
            "final. Une étape ultérieure de sélection de features (corrélation, importance de "
            "variables) sera menée une fois l'ensemble réalisé, avant la modélisation."
        ),
    )
    add_picture_centered(
        doc, f"{CHART_DIR}/feature_engineering_status.png",
        caption="Figure 5.3 — Avancement du feature engineering",
        description="2 familles sur 11 sont finalisées à ce stade du projet ; les 9 restantes sont planifiées.",
    )


# ===========================================================================
# 6. PREDICTION DU HEALTH INDEX PAR AUTOENCODEUR (A VENIR)
# ===========================================================================

def add_section6_prediction_hi(doc):
    doc.add_page_break()
    doc.add_heading("6. Prédiction du Health Index par autoencodeur (à venir)", level=1)
    doc.add_paragraph(
        "⚠️ Étape non encore réalisée. En s'appuyant directement sur la méthodologie validée par "
        "González-Muñiz et al. (2022), détaillée en section 1.7, les travaux prévus sont :"
    )

    doc.add_heading("6.1 Autoencodeur profond classique (DAE)", level=2)
    doc.add_paragraph(
        "Entraînement d'un autoencodeur sur les périodes de fonctionnement considérées comme "
        "saines (machine ON, hors épisodes anomaliques identifiés). Architecture envisagée par "
        "analogie avec FD001/FD003 (section 1.7.3) : encodeur (n_features → 8 → 4) → bottleneck "
        "(2) → décodeur symétrique (4 → 8 → n_features), fonction d'activation ReLU, optimiseur "
        "Adam."
    )

    doc.add_heading("6.2 Autoencodeur variationnel (VAE)", level=2)
    doc.add_paragraph(
        "Extension probabiliste à explorer en complément : encodeur produisant une distribution "
        "(μ, σ) plutôt qu'un point latent fixe, entraînement par perte ELBO "
        "(reconstruction + β × divergence KL). Piste d'amélioration ultérieure, plus coûteuse à "
        "régler (poids du terme KL) pour un gain incertain sur ce volume de données — à tester "
        "après validation du DAE classique."
    )

    doc.add_heading("6.3 Calcul du Health Index — erreur de reconstruction latente", level=2)
    doc.add_paragraph(
        "Conformément aux résultats de l'article de référence (section 1.7.4), le Health Index "
        "retenu sera calculé dans l'espace latent plutôt que dans l'espace d'entrée : "
        "HI = ε_NAP_LS(x) = ‖(d(x) − μ_X)ᵀ V Σ⁻¹‖₂, où d(x) = h_l(x) − ĥ_l(x) est l'écart entre "
        "la représentation latente de x et celle de sa reconstruction ré-encodée. Ce choix est "
        "justifié par la performance supérieure démontrée (monotonicité, trendabilité, "
        "prognosabilité) sur les 3 datasets de référence, quelle que soit l'architecture "
        "d'autoencodeur utilisée."
    )
    doc.add_paragraph(
        "Évaluation prévue selon les 3 métriques standards : monotonicité (régularité de la "
        "tendance du HI avec la dégradation), trendabilité (similarité entre trajectoires de "
        "dégradation sur les deux machines), prognosabilité (variabilité du HI en fin de vie "
        "relative à sa plage initiale-finale)."
    )
    doc.add_paragraph(
        "Ces travaux s'appuieront sur le dataset produit par les sections 3 et 4 de ce rapport — "
        "nettoyé et imputé avec traçabilité complète (colonnes de confiance/source) — et seront "
        "lancés une fois le feature engineering (section 5) finalisé."
    )


# ===========================================================================
# 7. VISUALISATION (A VENIR)
# ===========================================================================

def add_section7_visualisation(doc):
    doc.add_page_break()
    doc.add_heading("7. Visualisation (à venir)", level=1)
    doc.add_paragraph(
        "⚠️ Étape non encore réalisée. Les visualisations prévues pour restituer les résultats "
        "de la section 6 :"
    )
    bullets(doc, [
        "Trajectoires du Health Index dans le temps, par machine, avec zone critique surlignée",
        "Projection de l'espace latent en 2D (carte de dégradation colorée par HI ou par ancienneté)",
        "Comparaison visuelle des 3 familles de HI (erreur classique, RaPP complet, RaPP latent) sur les données i-SENSE, à l'image de la figure 1.4",
        "Tableau de bord de synthèse (seuils d'alerte, état ON/OFF, confiance des données imputées)",
    ])


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    generate_auxiliary_charts()

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    add_title_page(doc)
    add_toc_note(doc)
    add_section1_etat_art(doc)
    add_section2_contexte(doc)
    add_section3_traitement(doc)
    add_section4_eda(doc)
    add_section5_feature_engineering(doc)
    add_section6_prediction_hi(doc)
    add_section7_visualisation(doc)

    doc.save("Rapport_PFE_Monitoring_Huile_iSENSE.docx")
    print("Généré : Rapport_PFE_Monitoring_Huile_iSENSE.docx")


if __name__ == "__main__":
    main()
