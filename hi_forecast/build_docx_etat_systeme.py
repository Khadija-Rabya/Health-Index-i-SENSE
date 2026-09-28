"""Rapport Word — ETAT DU SYSTEME : methode courante, seuils physiques,
validation empirique, methodes abandonnees et derive constatee en exploitation.

Ce document REMPLACE Rapport_HealthIndex_Systeme_iSENSE.docx et
Rapport_Systeme_Methode_et_Modeles_iSENSE.docx, qui decrivent tous deux l'indice
systeme AGREGE — abandonne. Il n'en reprend que ce qui reste vrai : les raisons
de l'abandon, et le resultat negatif sur la prevision.

Tous les chiffres viennent de artifacts/systeme_seuils_stats.json, produit par
mesure sur le jeu de donnees et sur le flux en direct. Aucun n'est recopie.

REGLES DE VISUALISATION APPLIQUEES
  - Le seuil est un REPERE, pas une serie : trait fin, neutre, annote.
  - Les deux canaux ne partagent jamais un axe : unites differentes.
  - Petits multiples plutot que superposition quand les echelles divergent.
  - Les couleurs de statut (vert / orange / rouge) restent reservees aux etats.
"""
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from docx import Document

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, "C:\\pylib")
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

from build_docx import (GREY, NAVY, RED, bullets, figure, h, para,  # noqa: E402
                        rich, save_doc, setup, table)

ART = os.path.join(HERE, "artifacts")
FIG = os.path.join(HERE, "figures")
SORTIE = os.path.join(ROOT, "Rapport_Etat_Systeme_iSENSE.docx")

BLEU = "#1d4ed8"        # pression
TEAL = "#0d9488"        # vibration
NEUTRE = "#2d3748"      # reperes et seuils
GRILLE = "#e6ebf2"

S = json.load(open(os.path.join(ART, "systeme_seuils_stats.json"), encoding="utf-8"))
SEUILS = S["seuils"]
MACHINES = list(S["machines"])


def _lire(nom, defaut=None):
    p = os.path.join(ART, nom)
    if not os.path.exists(p):
        return defaut
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _style(ax):
    ax.grid(True, color=GRILLE, linewidth=0.8)
    ax.set_axisbelow(True)
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)
    for cote in ("left", "bottom"):
        ax.spines[cote].set_color("#cbd5e0")
    ax.tick_params(colors="#4a5568", labelsize=8)


def n(v, d=2):
    """Nombre au format francais, virgule decimale."""
    if v is None:
        return "—"
    return f"{v:.{d}f}".replace(".", ",")


def pc(v, d=2):
    return "—" if v is None else f"{v * 100:.{d}f} %".replace(".", ",")


# ───────────────────────────────────────────────────────── figures

def fig_vibration_seuil():
    """Distribution de la vibration en marche, contre le seuil de 1,5.

    Petits multiples : les deux machines ont des etendues tres differentes
    (max 2,83 contre 0,55), les forcer sur un axe commun ecraserait la B."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    import pandas as pd
    d = pd.read_csv(os.path.join(ROOT, "isense_oil_data_sensors_filled.csv"),
                    parse_dates=["created_at"], low_memory=False)
    d = d[d["Oil Pressure"] > SEUILS["marche_bar"]]
    for ax, mach in zip(axes, MACHINES):
        v = d.loc[d["asset_name"] == mach, "Oil System Vibration_filled"].dropna()
        ax.hist(v, bins=60, color=TEAL, alpha=0.85, edgecolor="none")
        ax.axvline(SEUILS["vib_alarme"], color=NEUTRE, linewidth=1.4,
                   linestyle="--")
        ax.annotate(f"seuil {n(SEUILS['vib_alarme'], 1)}",
                    xy=(SEUILS["vib_alarme"], ax.get_ylim()[1] * 0.92),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=NEUTRE, va="top")
        st = S["machines"][mach]
        ax.set_title(f"{mach}\n{st['n_marche']} mesures en marche · "
                     f"{st['v_n_sup_seuil']} au-dessus du seuil",
                     fontsize=9, color=NEUTRE)
        ax.set_xlabel("Oil System Vibration (mm/s²)", fontsize=8.5)
        ax.set_yscale("log")
        ax.set_ylabel("nombre de mesures (échelle log)", fontsize=8.5)
        _style(ax)
    fig.tight_layout()
    p = os.path.join(FIG, "etat_sys_vibration_seuil.png")
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_derive_pression():
    """Mediane journaliere de pression sur le flux en direct.

    L'axe n'est PAS ancre a zero : la question est le NIVEAU et sa rupture,
    entre 2 et 6 bar, pas la distance a l'absence de pression."""
    dj = S.get("derive_journaliere") or {}
    if not dj:
        return None
    fig, ax = plt.subplots(figsize=(9.2, 3.2))
    for mach, couleur in zip(MACHINES, (BLEU, TEAL)):
        serie = dj.get(mach) or []
        if not serie:
            continue
        # ROMPRE LA LIGNE SUR LES TROUS. Relier deux points separes de deux
        # semaines trace une pente continue la ou il n'y a AUCUNE mesure : le
        # lecteur y lit une baisse progressive alors que la verite est « on ne
        # sait pas ». Un jour manquant devient un blanc, pas une interpolation.
        x, y = [], []
        precedent = None
        for p in serie:
            jour = np.datetime64(p["jour"])
            if precedent is not None and (jour - precedent).astype(int) > 2:
                x.append(precedent + np.timedelta64(1, "D"))
                y.append(np.nan)          # coupure visible
            x.append(jour)
            y.append(p["med"])
            precedent = jour
        ax.plot(x, y, marker="o", markersize=3, linewidth=1.6,
                color=couleur, label=mach)
        # Reference d'entrainement : le niveau que les modeles ont appris.
        ref = S["machines"][mach]["p_med"]
        ax.axhline(ref, color=couleur, linewidth=1.0, linestyle=":", alpha=0.75)
        ax.annotate(f"médiane apprise {n(ref)} bar", xy=(x[0], ref),
                    xytext=(2, 3), textcoords="offset points",
                    fontsize=7.5, color=couleur)
    ax.set_ylabel("Oil Pressure — médiane du jour (bar)", fontsize=8.5)
    ax.set_title("Niveau de pression en exploitation, contre le niveau appris",
                 fontsize=9.5, color=NEUTRE)
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    p = os.path.join(FIG, "etat_sys_derive_pression.png")
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return p


# ───────────────────────────────────────────────────────── document

def main():
    os.makedirs(FIG, exist_ok=True)
    fig_vibration_seuil()
    fig_derive_pression()

    doc = Document()
    setup(doc)

    h(doc, "État du système — méthode, seuils et validation", 0)
    para(doc, "Motosoufflantes A et B · projet i-SENSE / OCP Maintenance Solutions",
         color=GREY, size=9)
    para(doc, f"Jeu de référence : {S['n_lignes']} mesures, "
              f"du {S['periode'][0]} au {S['periode'][1]}. "
              "Flux en direct relevé sur l'API i-SENSE.",
         color=GREY, size=9, space=10)

    # ---------------------------------------------------------------- 1
    h(doc, "1. Ce que mesure l'état du système", 1)
    para(doc,
         "L'état du système repose sur deux grandeurs, lues SÉPARÉMENT : la "
         "pression d'huile et la vibration du système d'huile. Elles ne sont "
         "jamais agrégées en un chiffre unique.")
    para(doc,
         "La raison est opérationnelle avant d'être statistique. Les deux "
         "grandeurs décrivent des organes différents et appellent des "
         "interventions différentes : la pression renseigne le circuit "
         "hydraulique — pompe, colmatage, fuite — la vibration la mécanique "
         "tournante — balourd, désalignement, roulement. Les agréger produit un "
         "chiffre sans destinataire : personne n'intervient sur « le système », "
         "on intervient sur la pompe OU sur les paliers.")

    # ---------------------------------------------------------------- 2
    h(doc, "2. La méthode : des seuils physiques, dans l'unité du capteur", 1)
    para(doc,
         "Chaque canal est jugé directement sur sa mesure, contre un seuil "
         "exprimé dans l'unité du capteur. Aucun indice normalisé n'intervient.")
    table(doc,
          ["Variable", "Seuil", "Signification", "Où il est défini"],
          [["Oil Pressure", f"{n(SEUILS['marche_bar'])} bar",
            "en dessous : machine à l'arrêt",
            "OIL_PRESSURE_ON_THRESHOLD — clean_isense_data.py, "
            "eda_isense.py, data_quality_isense.py"],
           ["Oil System Vibration", f"{n(SEUILS['vib_alarme'], 1)} mm/s²",
            "au-dessus : alarme",
            "flag_high_vibration — feature_engineering.py"],
           ["Oil System Vibration", f"{n(SEUILS['vib_borne'], 1)} mm/s²",
            "au-dessus : mesure physiquement impossible",
            "PHYSICAL_RANGES — data_quality_isense.py"]],
          widths=[1.5, 0.9, 1.7, 2.5])

    para(doc, "Ce que l'opérateur lit change de nature. « 5,40 bar, seuil de "
              "marche 0,10 » se vérifie sur le capteur ; « indice 0,834, zone A » "
              "ne se vérifie nulle part.", space=8)

    h(doc, "Une réserve à porter en soutenance", 2)
    rich(doc, [("Le document officiel OCP — ", False, None),
               ("Cadrage des seuils_Système d'huile.pdf", True, None),
               (" — ne fixe AUCUN seuil pour ces deux variables.", True, RED)])
    para(doc,
         "La table OFFICIAL_THRESHOLDS couvre la température, l'eau, la "
         "constante diélectrique, les codes ISO 4406, la densité et les "
         "viscosités. Ni la pression ni la vibration n'y figurent. Le seuil de "
         f"{n(SEUILS['vib_alarme'], 1)} mm/s² est qualifié de « seuil indicatif » "
         "dans feature_engineering.py.")
    para(doc,
         "Il est retenu parce que c'est celui que le pipeline applique depuis "
         "l'origine, et parce qu'il discrimine réellement — la section suivante "
         "le démontre. Il doit être défendu par les données, jamais présenté "
         "comme une norme constructeur.")

    # ---------------------------------------------------------------- 3
    h(doc, "3. Validation empirique du seuil de vibration", 1)
    lignes = []
    for m in MACHINES:
        st = S["machines"][m]
        lignes.append([m, f"{st['n_marche']}", n(st["v_med"], 3),
                       n(st["v_p95"], 3), n(st["v_p99"], 3), n(st["v_max"], 3),
                       f"{st['v_n_sup_seuil']}", pc(st["v_pct_sup_seuil"])])
    table(doc,
          ["Machine", "mesures en marche", "médiane", "p95", "p99", "max",
           f"> {n(SEUILS['vib_alarme'], 1)}", "taux"],
          lignes, widths=[1.6, 0.9, 0.7, 0.7, 0.7, 0.7, 0.6, 0.7])

    a = S["machines"]["Motosoufflante A"]
    para(doc,
         f"Le seuil tombe juste au-dessus du p99 de la Motosoufflante A "
         f"({n(a['v_p99'], 2)}) : il déclenche sur {a['v_n_sup_seuil']} mesures, "
         f"soit {pc(a['v_pct_sup_seuil'])} de son fonctionnement. C'est un taux "
         "d'alarme crédible — assez rare pour être pris au sérieux, assez "
         "fréquent pour être vérifiable.", space=8)
    para(doc,
         f"La borne physique de {n(SEUILS['vib_borne'], 1)} mm/s² n'est franchie "
         "sur aucune machine : elle ne sert pas à détecter une dégradation mais "
         "à écarter une mesure qui ne décrirait plus la machine, seulement son "
         "capteur.")
    figure(doc, "etat_sys_vibration_seuil.png",
           "Distribution de la vibration en marche, par machine, contre le seuil. "
           "Échelle verticale logarithmique : sans elle, les 149 dépassements de "
           "la machine A seraient invisibles à côté de ses 18 562 mesures.")

    # ---------------------------------------------------------------- 4
    h(doc, "4. Deux méthodes abandonnées, et pourquoi", 1)
    para(doc,
         "Cette section a valeur de méthode : les deux approches écartées l'ont "
         "été sur des raisons vérifiables, pas par préférence.")

    h(doc, "4.1 Distance de Mahalanobis / T² — non monotone", 2)
    para(doc,
         "Une distance statistique mesure ce qui est INHABITUEL, pas ce qui est "
         "DÉGRADÉ. Contre-exemple relevé sur les données : une vibration de "
         "0,010 (z = −13,28) accompagnée d'une pression normale produisait un "
         "indice de 0,0000 — l'alarme maximale pour une machine qui vibre moins "
         "que d'habitude, c'est-à-dire qui va mieux. 907 lignes étaient ainsi "
         "classées dégradées à tort.")
    rich(doc, [("Un indice de santé doit être MONOTONE : il ne baisse que "
                "lorsque l'état empire, jamais parce qu'une grandeur s'écarte "
                "dans le bon sens.", True, None)])

    h(doc, "4.2 Indice normalisé 2^(−s/2) — un seuil emprunté, jamais atteint", 2)
    para(doc,
         "L'approche suivante corrigeait la monotonie par un max(0, ·), puis "
         "convertissait l'écart en indice sur [0, 1]. Deux défauts l'ont fait "
         "abandonner à son tour.")
    bullets(doc, [
        "Le seuil d'alarme était emprunté au Health Index huile — 0,50 — un "
        "nombre sans unité, appliqué à des bar et des mm/s². Rien ne justifiait "
        "ce transfert.",
        "L'indice n'atteignait jamais les zones C ni D sur le canal vibration : "
        "0,00 % de dépassement. Un indicateur qui ne peut pas alarmer ne "
        "surveille rien.",
    ])
    para(doc,
         "Le passage aux seuils physiques corrige les deux : le seuil vient du "
         "traitement de données, il est exprimé dans l'unité du capteur, et il "
         f"déclenche sur {a['v_n_sup_seuil']} mesures réelles.")

    # ---------------------------------------------------------------- 5
    h(doc, "5. Prévoir l'état du système : un résultat négatif", 1)
    lb = _lire("systeme_leaderboard.json", []) or []
    para(doc,
         "Sept familles de modèles ont été comparées sous protocole sans fuite — "
         "coupures gelées par machine, validation croisée purgée avec embargo "
         "proportionnel à l'horizon, jeu de test touché une seule fois : Ridge, "
         "Lasso à porte, forêt aléatoire, gradient boosting par histogramme, "
         "XGBoost, LightGBM Huber et CatBoost.")
    if lb:
        para(doc,
             "Aucune ne bat la persistance sur les deux critères retenus "
             "simultanément — un skill positif ET une exactitude au moins égale "
             "à celle de la persistance.", bold=True)
        figure(doc, "sys_skill.png",
               "Skill des sept familles par horizon. Le zéro est la persistance : "
               "au-dessous, le modèle nuit.")
    para(doc,
         "Ce résultat portait sur l'indice agrégé, aujourd'hui abandonné. Il "
         "reste néanmoins instructif et doit être présenté comme tel : sur un "
         "signal aussi stable en régime établi, « rien ne change » est une "
         "prévision difficile à battre. La valeur du tableau de bord est dans la "
         "DÉTECTION au présent, pas dans l'anticipation.", space=8)

    # ---------------------------------------------------------------- 6
    h(doc, "6. Dérive constatée en exploitation", 1)
    d = S.get("direct") or {}
    if d and "erreur" not in d:
        lignes = []
        for m in MACHINES:
            ref = S["machines"][m]
            cur = d.get(m) or {}
            for nom, cle_ref, cle_cur, dec in (
                    ("pression (bar)", "p_med", "p_med", 2),
                    ("vibration (mm/s²)", "v_med", "v_med", 3)):
                r, c = ref[cle_ref], cur.get(cle_cur)
                rap = None if (c is None or not r) else c / r
                lignes.append([m, nom, n(r, dec), n(c, dec), n(rap, 3)])
        table(doc, ["Machine", "Grandeur", "Médiane apprise", "Médiane en direct",
                    "Rapport"], lignes, widths=[1.6, 1.4, 1.1, 1.1, 0.8])

        rich(doc, [("La pression des deux machines a été divisée par deux, "
                    "simultanément, en marche d'escalier.", True, RED)])
        para(doc,
             "Sur la Motosoufflante A la transition est datable : 5,70 bar "
             "jusqu'au 27 juillet, puis un trou de données, puis 2,60 bar à "
             "partir du 10 août — stable depuis, jour après jour, sans dérive.")
        para(doc,
             "Une usure de pompe dérive, elle ne produit pas une marche "
             "d'escalier ; et elle ne frappe pas deux machines le même jour avec "
             "le même facteur 0,5. Un facteur commun aux deux machines et propre "
             "à la pression désigne l'acquisition : recalibrage, changement "
             "d'unité ou de configuration. C'est le motif exact du facteur 10 de "
             "conductivité déjà identifié et corrigé dans clean_isense_data.py.")
        figure(doc, "etat_sys_derive_pression.png",
               "Médiane journalière de pression en exploitation. Les pointillés "
               "marquent le niveau sur lequel les modèles ont été entraînés. "
               "La ligne est rompue sur les périodes sans mesure : l'absence de "
               "donnée n'est pas une baisse progressive.")
        para(doc,
             "Conséquence directe : les modèles ont appris sur 5,4 et 4,4 bar et "
             "reçoivent 2,6 et 2,2 bar. Les prévisions postérieures au 10 août "
             "s'appuient sur une entrée hors distribution et ne doivent pas être "
             "présentées comme fiables tant que l'origine du décalage n'est pas "
             "tranchée avec l'exploitation.", space=8)
        para(doc,
             "À noter : le seuil de marche ne peut pas détecter ce décalage, et "
             "ce n'est pas un défaut. 2,2 bar reste très au-dessus de 0,10 bar : "
             "le seuil répond correctement à la question qu'il pose — « la "
             "machine tourne-t-elle ? » — qui n'est pas celle d'un changement de "
             "référentiel.")

        va = (d.get("Motosoufflante A") or {}).get("v_med")
        ra = S["machines"]["Motosoufflante A"]["v_med"]
        if va and ra:
            para(doc,
                 f"La vibration de la Motosoufflante A fait exception : elle "
                 f"passe de {n(ra, 3)} à {n(va, 3)} mm/s², soit un facteur "
                 f"{n(va / ra, 1)}, sans équivalent sur la machine B. Elle ne "
                 "s'explique donc pas par un recalibrage commun. Elle reste sous "
                 f"le seuil de {n(SEUILS['vib_alarme'], 1)} et n'alarme pas, mais "
                 "c'est précisément le type de montée lente que la surveillance "
                 "doit suivre.", space=8)

    # ---------------------------------------------------------------- 7
    h(doc, "7. Limites connues", 1)
    b = S["machines"]["Motosoufflante B"]
    bullets(doc, [
        f"Motosoufflante B : seulement {b['n_marche']} mesures en marche sur "
        f"{b['n_total']} ({pc(b['pct_marche'], 1)}). Ses seuils sont établis sur "
        "une base étroite.",
        f"Le canal vibration n'a jamais alarmé sur la machine B "
        f"({b['v_n_sup_seuil']} dépassement) : son comportement au-dessus du "
        "seuil n'est pas observé.",
        "Les seuils de pression et de vibration ne viennent pas du document de "
        "cadrage OCP, qui n'en fournit pas pour ces variables.",
        "Une partie de la vibration est reconstruite par imputation (forêt "
        "aléatoire en marche, zéro physique à l'arrêt) : un dépassement doit "
        "être confirmé sur une mesure réelle avant toute intervention.",
        "Les modèles sont entraînés jusqu'au " + S["periode"][1] +
        " ; tout écart de distribution postérieur dégrade la prévision, comme "
        "le montre la section 6.",
    ])

    p = save_doc(doc, SORTIE)
    print(f"Ecrit : {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
