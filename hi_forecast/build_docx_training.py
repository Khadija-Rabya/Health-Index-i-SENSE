"""TACHE 3 — Rapport d'entrainement complet : toute la chaine, de bout en bout."""
import json
import os
import sys

import numpy as np
import pandas as pd
from docx import Document

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from build_docx import (GREEN, GREY, NAVY, RED, bullets, figure, h, para, rich,
                        save_doc, setup, table)

U = pd.read_csv(os.path.join(HERE, "tableau_unifie_actuelle.csv"))
AEB = pd.read_csv(os.path.join(HERE, "ae_roleB_resultats.csv"))
ALS = pd.read_csv(os.path.join(HERE, "alerte_seuil_resultats.csv"))
with open(os.path.join(HERE, "verif_decisive.json"), encoding="utf-8") as f:
    DEC = json.load(f)
with open(os.path.join(HERE, "alertes_resultats.json"), encoding="utf-8") as f:
    ALR = json.load(f)
FEAS = pd.read_csv(os.path.join(HERE, "ae_faisabilite_fenetres.csv"))


def d4(v, n=4):
    return "—" if pd.isna(v) else f"{v:.{n}f}".replace(".", ",")


def s4(v, n=4):
    return "—" if pd.isna(v) else f"{v:+.{n}f}".replace(".", ",")


def main():
    doc = setup(Document(), landscape=False)
    h(doc, "Rapport d'entraînement complet — Health Index i-SENSE", 0)
    para(doc, "De l'acquisition des données au modèle déployé · 18 modèles comparés sous "
              "protocole identique", size=12, color=GREY)
    para(doc, "Projet i-SENSE / OCP Maintenance Solutions — UM6P   •   8 septembre 2026",
         size=9, color=GREY, space=12)

    # ================================================================ résumé
    h(doc, "Résumé", 1)
    rich(doc, [("Le R² de 0,9453 du pipeline d'origine mesurait l'autocorrélation, pas la "
                "prédiction.", True, RED),
               (" La persistance seule atteint 0,9427 à 10 minutes. Après correction de six "
                "fuites de données, mise en place d'un protocole gelé et 51 configurations "
                "évaluées, le gain réel sur la persistance est modeste mais réel — et il n'est "
                "pas là où on l'attendait.", False, None)])
    best = U.sort_values("test_skill", ascending=False).iloc[0]
    ret = U[U["famille"] == "LightGBM Huber (retenu)"].iloc[0]
    table(doc, ["", "CV acc", "CV skill", "Test acc", "Test R²", "Skill", "IC 95 % du skill"],
          [["Persistance (référence)", "0,7912", "0,0000", "0,8164", "0,1142", "0,0000", "—"],
           ["LightGBM Huber (retenu par le critère verrouillé)", d4(ret["cv_acc"]),
            d4(ret["cv_skill"]), d4(ret["test_acc"]), d4(ret["test_r2"]),
            s4(ret["test_skill"]), f"[{s4(ret['ic_bas'])} ; {s4(ret['ic_haut'])}] ✗"],
           [f"{best['famille']} (meilleur skill)", d4(best["cv_acc"]), d4(best["cv_skill"]),
            d4(best["test_acc"]), d4(best["test_r2"]), s4(best["test_skill"]),
            f"[{s4(best['ic_bas'])} ; {s4(best['ic_haut'])}] ✓"]],
          widths=[2.6, 0.75, 0.8, 0.75, 0.75, 0.75, 1.5], highlight={2})
    para(doc, "")
    rich(doc, [("Résultat central : le critère verrouillé (accuracy dans une bande de ±0,01) "
                "désigne un modèle dont le gain n'est PAS statistiquement significatif, tandis "
                "qu'un modèle 0,007 derrière sur ce critère a trois fois et demie plus de skill "
                "avec un intervalle qui exclut zéro.", True, RED)])

    # ====================================================== I. acquisition
    h(doc, "I. Acquisition des données", 1)
    table(doc, ["Élément", "Valeur"],
          [["Source", "API i-SENSE, extraction vers Excel (Api_to_excel.py)"],
           ["Format brut", "format long (1 ligne = 1 mesure), pivoté en large"],
           ["Volume final", "42 141 lignes × 171 colonnes"],
           ["Période", "29/12/2025 → 06/08/2026 (≈ 7 mois)"],
           ["Machines", "Motosoufflante A (19 896 lignes), Motosoufflante B (22 245)"],
           ["Cadence", "≈ 10 minutes (médiane 10, maximum 60)"],
           ["Capteurs", "14 grandeurs : température, H2O ppm et saturation, viscosités "
                        "(40 °C, cinématique, dynamique), densité, ISO 4/6/14, DC, "
                        "conductivité, vibration, pression"],
           ["Sessions", "352, définies par une rupture de plus de 60 minutes"]],
          widths=[1.8, 5.1])

    h(doc, "II. Analyse exploratoire", 1)
    bullets(doc, [
        "Distribution des sessions extrêmement déséquilibrée : médiane 3 lignes, 75ᵉ centile 38, "
        "maximum 7 904. 126 sessions sur 352 (35,8 %) ne comptent qu'une seule ligne.",
        "La machine est à l'arrêt 50 % des lignes. À l'arrêt, l'indice de santé ne bouge presque "
        "pas : |Δ| médian de 0,00037 contre 0,00439 en marche.",
        "Asymétrie forte entre machines : A tourne 93 % du temps, B est à l'arrêt 90 % du temps.",
        "Corrélations physiques attendues confirmées (μ = ν·ρ, loi ASTM D1298), ce qui a conduit "
        "à réintégrer densité et viscosité cinématique plutôt qu'à les éliminer comme redondantes.",
    ])

    # ================================================== III. qualité données
    h(doc, "III. Qualité des données — le constat déterminant", 1)
    rich(doc, [("39,96 % des lignes portent une viscosité physiquement impossible.", True, RED)])
    para(doc, "16 839 valeurs entre 3,0 et 19,9 cSt (médiane 8,9) pour une huile ISO VG 46 dont "
              "la référence est 46,0 cSt. La répartition n'est pas aléatoire : 100 % de ces "
              "valeurs viennent de la Motosoufflante B à l'arrêt, soit 84 % de ses lignes OFF.")
    table(doc, ["Machine", "État", "Lignes", "Viscosité < 20 cSt", "Médiane"],
          [["Motosoufflante A", "ON", "18 562", "0 (0,0 %)", "46,61 cSt"],
           ["Motosoufflante A", "OFF", "1 334", "0 (0,0 %)", "45,59 cSt"],
           ["Motosoufflante B", "ON", "2 204", "0 (0,0 %)", "41,61 cSt"],
           ["Motosoufflante B", "OFF", "20 041", "16 839 (84,0 %)", "7,54 cSt"]],
          widths=[1.8, 0.7, 1.1, 1.6, 1.1], highlight={3})
    para(doc, "")
    para(doc, "Le signe est physiquement inversé : à l'arrêt l'huile est plus froide, donc plus "
              "VISQUEUSE. C'est ce que fait la machine A (45,6 cSt à l'arrêt) et l'inverse exact "
              "de ce que fait B (7,5 cSt). Ces colonnes alimentent PCA_VARS : elles corrompent "
              "donc l'ÉTIQUETTE, pas seulement les variables d'entrée.")
    figure(doc, "07_viscosite.png",
           "Figure 1 — Distribution de la viscosité par machine et par état.", width=6.8)

    # ================================================ IV. étiquette
    h(doc, "IV. Construction de l'étiquette — ce qu'est vraiment health_index", 1)
    rich(doc, [("Ce n'est pas une mesure, c'est un score construit par le projet lui-même.",
                True, RED)])
    p = doc.add_paragraph()
    r = p.add_run("sévérité = max( T²_ACP / seuil_alarme , SPE_ACP / seuil_alarme , "
                  "IsolationForest / seuil_alarme )\nhealth_index = 1 / (1 + sévérité)")
    r.font.name, r.font.size, r.bold = "Consolas", 9, True
    para(doc, "ACP et Isolation Forest ajustées par machine sur les lignes sans aucun "
              "flag_high_* actif ; seuils d'alarme = 99ᵉ centile de ces lignes saines. "
              "Moyenne 0,5793, écart-type 0,0447, étendue 0,043 → 0,624. États dérivés : "
              "Normal 89,2 % / Surveillance 7,4 % / Alarme 3,4 %.")
    rich(doc, [("Conséquence structurante : ", True, None),
               ("il n'existe AUCUNE vérité terrain de dégradation dans ce jeu de données — ni "
                "panne, ni intervention, ni vidange, ni analyse d'huile en laboratoire. Le "
                "modèle prédit le score d'anomalie du projet, et son plafond est le bruit de ce "
                "score.", False, None)])

    # ================================================= V. audit des fuites
    h(doc, "V. Audit des fuites de données", 1)
    para(doc, "Le document de plan affirmait que toutes les variables étaient causales, "
              "« vérifié ». Trois ne l'étaient pas. Six problèmes distincts :")
    table(doc, ["#", "Fuite", "Correction"],
          [["F1", "L'étiquette elle-même : ACP, Isolation Forest et seuils ajustés sur tout "
                  "l'historique, test compris", "Réajustés sur la fenêtre d'entraînement seule"],
           ["F2", "vi_proxy : polyfit sur toutes les lignes ; se propage dans 7 colonnes",
            "Réajusté train-only, dérivées recalculées"],
           ["F3", "13 colonnes *_zscore standardisées avec moyenne/écart-type globaux",
            "Statistiques de la fenêtre d'entraînement"],
           ["F4", "HI_regles : référence de viscosité = médiane de tout l'historique",
            "Médiane de la fenêtre d'entraînement"],
           ["F5", "SimpleImputer ajusté avant le découpage",
            "Déplacé dans le pipeline, ajusté par pli"],
           ["F6", "Sélection du modèle effectuée sur le jeu de test",
            "Toute sélection sur CV ; test scoré une fois"]],
          widths=[0.4, 3.4, 3.1], size=8.5)
    para(doc, "")
    para(doc, "Vérifiés et sains : aucune ligne dupliquée, découpage déjà temporel, et utiliser "
              "health_index[t] pour prédire health_index[t+n] est de la prévision légitime.",
         size=9)

    # =============================================== VI. protocole gelé
    h(doc, "VI. Protocole d'évaluation gelé", 1)
    bullets(doc, [
        "Test gelé : les 20 % de lignes les plus récentes par machine (coupures 24/05/2026 pour "
        "A, 28/06/2026 pour B) — 7 795 lignes, scorées UNE SEULE FOIS par configuration.",
        "Validation croisée : 5 blocs temporels expansifs, purge et embargo de 3 h de chaque "
        "côté de la frontière. Sans embargo, les dernières lignes d'entraînement partagent leur "
        "cible avec le début de la validation et le score gonfle.",
        "Graine 42 partout. Toute sélection sur la CV, jamais sur le test.",
        "Accuracy définie comme la part de prédictions dans une bande de ±0,01 (≈ 22 % de "
        "l'écart-type de la cible), avec R² et skill rapportés systématiquement à côté.",
        "Significativité par bootstrap par blocs, l'unité de rééchantillonnage étant la SESSION "
        "et non la ligne — des lignes espacées de 10 minutes ne sont pas indépendantes.",
    ])

    # ====================================== VII. prétraitement
    h(doc, "VII. Imputation, normalisation, ingénierie de variables", 1)
    h(doc, "VII.1 Valeurs manquantes", 2)
    table(doc, ["Stratégie", "Accuracy CV"],
          [["Médiane", "0,7908"], ["Moyenne", "0,7914"],
           ["Médiane + indicateur", "0,7911"], ["MICE (IterativeImputer)", "0,7918"],
           ["KNN (référence sous-échantillonnée à 4 000)", "0,7927"]],
          widths=[3.4, 1.4])
    para(doc, "")
    para(doc, "Étendue totale de 0,0019 : le choix d'imputation est SANS EFFET mesurable. La "
              "médiane est retenue pour son coût.", size=9.5)
    h(doc, "VII.2 Mise à l'échelle", 2)
    para(doc, "Six variantes comparées (aucune, standard, robuste, min-max, quantile, "
              "Yeo-Johnson) : 0,7905 à 0,7919. Sans effet sur les modèles à arbres, "
              "déterminant en revanche pour les autoencodeurs, d'où l'ajustement DANS le pli.")
    h(doc, "VII.3 Variables construites", 2)
    bullets(doc, [
        "Calendaires : heure, jour, encodage cyclique sinus/cosinus.",
        "Session : temps écoulé, index de la mesure.",
        "Physico-chimiques : densité normalisée à 15 °C (ASTM D1298), écart de densité, "
        "ratio de sensibilité thermique, vi_proxy (résidu après retrait de l'effet température).",
        "Dynamique : lags 1-3, différence première, EWMA span 18 (≈ 3 h), pente sur 18 pas — "
        "toutes calculées DANS la session, jamais à cheval sur deux sessions.",
        "Métier : indice de contamination, écart au grade ISO VG, interaction température × "
        "viscosité, drapeaux de seuils.",
        "Confiance : score de provenance de la vibration, drapeau de comblement de capteur.",
    ])
    h(doc, "VII.4 Sélection de variables", 2)
    para(doc, "Quatre critères comparés (corrélation, information mutuelle, importance par "
              "permutation, SHAP) sur 10 jeux. Retenu en CV : SHAP top-20, qui passe de 70 à "
              "20 variables en gagnant 0,004 d'accuracy. Variables de tête : "
              "Oil System Vibration_filled_zscore_lf, health_index_lf, HI_isoforest_lf, "
              "time_in_session_h, Oil System Vibration_filled_diff1.")
    doc.add_page_break()

    # ====================================== VIII. autoencodeurs
    h(doc, "VIII. Autoencodeurs — trois architectures, deux rôles", 1)
    h(doc, "VIII.1 Faisabilité des fenêtres, avant tout entraînement", 2)
    f = FEAS[FEAS["W"] > 1]
    table(doc, ["W", "Sessions utilisables", "Échantillons", "% des lignes", "Test A", "Test B"],
          [[int(r["W"]), f"{int(r['sessions_ok'])} / 352",
            f"{int(r['echantillons']):,}".replace(",", " "),
            f"{r['pct_lignes']:.1%}".replace(".", ","), int(r["test_A"]), int(r["test_B"])]
           for _, r in f.iterrows()],
          widths=[0.5, 1.6, 1.3, 1.1, 0.8, 0.8], highlight={2})
    para(doc, "")
    para(doc, "Aucune fenêtre écartée, aucun remplissage. Deux réserves : 126 sessions ne "
              "comptent qu'une ligne et ne contribuent à rien ; à W = 18 les 5 plus grosses "
              "sessions fournissent 56,4 % des échantillons.", size=9.5)
    h(doc, "VIII.2 Rôle A — prévision", 2)
    para(doc, "Le latent et l'erreur de reconstruction sont ajoutés aux 20 variables SHAP, puis "
              "passés au régresseur d'écart. Les architectures séquentielles sont aussi testées "
              "en prévision directe (seq2seq). Sur 18 configurations (3 architectures × 2 "
              "étiquettes × variantes), 5 remplissent les deux critères — toutes DENSES. "
              "Aucune architecture séquentielle ne passe : leur intervalle de skill contient "
              "zéro sur la machine A et sur l'état ON.")
    rich(doc, [("Le contre-exemple le plus instructif : ", True, None),
               ("AE-3 Conv1D seq2seq direct obtient le MEILLEUR skill de test du banc d'essai "
                "(+0,2893) et il est rejeté, parce que son skill en validation croisée vaut "
                "−0,4071. Sans la règle « sélection en CV uniquement », c'est ce modèle qui "
                "aurait été déployé.", False, None)])
    h(doc, "VIII.3 Rôle B — reconstruction de l'étiquette (tableau séparé)", 2)
    rich(doc, [("Ces accuracies ne sont PAS comparables au rôle A : la cible n'est pas la même.",
                True, RED)])
    b = AEB[AEB["base"] == "etiquette actuelle"]
    table(doc, ["Construction de l'étiquette", "Écart-type", "Test acc", "Skill"],
          [[r["tag"].split(" (")[0], d4(r["sd_label"]), d4(r["test_acc"]), s4(r["test_skill"])]
           for _, r in b.iterrows()],
          widths=[3.2, 1.1, 1.0, 1.0], highlight={0})
    para(doc, "")
    para(doc, "Les étiquettes reconstruites par autoencodeur sont trois à quatre fois plus "
              "RUGUEUSES que la référence (écart-type 0,197-0,210 contre 0,046) et leur skill "
              "s'effondre. Remplacer ACP + Isolation Forest par un autoencodeur DÉGRADE "
              "l'étiquette — confirmation expérimentale, sous protocole sans fuite, de la "
              "décision de Phase 4 du projet d'origine.", size=9.5)
    doc.add_page_break()

    # ====================================== IX. tableau unifié
    h(doc, "IX. TABLEAU DE COMPARAISON UNIFIÉ", 1)
    rich(doc, [("Protocole strictement identique pour les 18 modèles. ", True, NAVY),
               ("Chacun fournit une prédiction d'écart ; l'architecture à porte et le facteur "
                "alpha sont réglés en CV de la même façon ; mêmes plis, même test gelé, même "
                "bootstrap par blocs. Les autoencodeurs figurent dans ce tableau au même titre "
                "que LightGBM, ElasticNet, Random Forest et la persistance. Aucun traitement de "
                "faveur.", False, None)])
    d = U.sort_values("test_skill", ascending=False)
    table(doc, ["Modèle", "Cat.", "CV acc ± sd", "CV skill", "Test acc", "Test R²", "RMSE",
                "Skill", "IC 95 % du skill", "Signif."],
          [[r["famille"], {"reference": "réf", "classique": "cl", "autoencodeur": "AE"}[r["categorie"]],
            f"{d4(r['cv_acc'])} ± {d4(r['cv_acc_sd'], 3)}", s4(r["cv_skill"]),
            d4(r["test_acc"]), d4(r["test_r2"]), d4(r["test_rmse"], 5), s4(r["test_skill"]),
            f"[{s4(r['ic_bas'], 3)} ; {s4(r['ic_haut'], 3)}]" if not pd.isna(r["ic_bas"]) else "—",
            "OUI" if r["significatif"] else "non"] for _, r in d.iterrows()],
          widths=[1.85, 0.32, 1.0, 0.62, 0.6, 0.55, 0.62, 0.6, 1.25, 0.5], size=7,
          highlight={0, 1, 2})
    para(doc, "")
    figure(doc, "21_tableau_unifie.png",
           "Figure 2 — À gauche : skill et intervalle de confiance de chaque modèle. À droite : "
           "les deux critères de sélection ne désignent pas le même gagnant.", width=7.0)

    h(doc, "IX.1 Ce que ce tableau révèle", 2)
    bullets(doc, [
        "Les modèles LINÉAIRES RÉGULARISÉS dominent : Ridge, Lasso et ElasticNet atteignent un "
        "skill de +0,332 à +0,333 avec des intervalles qui excluent nettement zéro. Une fois "
        "placés dans l'architecture à porte avec les 20 variables SHAP, ils passent de "
        "catastrophiques (accuracy CV de 0,40 à 0,48 en évaluation brute) à les meilleurs.",
        "Le modèle retenu par le critère verrouillé — LightGBM Huber — a la MEILLEURE accuracy "
        "en CV (0,8094) mais le skill le plus faible des modèles entraînés (+0,0941), et son "
        "intervalle CONTIENT ZÉRO [−0,022 ; +0,170]. Il n'est pas statistiquement significatif.",
        "Les autoencodeurs denses sont tous significatifs (+0,095 à +0,106) et améliorent "
        "l'accuracy de test (jusqu'à 0,8515 pour AE-2 LSTM), mais restent loin des linéaires en "
        "skill.",
        "Extra Trees, XGBoost, LightGBM (l2) et HistGB ont un skill de test élevé (+0,27 à "
        "+0,30) mais des intervalles très larges qui croisent zéro : leur performance est "
        "instable d'une session à l'autre.",
    ])
    rich(doc, [("Conclusion du tableau : ", True, RED),
               ("le critère d'accuracy dans une bande de ±0,01, verrouillé au début du projet, "
                "sélectionne un modèle non significatif. Le critère de skill — celui qui compte "
                "pour une alerte — désigne Lasso ou ElasticNet, avec 3,5 fois plus de gain et "
                "un intervalle qui exclut zéro.", False, None)])
    doc.add_page_break()

    # ---------------------------------------- IX.2 toutes les métriques de régression
    h(doc, "IX.2 Toutes les métriques de régression", 2)
    para(doc, "Onze métriques par modèle, mêmes lignes de test. Une seule métrique ne suffit "
              "jamais à désigner un gagnant : les colonnes ci-dessous ne s'accordent pas.")
    dr = U.sort_values("test_skill", ascending=False)
    table(doc, ["Modèle", "acc ±0,01", "R²", "R² aj.", "RMSE", "MAE", "MedAE", "MAPE %",
                "Var. expl.", "Err. max", "Biais", "Skill"],
          [[r["famille"], d4(r["test_acc"]), d4(r["test_r2"]), d4(r["adj_r2"]),
            d4(r["test_rmse"]), d4(r["mae"]), d4(r["medae"]), d4(r["mape"], 3),
            d4(r["var_expliquee"]), d4(r["erreur_max"]), s4(r["biais"]), s4(r["test_skill"])]
           for _, r in dr.iterrows()],
          widths=[1.75, 0.62, 0.5, 0.52, 0.55, 0.52, 0.55, 0.55, 0.6, 0.58, 0.55, 0.55],
          size=6.5, highlight={0, 1, 2})

    # ------------------------------- IX.3 métriques de classification, AVEC persistance
    h(doc, "IX.3 Métriques de classification de l'état à t+3 h", 2)
    para(doc, "L'indice prédit est converti en état par les mêmes seuils que le projet "
              "(centiles 5 % et 1 % de la fenêtre d'entraînement). C'est la traduction directe "
              "de la logique du système : indice + seuils = alerte. La ligne PERSISTANCE est "
              "présente — sans elle, ces colonnes classeraient les modèles entre eux sans dire "
              "si l'un bat « ne rien faire ».")
    dc = U.sort_values("alerte_f1", ascending=False)
    table(doc, ["Modèle", "Acc. état", "Acc. équil.", "F1 macro", "MCC", "Kappa",
                "Rappel alerte", "Préc. alerte", "F1 alerte"],
          [[r["famille"], d4(r["etat_accuracy"]), d4(r["etat_bal_accuracy"]),
            d4(r["etat_f1_macro"]), d4(r["etat_mcc"]), d4(r["etat_kappa"]),
            d4(r["alerte_rappel"]), d4(r["alerte_precision"]), d4(r["alerte_f1"])]
           for _, r in dc.iterrows()],
          widths=[1.9, 0.72, 0.75, 0.68, 0.62, 0.62, 0.78, 0.72, 0.68], size=7,
          highlight={i for i, (_, r) in enumerate(dc.iterrows())
                     if r["categorie"] == "reference"})
    para(doc, "")
    rich(doc, [("Ce que ces colonnes disent pour l'alerte — et c'est net.", True, NAVY)])
    pers = U[U["categorie"] == "reference"].iloc[0]
    mods = U[U["categorie"] != "reference"]
    table(doc, ["Métrique", "Persistance", "Meilleur modèle", "Nb. de modèles qui battent"],
          [["Accuracy de l'état", d4(pers["etat_accuracy"]),
            f"{d4(mods['etat_accuracy'].max())} (Random Forest)",
            f"{int((mods['etat_accuracy'] > pers['etat_accuracy']).sum())} / 18"],
           ["Accuracy équilibrée", d4(pers["etat_bal_accuracy"]),
            f"{d4(mods['etat_bal_accuracy'].max())} (AE-2 GRU seq2seq)",
            f"{int((mods['etat_bal_accuracy'] > pers['etat_bal_accuracy']).sum())} / 18"],
           ["MCC", d4(pers["etat_mcc"]), f"{d4(mods['etat_mcc'].max())} (Random Forest)",
            f"{int((mods['etat_mcc'] > pers['etat_mcc']).sum())} / 18"],
           ["RAPPEL d'alerte", d4(pers["alerte_rappel"]),
            f"{d4(mods['alerte_rappel'].max())} (AE-2 GRU seq2seq)",
            f"{int((mods['alerte_rappel'] > pers['alerte_rappel']).sum())} / 18"],
           ["PRÉCISION d'alerte", d4(pers["alerte_precision"]),
            f"{d4(mods['alerte_precision'].max())} (Random Forest)",
            f"{int((mods['alerte_precision'] > pers['alerte_precision']).sum())} / 18"],
           ["F1 d'alerte", d4(pers["alerte_f1"]),
            f"{d4(mods['alerte_f1'].max())} (Random Forest)",
            f"{int((mods['alerte_f1'] > pers['alerte_f1']).sum())} / 18"]],
          widths=[1.7, 1.2, 2.1, 1.9], highlight={3, 4})
    para(doc, "")
    bullets(doc, [
        "AUCUN modèle sur 18 ne bat la persistance sur le RAPPEL d'alerte (0,9354). Le meilleur "
        "atteint 0,9347. Autrement dit : aucun modèle ne détecte PLUS de dégradations que "
        "« l'état de dans 3 h sera celui d'aujourd'hui ».",
        "17 modèles sur 18 battent la persistance sur la PRÉCISION d'alerte. Random Forest "
        "monte à 0,9688 contre 0,9358 — soit un tiers de fausses alertes en moins.",
        "Un seul modèle dépasse la persistance en accuracy équilibrée, et de 0,0005 : c'est du "
        "bruit, pas un gain.",
        "La valeur ajoutée des modèles, sur la décision d'alerte, est donc dans la RÉDUCTION DES "
        "FAUSSES ALERTES, pas dans la détection de dégradations supplémentaires. C'est utile — "
        "moins d'interventions inutiles — mais il faut le dire exactement ainsi.",
    ])
    rich(doc, [("La seule exception connue est le classifieur de TRANSITION (section X) : ", True, GREEN),
               ("il capte 88 basculements sur 130 avec trois heures d'avance, là où la "
                "persistance en capte structurellement zéro. C'est le seul dispositif du projet "
                "qui apporte du RAPPEL que la persistance ne peut pas fournir.", False, None)])
    doc.add_page_break()

    # ====================================== X. alertes
    h(doc, "X. Prédiction d'alertes — la reformulation", 1)
    para(doc, "L'objectif du système n'est pas de suivre la valeur de l'indice mais de "
              "déclencher une alerte. Or l'indice porte déjà des seuils : prédire l'indice "
              "prédit donc l'alerte. Cette logique a été suivie jusqu'au bout, avec réglage du "
              "seuil de décision EN CV — et la persistance reçoit exactement le même traitement.")
    a = ALS.copy()
    table(doc, ["Étiquette", "Décision", "Critère", "Rappel", "Précision", "F1",
                "Rappel persist.", "Gain", "Signif."],
          [[r["etiquette"].replace("etiquette ", ""), r["decision"][:26], r["critere"],
            d4(r["rappel"], 3), d4(r["precision"], 3), d4(r["f1"], 3),
            d4(r["rappel_p"], 3), s4(r["gain_rappel"], 3) if not pd.isna(r["gain_rappel"]) else "—",
            "OUI" if r["signif"] else "non"] for _, r in a.iterrows()],
          widths=[1.0, 1.7, 1.0, 0.6, 0.65, 0.5, 0.75, 0.6, 0.5], size=7)
    para(doc, "")
    bullets(doc, [
        "Pour « sera-t-il en alerte dans 3 h ? », la logique fonctionne : sur l'étiquette à "
        "viscosité masquée, le rappel passe de 0,885 (persistance) à 0,925, gain significatif.",
        "Pour détecter une TRANSITION (Normal → alerte), le seuil sur l'indice prédit ÉCHOUE "
        "complètement : rappel 0,000 sur l'étiquette actuelle. Le modèle prédit correctement que "
        "l'indice va baisser un peu, mais pas assez pour franchir le seuil.",
        "En revanche un CLASSIFIEUR DÉDIÉ à la transition obtient un rappel de 0,677 avec une "
        "précision de 0,277 et un taux de fausse alerte de 11,8 % — soit 88 basculements "
        "détectés sur 130, trois heures à l'avance, là où la persistance en détecte ZÉRO par "
        "construction.",
    ])
    rich(doc, [("Il faut donc les deux modèles, pas l'un ou l'autre : ", True, GREEN),
               ("l'indice prédit avec seuil répond à « quel sera l'état », le classifieur de "
                "transition répond à « va-t-il basculer ». Le signal de basculement est dans la "
                "DYNAMIQUE, pas dans le NIVEAU prédit — c'est pourquoi un régresseur optimisé "
                "sur l'erreur de valeur n'a aucune raison de bien placer les points par rapport "
                "à un seuil.", False, None)])

    # ====================================== XI. expérience décisive
    h(doc, "XI. L'expérience décisive — masquage de la viscosité", 1)
    av, ap = DEC["decisive"]["avant"], DEC["decisive"]["apres"]
    table(doc, ["Indicateur", "Avant", "Après"],
          [["Motosoufflante A — skill", s4(av["Motosoufflante A"]["skill"]),
            s4(ap["Motosoufflante A"]["skill"])],
           ["Motosoufflante B — skill", s4(av["Motosoufflante B"]["skill"]),
            s4(ap["Motosoufflante B"]["skill"])],
           ["Global — écart d'accuracy", s4(av["global"]["delta_acc"]),
            s4(ap["global"]["delta_acc"])],
           ["Global — IC 95 %",
            f"[{av['global']['ci_acc'][0]:+.4f} ; {av['global']['ci_acc'][1]:+.4f}] ✗".replace(".", ","),
            f"[{ap['global']['ci_acc'][0]:+.4f} ; {ap['global']['ci_acc'][1]:+.4f}] ✓".replace(".", ",")],
           ["Global — taille d'échantillon effective", f"{av['global']['n_eff']:.0f}",
            f"{ap['global']['n_eff']:.0f}"]],
          widths=[2.6, 2.1, 2.1], highlight={0, 3, 4})
    para(doc, "")
    para(doc, "L'artefact du viscosimètre DÉTRUISAIT la Motosoufflante A : son skill passe de "
              "−0,119 à +0,086. Et le bénéfice global devient statistiquement significatif pour "
              "la première fois, la taille d'échantillon effective passant de 98 à 1 707 — sans "
              "une seule observation supplémentaire, simplement parce que les deux machines "
              "cessent d'avoir des écarts de signes opposés.")
    figure(doc, "09_experience_decisive.png",
           "Figure 3 — Intervalles de confiance et taille d'échantillon effective, avant et "
           "après masquage.", width=6.8)

    # ====================================== XII. limites
    h(doc, "XII. Limites", 1)
    bullets(doc, [
        "La cible est un CONSTRUIT sans vérité terrain : aucune panne, intervention, vidange ou "
        "analyse de laboratoire n'est enregistrée. Le modèle prédit le score d'anomalie du "
        "projet, pas la santé réelle.",
        "76,8 % des lignes ne bougent pas de plus de 0,01 en 3 h. Sur les lignes qui bougent "
        "vraiment, le R² du modèle n'est que de 0,039 et la corrélation entre écart réel et "
        "écart prédit de 0,394 — soit 15,5 % de la variance du mouvement expliquée.",
        "Le jeu de test ne contient que 11 sessions. Les intervalles de confiance reposent sur "
        "5 à 11 unités de rééchantillonnage : ils indiquent un accord de signe, pas une "
        "couverture garantie à 95 %.",
        "Le test contient MOINS DE 20 lignes de la Motosoufflante B en marche : tout ce qui est "
        "affirmé sur B est affirmé sur B À L'ARRÊT. Le modèle n'a jamais été évalué sur B en "
        "charge.",
        "39,96 % des viscosités sont fausses et corrompent l'étiquette. C'est un problème de "
        "collecte, pas de modélisation.",
        "L'étiquette dérive : sa référence « saine » est ajustée sur la fenêtre d'entraînement "
        "et vieillit avec l'huile. Avec l'étiquette sans fuite, la période de test contient "
        "73,3 % de lignes en alerte contre une période d'entraînement majoritairement Normale.",
    ])
    h(doc, "XII.1 Recommandations, par ordre d'impact", 2)
    bullets(doc, [
        "MASQUER LA VISCOSITÉ À L'ARRÊT — déjà prouvé expérimentalement, c'est la seule "
        "modification qui rend le bénéfice mesurable.",
        "BASCULER SUR LASSO OU ELASTICNET dans l'architecture à porte : skill +0,332 avec "
        "intervalle significatif, contre +0,094 non significatif pour le modèle actuellement "
        "retenu.",
        "AJOUTER LE CLASSIFIEUR DE TRANSITION comme second modèle : c'est le seul qui apporte "
        "une alerte précoce que la persistance ne peut structurellement pas fournir.",
        "OBTENIR UNE VÉRITÉ TERRAIN : dates de vidange, interventions, analyses d'huile. Sans "
        "elle, aucun réglage de modèle ne relèvera le plafond.",
        "ÉVALUER LA MOTOSOUFFLANTE B EN MARCHE : régime jamais testé, et c'est celui où un "
        "système d'alarme doit fonctionner.",
    ])
    h(doc, "XII.2 Ce que le système apporte réellement, en une phrase", 2)
    rich(doc, [("Sur la valeur de l'indice : ", True, None),
               ("un gain réel mais modeste — 5 à 6 % d'erreur en moins, jusqu'à +0,33 de skill "
                "pour Ridge, statistiquement significatif.", False, None)])
    rich(doc, [("Sur la décision d'alerte : ", True, None),
               ("une réduction des fausses alertes (précision 0,9688 contre 0,9358), mais AUCUN "
                "gain de rappel — aucun des 18 modèles ne détecte plus de dégradations que la "
                "persistance.", False, None)])
    rich(doc, [("Sur l'alerte précoce : ", True, None),
               ("le classifieur de transition est le seul apport net — 88 basculements sur 130 "
                "détectés trois heures à l'avance, contre zéro pour la persistance.",
                False, None)])

    return save_doc(doc, os.path.join(ROOT, "Rapport_Entrainement_Complet_iSENSE.docx"))


if __name__ == "__main__":
    p = main()
    print(f"Document genere : {p}  ({os.path.getsize(p)/1e6:.2f} Mo)")
