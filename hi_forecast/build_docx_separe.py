"""Rapport Word — entrainement SEPARE par machine, deux validations, toutes metriques."""
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

D = pd.read_csv(os.path.join(HERE, "separe_par_machine_complet.csv"))
with open(os.path.join(HERE, "artifacts", "reco_separe.json"), encoding="utf-8") as f:
    RECO = json.load(f)
with open(os.path.join(HERE, "plis_detail.json"), encoding="utf-8") as f:
    PLIS = json.load(f)
with open(os.path.join(HERE, "porte_stats.json"), encoding="utf-8") as f:
    PORTE = json.load(f)
CV = "croisee (blocs expansifs)"
WF = "glissante (walk-forward)"
G = "G* porte réglée"
MACH = sorted(D["machine"].unique())

DESC = {
    "Ridge": ("Linéaire régularisée L2",
              "Rétrécit tous les coefficients sans en annuler aucun. Adapté quand les variables "
              "sont nombreuses et corrélées : il répartit le poids plutôt que d'en choisir un.",
              "R², RMSE, MAE — un modèle linéaire s'évalue d'abord sur l'erreur quadratique, "
              "sa métrique d'entraînement."),
    "Lasso": ("Linéaire régularisée L1",
              "Annule les coefficients inutiles : il sélectionne un sous-ensemble parcimonieux "
              "de variables. Utile ici où 20 variables sur 163 suffisent.",
              "R², RMSE + nombre de variables retenues — la parcimonie fait partie de la qualité."),
    "ElasticNet": ("Linéaire L1 + L2",
                   "Compromis : sélectionne comme le Lasso, stabilise comme le Ridge sur les "
                   "groupes de variables corrélées (les lags d'une même grandeur, par exemple).",
                   "Mêmes métriques que Ridge et Lasso, plus la stabilité entre plis."),
    "Random Forest": ("Arbres — bagging",
                      "Forêt d'arbres décorrélés par double échantillonnage, agrégés par moyenne. "
                      "Capture non-linéarités et interactions sans réglage fin.",
                      "MCC et F1 d'alerte — un modèle à arbres découpe l'espace en régions, ce qui "
                      "le rend naturellement bon sur des décisions par seuil."),
    "CatBoost": ("Arbres — boosting",
                 "Boosting de gradient à arbres symétriques, corrige séquentiellement le résidu. "
                 "Robuste sans réglage.",
                 "R², RMSE et MCC — il optimise une perte quadratique mais reste un modèle de "
                 "décision par régions."),
    "LightGBM Huber": ("Arbres — boosting, perte robuste",
                       "Croissance en profondeur d'abord, avec perte de Huber : il estime une "
                       "MÉDIANE conditionnelle, pas une moyenne. Il prédit donc des écarts petits, "
                       "ce que la bande de tolérance récompense.",
                       "accuracy @ ±0,01 et MedAE — ce sont les métriques que sa perte optimise "
                       "réellement. Le juger au R² serait le juger sur un objectif qui n'est pas "
                       "le sien."),
    "AE-1 dense débruiteur": ("Autoencodeur dense débruiteur",
                              "Comprime les 70 variables capteur en 16 dimensions latentes, en "
                              "apprenant à reconstruire une entrée BRUITÉE. Le latent et l'erreur "
                              "de reconstruction alimentent ensuite le régresseur.",
                              "accuracy, MedAE et erreur de reconstruction — un autoencodeur "
                              "s'évalue aussi sur sa capacité à reconstruire, pas seulement à prédire."),
    "Ridge + AE": ("Hybride : compression non linéaire + décision linéaire",
                   "Les variables de l'autoencodeur passent à une régression Ridge : "
                   "représentation non linéaire, décision linéaire régularisée.",
                   "R² et RMSE, comme Ridge, avec la stabilité inter-plis en critère secondaire."),
}


def d4(v, n=4):
    return "—" if pd.isna(v) else f"{v:.{n}f}".replace(".", ",")


def s4(v, n=4):
    return "—" if pd.isna(v) else f"{v:+.{n}f}".replace(".", ",")


def main():
    doc = setup(Document())
    h(doc, "Modèles de prédiction par motosoufflante", 0)
    para(doc, "Entraînement séparé · deux schémas de validation · toutes les métriques",
         size=12, color=GREY)
    para(doc, "Projet i-SENSE / OCP Maintenance Solutions — UM6P   •   9 septembre 2026",
         size=9, color=GREY, space=12)

    # ------------------------------------------------------------ 1. principe
    h(doc, "1. Ce qui a été fait, en clair", 1)
    para(doc, "Chaque motosoufflante reçoit ses PROPRES modèles, entraînés uniquement sur ses "
              "lignes et validés sur sa propre chronologie. Huit familles de modèles sont "
              "comparées, sous deux schémas de validation et trois configurations de décision, "
              "soit 96 configurations évaluées au total.")
    table(doc, ["Élément", "Motosoufflante A", "Motosoufflante B"],
          [["Lignes d'entraînement", "15 232", "16 437"],
           ["Lignes de test (gelé)", "3 518", "4 277"],
           ["Part du temps en marche", "93 %", "10 %"],
           ["Accuracy de la persistance", "0,9258", "0,7264"]],
          widths=[2.3, 2.3, 2.3])

    h(doc, "2. Les deux schémas de validation", 1)
    rich(doc, [("Validation croisée par blocs expansifs. ", True, NAVY),
               ("Le bloc d'entraînement grandit à chaque pli : on entraîne sur [0..i] et on "
                "valide sur le bloc i+1. Elle répond à la question « avec TOUT l'historique "
                "disponible, que vaut le modèle ? ».", False, None)])
    rich(doc, [("Validation glissante (walk-forward). ", True, NAVY),
               ("La fenêtre d'entraînement garde une TAILLE FIXE et avance dans le temps. Elle "
                "répond à « avec un historique récent et borné, que vaut le modèle ? » — c'est "
                "le régime réel d'un système réentraîné périodiquement en production.",
                False, None)])
    para(doc, "Les deux appliquent la même purge et le même embargo de 3 heures de chaque côté "
              "de la frontière de validation : sans cela, les dernières lignes d'entraînement "
              "partagent leur cible avec le début de la validation et le score est faussé.")
    rich(doc, [("Pourquoi comparer les deux : ", True, None),
               ("la validation glissante est plus sévère parce qu'elle dispose de moins de "
                "données. Si un modèle ne tient que sous validation croisée, c'est qu'il a "
                "besoin de tout l'historique — information utile avant de le déployer.",
                False, None)])
    figure(doc, "23_separe_validations.png",
           "Figure 1 — Les deux schémas de validation, par machine et par famille.", width=6.8)

    h(doc, "2.1 Ce qu'est un « pli », concrètement", 2)
    para(doc, "Un PLI (fold) est une des découpes de la validation : un morceau de données mis "
              "de côté pour évaluer, pendant que le modèle apprend sur le reste. « 5 plis » "
              "signifie qu'on répète l'opération cinq fois, avec un morceau différent à chaque "
              "fois. Les lignes d'entraînement de chaque machine sont découpées par ordre "
              "CHRONOLOGIQUE en six blocs égaux ; chaque pli entraîne sur tout ce qui précède et "
              "valide sur le bloc suivant.")
    for mach in MACH:
        info = PLIS.get(mach)
        if not info:
            continue
        h(doc, f"Les 5 plis de la {mach} ({info['n_train']:,} lignes)".replace(",", " "), 3)
        table(doc, ["Pli", "Lignes d'entraînement", "Lignes de validation",
                    "Période validée", "Accuracy Lasso", "Accuracy RF"],
              [[r["pli"], f"{r['n_train']:,}".replace(",", " "),
                f"{r['n_val']:,}".replace(",", " "),
                f"{r['debut']} → {r['fin']}", d4(r["acc_lasso"]), d4(r["acc_rf"])]
               for r in info["plis"]]
              + [["", "", "", "moyenne ± écart-type",
                  f"{d4(info['lasso_moy'])} ± {d4(info['lasso_sd'], 3)}",
                  f"{d4(info['rf_moy'])} ± {d4(info['rf_sd'], 3)}"]],
              widths=[0.45, 1.5, 1.4, 1.85, 1.15, 1.05], size=8.5,
              highlight={len(info["plis"])})
        para(doc, "")

    rich(doc, [("Trois propriétés à remarquer.", True, NAVY)])
    bullets(doc, [
        "Le bloc d'entraînement GRANDIT à chaque pli (de 2 506 à 12 658 lignes pour la "
        "Motosoufflante A). C'est voulu : on simule un système qui accumule de l'historique.",
        "La validation est TOUJOURS dans le futur du bloc d'entraînement, jamais l'inverse.",
        "Entre les deux, 3 heures de vide (purge + embargo). La cible étant à t+3 h, sans ce "
        "vide les dernières lignes d'entraînement auraient leur cible À L'INTÉRIEUR du bloc de "
        "validation — une fuite qui gonflerait le score.",
    ])
    rich(doc, [("Pourquoi l'écart-type compte autant que la moyenne. ", True, RED),
               ("Sur la Motosoufflante B, Lasso obtient 0,8379 ± 0,0392 et Random Forest "
                "0,8169 ± 0,0582 : Lasso est meilleur ET plus régulier. Un modèle dont "
                "l'écart-type est élevé marche bien certains mois et s'effondre d'autres — en "
                "production on ne saura pas lesquels à l'avance. C'est ce que le « ± sd » sert "
                "à détecter, et c'est pourquoi un écart de moyenne plus petit que l'écart-type "
                "n'est PAS une différence réelle entre deux modèles.", False, None)])

    h(doc, "3. Les trois configurations de décision", 1)
    p = doc.add_paragraph()
    r = p.add_run("prédiction = health_index[t] + α · Δ̂ · 1[ P(mouvement) > seuil ]")
    r.font.name, r.font.size, r.bold = "Consolas", 10, True
    r.font.color.rgb = NAVY
    table(doc, ["Configuration", "Réglage", "Ce qu'elle teste"],
          [["G0 — persistance", "α = 0", "Borne basse : ne rien corriger du tout. Toute famille "
                                         "doit battre cette ligne pour exister."],
           ["G1 — sans porte", "α = 1, correction systématique",
            "Le modèle corrige toujours, sans filtre. Mesure la valeur brute de la prédiction."],
           ["G* — porte réglée", "α et seuil choisis en validation",
            "Le modèle ne corrige que lorsqu'un second modèle juge un mouvement probable."]],
          widths=[1.5, 1.9, 3.5])
    para(doc, "")
    h(doc, "3.1 Porte ouverte ou fermée : ce que cela signifie", 2)
    para(doc, "La porte est un SECOND modèle, distinct de celui qui prédit la valeur. Son unique "
              "rôle est de répondre à une question binaire : le health index va-t-il bouger de "
              "plus que la tolérance de ±0,01 d'ici l'horizon ?")
    table(doc, ["État", "Ce qui se passe", "Prédiction affichée"],
          [["FERMÉE", "Le modèle estime que rien ne bougera. Aucune correction n'est appliquée.",
            "= persistance, écart de +0,00000"],
           ["OUVERTE", "Un mouvement est jugé probable. La correction du régresseur est appliquée.",
            "= persistance + écart prédit"]],
          widths=[1.0, 3.7, 2.2], highlight={0})
    para(doc, "")
    rich(doc, [("Pourquoi cette porte existe. ", True, NAVY),
               ("C'est le résultat central de l'audit. Un balayage du facteur de correction α "
                "sur cinq familles de modèles donne un α optimal en validation croisée égal à "
                "ZÉRO pour 13 familles sur 14 : corriger systématiquement la persistance "
                "DÉGRADE la précision. La raison est arithmétique — la persistance est déjà "
                "juste sur environ 79 % des lignes, et une correction bruitée fait sortir de la "
                "bande ±0,01 une partie de ces 79 % plus vite qu'elle n'y fait rentrer une "
                "partie des 21 % restants. La porte résout cela : on ne corrige que là où il y "
                "a quelque chose à corriger.", False, None)])
    rich(doc, [("Fréquence réelle d'ouverture, sur le test gelé, pour les modèles retenus :",
                True, None)])
    table(doc, ["Machine", "Seuil", "Porte ouverte", "Mouvement réel",
                "Précision (ouverte → ça bouge)", "Rappel (mouvements captés)",
                "Fermée à raison"],
          [[mach.replace("Motosoufflante ", "Motosoufflante "),
            d4(PORTE[mach]["seuil"], 2),
            f"{PORTE[mach]['pct_ouverte']:.1%}".replace(".", ","),
            f"{PORTE[mach]['pct_bouge_reel']:.1%}".replace(".", ","),
            f"{PORTE[mach]['precision']:.1%}".replace(".", ","),
            f"{PORTE[mach]['rappel']:.1%}".replace(".", ","),
            f"{PORTE[mach]['exactitude_fermee']:.1%}".replace(".", ",")]
           for mach in MACH if mach in PORTE],
          widths=[1.6, 0.6, 1.0, 1.0, 1.5, 1.3, 1.0], size=8)
    para(doc, "")
    bullets(doc, [
        "Une porte fermée est le cas NORMAL et MAJORITAIRE : elle reste fermée sur 90 % des "
        "lignes de la Motosoufflante A et 83 % de celles de la Motosoufflante B. Une porte qui "
        "s'ouvrirait en permanence serait le signal inquiétant — machine en dégradation active, "
        "ou classifieur mal calibré qui crie au loup.",
        "Quand elle reste fermée, elle a raison dans 94,3 % des cas sur la Motosoufflante A et "
        "80,7 % sur la B : l'indice ne bouge effectivement pas.",
        "Quand elle s'ouvre, elle a raison dans 66,0 % des cas sur la Motosoufflante B, mais "
        "seulement 22,7 % sur la A. C'est cohérent avec le fait que A bouge rarement (7,4 % des "
        "lignes contre 27,4 % pour B) : détecter un événement rare est intrinsèquement plus "
        "difficile, et la porte de A ouvre donc souvent pour rien.",
        "Le rappel reste modeste (30,3 % sur A, 41,6 % sur B) : la porte laisse passer la "
        "majorité des mouvements réels. C'est un choix assumé — le seuil est réglé en validation "
        "pour maximiser l'accuracy finale, pas pour tout capter. Ouvrir plus souvent "
        "rattraperait des mouvements mais casserait davantage de lignes stables.",
    ])
    para(doc, "Les valeurs de skill et d'accuracy affichées par le tableau de bord sont "
              "HISTORIQUES : mesurées une fois pour toutes sur le test gelé, elles disent ce que "
              "vaut le modèle en général et ne changent pas d'un rafraîchissement à l'autre.",
         size=9.5, italic=True, color=GREY)

    figure(doc, "24_separe_configs.png",
           "Figure 2 — Les trois configurations. La porte réglée domine partout ; la correction "
           "systématique (G1) dégrade la persistance.", width=6.8)

    # ------------------------------------------------- 4. les modèles
    doc.add_page_break()
    h(doc, "4. Les huit modèles et leurs métriques appropriées", 1)
    para(doc, "Chaque famille optimise une fonction de coût différente : la juger sur une "
              "métrique étrangère à son objectif est une erreur d'interprétation courante. "
              "Le tableau ci-dessous indique, pour chacune, la métrique qui lui correspond.")
    for fam, (cat, principe, metr) in DESC.items():
        h(doc, f"{fam} — {cat}", 3)
        rich(doc, [("Principe. ", True, None), (principe, False, None)], size=9.5)
        rich(doc, [("Métriques appropriées. ", True, GREEN), (metr, False, None)], size=9.5)

    # ------------------------------------------------- 5. résultats par machine
    doc.add_page_break()
    h(doc, "5. Comment lire les tableaux de résultats", 1)
    para(doc, "Trois colonnes portent l'essentiel de la décision. Elles ne mesurent pas la même "
              "chose et ne servent pas au même usage.")
    table(doc, ["Colonne", "Ce qu'elle mesure", "À quoi elle sert"],
          [["Val acc ± sd",
            "Accuracy en validation : part des prédictions tombant dans la bande de ±0,01 "
            "autour de la vraie valeur. Moyenne ± écart-type sur les 5 plis.",
            "CRITÈRE DE SÉLECTION. L'écart-type compte autant que la moyenne : il dit si le "
            "modèle est régulier ou s'il marche certains mois et s'effondre d'autres."],
           ["Val skill",
            "Gain sur la persistance, en validation :\nskill = 1 − MSE(modèle) / MSE(persistance).\n"
            "0 = aussi bon que « rien ne change en 3 h ». +0,20 = 20 % de l'erreur quadratique "
            "retirée. Négatif = pire que ne rien prédire.",
            "MESURE HONNÊTE DU GAIN. Le R² se compare à la moyenne de la cible — référence "
            "sans intérêt pour une série aussi autocorrélée. Le skill se compare à la "
            "persistance, qui est la vraie alternative."],
           ["Test acc",
            "La même accuracy, mais sur le test gelé (7 795 lignes jamais vues), scorée UNE "
            "SEULE FOIS par configuration.",
            "VÉRIFICATION, jamais sélection. Si on choisissait d'après elle, elle cesserait "
            "d'être une mesure indépendante."]],
          widths=[1.1, 2.8, 3.0], size=8.5)
    para(doc, "")
    rich(doc, [("Règle appliquée : ", True, NAVY),
               ("le modèle retenu est le PREMIER EN VALIDATION PARMI LES SIGNIFICATIFS — et non "
                "le premier tout court. Un modèle dont l'intervalle de confiance contient zéro "
                "n'a pas démontré qu'il fait mieux que ne rien prédire, quelle que soit sa place "
                "au classement.", False, None)])

    doc.add_page_break()
    h(doc, "6. Résultats détaillés par machine", 1)
    figure(doc, "22_separe_classement.png",
           "Figure 3 — Validation croisée contre test gelé, par machine. Vert : gain "
           "statistiquement significatif.", width=7.0)

    for mach in MACH:
        h(doc, f"6.{MACH.index(mach)+1} {mach}", 2)
        for sname, lib in [(CV, "Validation croisée (blocs expansifs)"),
                           (WF, "Validation glissante (walk-forward)")]:
            h(doc, lib, 3)
            s = D[(D.machine == mach) & (D.validation == sname) &
                  (D.config == G)].sort_values("val_acc", ascending=False)
            table(doc, ["Famille", "Val acc ± sd", "Val skill", "Test acc", "R²", "RMSE",
                        "MAE", "MedAE", "MAPE", "Skill", "MCC", "F1 al.", "Sig."],
                  [[r["famille"], f"{d4(r['val_acc'])}±{d4(r['val_sd'], 3)}",
                    s4(r["val_skill"]), d4(r["acc_tol"]), d4(r["r2"]), d4(r["rmse"]),
                    d4(r["mae"]), d4(r["medae"]), d4(r["mape"], 2), s4(r["skill"]),
                    d4(r["etat_mcc"]), d4(r["alerte_f1"]),
                    "OUI" if r["significatif"] else "non"] for _, r in s.iterrows()],
                  widths=[1.35, 0.85, 0.6, 0.55, 0.5, 0.55, 0.55, 0.55, 0.5, 0.55, 0.5, 0.5, 0.4],
                  size=6.5, highlight={0})
            para(doc, f"Référence persistance : accuracy {d4(s['acc_persist'].iloc[0])}.",
                 italic=True, size=8.5, color=GREY)
        rec = RECO.get(mach, {})
        if rec:
            rich(doc, [(f"Modèle retenu pour {mach} : ", True, GREEN),
                       (f"{rec['famille']} — validation {d4(rec['val_acc'])}, test "
                        f"{d4(rec['test_acc'])}, skill {s4(rec['skill'])}, "
                        f"{'significatif' if rec.get('significatif') else 'NON significatif'}.",
                        False, None)])
        para(doc, "", space=8)

    doc.add_page_break()
    h(doc, "7. Pourquoi Lasso est retenu sur les deux machines", 1)
    para(doc, "Les deux machines désignent la même famille, mais pour des raisons différentes. "
              "Le détail mérite d'être posé, car le cas de la Motosoufflante A n'est pas "
              "évident au premier regard.")

    for mach in MACH:
        s = D[(D.machine == mach) & (D.validation == CV) &
              (D.config == G)].sort_values("val_acc", ascending=False)
        h(doc, f"7.{MACH.index(mach)+1} {mach}", 2)
        table(doc, ["Famille", "Val acc", "Val sd", "Val skill", "Test acc", "Skill",
                    "IC bas", "Significatif"],
              [[r["famille"], d4(r["val_acc"]), d4(r["val_sd"]), s4(r["val_skill"]),
                d4(r["acc_tol"]), s4(r["skill"]), s4(r["ic_bas"]),
                "OUI" if r["significatif"] else "non"] for _, r in s.iterrows()],
              widths=[1.6, 0.75, 0.7, 0.8, 0.75, 0.75, 0.75, 0.9], size=8,
              highlight={i for i, (_, r) in enumerate(s.iterrows())
                         if r["famille"] == "Lasso"})
        para(doc, "")
        if mach.endswith("B"):
            rich(doc, [("Le cas clair. ", True, GREEN),
                       ("Lasso est premier sur les quatre critères à la fois : meilleure "
                        "accuracy de validation, plus faible écart-type, meilleur skill de "
                        "validation, meilleur skill de test. C'est rare dans ce projet — les "
                        "métriques se contredisent d'habitude. Ici elles convergent, ce qui rend "
                        "le choix incontestable.", False, None)])
        else:
            rich(doc, [("Le cas qui demande une explication. ", True, RED),
                       ("Sur cette machine, Lasso n'est que TROISIÈME en accuracy de validation "
                        "(0,7789 contre 0,7854 pour LightGBM Huber). Trois raisons le font "
                        "retenir malgré tout :", False, None)])
            bullets(doc, [
                "Son skill de validation est de loin le meilleur : +0,2024 contre +0,1232 pour "
                "le premier — soit 64 % de gain en plus sur ce qui compte vraiment.",
                "Il est le seul, avec ElasticNet et Ridge, dont l'intervalle de confiance exclut "
                "zéro. Les deux modèles placés devant lui ne sont PAS statistiquement "
                "significatifs : leur gain peut être nul.",
                "L'écart d'accuracy qui le sépare du premier est de 0,0065, contre un écart-type "
                "de 0,08 — largement à l'intérieur du bruit. Ce n'est pas une différence réelle.",
            ])
            para(doc, "Retenir LightGBM Huber sur cette machine reviendrait à déployer un modèle "
                      "dont on ne peut pas affirmer qu'il fait mieux que ne rien prédire — son "
                      "skill de test est même négatif (−0,0128).", size=9.5)

    h(doc, "7.3 La raison de fond", 2)
    para(doc, "Lasso pénalise en L1 : il ANNULE les coefficients inutiles au lieu de simplement "
              "les rétrécir. Sur ce problème, l'écart à prédire est de très faible amplitude "
              "(|Δ| médian de 0,003) et approximativement linéaire dans les variables de "
              "dynamique récente. Une régularisation forte évite d'apprendre le bruit — "
              "exactement ce dont on a besoin quand 77 % des lignes ne bougent pas de plus que "
              "la tolérance.")
    rich(doc, [("L'argument le plus solide : ", True, GREEN),
               ("les DEUX machines choisissent la même famille alors que leurs régimes sont "
                "opposés — la Motosoufflante A tourne 93 % du temps, la Motosoufflante B est à "
                "l'arrêt 90 % du temps. Quand deux populations aussi différentes convergent vers "
                "le même modèle, ce n'est pas un hasard d'échantillonnage.", False, None)])

    doc.add_page_break()
    h(doc, "8. Toutes les métriques, vue d'ensemble", 1)
    para(doc, "Quinze métriques par modèle et par machine : onze de régression (accuracy dans "
              "la bande, R², RMSE, MAE, MedAE, MAPE, variance expliquée, erreur maximale, biais, "
              "skill) et huit de décision d'état (accuracy, accuracy équilibrée, F1 macro, MCC, "
              "kappa, rappel, précision et F1 d'alerte).")
    figure(doc, "25_separe_heatmap.png",
           "Figure 4 — Carte thermique de toutes les métriques, par machine. La couleur donne "
           "le rang relatif : aucune ligne n'est uniformément verte.", width=7.0)
    figure(doc, "26_separe_significativite.png",
           "Figure 5 — Skill et intervalle de confiance à 95 % par bootstrap de sessions.",
           width=6.8)

    # ------------------------------------------------- 7. conclusions
    h(doc, "9. Ce qu'il faut retenir", 1)
    lines = []
    for mach in MACH:
        rec = RECO.get(mach, {})
        if rec:
            lines.append(f"{mach} : {rec['famille']} retenu (validation {d4(rec['val_acc'])}, "
                         f"test {d4(rec['test_acc'])}, skill {s4(rec['skill'])}).")
    bullets(doc, lines + [
        "Les DEUX machines retiennent le même modèle — Lasso — alors qu'elles ont des régimes "
        "opposés (A tourne 93 % du temps, B est à l'arrêt 90 % du temps). C'est un argument "
        "fort en faveur d'une régularisation linéaire simple plutôt que d'une architecture "
        "complexe.",
        "La porte réglée (G*) bat systématiquement la correction brute (G1) : corriger la "
        "persistance sans filtre DÉGRADE la précision, parce que la persistance est déjà juste "
        "sur environ 79 % des lignes.",
        "La validation glissante donne des scores un peu plus faibles que la validation croisée, "
        "ce qui est attendu : elle dispose de moins de données d'entraînement. Le classement "
        "des familles reste globalement le même, ce qui indique que les conclusions ne dépendent "
        "pas du schéma de validation choisi.",
        "Aucune métrique ne suffit seule : selon celle qu'on regarde, le gagnant change. Un "
        "modèle à perte de Huber (LightGBM) domine sur la bande de tolérance et la MedAE ; un "
        "modèle linéaire domine sur R², RMSE et skill ; un modèle à arbres domine sur MCC et F1 "
        "d'alerte. C'est pourquoi le tableau les présente toutes.",
    ])
    h(doc, "9.1 Réserve méthodologique", 2)
    para(doc, "Neuf valeurs de skill en validation glissante ont divergé (jusqu'à −5×10²²) et "
              "ont été marquées non calculables. La cause est identifiée : sur certains blocs de "
              "validation, l'indice ne bouge pratiquement pas, la MSE de la persistance tend "
              "vers zéro et le rapport qui définit le skill diverge. Ces valeurs sont neutralisées "
              "dans le tableau plutôt que laissées telles quelles ; elles ne concernent aucun "
              "des modèles retenus, et l'accuracy de validation — le critère de sélection — "
              "n'est pas affectée.", size=9.5)
    para(doc, "Sources reproductibles : run_26_separe_complet.py (entraînement et validation), "
              "make_figures_separe.py (figures), separe_par_machine_complet.csv (96 lignes "
              "× 40 colonnes), artifacts/modeles_separes_par_machine.joblib.",
         italic=True, size=8.5, color=GREY)

    return save_doc(doc, os.path.join(ROOT, "Rapport_Modeles_Par_Motosoufflante.docx"))


if __name__ == "__main__":
    p = main()
    print(f"Document genere : {p}  ({os.path.getsize(p)/1e6:.2f} Mo)")
