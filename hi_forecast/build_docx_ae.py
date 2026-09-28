"""Rapport Word dedie aux autoencodeurs (roles A et B)."""
import json
import os
import sys

import pandas as pd
from docx import Document

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from build_docx import (GREEN, GREY, NAVY, RED, bullets, figure, h, para, rich,
                        save_doc, setup, table)

A = json.load(open(os.path.join(HERE, "ae_roleA_resultats.json"), encoding="utf-8"))
dA = pd.DataFrame([{k: v for k, v in r.items() if k != "scopes"} for r in A])
B = pd.read_csv(os.path.join(HERE, "ae_roleB_resultats.csv"))
feas = pd.read_csv(os.path.join(HERE, "ae_faisabilite_fenetres.csv"))


def d4(v, n=4):
    return f"{v:.{n}f}".replace(".", ",")


def s4(v, n=4):
    return f"{v:+.{n}f}".replace(".", ",")


def main():
    doc = setup(Document())
    h(doc, "Autoencodeurs pour la prédiction du Health Index", 0)
    para(doc, "Trois architectures · deux rôles · deux étiquettes · protocole d'audit inchangé",
         size=12, color=GREY)
    para(doc, "Projet i-SENSE / OCP Maintenance Solutions — UM6P   •   8 septembre 2026",
         size=9, color=GREY, space=12)

    # ---------------------------------------------------------- 1. protocole
    h(doc, "1. Protocole et critère de réussite", 1)
    para(doc, "Le protocole est celui de l'audit, sans aucune exception :")
    bullets(doc, [
        "étiquette gelée, test gelé (7 795 lignes), CV 5 blocs temporels expansifs avec purge "
        "et embargo de 3 h, graine 42 ;",
        "autoencodeur, imputeur et mise à l'échelle ajustés DANS chaque pli, sur les seules "
        "lignes d'entraînement du pli — les autoencodeurs sont sensibles à l'échelle, "
        "contrairement aux modèles à arbres, donc ce point devient déterminant ici ;",
        "pour le rôle B, l'autoencodeur n'est entraîné que sur les lignes SAINES (aucun "
        "flag_high_* actif) de la fenêtre d'entraînement ;",
        "test scoré UNE SEULE FOIS par configuration, aucune sélection sur le test.",
    ])
    rich(doc, [("Critère de réussite. ", True, NAVY),
               ("« Dépasser 80 % » ne compte pas : la persistance atteint déjà 81,6 %. Le "
                "critère retenu est double — (a) battre la persistance en CV sur acc@±0,01 ET "
                "sur le skill ; (b) intervalle de confiance à 95 % du gain de skill (bootstrap "
                "par blocs de session) au-dessus de zéro, en global, par machine, ET par état "
                "ON/OFF. L'accuracy est rapportée à côté mais ne suffit jamais à valider.",
                False, None)])

    # ------------------------------------------------------- 2. faisabilite
    h(doc, "2. Faisabilité des fenêtres — avant tout entraînement", 1)
    para(doc, "La médiane de session est de 3 lignes. Une fenêtre ne franchit JAMAIS une "
              "frontière de session, et aucune n'est complétée par remplissage. Un échantillon "
              "à l'instant t exige les W lignes t−W+1…t et la cible à t+18 dans la même "
              "session, soit max(0, L − W − 17) échantillons pour une session de longueur L.")
    f = feas[feas["W"] > 1]
    table(doc, ["W", "Sessions utilisables", "Échantillons", "% des lignes", "Test A",
                "Test B", "Décision"],
          [[int(r["W"]), f"{int(r['sessions_ok'])} / 352", f"{int(r['echantillons']):,}".replace(",", " "),
            f"{r['pct_lignes']:.1%}".replace(".", ","), int(r["test_A"]), int(r["test_B"]),
            "retenue" + (" (utilisée)" if int(r["W"]) == 18 else "")]
           for _, r in f.iterrows()],
          widths=[0.5, 1.5, 1.2, 1.0, 0.7, 0.7, 1.3], highlight={2})
    para(doc, "")
    rich(doc, [("Aucune fenêtre n'a dû être écartée, mais deux réserves comptent.", True, RED)])
    bullets(doc, [
        "126 sessions sur 352 (35,8 %) ne comptent qu'une seule ligne et ne contribuent à rien, "
        "quelle que soit W. Les 89,5 % de lignes conservées viennent d'un petit nombre de "
        "longues sessions.",
        "Concentration extrême : à W = 18, les 5 plus grosses sessions fournissent 56,4 % des "
        "échantillons, et la session Motosoufflante A_S38 en fournit 7 869 à elle seule. Les "
        "modèles séquentiels apprennent donc surtout la dynamique de quelques sessions, pas "
        "celle du parc.",
    ])
    figure(doc, "17_ae_faisabilite.png",
           "Figure 1 — Sessions et échantillons utilisables selon la longueur de fenêtre.",
           width=6.6)
    doc.add_page_break()

    # ---------------------------------------------------------- 3. ROLE A
    h(doc, "3. RÔLE A — prévision de health_index à t+18", 1)
    para(doc, "Le vecteur latent et l'erreur de reconstruction de l'autoencodeur sont AJOUTÉS "
              "aux 20 variables sélectionnées par SHAP, puis passés au régresseur d'écart de "
              "l'architecture à porte. Les architectures séquentielles sont en outre testées "
              "en prévision DIRECTE de l'écart (seq2seq).")

    h(doc, "3.1 Les cinq configurations qui remplissent (a) et (b)", 2)
    p = dA[dA["passe"]].sort_values("test_skill", ascending=False)
    table(doc, ["Architecture", "Étiquette", "CV acc", "CV skill", "Test acc", "Test R²",
                "Test skill"],
          [[r["nom"], r["etiquette"].replace("etiquette ", ""), d4(r["cv_acc"]),
            s4(r["cv_skill"]), d4(r["test_acc"]), d4(r["test_r2"]), s4(r["test_skill"])]
           for _, r in p.iterrows()],
          widths=[2.1, 1.3, 0.7, 0.7, 0.7, 0.7, 0.75], size=8, highlight={0, 1})
    para(doc, "")
    rich(doc, [("Comparaison avec le modèle retenu jusqu'ici. ", True, None),
               ("La porte LightGBM sans autoencodeur atteignait 0,8259 d'accuracy de test et "
                "+0,0941 de skill, mais ÉCHOUAIT le critère (b) : son IC global "
                "[−0,005 ; +0,166] et celui de la machine A [−0,333 ; +0,495] contenaient zéro. "
                "L'autoencodeur dense débruiteur est la première configuration du projet à "
                "franchir les deux critères sur tous les périmètres.", False, None)])

    best = max(A, key=lambda r: (r["passe"], r["test_skill"] if r["etiquette"] ==
                                 "etiquette actuelle" else -9))
    h(doc, "3.2 Détail du critère (b) — meilleur autoencodeur, étiquette actuelle", 2)
    table(doc, ["Périmètre", "Skill", "IC 95 %", "Blocs", "> 0 ?"],
          [[k, s4(v["skill"]),
            f"[{v['ci_skill'][0]:+.4f} ; {v['ci_skill'][1]:+.4f}]".replace(".", ","),
            v["n_blocks"], "OUI" if v["skill_clears_zero"] else "non"]
           for k, v in best["scopes"].items() if k in ("global", "A", "B", "ON", "OFF")],
          widths=[1.2, 1.0, 2.0, 0.7, 0.7], highlight={0})
    para(doc, "")
    para(doc, "⚠️ La réserve sur le nombre de blocs reste entière : 5 à 11 unités de "
              "rééchantillonnage ne produisent pas un intervalle de couverture garantie. Ces "
              "intervalles signifient « les sessions s'accordent en signe », pas « 95 % de "
              "couverture ».", size=9, color=RED)
    figure(doc, "19_ae_criteres.png",
           "Figure 2 — Critère (b) par périmètre : autoencodeur contre modèle à porte "
           "précédent.", width=6.6)

    h(doc, "3.3 Ce qui échoue, et pourquoi c'est instructif", 2)
    bullets(doc, [
        "AUCUNE architecture séquentielle ne passe. GRU, LSTM et Conv1D franchissent tous le "
        "critère (a) mais échouent (b) : leur intervalle de skill contient zéro sur la machine "
        "A et sur l'état ON — les deux périmètres où la persistance est déjà très forte. Le "
        "gain se concentre sur B/OFF, exactement comme pour les modèles à arbres.",
        "La prévision directe seq2seq échoue le critère (a), avec un skill en CV négatif : "
        "−0,0248 (GRU, étiquette actuelle), −0,4368 (GRU, masquée), −0,4071 (Conv1D).",
        "Le cas le plus instructif est AE-3 Conv1D seq2seq direct sur l'étiquette actuelle : "
        "c'est le MEILLEUR skill de test de tout le banc d'essai (+0,2893), et il est rejeté "
        "parce que son skill en CV vaut −0,4071. Sans la discipline « sélection en CV "
        "uniquement », c'est ce modèle qui aurait été déployé.",
        "Lecture d'ensemble : compresser l'ÉTAT INSTANTANÉ aide ; modéliser la SÉQUENCE n'aide "
        "pas. C'est cohérent avec la structure des données — la plupart des sessions sont trop "
        "courtes pour porter une dynamique, et le peu de dynamique exploitable est déjà capturé "
        "par les variables _diff1, _ewma et _slope_3h.",
    ])
    figure(doc, "18_ae_roleA_skill.png",
           "Figure 3 — Skill de chaque architecture, par étiquette. Vert : passe (a) et (b) ; "
           "orange : passe (a) seulement ; rouge : échoue (a).", width=7.0)
    doc.add_page_break()

    # ---------------------------------------------------------- 4. ROLE B
    h(doc, "4. RÔLE B — reconstruction de l'ÉTIQUETTE", 1)
    rich(doc, [("AVERTISSEMENT — ces accuracies ne sont PAS comparables à celles du rôle A.",
                True, RED)])
    para(doc, "La cible n'est pas la même. Une étiquette plus lisse fait monter mécaniquement "
              "l'accuracy sans qu'aucune prévision se soit améliorée ; une étiquette plus "
              "rugueuse la fait chuter. La colonne « écart-type de l'étiquette » rend cet effet "
              "visible. Seul le SKILL, mesuré contre la persistance de la MÊME étiquette, reste "
              "interprétable d'une ligne à l'autre. Ce tableau ne rejoint jamais celui du rôle A.")
    para(doc, "Sévérité = erreur de reconstruction / centile 99 des lignes saines "
              "d'entraînement, puis health_index = 1 / (1 + sévérité). L'autoencodeur n'est "
              "entraîné que sur les lignes saines de la fenêtre d'entraînement.", size=9.5)
    table(doc, ["Base", "Construction de l'étiquette", "Écart-type", "CV acc", "Test acc",
                "Test acc persist.", "Skill"],
          [[r["base"].replace("etiquette ", ""),
            r["tag"].split(" (")[0], d4(r["sd_label"]), d4(r["cv_acc"]), d4(r["test_acc"]),
            d4(r["test_acc_persist"]), s4(r["test_skill"])] for _, r in B.iterrows()],
          widths=[1.1, 2.3, 0.8, 0.65, 0.7, 0.9, 0.7], size=7.5, highlight={0, 7})
    para(doc, "")
    rich(doc, [("Verdict : remplacer PCA T²/SPE + Isolation Forest par une erreur de "
                "reconstruction d'autoencodeur DÉGRADE l'étiquette.", True, RED)])
    bullets(doc, [
        "Les étiquettes « AE seul » sont trois à quatre fois plus rugueuses (écart-type 0,157 "
        "à 0,210 contre 0,046 à 0,048 pour la référence) et leur skill s'effondre à ±0,01, "
        "voire devient négatif.",
        "La combinaison OR (AE ou PCA) limite les dégâts mais reste toujours en dessous de la "
        "référence.",
        "C'est une confirmation expérimentale, sous protocole sans fuite, de la décision de "
        "Phase 4 du projet d'origine, qui écartait l'autoencodeur sur la base de trois tests de "
        "validation. Cette fois la décision repose sur une mesure de prévisibilité, pas "
        "seulement sur des tests de cohérence.",
        "Noter que l'effet redouté — « une étiquette plus lisse gonfle l'accuracy » — se produit "
        "ici EN SENS INVERSE : les étiquettes AE étant plus rugueuses, l'accuracy chute (0,32 à "
        "0,69 contre 0,84 pour la référence). L'avertissement reste néanmoins nécessaire pour "
        "la combinaison OR, dont l'écart-type se rapproche de celui de la référence.",
    ])
    figure(doc, "20_ae_roleB.png",
           "Figure 4 — Rôle B : écart-type de l'étiquette reconstruite et accuracy associée.",
           width=7.0)
    doc.add_page_break()

    # ---------------------------------------------------------- 5. synthese
    h(doc, "5. Synthèse", 1)
    bullets(doc, [
        "Un autoencodeur améliore réellement la prévision — mais uniquement en rôle A, "
        "uniquement en architecture DENSE, et uniquement comme fournisseur de variables "
        "(latent + erreur de reconstruction) pour le régresseur d'écart.",
        "La variante DÉBRUITEUSE est la meilleure (latent 16, bruit 0,3), ce qui est cohérent "
        "avec un jeu de capteurs bruité : forcer la reconstruction à partir d'une entrée "
        "dégradée produit un latent plus robuste.",
        "C'est la première configuration du projet à remplir les deux critères sur TOUS les "
        "périmètres, là où le modèle à porte LightGBM échouait sur la machine A et en global.",
        "Les architectures séquentielles n'apportent rien ici, et la prévision directe seq2seq "
        "est franchement mauvaise en validation croisée malgré des scores de test parfois "
        "flatteurs.",
        "En rôle B, l'autoencodeur dégrade l'étiquette et ne doit pas remplacer PCA + "
        "Isolation Forest.",
    ])
    h(doc, "5.1 Recommandation", 2)
    rich(doc, [("Retenir l'AE-1 dense débruiteur (latent 16, bruit 0,3) comme extracteur de "
                "variables, en amont du régresseur d'écart et de la porte.", True, GREEN)])
    para(doc, "Sur l'étiquette actuelle il porte l'accuracy de test de 0,8259 à 0,8499 et le "
              "skill de +0,0941 à +0,1064, tout en devenant significatif sur tous les "
              "périmètres. Sur l'étiquette à viscosité masquée — celle que recommande l'audit — "
              "il atteint un R² de test de 0,5079 pour un skill de +0,1088.")
    para(doc, "Sources reproductibles : run_15_ae_faisabilite.py, ae_models.py, "
              "run_16_ae_roleA.py, run_16b_ae_roleA_driver.py, run_17_ae_roleB.py ; résultats "
              "dans ae_roleA_resultats.csv/.json et ae_roleB_resultats.csv/.json.",
         italic=True, size=8.5, color=GREY)

    return save_doc(doc, os.path.join(ROOT, "Rapport_Autoencodeurs_iSENSE.docx"))


if __name__ == "__main__":
    p = main()
    print(f"Document genere : {p}  ({os.path.getsize(p)/1e6:.2f} Mo)")
