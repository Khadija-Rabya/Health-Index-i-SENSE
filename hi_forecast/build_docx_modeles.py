"""Rapport Word dedie : comparaison detaillee des 14 familles de modeles."""
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

lb = pd.read_csv(os.path.join(HERE, "leaderboard_familles.csv"))
PERS_ACC, PERS_R2, PERS_CV = 0.8164207825529185, 0.11423529583335357, 0.7912


def d4(v, n=4):
    return f"{v:.{n}f}".replace(".", ",")


def s4(v, n=4):
    return f"{v:+.{n}f}".replace(".", ",")


def row(name):
    return lb[lb["name"] == name].iloc[0]


# ------------------------------------------------------- fiches par famille
FICHES = [
    ("ElasticNet", "Linéaire régularisée",
     "ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=5000)",
     "Régression linéaire pénalisée par une combinaison de L1 (Lasso) et L2 (Ridge). "
     "Le terme L1 met des coefficients exactement à zéro (sélection de variables), le terme "
     "L2 stabilise les coefficients des variables corrélées — ce qui compte ici, car le jeu "
     "contient des familles entières de variables quasi colinéaires (lags, EWMA, pentes d'une "
     "même grandeur).",
     "Meilleur R² de test de tout le classement (0,4136) et meilleure RMSE. Mais son R² en CV "
     "vaut −0,68 avec un écart-type de 1,58 : la performance est très inégale d'un bloc "
     "temporel à l'autre. C'est précisément pour cela que la CV ne l'a pas retenue, et la "
     "recommandation initiale de la déployer a été retirée après vérification."),
    ("Lasso", "Linéaire régularisée",
     "Lasso(alpha=1e-4, max_iter=5000)",
     "Régression linéaire pénalisée L1 pure. Sélectionne un sous-ensemble parcimonieux de "
     "variables en annulant les coefficients des autres.",
     "Meilleure accuracy de test du classement (0,8381) et deuxième R² (0,4082). Même "
     "faiblesse qu'ElasticNet : écart-type de 0,95 sur le R² en CV. Confirme que sur ce "
     "problème la régularisation forte est plus utile que la capacité du modèle."),
    ("Random Forest", "Arbres — bagging",
     "RandomForestRegressor(n_estimators=300, max_depth=14, max_features='sqrt')",
     "Forêt d'arbres décorrélés par double échantillonnage (lignes et variables), agrégés par "
     "moyenne. max_features='sqrt' — au lieu du défaut 1.0 — décorrèle davantage les arbres et "
     "divise le temps d'entraînement par quatre sur ce jeu très redondant.",
     "R² de test 0,4082, à égalité avec Lasso, pour 54 s d'entraînement. C'est la famille du "
     "modèle d'origine du projet ; sous le protocole sans fuite son R² en CV est de −1,19 "
     "avec un écart-type de 1,96, ce qui révèle l'instabilité que le découpage unique du "
     "pipeline initial masquait."),
    ("Ridge", "Linéaire régularisée",
     "Ridge(alpha=10.0)",
     "Régression linéaire pénalisée L2 : rétrécit tous les coefficients sans en annuler aucun.",
     "R² de test 0,4010, mais accuracy nettement plus faible (0,7119) : le modèle prédit des "
     "écarts non nuls sur presque toutes les lignes, ce qui fait sortir de la bande ±0,01 des "
     "prédictions que la persistance réussissait. Illustration nette du fait que R² et bande "
     "de tolérance ne récompensent pas le même comportement."),
    ("CatBoost", "Arbres — boosting",
     "CatBoostRegressor(iterations=400, depth=6, learning_rate=0.05)",
     "Boosting de gradient à arbres symétriques (oblivious trees), avec un schéma d'ordonnancement "
     "qui limite le biais de cible. Réputé robuste sans réglage.",
     "R² de test 0,3942, meilleur des quatre bibliothèques de boosting testées. Son R² en CV "
     "(−2,07 ± 3,87) reste très instable."),
    ("Gradient Boosting", "Arbres — boosting",
     "GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05)",
     "Implémentation historique de scikit-learn, arbres peu profonds ajoutés séquentiellement "
     "pour corriger le résidu courant.",
     "R² de test 0,3925 pour 629 s — la famille la plus coûteuse du banc d'essai, sans "
     "contrepartie en performance."),
    ("Extra Trees", "Arbres — bagging",
     "ExtraTreesRegressor(n_estimators=300, max_depth=14, max_features='sqrt')",
     "Forêt d'arbres extrêmement aléatoires : les seuils de coupure sont tirés au hasard plutôt "
     "qu'optimisés, ce qui augmente le biais et réduit fortement la variance.",
     "LA SEULE FAMILLE DONT LE R² EN CV EST POSITIF (+0,2671 ± 0,1452) et la seule dont le "
     "skill en CV est positif (+0,0190). Son R² de test (0,3855) est un peu inférieur à celui "
     "d'ElasticNet, mais c'est elle que désigne la CV sous un objectif R² — et son écart-type "
     "en CV est le plus faible du classement après les méthodes à noyau."),
    ("Linear (OLS)", "Linéaire",
     "LinearRegression() (aucune régularisation)",
     "Moindres carrés ordinaires, sans pénalisation.",
     "R² CV de −135,6 avec un écart-type de 302 : effondrement complet. Avec ~70 variables "
     "fortement colinéaires et des blocs de validation hors du domaine d'entraînement, "
     "l'extrapolation devient arbitraire. C'est le contre-exemple qui justifie la "
     "régularisation de toutes les autres variantes linéaires."),
    ("SVM (RBF, Nystroem)", "Noyau / distance",
     "Nystroem(gamma=0.01, n_components=300) + Ridge(alpha=1.0)",
     "Approximation de faible rang du noyau gaussien (méthode de Nyström) suivie d'une "
     "régression Ridge : équivalent approché d'une SVM à noyau RBF, en coût linéaire.",
     "Accuracy de test honorable (0,8114) et surtout l'un des plus faibles écarts-types en CV "
     "(0,1587). C'est la seule famille dont le α optimal en CV n'est pas nul (0,05) — donc la "
     "seule qui, sans porte, apporte quelque chose, même infime."),
    ("LightGBM", "Arbres — boosting",
     "LGBMRegressor(objective='l2', n_estimators=400, learning_rate=0.05, num_leaves=31)",
     "Boosting par croissance en profondeur d'abord (leaf-wise), avec regroupement des "
     "variables en histogrammes. Très rapide.",
     "R² de test 0,2937 avec la perte quadratique par défaut. C'est pourtant cette famille qui "
     "constitue le modèle final — mais avec une perte de Huber et une porte, configuration qui "
     "ne figure pas dans ce banc d'essai car elle n'est pas une « famille » au même sens."),
    ("XGBoost", "Arbres — boosting",
     "XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8)",
     "Boosting de gradient régularisé, croissance en largeur d'abord (level-wise).",
     "R² de test 0,2221 et R² en CV de −4,40 ± 7,89 : la plus instable des bibliothèques de "
     "boosting. C'était l'un des deux modèles du pipeline d'origine."),
    ("KNN (k=25)", "Noyau / distance",
     "KNeighborsRegressor(n_neighbors=25, weights='distance')",
     "Moyenne pondérée des 25 voisins les plus proches dans l'espace des variables standardisées.",
     "MEILLEURE ACCURACY EN CV DE TOUT LE BANC D'ESSAI (0,6419 ± 0,1565) alors que son R² de "
     "test n'est que de 0,2157. Un estimateur local prédit des écarts petits et prudents, ce "
     "que la bande ±0,01 récompense et que le R² ignore — la démonstration la plus nette que "
     "le choix de la métrique décide du classement."),
    ("MLP (128,64)", "Réseau de neurones",
     "MLPRegressor(hidden_layer_sizes=(128,64), max_iter=300, early_stopping=True)",
     "Perceptron multicouche à deux couches cachées, arrêt anticipé sur une fraction de "
     "validation interne.",
     "R² de test −0,3091 : NETTEMENT PIRE QUE LA PERSISTANCE. Le réseau apprend une fonction "
     "trop libre pour un signal dont l'amplitude utile est de l'ordre de 0,003, et le "
     "changement de régime entre blocs temporels le fait diverger."),
    ("SVM (RBF exact)", "Noyau / distance",
     "SVR(kernel='rbf', C=1.0, gamma='scale') — sous-échantillon de 6 000 lignes",
     "Machine à vecteurs de support à noyau gaussien exact. Le coût étant quadratique en "
     "nombre de lignes, l'entraînement a été fait sur un sous-échantillon de 6 000 lignes, "
     "limitation signalée dans le classement.",
     "PIRE FAMILLE DU BANC D'ESSAI : R² de test −0,7968, accuracy 0,2153, skill −1,03. Le "
     "sous-échantillonnage explique une part du résultat, mais l'ordre de grandeur de l'échec "
     "(prédictions deux fois moins bonnes que de ne rien faire) ne s'explique pas par cela seul."),
]


def main():
    doc = setup(Document())
    h(doc, "Comparaison détaillée des modèles de prédiction du Health Index", 0)
    para(doc, "14 familles de modèles évaluées sous un protocole d'évaluation unique et gelé",
         size=12, color=GREY)
    para(doc, "Projet i-SENSE / OCP Maintenance Solutions — UM6P   •   8 septembre 2026",
         size=9, color=GREY, space=12)

    # ------------------------------------------------------------- 1. cadre
    h(doc, "1. Ce qui est comparé, et dans quelles conditions", 1)
    para(doc, "Toutes les familles prédisent la même cible — le health_index à t+18 pas "
              "(environ 3 heures) — à partir des mêmes variables, sur les mêmes lignes, avec "
              "le même découpage. Les conditions sont strictement identiques d'une famille à "
              "l'autre :")
    table(doc, ["Élément du protocole", "Valeur commune à toutes les familles"],
          [["Cible", "health_index[t+18], valeur continue dans [0, 1]"],
           ["Formulation", "prédiction de l'écart δ = y[t+18] − y[t], puis reconstruction"],
           ["Variables", "70 colonnes après retrait des constantes, doublons et |r| > 0,995"],
           ["Imputation", "médiane, ajustée à l'intérieur de chaque pli"],
           ["Mise à l'échelle", "StandardScaler pour les familles linéaires, à noyau et le MLP"],
           ["Validation croisée", "5 blocs temporels expansifs, purge et embargo de 3 h"],
           ["Test gelé", "les 20 % de lignes les plus récentes par machine — 7 795 lignes"],
           ["Graine aléatoire", "42"],
           ["Facteur de rétrécissement", "α = 1 (correction brute, sans porte)"]],
          widths=[2.3, 4.6])
    para(doc, "")
    rich(doc, [("Pourquoi α = 1 pour le classement. ", True, None),
               ("Le facteur de rétrécissement optimal en validation croisée vaut 0 pour 13 "
                "familles sur 14 : à ce réglage, toutes se réduiraient exactement à la "
                "persistance et le classement serait plat. Les comparer à α = 1 est la seule "
                "façon de mesurer ce que chaque famille apporte réellement. Ce constat est "
                "développé en section 5.", False, None)])
    para(doc, "Naive Bayes ne figure pas dans ce banc d'essai : c'est un classifieur, sans "
              "équivalent de régression applicable à une cible continue. La persistance "
              "(« la valeur dans 3 h sera celle d'aujourd'hui ») sert de référence à battre "
              "et non de famille concurrente : elle atteint 0,8164 d'accuracy et 0,1142 de R² "
              "sur le test gelé.", size=9.5)

    # -------------------------------------------------------- 2. vue globale
    h(doc, "2. Vue d'ensemble", 1)
    d = lb.sort_values("test_r2", ascending=False)
    table(doc, ["Rang", "Famille", "Catégorie", "CV acc", "CV R²", "TEST acc", "TEST R²",
                "RMSE", "skill", "s"],
          [[i + 1, r["name"],
            {"Linear (OLS)": "Linéaire", "Ridge": "Linéaire", "Lasso": "Linéaire",
             "ElasticNet": "Linéaire", "Random Forest": "Bagging", "Extra Trees": "Bagging",
             "Gradient Boosting": "Boosting", "XGBoost": "Boosting", "LightGBM": "Boosting",
             "CatBoost": "Boosting", "SVM (RBF, Nystroem)": "Noyau",
             "SVM (RBF exact)": "Noyau", "KNN (k=25)": "Distance",
             "MLP (128,64)": "Réseau"}[r["name"]],
            d4(r["cv_acc_tol_mean"]), d4(r["cv_r2_mean"]), d4(r["test_acc_tol"]),
            d4(r["test_r2"]), d4(r["test_rmse"]), s4(r["test_skill_vs_persist"]),
            f"{r['seconds']:.0f}"]
           for i, (_, r) in enumerate(d.iterrows())],
          widths=[0.4, 1.5, 0.85, 0.65, 0.7, 0.65, 0.65, 0.65, 0.65, 0.4], size=7.5,
          highlight={0, 1, 2})
    para(doc, "")
    para(doc, f"Référence persistance : accuracy {d4(PERS_ACC)} · R² {d4(PERS_R2)} · "
              f"accuracy en CV {d4(PERS_CV)}.", italic=True, size=9, color=GREY)
    figure(doc, "10_familles_classement.png",
           "Figure 1 — Classement sur le test gelé selon deux métriques. L'ordre change : "
           "Lasso mène sur la bande de tolérance, ElasticNet sur le R².", width=7.0)

    rich(doc, [("Trois observations immédiates.", True, NAVY)])
    bullets(doc, [
        "Les modèles linéaires régularisés (ElasticNet, Lasso, Ridge) occupent trois des "
        "quatre premières places au R². Sur ce problème, la régularisation vaut mieux que la "
        "capacité : l'écart à prédire est de faible amplitude et approximativement linéaire "
        "dans les variables de dynamique récente.",
        "Deux familles font moins bien que de ne rien prédire : MLP (R² −0,3091) et SVM à "
        "noyau exact (R² −0,7968, skill −1,03). Une puissance de modélisation non contrainte "
        "se retourne contre le modèle quand le régime change entre les blocs.",
        "Le temps de calcul n'achète rien : Gradient Boosting coûte 629 s pour un R² de "
        "0,3925, Ridge coûte 20 s pour 0,4010.",
    ])
    figure(doc, "13_cout_vs_performance.png",
           "Figure 2 — Coût de calcul contre performance. Aucune tendance : les familles les "
           "plus lentes ne sont pas les meilleures.", width=6.4)
    doc.add_page_break()

    # ----------------------------------------- 3. metrique = classement
    h(doc, "3. Le choix de la métrique change le classement", 1)
    para(doc, "Les deux métriques principales ne récompensent pas le même comportement. Le R² "
              "mesure la part de variance reconstruite : il valorise les modèles qui suivent "
              "les grandes excursions. L'accuracy dans une bande de ±0,01 compte les lignes "
              "où l'erreur est petite : elle valorise les modèles prudents, qui prédisent des "
              "écarts proches de zéro.")
    comp = [("KNN (k=25)", "meilleure accuracy en CV (0,6419) mais R² de test de seulement 0,2157"),
            ("Ridge", "R² de test élevé (0,4010) mais accuracy faible (0,7119)"),
            ("Lasso", "premier sur les deux — le seul cas d'accord")]
    table(doc, ["Famille", "Comportement contrasté"],
          [[a, b] for a, b in comp], widths=[1.8, 5.1])
    para(doc, "")
    rich(doc, [("Le cas de KNN est le plus instructif. ", True, None),
               ("Un estimateur local moyenne 25 voisins : ses prédictions sont naturellement "
                "petites et prudentes. La bande de tolérance le classe premier en validation "
                "croisée, le R² le classe douzième. Aucun des deux chiffres n'est faux — ils "
                "mesurent deux qualités différentes, et il faut choisir laquelle correspond à "
                "l'usage. Pour un système d'alarme, qui se déclenche sur un franchissement de "
                "seuil, c'est le R² et le skill qui comptent.", False, None)])
    figure(doc, "12_heatmap_metriques.png",
           "Figure 3 — Les neuf métriques de test pour les 14 familles. La couleur donne le "
           "rang relatif (vert = meilleur). Les lignes ne sont pas uniformes : aucune famille "
           "ne domine sur tous les critères.", width=6.8)
    doc.add_page_break()

    # ------------------------------------------- 4. CV contre test
    h(doc, "4. Validation croisée contre test gelé : un désaccord massif", 1)
    rich(doc, [("C'est le résultat le plus important de ce banc d'essai.", True, RED)])
    para(doc, "Le classement obtenu sur le test gelé n'est pas reproduit par la validation "
              "croisée, et les écarts-types en CV sont si larges que la colonne CV ne sépare "
              "pas les familles.")
    sd = lb["cv_acc_tol_std"]
    table(doc, ["Indicateur de la colonne « accuracy en CV »", "Valeur"],
          [["Étendue des moyennes (max − min)", d4(lb["cv_acc_tol_mean"].max()
                                                   - lb["cv_acc_tol_mean"].min())],
           ["Écart-type minimal", d4(sd.min())],
           ["Écart-type médian", d4(sd.median())],
           ["Écart-type maximal", d4(sd.max())]],
          widths=[3.6, 1.6], highlight={0, 2})
    para(doc, "")
    para(doc, "L'étendue des moyennes (0,53) est du même ordre que l'écart-type médian (0,28) : "
              "presque tous les écarts deux à deux entre familles tiennent à l'intérieur d'un "
              "écart-type. Il en découle une règle de lecture stricte : le classement relatif "
              "des familles se lit sur les colonnes de test, tandis que la colonne CV ne "
              "démontre qu'une chose — que toutes les familles sans porte sont instables d'un "
              "bloc temporel à l'autre.")
    figure(doc, "11_cv_vs_test_familles.png",
           "Figure 4 — À gauche : R² en CV contre R² de test, sans relation exploitable. "
           "À droite : accuracy en CV avec barres d'erreur, qui se recouvrent presque toutes.",
           width=7.0)
    rich(doc, [("Conséquence méthodologique. ", True, None),
               ("Une famille ne peut pas être choisie sur son score de test — ce serait une "
                "sélection sur le jeu de test, la faute que l'audit du projet a identifiée dans "
                "le pipeline d'origine. Et elle ne peut pas non plus être choisie sur la CV, "
                "puisque celle-ci ne discrimine pas. La conclusion honnête est qu'aucune "
                "famille brute n'est identifiable comme gagnante, et c'est ce qui a conduit à "
                "changer d'architecture plutôt que de famille.", False, None)])
    doc.add_page_break()

    # ------------------------------------------- 5. alpha
    h(doc, "5. Le constat qui a fait abandonner l'approche « choisir la meilleure famille »", 1)
    para(doc, "Pour chaque famille, la prédiction finale s'écrit hi[t] + α · δ̂, où α est un "
              "facteur de rétrécissement appliqué à l'écart prédit. α = 0 revient exactement à "
              "la persistance, α = 1 au modèle brut. Le balayage de α en validation croisée "
              "donne un résultat sans ambiguïté.")
    figure(doc, "14_courbes_alpha.png",
           "Figure 5 — Accuracy en CV en fonction du facteur de rétrécissement. Les courbes "
           "décroissent de façon monotone : plus on corrige, plus on dégrade.", width=6.6)
    rich(doc, [("α optimal = 0 pour 13 familles sur 14.", True, RED)])
    para(doc, "Le mécanisme est arithmétique. La persistance est déjà à l'intérieur de la "
              "bande ±0,01 sur environ 79 % des lignes. Une correction bruitée fait sortir de "
              "la bande une partie de ces 79 % plus vite qu'elle n'y fait rentrer une partie "
              "des 21 % restants. Les pertes robustes (erreur absolue, Huber) dégradent plus "
              "lentement que la perte quadratique — elles estiment une médiane conditionnelle, "
              "proche de zéro — mais aucune ne franchit la ligne de la persistance.")
    para(doc, "Seule la SVM à noyau approché (Nyström) obtient un α optimal non nul, à 0,05, "
              "pour un gain de l'ordre du bruit.")
    rich(doc, [("D'où la solution retenue : une porte. ", True, GREEN),
               ("Plutôt que de rétrécir uniformément la correction, on entraîne un second "
                "modèle chargé de décider SI l'indice va bouger de plus que la tolérance, et "
                "l'on n'applique la correction que lorsqu'il se déclenche. C'est la première "
                "configuration du projet à battre la persistance en validation croisée.",
                False, None)])
    table(doc, ["Configuration", "CV acc", "CV écart-type", "TEST acc", "TEST R²"],
          [["Persistance", d4(PERS_CV), "0,0391", d4(PERS_ACC), d4(PERS_R2)],
           ["Meilleure famille brute (ElasticNet)", d4(row("ElasticNet")["cv_acc_tol_mean"]),
            d4(row("ElasticNet")["cv_acc_tol_std"]), d4(row("ElasticNet")["test_acc_tol"]),
            d4(row("ElasticNet")["test_r2"])],
           ["Modèle à porte (retenu)", "0,8094", "0,0370", "0,8259", "0,1975"]],
          widths=[2.6, 1.1, 1.3, 1.0, 1.0], highlight={2})
    para(doc, "")
    para(doc, "L'écart-type en CV du modèle à porte (0,0370) est de quatre à neuf fois "
              "inférieur à celui de n'importe quelle famille brute. C'est ce gain de stabilité, "
              "et non un meilleur score de test, qui justifie le choix.", size=9.5)
    figure(doc, "15_familles_vs_porte.png",
           "Figure 6 — Les quatre meilleures familles face à l'architecture retenue. À gauche "
           "le critère de sélection (CV), à droite le test gelé.", width=7.0)
    doc.add_page_break()

    # ------------------------------------------- 6. par machine
    h(doc, "6. Comportement par machine", 1)
    para(doc, "Le classement global masque une asymétrie forte entre les deux motosoufflantes.")
    figure(doc, "16_familles_par_machine.png",
           "Figure 7 — Accuracy par machine et par famille, avec les niveaux de persistance. "
           "Aucune barre bleue n'atteint la ligne bleue.", width=6.8)
    d6 = lb.sort_values("test_r2", ascending=False).head(6)
    table(doc, ["Famille", "acc — A", "acc — B", "R² — A", "R² — B"],
          [[r["name"], d4(r["test_acc_A"]), d4(r["test_acc_B"]),
            d4(r["test_r2_A"]), d4(r["test_r2_B"])] for _, r in d6.iterrows()]
          + [["Persistance", "0,9258", "0,7264", "0,8841", "−0,2719"]],
          widths=[1.9, 1.1, 1.1, 1.1, 1.1], highlight={6})
    para(doc, "")
    bullets(doc, [
        "Sur la Motosoufflante A, AUCUNE famille ne bat la persistance en accuracy (0,9258). "
        "La machine tourne 93 % du temps et son indice est très stable : il n'y a presque rien "
        "à corriger, et toute correction dégrade.",
        "Sur la Motosoufflante B, la plupart des familles dépassent la persistance (0,7264). "
        "C'est là que se concentre tout le bénéfice mesurable.",
        "L'audit a montré depuis que cette asymétrie provenait en grande partie d'un artefact "
        "du viscosimètre de la Motosoufflante B à l'arrêt. Après masquage de ces valeurs, "
        "l'écart entre les deux machines se réduit fortement et le bénéfice global devient "
        "statistiquement significatif — voir le rapport d'audit complet, section III.4bis.",
    ])
    doc.add_page_break()

    # ------------------------------------------- 7. fiches
    h(doc, "7. Fiche détaillée par famille", 1)
    para(doc, "Les familles sont présentées dans l'ordre du classement par R² de test.",
         italic=True, size=9, color=GREY)
    for name, cat, hp, principe, resultat in FICHES:
        r = row(name)
        h(doc, f"{name}  —  {cat}", 2)
        para(doc, f"Configuration : {hp}", size=8.5, color=GREY, italic=True)
        table(doc, ["CV acc", "CV R²", "TEST acc", "TEST R²", "RMSE", "MAE", "skill", "α*", "s"],
              [[d4(r["cv_acc_tol_mean"]) + " ± " + d4(r["cv_acc_tol_std"], 3),
                d4(r["cv_r2_mean"]), d4(r["test_acc_tol"]), d4(r["test_r2"]),
                d4(r["test_rmse"]), d4(r["test_mae"]), s4(r["test_skill_vs_persist"]),
                d4(r["alpha_cv_best"], 2), f"{r['seconds']:.0f}"]],
              widths=[1.0, 0.7, 0.7, 0.7, 0.7, 0.7, 0.75, 0.5, 0.4], size=8)
        para(doc, "")
        rich(doc, [("Principe. ", True, None), (principe, False, None)], size=9.5)
        rich(doc, [("Résultat. ", True, None), (resultat, False, None)], size=9.5)
        para(doc, "", space=8)

    doc.add_page_break()
    # ------------------------------------------- 8. conclusion
    h(doc, "8. Conclusion", 1)
    bullets(doc, [
        "Sur le test gelé, les modèles linéaires régularisés dominent : ElasticNet obtient le "
        "meilleur R² (0,4136) et Lasso la meilleure accuracy (0,8381). La régularisation forte "
        "vaut mieux que la capacité sur un signal de faible amplitude.",
        "Ce classement n'est pas confirmé par la validation croisée, dont les écarts-types "
        "(0,12 à 0,34 pour une étendue de moyennes de 0,53) ne permettent de séparer aucune "
        "famille. Choisir une famille sur son score de test serait une sélection sur le jeu de "
        "test, faute que cet audit corrige par ailleurs.",
        "Le facteur de rétrécissement optimal en CV vaut zéro pour 13 familles sur 14 : sans "
        "mécanisme de porte, la validation croisée demande de NE PAS corriger la persistance.",
        "Extra Trees est la seule famille avec un R² et un skill positifs en validation "
        "croisée, mais elle n'est pas séparable du modèle finalement retenu (R² CV +0,2671 ± "
        "0,1452 contre +0,2554 ± 0,2475).",
        "Deux familles font moins bien que l'absence de prédiction : MLP et SVM à noyau exact.",
        "La conclusion opérationnelle n'est donc pas « telle famille est la meilleure » mais "
        "« aucune famille brute n'est identifiable comme gagnante », ce qui a conduit à changer "
        "d'architecture — correction sélective par porte — plutôt qu'à continuer de comparer "
        "des familles entre elles.",
    ])
    para(doc, "")
    para(doc, "Sources reproductibles : hi_forecast/run_04_leaderboard.py (banc d'essai), "
              "hi_forecast/run_03_shrink_loss.py (balayage de α), "
              "hi_forecast/leaderboard_familles.csv et livrable2_leaderboard.md (chiffres "
              "bruts), hi_forecast/make_figures_modeles.py (figures).",
         italic=True, size=8.5, color=GREY)

    return save_doc(doc, os.path.join(ROOT, "Rapport_Comparaison_Modeles_iSENSE.docx"))


if __name__ == "__main__":
    p = main()
    print(f"Document genere : {p}  ({os.path.getsize(p)/1e6:.2f} Mo)")
