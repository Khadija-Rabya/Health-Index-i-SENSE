"""Rapport Word — construction du tableau de bord, etape par etape.

Recit chronologique : pour chaque etape, le probleme rencontre, ce qui a ete
fait, et le resultat mesure. Complementaire du rapport A a Z, qui decrit l'etat
final ; celui-ci decrit le CHEMIN, y compris les impasses — elles se defendent
mieux qu'elles ne se cachent.

Les chiffres viennent des artefacts quand ils y sont ; les inventaires de
fichiers sont releves sur le code a l'execution.
"""
import json
import os
import re
import sys
import warnings

from docx import Document

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, "C:\\pylib")
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

from build_docx import (GREEN, GREY, NAVY, RED, bullets, h, para, rich,  # noqa: E402
                        save_doc, setup, table)

ART = os.path.join(HERE, "artifacts")
BACKEND = os.path.join(ROOT, "dashboard", "backend")
FRONT = os.path.join(ROOT, "dashboard", "frontend", "src")


def _lire(nom, defaut=None):
    p = os.path.join(ART, nom)
    if not os.path.exists(p):
        return defaut
    with open(p, encoding="utf-8") as f:
        return json.load(f)


ZON = _lire("systeme_zones.json", {}) or {}
LB = _lire("systeme_leaderboard.json", []) or []
CAL = _lire("systeme_calibrage.json", {}) or {}


def etape(doc, numero, titre, probleme, fait, resultat, couleur=None):
    """Bloc uniforme : ce qui n'allait pas, ce qu'on a fait, ce que ca a donne."""
    h(doc, f"Étape {numero} — {titre}", 1)
    rich(doc, [("Le problème. ", True, RED), (probleme, False, None)])
    rich(doc, [("Ce qui a été fait. ", True, NAVY), (fait, False, None)])
    rich(doc, [("Le résultat. ", True, GREEN), (resultat, False, couleur)])


def main():
    doc = setup(Document())
    h(doc, "Construction du tableau de bord", 0)
    para(doc, "Étape par étape — problème, action, résultat",
         size=12, color=GREY)
    para(doc, "Projet i-SENSE / OCP Maintenance Solutions — UM6P   •   Motosoufflantes A et B",
         size=9, color=GREY, space=12)

    para(doc, "Ce document décrit le CHEMIN parcouru, impasses comprises. Deux méthodes "
              "ont été construites puis abandonnées pour l'indice système : elles sont "
              "décrites ici avec la raison de leur abandon. Une démarche qui montre ses "
              "essais infructueux se défend mieux qu'une démarche qui les dissimule.",
         italic=True, size=9.5, color=GREY, space=10)

    # ════════════════════════════════════════════════════════ 0
    h(doc, "Point de départ", 1)
    para(doc, "Un tableau de bord existait : trois onglets (prévision, qualité des "
              "données, diagnostics), un seul Health Index — celui de l'huile — et trois "
              "horizons de prévision (20 min, 3 h, 24 h). Les données étaient rejouées "
              "depuis un export figé.")

    # ════════════════════════════════════════════════════════ 1
    etape(doc, 1, "Faire démarrer l'application",
          "Le backend refusait de se lancer : « ModuleNotFoundError: No module named "
          "'fastapi' », puis « Could not import module main ».",
          "Diagnostic de l'environnement : le venv C:\\isense_venv ne contient ni "
          "fastapi, ni lightgbm, ni shap — les dépendances sont dans le Python par "
          "défaut. Et uvicorn doit être lancé depuis dashboard/backend.",
          "Application fonctionnelle. Règle retenue : ne jamais activer ce venv pour le "
          "tableau de bord.")

    # ════════════════════════════════════════════════════════ 2
    etape(doc, 2, "Séparer l'état de l'huile de l'état de la machine",
          "Un seul indice mélangeait deux questions de maintenance distinctes : "
          "« faut-il vidanger ? » et « faut-il intervenir mécaniquement ? ».",
          "Construction d'un SECOND indice, sur les deux seules variables qui décrivent "
          "la machine et non son lubrifiant : Oil Pressure et Oil System Vibration. "
          "La distinction était déjà posée dans une note du projet, laissée à "
          "implémenter.",
          f"Deux indices indépendants. Corrélation mesurée : "
          f"{('%.2f' % ZON.get('correlation_huile_systeme', 0)).replace('.', ',')} — "
          "assez pour rester cohérents, assez peu pour porter chacun une information "
          "propre.")

    # ════════════════════════════════════════════════════════ 3
    h(doc, "Étape 3 — Trois méthodes pour l'indice système, deux abandonnées", 1)
    para(doc, "C'est l'étape la plus instructive du projet. Trois constructions "
              "successives, deux écartées pour des raisons de fond.")

    h(doc, "3.1  Première tentative — zones ISO 20816-3 recalibrées", 2)
    rich(doc, [("Abandonnée. ", True, RED),
               ("La norme définit ses zones sur la VITESSE de vibration, en mm/s. L'API "
                "i-SENSE déclare des mm/s² — une accélération. La conversion exige la "
                "fréquence de rotation, absente des données. Et dans l'unité déclarée, "
                "la médiane observée vaut ≈ 1,2·10⁻⁵ g, physiquement impossible : "
                "l'étiquette d'unité est elle-même douteuse. Appliqués tels quels, les "
                "seuils normatifs classaient 100 % des relevés en zone A.", False, None)])

    h(doc, "3.2  Deuxième tentative — distance de Mahalanobis (T² de Hotelling)", 2)
    para(doc, "Méthode statistique standard, insensible à l'unité par construction. Elle "
              "semblait résoudre le problème précédent.")
    rich(doc, [("Abandonnée, et pour une raison plus grave. ", True, RED),
               ("Une distance est SYMÉTRIQUE : elle pénalise autant une machine qui "
                "vibre MOINS que la normale qu'une qui vibre plus.", False, None)])
    table(doc, ["Vibration", "Pression", "Indice rendu"],
          [["0,010 — soit 13 σ EN DESSOUS de la normale, machine exceptionnellement douce",
            "4,30 bar — parfaitement normale", "0,0000"]],
          widths=[3.2, 2.2, 1.2])
    para(doc, "Le pire score possible pour une machine qui allait très bien. Et 907 "
              "relevés étaient dégradés pour la seule raison d'être plus calmes que "
              "d'habitude.", bold=True)

    h(doc, "3.3  Méthode retenue — sous-indices monotones, maillon faible", 2)
    para(doc, "Principe directeur : UN INDICE DE SANTÉ DOIT ÊTRE MONOTONE. Il ne baisse "
              "que lorsque l'état empire — jamais parce qu'une grandeur s'écarte dans le "
              "bon sens.", bold=True)
    para(doc, "s = max( 0 , (x − x₅₀) / (x₉₅ − x₅₀) )      d = 2 ^ ( − s / 2 )      "
              "HI = min( d_vibration , d_pression )", bold=True, size=10.5, space=6)
    bullets(doc, [
        "Vibration : à SENS UNIQUE, plus haut = pire. Traitée en logarithme, son "
        "asymétrie passant de +2,05 à −0,29.",
        "Pression : BILATÉRALE — trop basse signale une fuite ou une usure de pompe, "
        "trop haute un colmatage. Laissée brute, elle est déjà symétrique.",
        "Agrégation par le MAILLON FAIBLE : une machine ne vaut pas mieux que son pire "
        "symptôme. Une moyenne laisserait une pression effondrée être compensée par une "
        "vibration parfaite.",
        "Le max(0, ·) est le cœur du correctif : être meilleur que la médiane saine ne "
        "rapporte rien, mais ne coûte rien.",
    ])
    rows = []
    for m, c in CAL.items():
        if "erreur" in c:
            continue
        ct = c.get("controle", {})
        rows.append([m, f"{ct.get('taux_alarme_population_saine', 0)*100:.2f} %".replace(".", ","),
                     f"{ct.get('pilote_vibration', 0)*100:.0f} %",
                     f"{ct.get('pilote_pression', 0)*100:.0f} %"])
    if rows:
        table(doc, ["Machine", "Faux positifs sur population saine",
                    "Limité par la vibration", "Limité par la pression"], rows,
              widths=[1.7, 2.0, 1.5, 1.4])
    para(doc, "Le taux de faux positifs est le contrôle qui dit si le seuil est "
              "utilisable : il mesure la proportion de relevés CONNUS SAINS que l'indice "
              "classerait en alarme. Il manquait aux deux méthodes précédentes.",
         size=9, color=GREY)

    # ════════════════════════════════════════════════════════ 4
    etape(doc, 4, "Prévoir l'indice système — et accepter un résultat négatif",
          "Aucun modèle n'existait pour prévoir l'état mécanique.",
          "Comparaison de SEPT familles (Ridge, Lasso + porte, forêt aléatoire, "
          "HistGradientBoosting, XGBoost, LightGBM Huber, CatBoost) aux quatre horizons, "
          "sous le protocole gelé : validation croisée purgée, sélection sur "
          "l'entraînement seul, test touché une seule fois.",
          "AUCUN modèle ne bat la persistance en précision, à aucun horizon. "
          "Le constat est robuste : il a été retrouvé à l'identique sur les TROIS "
          "définitions successives de l'indice.", RED)
    rows = []
    for hp, lab in [(1, "10 min"), (2, "20 min"), (18, "3 h"), (144, "24 h")]:
        ev = [r for r in LB if r["horizon"] == hp
              and r["machine"] == "Motosoufflante A" and r.get("evaluable")]
        if not ev:
            continue
        best = max(ev, key=lambda r: r.get("test_acc") or 0)
        rows.append([lab,
                     f"{best.get('acc_persist', 0):.4f}".replace(".", ","),
                     best["famille"].replace("_", " "),
                     f"{best.get('test_acc', 0):.4f}".replace(".", ",")])
    if rows:
        table(doc, ["Horizon", "Persistance", "Meilleur modèle", "Sa précision"], rows,
              widths=[1.0, 1.4, 2.4, 1.4])
    para(doc, "Trois familles (XGBoost, LightGBM, CatBoost) ont été ajoutées après coup : "
              "le pipeline de l'indice HUILE avait été évalué contre LightGBM et CatBoost, "
              "pas l'indice système. Les deux indices étaient donc jugés sur des jeux de "
              "familles différents — une incohérence, corrigée.", size=9, color=GREY)

    # ════════════════════════════════════════════════════════ 5
    etape(doc, 5, "Resserrer la porte de fiabilité",
          "Un horizon était déclaré fiable sur le seul skill (erreur quadratique). Or sur "
          "l'indice système, les modèles obtiennent un skill jusqu'à +0,45 tout en tombant "
          "MOINS souvent que la persistance dans la bande utile de ±0,01.",
          "Double critère : fiable ⟺ skill > 0 ET précision ≥ précision de la "
          "persistance. Le critère qui compte pour décider d'une intervention est la "
          "précision.",
          "Conséquence assumée : aucune prévision de l'indice système n'est affichée, et "
          "l'horizon 24 h de l'indice huile sur la machine A non plus. L'indice est "
          "montré, sa prévision ne l'est pas.")

    # ════════════════════════════════════════════════════════ 6
    etape(doc, 6, "Corriger la cadence d'affichage",
          "Les valeurs changeaient plusieurs fois par seconde. Trois réglages se "
          "cumulaient : rejeu à vitesse ×1800, une image toutes les 2 secondes, un "
          "redessin toutes les 500 ms — et chaque image distante de 1,23 jour de données. "
          "L'écran faisait défiler une journée d'archives toutes les deux secondes.",
          "Passage à la cadence réelle des capteurs : le tableau de bord affiche le "
          "dernier instant mesuré et se réinterroge toutes les 10 minutes. La barre de "
          "rejeu est supprimée.",
          "Comportement d'un écran de supervision : entre deux mesures, rien ne bouge.")

    # ════════════════════════════════════════════════════════ 7
    etape(doc, 7, "Refaire les courbes",
          "Les graphiques montraient sept mois d'archives, et la valeur prédite était "
          "tracée à l'instant où la prédiction avait été faite — donc dans le passé. "
          "Aucune projection vers le futur n'apparaissait.",
          "Fenêtre récente à gauche, prévision projetée à droite jusqu'à +24 h, repère "
          "vertical « maintenant », et un VRAI axe de temps numérique — sans quoi les "
          "échéances 10 min, 20 min, 3 h et 24 h seraient espacées à intervalles égaux et "
          "le graphique mentirait sur la distance au futur.",
          "Deux fenêtres distinctes : l'huile sur 48 h calendaires, le système sur tout "
          "l'historique de FONCTIONNEMENT — son indice n'existant qu'en marche, une "
          "fenêtre calendaire sur une machine arrêtée ne contiendrait aucun point.")

    # ════════════════════════════════════════════════════════ 8
    etape(doc, 8, "Passer du recalcul global à l'inférence incrémentale",
          "Le service appelait prepare(), qui relit les 42 141 lignes et RÉAJUSTE tous "
          "les paramètres : z-scores, coefficients, ACP, Isolation Forest, ancrages. "
          "Coûteux, et surtout FAUX : des paramètres réajustés dérivent avec les données "
          "qu'ils jugent, et deux lignes identiques recevraient deux indices différents "
          "selon le lot dans lequel elles arrivent.",
          "Gel de tous les paramètres dans un artefact unique, puis application à un "
          "TAMPON de 600 lignes par machine. Conservation explicite de l'état de session "
          "(horodatage de début, lignes évincées) : les variables temporelles se comptent "
          "depuis la première ligne de la session, souvent déjà sortie du tampon.",
          "Test de non-régression en ligne contre hors ligne : écart maximum "
          "0,000e+00 sur les deux indices. Échec du test = blocage du déploiement.")

    # ════════════════════════════════════════════════════════ 9
    etape(doc, 9, "Brancher l'ingestion temps réel",
          "Il n'existait aucun flux live : le tableau de bord rejouait un export figé au "
          "6 août.",
          "Écriture du connecteur LiveFeed, de même interface que le rejeu — aucun étage "
          "en aval ne change selon la source. Quatre bugs ont été découverts en le "
          "confrontant au vrai serveur : session requests manquante, fenêtre de relevé "
          "trop large provoquant un 504, horodatages ISO 8601 suffixés Z, et une colonne "
          "session_id créée en nombre puis remplie de texte.",
          "846 lignes relevées, 0 erreur, boucle active, dernière mesure au 14/09/2026. "
          "Le tableau de bord tourne sur des données réelles.")

    # ════════════════════════════════════════════════════════ 10
    h(doc, "Étape 10 — Ce que le système a détecté le premier jour", 1)
    para(doc, "Dès la mise en service, le tableau de bord a affiché « Alarme » sur la "
              "Motosoufflante A avec un Health Index huile de 0,0236 — une valeur qui "
              "n'apparaît nulle part dans sept mois d'historique.")
    para(doc, "La vérification a tranché :", space=6)
    table(doc, ["Mesure", "Valeur au 14/09", "Plage d'entraînement", "Verdict"],
          [["Viscosity at 40 °C", "18,88 cSt", "42,96 .. 47,95", "HORS PLAGE"],
           ["Kinematic Viscosity", "51,33 cSt", "28,04 .. 67,04", "normale"],
           ["Dynamic Viscosity", "43,86 cP", "23,71 .. 57,53", "normale"]],
          widths=[1.9, 1.4, 1.8, 1.4])
    para(doc, "À 40 °C, ces grandeurs décrivent la même chose. Un capteur qui annonce "
              "18,88 pendant que son voisin annonce 51,33 sur la même huile se contredit "
              "lui-même. De plus 18,88 est sous le seuil des 20 cSt que le pipeline juge "
              "physiquement impossible pour une ISO VG 46.", bold=True)
    rich(doc, [("Ce n'est pas une dégradation de l'huile, c'est un défaut de "
                "viscosimètre. ", True, RED),
               ("Et le système l'avait déjà dit : sa porte de qualité classait la ligne "
                "« rejetée », motif « viscosité impossible en marche ». Un pipeline naïf "
                "aurait déclenché une vidange sur un capteur cassé.", False, None)])

    # ════════════════════════════════════════════════════════ 11
    etape(doc, 11, "Épurer, puis réorganiser l'écran",
          "Le tableau de bord portait trois onglets et des panneaux d'analyse qui ne "
          "relevaient pas du suivi opérationnel.",
          "Retrait des onglets qualité et diagnostics, de la comparaison de modèles "
          "(déplacée dans les rapports) et des tuiles de KPI. Puis réorganisation en deux "
          "niveaux : une grille de deux cases avec la photo de la machine en fond, et une "
          "vue de détail au clic, ouverte par le schéma de l'équipement.",
          "Export statique réduit de 2,16 Mo à 1,08 Mo, trois fichiers au lieu de quinze.")

    # ════════════════════════════════════════════════════════ 12
    etape(doc, 12, "Sortir les identifiants du code",
          "L'adresse et le mot de passe de l'API i-SENSE étaient en clair dans "
          "Api_to_excel.py, fichier suivi en version.",
          "Lecture dans l'environnement (ISENSE_EMAIL, ISENSE_PASSWORD), avec repli sur "
          "une saisie masquée. La demande est différée et non faite au chargement du "
          "module : le connecteur importe ce fichier, une invite bloquante suspendrait le "
          "serveur.",
          "Le mot de passe RESTE présent dans l'historique du dépôt : le retirer du "
          "fichier ne le rend pas irrécupérable. Seule une rotation côté i-SENSE ferme "
          "la brèche — décision laissée à l'équipe.", RED)

    # ════════════════════════════════════════════════════════ bilan
    h(doc, "Ce que le tableau de bord montre aujourd'hui", 1)
    bullets(doc, [
        "Une grille de deux machines, photo en fond, état et barres par indice.",
        "Au clic : le schéma de l'équipement, les deux Health Index calculés, les quatre "
        "horizons avec persistance et écart, le bloc système avec sa zone et son canal "
        "limitant, et le bandeau de qualité quand la donnée est rejetée.",
        "Deux courbes : mesuré à gauche du repère « maintenant », prévision projetée à "
        "droite.",
        "Des données réelles, relevées toutes les 10 minutes depuis l'API i-SENSE.",
    ])

    h(doc, "Ce qui reste ouvert", 1)
    bullets(doc, [
        "L'unité réelle de Oil System Vibration, à confirmer auprès du constructeur.",
        "Le viscosimètre de la Motosoufflante A, en défaut depuis au moins le 12/09.",
        "Le viscosimètre de la Motosoufflante B, qui s'effondre à 11,9 cSt de médiane dès "
        "l'arrêt — 40 % du jeu complet sous le seuil d'impossibilité physique.",
        "Des données de la Motosoufflante B en fonctionnement : sa fenêtre de test ne "
        "contient aucune ligne en marche, son indice système reste invérifiable.",
        "Un réentraînement sur données récentes : les modèles sont ajustés au 6 août.",
    ])

    out = os.path.join(ROOT, "Rapport_Construction_Dashboard_iSENSE.docx")
    return save_doc(doc, out)


if __name__ == "__main__":
    p = main()
    print(f"Document Word genere : {p}   ({os.path.getsize(p)/1024:.0f} Ko)")
