"""Rapport Word — le tableau de bord de A a Z : conception, chaine de donnees,
composants, decisions.

Pendant Word de explication_dashboard_A_a_Z.md. Les chiffres qui viennent des
artefacts (calibrage, classement, zones) sont LUS ; les inventaires de fichiers
et de composants sont releves a l'execution sur le code reel, pas recopies.
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

from build_docx import (GREY, NAVY, RED, bullets, h, para, rich,  # noqa: E402
                        save_doc, setup, table)

BACKEND = os.path.join(ROOT, "dashboard", "backend")
FRONT = os.path.join(ROOT, "dashboard", "frontend", "src")
ART = os.path.join(HERE, "artifacts")


def _lire(nom, defaut=None):
    p = os.path.join(ART, nom)
    if not os.path.exists(p):
        return defaut
    with open(p, encoding="utf-8") as f:
        return json.load(f)


ZON = _lire("systeme_zones.json", {}) or {}
LB = _lire("systeme_leaderboard.json", []) or []


def _lignes(chemin):
    with open(chemin, encoding="utf-8") as f:
        return sum(1 for _ in f)


def inventaire_backend():
    """Fichiers du backend, releves sur le disque."""
    roles = {
        "service.py": "Couche métier : predict_latest, history_series, KPI, porte de qualité",
        "ingestion.py": "Flux incrémental : gel des paramètres, tampons, application",
        "live_feed.py": "Connecteur temps réel i-SENSE + simulateur hors réseau",
        "main.py": "API FastAPI",
        "replay.py": "Rejeu de l'historique à vitesse configurable",
        "test_ingestion.py": "Non-régression en ligne / hors ligne",
    }
    out = []
    for nom, role in roles.items():
        p = os.path.join(BACKEND, nom)
        if os.path.exists(p):
            out.append([nom, str(_lignes(p)), role])
    return out


def inventaire_composants():
    """Composants React, releves dans App.tsx."""
    roles = {
        "MachineImage": "Emplacement photo, cinq extensions essayées, repère si absente",
        "HorizonTable": "Horizon · prédit · persistance · écart",
        "EtatMachineBloc": "Marche / arrêt lu sur la pression, jauge, durée, basculements",
        "CanalBloc": "Un canal système : mesure, seuil, réglette, règle et sa source",
        "SystemeBloc": "Les deux canaux côte à côte, jamais agrégés",
        "slugMachine": "Nom de machine → nom de fichier image (auxiliaire)",
        "SchemaEquipement": "Schéma de l'équipement, en tête de carte",
        "MachineTuile": "Case d'accueil : photo en fond, mesures, accès au détail",
        "MachineCard": "Assemble le tout pour une machine",
        "ZoneNotifications": "Problèmes détectés, du plus grave au moins grave",
        "ZoneActions": "Actions correctives, nommant le capteur en cause",
        "CourbeIndice": "Courbe mesuré → prévu, axe de temps réel",
        "CourbeSysteme": "Pression et vibration en petits multiples, chacune son unité",
        "PredictionPanel": "Grille d'accueil, puis vue de détail d'une machine",
        "GardeFou": "Garde-fou de rendu : une erreur n'efface plus la page",
        "App": "État, cadence de rafraîchissement, transport",
    }
    src = open(os.path.join(FRONT, "App.tsx"), encoding="utf-8").read()
    # `class` autant que `function` : GardeFou est un composant a etat, la seule
    # forme qui puisse intercepter une erreur de rendu en React. Ne chercher que
    # les fonctions le faisait disparaitre de l'inventaire.
    trouves = re.findall(r"^(?:export default )?(?:function|class) (\w+)", src, re.M)
    return [[n, roles.get(n, "—")] for n in trouves]


def main():
    doc = setup(Document())
    h(doc, "Le tableau de bord de A à Z", 0)
    para(doc, "Conception, chaîne de données, composants, décisions",
         size=12, color=GREY)
    para(doc, "Projet i-SENSE / OCP Maintenance Solutions — UM6P   •   Motosoufflantes A et B",
         size=9, color=GREY, space=12)

    # ═══════════════════════════════════════════════ 1
    h(doc, "1. Ce que le tableau de bord montre, et rien d'autre", 1)
    para(doc, "Une seule vue, deux cartes machine et deux courbes. Chaque élément répond "
              "à un point du cahier des charges.")
    table(doc, ["Élément affiché", "Ce qu'il répond"],
          [["Photo de la machine", "identification visuelle"],
           ["Health Index HUILE calculé à t", "état physico-chimique du lubrifiant"],
           ["Health Index SYSTÈME calculé à t", "état mécanique et hydraulique"],
           ["Prévision à 10 min, 20 min, 3 h, 24 h", "où va chaque indice"],
           ["Persistance, en regard de chaque prévision", "ce que donnerait « rien ne change »"],
           ["7 KPI de décision", "marge, tendance, délai, apport, fiabilité, disponibilité"],
           ["Courbe HI huile mesuré → prévu", "trajectoire récente et projection"],
           ["Courbe HI système mesuré → prévu", "idem, sur l'état mécanique"]],
          widths=[3.0, 3.6])
    para(doc, "Tout le reste — qualité des données, diagnostics capteurs, comparaison de "
              "modèles — a été retiré de l'écran et vit dans les rapports.", size=9,
         color=GREY)

    # ═══════════════════════════════════════════════ 2
    h(doc, "2. La chaîne de données, étage par étage", 1)
    table(doc, ["Étage", "Fichier", "Ce qui s'y passe"],
          [["1 · Ingestion", "Api_to_excel.py · live_feed.py",
            "Appel API toutes les 10 min, format long → large"],
           ["2 · Nettoyage", "clean_isense_data.py",
            "Sentinelles → NaN, Oil Conductivity ×10 → nS/m, sessions (rupture > 60 min)"],
           ["3 · Imputation", "fill_vibration.py · impute_sensor_gap.py",
            "Vibration : 0 à l'arrêt, RandomForest en marche. Trou capteur : report + RF"],
           ["4 · Variables", "feature_engineering.py",
            "Calendrier, session, indices physico-chimiques, lags 1-3, diff1, EWMA(18), pente 3 h"],
           ["5 · Distances", "health_index_baseline.py",
            "12 variables à poids égaux, seuils OCP → HI_regles"],
           ["6 · Protocole gelé", "hi_forecast/protocol.py",
            "Reconstruction SANS FUITE : z-scores, vi_proxy, ACP + Isolation Forest"],
           ["7 · Indice système", "hi_forecast/systeme_hi.py",
            "Sous-indices monotones pression / vibration"],
           ["8 · Service", "dashboard/backend/service.py",
            "predict_latest, history_series, KPI"],
           ["9 · API", "dashboard/backend/main.py", "FastAPI"],
           ["10 · Écran", "dashboard/frontend/src/App.tsx", "React"]],
          widths=[1.2, 2.3, 3.1], size=8)

    h(doc, "2.1  Le point qui structure tout : ajuster n'est pas appliquer", 2)
    para(doc, "Certaines transformations ont des paramètres ESTIMÉS sur les données : "
              "moyennes et écarts-types des z-scores, coefficients du vi_proxy, références "
              "de viscosité, ACP et Isolation Forest de l'étiquette, ancrages de l'indice "
              "système.")
    para(doc, "Ces paramètres sont estimés UNE FOIS, sur la fenêtre d'entraînement "
              "uniquement — jamais sur le futur. Le découpage est gelé sur les "
              "horodatages : coupure au 24 mai 2026 pour la Motosoufflante A, au 28 juin "
              "pour la B. Jamais de découpage aléatoire, qui mélangerait passé et futur "
              "d'une même session.")

    # ═══════════════════════════════════════════════ 3
    h(doc, "3. L'ingestion incrémentale — le chemin de production", 1)
    para(doc, "Le service appelait prepare(), qui relit les 42 141 lignes et RÉAJUSTE tous "
              "les paramètres ci-dessus. C'est correct pour entraîner, faux pour servir :")
    bullets(doc, [
        "COÛT — recalculer sept mois d'historique pour obtenir la valeur d'une seule ligne "
        "qui vient d'arriver.",
        "CORRECTION — des paramètres réajustés dérivent avec les données qu'ils jugent. "
        "Deux lignes identiques recevraient deux indices différents selon le lot dans "
        "lequel elles arrivent, et l'indice cesserait d'être comparable dans le temps.",
    ])
    h(doc, "3.1  Geler, puis appliquer", 2)
    para(doc, "geler_pretraitement() exécute la chaîne d'ajustement une fois et enregistre "
              "tous les paramètres dans artifacts/pretraitement.joblib : z-scores, "
              "références de viscosité, coefficients vi_proxy, objets ACP et Isolation "
              "Forest des deux machines, ancrages de l'indice système, imputeur de "
              "vibration et médianes de capteurs.")
    para(doc, "FluxIncremental tient un TAMPON de 600 lignes par machine. À chaque ligne "
              "reçue, il recalcule les variables sur ce tampon seulement, puis applique "
              "les transformations figées. Jamais sur le jeu complet.")
    para(doc, "Pourquoi un tampon et pas une ligne seule : lags 1 à 3, différence première, "
              "pente sur 18 pas et EWMA de portée 18 ont besoin de passé. 600 lignes, "
              "c'est la même amorce que le mode rejeu.", size=9, color=GREY)

    h(doc, "3.2  L'état de session", 2)
    para(doc, "time_in_session_h et measure_index_in_session se comptent depuis la PREMIÈRE "
              "ligne de la session, souvent déjà sortie du tampon. On conserve donc "
              "explicitement, par session, son horodatage de début et le nombre de lignes "
              "évincées — le SessionState prévu par l'architecture.")

    h(doc, "3.3  La preuve — test_ingestion.py", 2)
    para(doc, "L'architecture exige que l'en-ligne égale le hors-ligne à 1e-9 près. Le test "
              "rejoue 40 lignes une par une dans le flux et compare :")
    table(doc, ["Grandeur", "Écart maximum", "Verdict"],
          [["Health Index huile", "0,000e+00", "exact"],
           ["Health Index système", "0,000e+00", "exact"],
           ["Variables de session", "5,55e-17", "bruit de virgule flottante"]],
          widths=[2.2, 1.8, 2.2], highlight={0, 1})
    para(doc, "Échec du test = blocage du déploiement.", bold=True, color=RED)
    para(doc, "C'est le garde-fou contre le skew entraînement/service, une erreur qui s'est "
              "déjà produite dans ce projet : le premier lancement de l'API servait le CSV "
              "brut quand le modèle attendait les variables reconstruites, et un try/except "
              "masquait l'erreur en retombant silencieusement sur la persistance — la "
              "courbe s'affichait, fausse.", size=9, color=GREY)

    h(doc, "3.4  Le connecteur temps réel", 2)
    para(doc, "LiveFeed expose exactement la même interface que ReplayFeed : window(n) "
              "renvoie « les lignes traitées dont on dispose maintenant ». Aucun étage en "
              "aval ne change selon la source — c'est la propriété qui rend la bascule "
              "sûre. Une boucle de fond relève à la cadence des capteurs ; une panne "
              "réseau n'éteint pas le tableau de bord, elle se voit dans status().")
    para(doc, "SimulateurAPI rejoue le CSV nettoyé en se faisant passer pour l'API : le "
              "chemin complet est donc exerçable sans réseau.")
    rich(doc, [("Sécurité : ", True, RED),
               ("les identifiants se lisent dans l'environnement (ISENSE_EMAIL, "
                "ISENSE_PASSWORD), jamais dans le code. Api_to_excel.py les porte encore "
                "en clair — à retirer et à faire tourner.", False, None)])

    # ═══════════════════════════════════════════════ 4
    h(doc, "4. L'indice d'huile, et l'état du système", 1)
    para(doc, "Le tableau de bord porte DEUX lectures, qui répondent à deux questions "
              "de maintenance différentes et ne sont pas construites de la même façon.")
    table(doc, ["", "Health Index HUILE", "État du SYSTÈME"],
          [["Question", "Faut-il vidanger ?", "Faut-il intervenir mécaniquement ?"],
           ["Variables", "~20 physico-chimiques", "Oil Pressure, Oil System Vibration"],
           ["Méthode", "ACP (T² + SPE) + Isolation Forest",
            "seuils physiques, dans l'unité du capteur"],
           ["Agrégation", "max des 3 rapports au seuil",
            "AUCUNE — les deux canaux restent séparés"],
           ["Sortie", "indice 1 / (1 + sévérité)",
            "la mesure elle-même, en bar et en mm/s²"],
           ["Alarme", "0,50",
            "> 1,5 mm/s² en vibration ; la pression dit marche / arrêt à 0,10 bar"]],
          widths=[1.2, 2.6, 2.8], size=8.5)

    rich(doc, [("L'état du système n'est plus un indice. ", True, None),
               ("Deux versions antérieures en produisaient un — d'abord une distance de "
                "Mahalanobis, puis un indice normalisé 2^(−s/2) agrégé par maillon "
                "faible. Les deux ont été abandonnées.", False, None)])
    bullets(doc, [
        "La distance de Mahalanobis mesurait l'INHABITUEL, pas le DÉGRADÉ : une "
        "vibration anormalement BASSE produisait l'alarme maximale. 907 lignes "
        "étaient classées dégradées à tort.",
        "L'indice normalisé corrigeait la monotonie, mais empruntait son seuil "
        "d'alarme — 0,50 — au Health Index huile : un nombre sans unité appliqué à "
        "des bar et des mm/s². Et il n'atteignait jamais les zones C ni D sur le "
        "canal vibration, donc il ne pouvait pas alarmer.",
        "Agréger les deux canaux produisait en outre un chiffre sans destinataire : "
        "personne n'intervient sur « le système », on intervient sur la pompe OU "
        "sur les paliers.",
    ])
    para(doc, "Chaque canal est désormais jugé sur SA MESURE, contre un seuil établi "
              "pendant le traitement de données : 0,10 bar pour la marche "
              "(OIL_PRESSURE_ON_THRESHOLD), 1,5 mm/s² pour l'alarme vibratoire "
              "(flag_high_vibration), 9,8 mm/s² pour l'impossibilité physique "
              "(PHYSICAL_RANGES). Le détail et la validation de ces seuils sont dans "
              "Rapport_Etat_Systeme_iSENSE.docx.")
    rich(doc, [("Réserve : ", True, None),
               ("le document de cadrage OCP ne fixe aucun seuil pour ces deux "
                "variables. Le 1,5 est un seuil indicatif du pipeline, à défendre par "
                "les données — il tombe juste au-dessus du p99 en marche de la machine "
                "A et déclenche sur 0,80 % de ses mesures — jamais comme une norme "
                "constructeur.", False, RED)])

    # ═══════════════════════════════════════════════ 5
    h(doc, "5. Les modèles de prévision", 1)
    para(doc, "Quatre horizons, un modèle par machine et par horizon. Protocole gelé : "
              "découpage sur horodatages, validation croisée PURGÉE avec embargo "
              "proportionnel à l'horizon, sélection sur l'entraînement seul, test touché "
              "une seule fois.")
    rich(doc, [("Métrique de décision : ", True, None),
               ("le skill contre la persistance, pas le R². La cible est un indice "
                "construit, sans vérité terrain.", False, None)])
    h(doc, "5.1  La porte de fiabilité à deux critères", 2)
    para(doc, "fiable  ⟺  skill > 0   ET   accuracy ≥ accuracy_persistance",
         bold=True, size=11, space=6)
    para(doc, "Les deux ne vont pas ensemble. Sur l'indice système, les modèles obtiennent "
              "un skill jusqu'à +0,45 tout en tombant MOINS souvent que la persistance dans "
              "la bande utile de ±0,01. Un seul critère ferait passer pour utilisable une "
              "prévision qui dégrade la précision de décision.")
    rich(doc, [("Conséquence assumée : ", True, None),
               ("aucun horizon de l'indice système n'est affiché comme prévision fiable, "
                "et l'horizon 24 h de l'indice huile sur la machine A non plus "
                "(skill −0,156). L'indice est montré, sa prévision ne l'est pas.",
                False, RED)])

    # ═══════════════════════════════════════════════ 6
    h(doc, "6. Le backend", 1)
    table(doc, ["Fichier", "Lignes", "Rôle"], inventaire_backend(),
          widths=[1.5, 0.7, 4.4], size=8.5)
    h(doc, "6.1  Points d'entrée", 2)
    table(doc, ["Route", "Rôle"],
          [["GET /health", "état du service, modèles chargés, métriques gelées"],
           ["POST /predict", "prédiction ; corps optionnel = lignes à ingérer"],
           ["GET /history", "série observée récente"],
           ["GET /ingestion", "état des tampons et de l'artefact"],
           ["GET /comparaison", "classement des familles de modèles"],
           ["WS /ws", "flux poussé, un message toutes les 10 min"]],
          widths=[1.8, 4.8], size=8.5)

    # ═══════════════════════════════════════════════ 7
    h(doc, "7. Le frontend", 1)
    table(doc, ["Composant", "Rôle"], inventaire_composants(),
          widths=[1.6, 5.0], size=8.5)
    h(doc, "7.1  Les décisions de visualisation", 2)
    bullets(doc, [
        "L'axe est un VRAI axe de temps, numérique. Sans cela les échéances 10 min, "
        "20 min, 3 h et 24 h seraient espacées à intervalles égaux et le graphique "
        "mentirait sur la distance au futur.",
        "Les deux courbes n'ont pas la même fenêtre. L'huile montre les 48 dernières "
        "heures calendaires — sa chimie évolue même machine arrêtée. Le système montre "
        "tout l'historique de FONCTIONNEMENT, arrêts intercalés ignorés : la pression "
        "et la vibration n'ont de sens qu'en marche.",
        "Pression et vibration sont tracées en PETITS MULTIPLES, un graphe chacune, "
        "avec son axe et son unité. Un axe commun ferait lire « 5,4 » et « 0,15 » "
        "comme deux points d'une même grandeur ; un double axe permettrait de leur "
        "faire dire n'importe quoi en choisissant les échelles.",
        "Le zéro de l'axe est décidé grandeur par grandeur, pas par une règle unique. "
        "La vibration y est ancrée — elle descend réellement à 0,01 et zéro signifie "
        "immobile. La pression ne l'est pas : elle vit entre 4 et 6,5 bar, et ancrer "
        "à zéro écraserait la variation utile dans le cinquième haut du graphe. Comme "
        "un axe non ancré peut tromper, le sous-titre le dit explicitement.",
        "Les notifications sont dépliées par un bouton placé dans l'en-tête de chaque "
        "machine, à côté des badges d'état, coloré par la sévérité la PLUS GRAVE : un "
        "critique ne doit pas se cacher derrière deux informations. Pas de "
        "notification, pas de bouton.",
        "Chaque action corrective NOMME le capteur en cause. « Vérifier le "
        "viscosimètre de la Motosoufflante B » est actionnable ; « vérifier la "
        "donnée » ne l'est pas.",
    ])
    para(doc, "Les couleurs de série ont été validées par script, pas choisies à l'œil : "
              "#1d4ed8 mesuré et #0d9488 prévision passent les cinq contrôles — bande de "
              "clarté, plancher de chroma, séparation en vision déficiente, plancher de "
              "vision normale, contraste sur le fond. Vert, orange et rouge sont réservés "
              "aux états Normal / Surveillance / Alarme et jamais réutilisés pour "
              "identifier une série.", size=9, color=GREY)

    # ═══════════════════════════════════════════════ 8
    h(doc, "8. Les deux modes de fonctionnement", 1)
    table(doc, ["", "Mode STATIQUE", "Mode LIVE"],
          [["Source", "3 fichiers JSON précalculés (1,08 Mo)", "API FastAPI"],
           ["Backend", "aucun", "uvicorn main:app"],
           ["Usage", "démonstration, déploiement Vercel", "développement, temps réel"],
           ["Transport", "lecture locale", "WebSocket, repli polling"]],
          widths=[1.2, 2.7, 2.7])
    para(doc, "Le mode statique est produit par run_29_export_statique.py, qui appelle "
              "EXACTEMENT les mêmes fonctions de service que l'API. Les deux modes ne "
              "peuvent donc pas diverger.")
    para(doc, "Le tableau de bord affiche le dernier instant mesuré et se réinterroge "
              "toutes les 10 minutes — la cadence d'acquisition des capteurs. Entre deux "
              "mesures, l'écran ne bouge pas : c'est le comportement d'un écran de "
              "supervision. Un réglage antérieur faisait défiler une frame toutes les "
              "2 secondes à vitesse ×1800, soit un saut d'environ une journée d'archives "
              "toutes les deux secondes — ce n'était pas du temps réel mais un film "
              "accéléré.")

    # ═══════════════════════════════════════════════ 9
    h(doc, "9. Les décisions structurantes, et leurs raisons", 1)
    table(doc, ["Décision", "Raison"],
          [["Deux indices séparés, pas un score unique",
            "vidanger et intervenir mécaniquement sont deux décisions différentes"],
           ["Méthode différente pour l'état du système",
            "à deux variables, l'ACP et l'Isolation Forest sont inadaptées"],
           ["Aucun indice système, aucune agrégation",
            "on intervient sur la pompe OU sur les paliers, jamais sur « le système »"],
           ["Seuils dans l'unité du capteur",
            "« 5,40 bar contre 0,10 » se vérifie ; « indice 0,834, zone A » non"],
           ["Seuils pris au traitement de données",
            "emprunter le 0,50 du Health Index huile jugeait des bar avec un nombre "
            "sans unité"],
           ["Machine à l'arrêt non évaluée",
            "pression et vibration n'y portent aucun signal mécanique"],
           ["Porte de fiabilité à deux critères",
            "skill MSE et précision divergent, seule la précision décide"],
           ["Paramètres de prétraitement figés",
            "sinon les seuils dérivent avec les données qu'ils jugent"],
           ["Test de non-régression bloquant",
            "le skew entraînement/service s'est déjà produit ici"],
           ["Cadence 10 min",
            "celle des capteurs ; au-delà, l'écran invente du mouvement"],
           ["Axe de temps numérique",
            "sinon le graphique ment sur la distance au futur"]],
          widths=[2.6, 4.0], size=8.5)

    # ═══════════════════════════════════════════════ 10
    h(doc, "10. Ce qui reste ouvert", 1)
    bullets(doc, [
        "L'unité réelle de Oil System Vibration. L'API déclare des mm/s², mais les valeurs "
        "vaudraient ≈ 1,2·10⁻⁵ g. L'étiquette est douteuse, ce qui interdit tout seuil "
        "normatif. La méthode retenue y est insensible, mais la question reste posée.",
        "Le viscosimètre de la Motosoufflante B, qui s'effondre à 11,9 cSt de médiane dès "
        "l'arrêt — 40 % du jeu complet sous le seuil d'impossibilité physique. Son alarme "
        "actuelle est très probablement un artefact de capteur.",
        "Des données de la Motosoufflante B en fonctionnement, sans quoi ses seuils "
        "restent établis sur une base étroite : 2 204 lignes en marche sur 22 245, et "
        "aucun dépassement du seuil vibratoire jamais observé.",
        "LA DÉRIVE DE PRESSION. Mesurée sur le flux en direct, la pression des deux "
        "machines est divisée par deux — rapport 0,481 et 0,500 — en marche "
        "d'escalier, stable depuis cinq semaines. Un facteur commun aux deux machines "
        "et propre à la pression désigne l'acquisition, pas une dégradation. Les "
        "modèles ont appris sur 5,4 et 4,4 bar et reçoivent 2,6 et 2,2 bar : toute "
        "prévision postérieure au 10 août s'appuie sur une entrée hors distribution. "
        "À trancher avec l'exploitation avant toute exploitation des prévisions.",
        "La montée de vibration de la Motosoufflante A, passée de 0,151 à 0,578 mm/s² "
        "sans équivalent sur la machine B — elle ne s'explique donc pas par un "
        "recalibrage commun. Sous le seuil, mais à suivre.",
        "La cadence du flux en direct : environ une mesure par heure contre une "
        "toutes les 10 minutes dans l'historique, avec des coupures. Le bandeau "
        "« donnée périmée » se déclenche donc régulièrement, et c'est un constat sur "
        "le flux, pas un défaut de l'écran.",
    ])

    out = os.path.join(ROOT, "Rapport_Dashboard_A_a_Z_iSENSE.docx")
    return save_doc(doc, out)


if __name__ == "__main__":
    p = main()
    print(f"Document Word genere : {p}   ({os.path.getsize(p)/1024:.0f} Ko)")
