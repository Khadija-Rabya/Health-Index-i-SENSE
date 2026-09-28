"""
Genere un document Word listant les points a fixer / questions a poser a
l'equipe i-SENSE, organises par interlocuteur. Mis a jour pour couvrir toute
la refonte physico-chimique du pipeline, l'imputation finale de la vibration,
le feature engineering complet, et la construction du Health Index
(Phases 1 a 5) - pas seulement l'etat initial du projet.

Sortie : Questions_Reunion_Equipe_iSENSE.docx
"""

from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH


def add_intro_line(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(10)
    r.font.color.rgb = RGBColor(0x40, 0x40, 0x40)
    p.paragraph_format.space_after = Pt(10)


def add_questions(doc, items):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(it)


def add_title_page(doc):
    title = doc.add_heading("Points à fixer et questions à poser à l'équipe", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title2 = doc.add_heading("Projet i-SENSE — Monitoring de l'huile de lubrification", level=1)
    title2.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = sub.add_run(
        "Mis à jour après la refonte physico-chimique du pipeline (Data Quality, nettoyage, "
        "imputation de la vibration par approche hybride, feature engineering complet) et la "
        "construction du Health Index (Phases 1 à 5) — organisé par interlocuteur : spécialiste "
        "métier, data scientist, responsable API i-SENSE, et questions transverses."
    )
    run.italic = True
    run.font.size = Pt(12)
    doc.add_page_break()


def main():
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    add_title_page(doc)

    # ------------------------------------------------------------------
    doc.add_heading("1. Questions pour le spécialiste métier (huile / machines)", level=1)
    add_intro_line(
        doc,
        "Objectif : valider les hypothèses physiques et opérationnelles utilisées dans le "
        "nettoyage, l'audit Data Quality, les seuils d'alerte, et le Health Index."
    )
    add_questions(doc, [
        "⚠️ [NOUVEAU — priorité haute] 88,4 % des mesures de constante diélectrique (DC) de "
        "Motosoufflante B sont physiquement impossibles (ε_r < 1, en dessous du vide), contre 0 % "
        "sur Motosoufflante A — probable défaut de calibration du capteur DC sur B. Faut-il "
        "corriger, recalibrer, ou exclure cette variable pour B ? Elle est actuellement exclue du "
        "Health Index en attendant votre confirmation.",
        "⚠️ [NOUVEAU] Motosoufflante B présente aussi un écart systématique de viscosité aux "
        "grades ISO VG nominaux (22/32/46/68/100/150) — bien plus marqué que sur A. Est-ce "
        "cohérent avec le type d'huile réellement utilisé sur B, ou faut-il l'investiguer comme "
        "une seconde anomalie capteur ?",
        "⚠️ [NOUVEAU] Sans historique de panne, nous avons approximé une période « saine » de "
        "référence par l'absence de tout dépassement de seuil (~97-98 % du dataset) pour "
        "entraîner les méthodes de détection d'anomalie du Health Index (PCA, Isolation Forest). "
        "Cette approximation vous semble-t-elle raisonnable, ou disposez-vous d'une période de "
        "référence plus fiable (ex. juste après une vidange connue) ?",
        "⚠️ [NOUVEAU] Le Health Index à base de règles métier pondère à parts égales (25 % "
        "chacun) la contamination ISO 4406, la teneur en eau, l'écart de viscosité au grade et "
        "l'écart de densité — quelle pondération recommanderiez-vous en fonction de la criticité "
        "réelle de chaque paramètre ?",
        "Les seuils d'alerte actuellement utilisés (contamination_index > 20, Oil H2O ppm > 100, "
        "Oil Temperature > 55°C, vibration > 1,5 mm/s²) restent des valeurs indicatives, tout "
        "comme les seuils Surveillance/Alarme du Health Index (calibrés statistiquement par "
        "percentile faute de mieux) — quels sont les seuils réels validés pour ces deux machines ?",
        "Quel est le grade d'huile réellement utilisé sur chaque machine (pour calibrer "
        "correctement l'écart au grade ISO VG) ?",
        "Machine A a une viscosité moyenne de ~46 cSt et Machine B de ~18 cSt (2,5x plus faible) — "
        "s'agit-il d'huiles différentes, d'un problème de dilution sur B, ou d'un artefact capteur ?",
        "Motosoufflante B est à l'arrêt la grande majorité du temps (90,1 % des mesures, pression "
        "quasi nulle) — est-ce un arrêt de maintenance planifié, une panne, ou un choix "
        "opérationnel ? Une reprise est-elle prévue ?",
        "Existe-t-il un historique de changement d'huile, de maintenance ou d'incidents connus sur "
        "ces deux machines pouvant servir de vérité terrain ? **C'est le point qui limite le plus "
        "la validation du Health Index actuel** : sans aucun événement de panne confirmé, nous ne "
        "pouvons le valider que par des tests indirects (cohérence avec les seuils, injection de "
        "défauts synthétiques, stabilité en fonctionnement stable — cf. rapport joint), jamais par "
        "une vraie dégradation observée.",
        "Le trou de données synchrone (994 mesures, 9 variables indisponibles simultanément) "
        "correspond-il à une opération connue (calibration, maintenance capteur, coupure réseau) ?",
        "Existe-t-il des cas historiques documentés de « changement d'huile requis » pouvant servir "
        "de référence pour entraîner ou valider une classification Bonne / Dégradée / À changer ?",
        "Le paramètre « particules d'usure » (un des 11 paramètres cibles du capteur IoT) est absent "
        "du dataset actuel — est-il prévu, disponible via un autre capteur, ou hors périmètre du "
        "projet ?",
        "Le module de vibration transmet-il en continu, ou uniquement sur déclenchement / plage "
        "horaire programmée ? Nous avons contourné cette incertitude par une décision métier "
        "directe (0 à l'arrêt machine, prédiction statistique en marche) plutôt que d'attendre la "
        "réponse — une confirmation permettrait de valider ou raffiner ce choix.",
        "⚠️ [NOUVEAU 2026-09-01] Le document « Cadrage des seuils_Système d'huile.pdf » calibre "
        "les seuils de densité et de viscosité sur une référence d'huile TD46 (Shell Turbo T46, "
        "ISO VG 46). Motosoufflante A colle raisonnablement à cette référence, mais Motosoufflante "
        "B en sort massivement (viscosité à 40°C observée jusqu'à 3 cSt, contre une plage normale "
        "43,7-48,3 cSt) — B utilise-t-elle une huile différente, ou est-ce la confirmation d'un "
        "problème de capteur/calibration déjà suspecté (cf. anomalie DC) ?",
        "⚠️ [NOUVEAU 2026-09-01] Les seuils officiels de `DC` indiquent qu'une valeur élevée "
        "(>2,6) est critique — mais les diapositives i-SENSE indiquent l'inverse (DC élevée = bon "
        "état, DC faible = dégradation). Ces deux documents se contredisent sur le sens de "
        "dégradation : lequel fait foi ? Nous avons retenu le tableau de seuils tel quel dans le "
        "Health Index, en attendant votre confirmation.",
    ])

    # ------------------------------------------------------------------
    doc.add_page_break()
    doc.add_heading("2. Questions pour l'équipe Data Science", level=1)
    add_intro_line(
        doc,
        "Objectif : challenger les choix méthodologiques faits pendant la refonte physico-chimique "
        "et la construction du Health Index, et cadrer les prochaines étapes."
    )
    add_questions(doc, [
        "⚠️ [NOUVEAU] La synthèse bibliographique reçue (autoencodeur en espace latent) mentionne "
        "« ~20 392 lignes ML-ready, 14 variables physiques » et « un pic H2O daté du 7 février "
        "2026 ». Ces chiffres ne correspondent à aucun résultat produit dans notre pipeline actuel "
        "(42 141 lignes, 147 colonnes de features, aucun pic H2O identifié à cette date dans nos "
        "analyses). Pouvez-vous préciser la source de ces chiffres avant qu'on ne les utilise pour "
        "dimensionner un futur modèle ?",
        "⚠️ [NOUVEAU] L'autoencodeur (Phase 4 du plan de construction du Health Index) n'a "
        "finalement pas été développé : la PCA linéaire (T²/SPE) et l'Isolation Forest ont passé "
        "avec de larges marges les 3 tests de validation prévus (cohérence avec les flags métier, "
        "injection de défauts synthétiques, stabilité en fonctionnement stable — cf. rapport "
        "joint). Confirmez-vous qu'il n'est pas prioritaire de le développer tant qu'aucune limite "
        "de la PCA n'est démontrée sur ces données ?",
        "⚠️ [NOUVEAU] Le remplacement de `Oil System Vibration` a été résolu par une approche "
        "hybride : 0 quand la machine est à l'arrêt (décision métier), prédiction par Random "
        "Forest quand elle tourne (meilleure méthode parmi 8 comparées : RF, XGBoost, KNN, "
        "interpolation, moyenne/médiane simple et conditionnelle — R² = 0,438). Ce niveau de "
        "performance vous semble-t-il suffisant pour un usage en aval (Health Index, futurs "
        "modèles), ou faudrait-il exiger un R² plus élevé avant utilisation ?",
        "Le dataset i-SENSE ne contient aucun événement de panne connu (contrairement à NASA "
        "C-MAPSS qui offre un run-to-failure complet) — comment définir une variable cible RUL ou "
        "un label de dégradation en l'absence de vérité terrain ? Sans cela, la trendabilité et la "
        "prognosabilité (au sens de Coble, 2010) du Health Index ne peuvent pas être calculées, "
        "seule la monotonicité serait envisageable.",
        "Le feature engineering est maintenant complet (147 colonnes, 114 features ajoutées : "
        "calendaire, session, indices physico-chimiques, tendance, lags, EWMA, indices métier, "
        "z-score par machine, flags de seuils, confiance vibration/capteur, encodage catégoriel). "
        "Une famille supplémentaire (statistiques en fenêtre glissante, 78 colonnes) a été retirée "
        "après un test d'importance de features l'ayant jugée non prioritaire pour le Health Index "
        "retenu. Voyez-vous une famille manquante utile pour un futur modèle supervisé (RUL, "
        "classification) ?",
        "Faut-il exclure entièrement les données en état OFF de Motosoufflante B de l'entraînement "
        "d'un futur modèle de dégradation, ou les conserver pour un usage différent (détection "
        "d'arrêt, suivi de disponibilité) ?",
        "42 141 lignes sur environ 8 mois pour 2 machines : ce volume vous semble-t-il suffisant "
        "pour un futur modèle plus complexe (deep learning), ou faut-il envisager d'attendre plus "
        "de données ?",
        "Le remplacement des sentinelles, le forward-fill, les modèles Random Forest utilisés pour "
        "l'imputation des capteurs, et la PCA/Isolation Forest pour le Health Index vous "
        "semblent-ils adaptés, ou recommanderiez-vous d'autres approches (imputation multiple, "
        "modèles bayésiens, autres méthodes de détection d'anomalie) ?",
    ])

    # ------------------------------------------------------------------
    doc.add_page_break()
    doc.add_heading("3. Questions pour le responsable API / infrastructure de données i-SENSE", level=1)
    add_intro_line(
        doc,
        "Objectif : comprendre l'origine des problèmes de qualité rencontrés côté capteurs/API et "
        "anticiper les évolutions de l'infrastructure de données."
    )
    add_questions(doc, [
        "Pourquoi certains capteurs (DC, Density, Dynamic Viscosity, Kinematic Viscosity, ISO 4, "
        "Oil H2O Saturation/ppm, Oil Temperature, Viscosity at 40°C) tombent-ils en panne "
        "exactement en même temps (994 mesures), alors que d'autres (ISO 6, ISO 14, "
        "Oil Conductivity, Oil Pressure) restent disponibles ? Partagent-ils un module matériel "
        "commun ?",
        "L'API actuellement utilisée (v3back-demo.i-sense.io) est-elle un environnement de "
        "démonstration ? Une API de production avec un SLA de disponibilité différent est-elle "
        "prévue, et sous quel délai ?",
        "Est-il possible d'obtenir des métadonnées complémentaires via l'API (journal de "
        "maintenance, alarmes déclenchées, dates de changement d'huile, changements de "
        "configuration capteur) pour enrichir le contexte des données déjà collectées ? C'est le "
        "point qui débloquerait le plus la validation du Health Index (cf. section 1).",
        "Existe-t-il une limite de rétention/historique sur l'API, ou peut-on remonter au-delà de "
        "la période actuellement récupérée ?",
        "Un accès en flux (streaming / webhook) est-il possible, ou seul un pull périodique par "
        "requête HTTP est-il supporté pour un futur déploiement en production ?",
        "`Oil System Vibration` n'est mesurée que ~34 % du temps — s'agit-il d'une limitation "
        "matérielle du capteur, d'une consigne d'échantillonnage volontaire, ou d'un "
        "dysfonctionnement à corriger côté i-SENSE ? Nous avons contourné ce point par un "
        "remplissage statistique (0 à l'arrêt, prédiction en marche) faute de réponse, mais la "
        "cause reste à clarifier pour juger si cette approche restera nécessaire à l'avenir.",
        "Les identifiants d'authentification actuellement utilisés (email/mot de passe en clair "
        "dans le script) sont-ils appropriés pour la suite du projet, ou faut-il migrer vers un "
        "mécanisme d'authentification plus robuste (clé API, OAuth) avant un déploiement réel ?",
        "⚠️ [NOUVEAU 2026-09-01] `Oil Conductivity` contient une sentinelle non standard "
        "(-0,09999, différente du -99,99 générique des autres colonnes) sur les 994 lignes du "
        "trou capteur synchrone déjà signalé — non détectée par notre pipeline jusqu'à "
        "présent. Pouvez-vous confirmer cette valeur sentinelle côté export API ?",
        "⚠️ [NOUVEAU 2026-09-01] `Oil Conductivity` semble être dans une unité différente de "
        "celle attendue (nS/m) : nos valeurs réelles se regroupent entre 0,062 et 0,072, hors de "
        "toute plage physique documentée. Nous avons déterminé par analyse qu'un facteur ×10 les "
        "ramène dans la plage normale attendue (0,4-2,0 nS/m) — pouvez-vous confirmer l'unité "
        "réelle de cette mesure et ce facteur ?",
    ])

    # ------------------------------------------------------------------
    doc.add_page_break()
    doc.add_heading("4. Questions transverses (toute l'équipe)", level=1)
    add_intro_line(
        doc,
        "Objectif : cadrer les attentes globales du projet avant d'investir dans un futur modèle "
        "de dégradation ou un déploiement en production du Health Index."
    )
    add_questions(doc, [
        "⚠️ [NOUVEAU] Les seuils Normal/Surveillance/Alarme du Health Index sont actuellement "
        "calibrés statistiquement (percentiles 95/99 sur une période saine approximée, par "
        "machine) — qui doit valider/ajuster ces seuils avant un usage en production : le data "
        "scientist, le spécialiste métier, ou une validation conjointe ?",
        "Quel est l'objectif final concret attendu par OCP : le Health Index continu déjà produit "
        "(0-1, avec 3 états Normal/Surveillance/Alarme), une RUL en heures, ou une combinaison des "
        "deux ?",
        "Quelle fréquence de mise à jour et quelle latence d'inférence sont attendues pour un "
        "déploiement en production (temps réel, horaire, journalier) ?",
        "D'autres machines ou sites OCP seront-ils intégrés à terme, et le pipeline actuel "
        "(Data Quality, nettoyage, imputation, feature engineering, Health Index) doit-il être "
        "pensé pour généraliser à d'autres capteurs i-SENSE dès maintenant ?",
    ])

    doc.save("Questions_Reunion_Equipe_iSENSE.docx")
    print("Généré : Questions_Reunion_Equipe_iSENSE.docx")


if __name__ == "__main__":
    main()
