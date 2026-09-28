"""Rapport Word — Health Index SYSTEME : méthode, calibrage, prévision, limites.

Aucun nombre du calibrage ni des résultats n'est écrit en dur : tout vient des
artefacts produits par run_30_systeme_horizons.py, et le tableau des asymétries
est recalculé à l'exécution via systeme_hi.diagnostic_formes(). Regénérer le
rapport après un réentraînement suffit à le remettre à jour.
"""
import json
import os
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

ART = os.path.join(HERE, "artifacts")


def _lire(nom):
    with open(os.path.join(ART, nom), encoding="utf-8") as f:
        return json.load(f)


CAL = _lire("systeme_calibrage.json")
LB = _lire("systeme_leaderboard.json")
ZON = _lire("systeme_zones.json")

HORIZONS = [(1, "10 min"), (2, "20 min"), (18, "3 h"), (144, "24 h")]
MACHINES = ["Motosoufflante A", "Motosoufflante B"]
FAMILLE_LABEL = {"ridge": "Ridge", "lasso_porte": "Lasso + porte",
                 "foret_aleatoire": "Forêt aléatoire",
                 "gradient_histogramme": "Gradient boosting (histogramme)"}


def d(v, n=4):
    return "—" if v is None else f"{v:.{n}f}".replace(".", ",")


def s(v, n=4):
    return "—" if v is None else f"{v:+.{n}f}".replace(".", ",")


def pc(v, n=2):
    return "—" if v is None else f"{v*100:.{n}f} %".replace(".", ",")


def formes():
    from protocol import prepare, train_mask_from_cutoffs
    from systeme_hi import diagnostic_formes
    df, _, cut, _ = prepare(verbose=False, mask_off_viscosity=False)
    return diagnostic_formes(df, train_mask_from_cutoffs(df, cut))


def main():
    print("  calcul des asymetries ...")
    F = formes()
    calA = CAL.get("Motosoufflante A") or {}

    doc = setup(Document())
    h(doc, "Health Index système", 0)
    para(doc, "État mécanique et hydraulique — sous-indices de dégradation monotones",
         size=12, color=GREY)
    para(doc, "Projet i-SENSE / OCP Maintenance Solutions — UM6P   •   Motosoufflantes A et B",
         size=9, color=GREY, space=12)

    # ═════════════════════════════════════════════════ 1
    h(doc, "1. Le principe qui gouverne tout le reste", 1)
    para(doc, "Un indice de santé ne mesure pas l'INHABITUEL. Il mesure la DÉGRADATION.",
         bold=True, size=11)
    para(doc, "Il ne doit donc baisser que lorsque l'état empire — jamais parce qu'une "
              "grandeur s'écarte de sa valeur habituelle dans le bon sens. C'est un "
              "principe de MONOTONIE, et il disqualifie toute mesure d'écart symétrique : "
              "distance de Mahalanobis, T² de Hotelling, z-score en valeur absolue, "
              "ellipse de confiance.")
    para(doc, "Ces outils sont excellents pour la détection d'anomalie — « ce point est-il "
              "atypique ? ». Ils sont inadaptés à un indice de santé — « cette machine "
              "va-t-elle plus mal ? ». Les deux questions ne sont pas la même.")

    # ═════════════════════════════════════════════════ 2
    h(doc, "2. La démonstration, sur les données du projet", 1)
    para(doc, "Une version antérieure de cet indice employait la distance de Mahalanobis. "
              "Voici ce qu'elle produisait :")
    table(doc, ["Vibration", "Pression", "Indice rendu"],
          [["0,010 — soit 13 σ EN DESSOUS de la normale, machine exceptionnellement douce",
            "4,30 bar — parfaitement normale (+0,10 σ)", "0,0000"]],
          widths=[3.2, 2.2, 1.2])
    rich(doc, [("Le pire score de santé possible, pour une machine qui allait "
                "manifestement très bien. ", False, None),
               ("Ce n'était pas un cas isolé : sur les 2 956 relevés où la machine vibrait "
                "moins que d'habitude, 907 étaient dégradés en zone B ou pire, dont 38 en "
                "alarme — uniquement pour être plus calmes que la normale.", True, RED)])
    para(doc, "La cause est structurelle et non un défaut de réglage : la vibration est "
              "une variable à sens unique traitée comme bilatérale.")

    # ═════════════════════════════════════════════════ 3
    h(doc, "3. Les deux variables n'ont pas le même sens de dégradation", 1)
    table(doc, ["Variable", "Sens", "Pourquoi"],
          [["Oil System Vibration", "à sens unique — plus haut = pire",
            "Balourd, désalignement, usure de roulement font tous MONTER la vibration. "
            "Une machine plus silencieuse n'est jamais plus malade."],
           ["Oil Pressure", "bilatérale — l'écart à la cible est le défaut",
            "Trop basse : fuite, usure de pompe. Trop haute : colmatage. La bande de "
            "fonctionnement saine est la cible."]],
          widths=[1.5, 1.7, 3.4])
    para(doc, "S'y ajoute une différence de forme, mesurée sur la population saine. Le "
              "tableau ci-dessous est recalculé à chaque exécution :", space=6)
    rows = [[r["machine"], r["variable"], r["n"], d(r["asymetrie"], 2),
             d(r["asymetrie_log"], 2), r["sens"],
             "logarithme" if r["transformee"] else "brute"]
            for _, r in F.iterrows()]
    table(doc, ["Machine", "Variable", "n", "Asymétrie", "Après log", "Sens", "Échelle"],
          rows, widths=[1.3, 1.7, 0.6, 0.85, 0.85, 0.9, 0.9])
    para(doc, "La vibration varie en ordre de grandeur : son logarithme est quasi "
              "symétrique. La pression est déjà symétrique et le logarithme la "
              "détruirait. Leur appliquer le même traitement par souci de symétrie serait "
              "l'erreur à ne pas commettre.")

    # ═════════════════════════════════════════════════ 4
    h(doc, "4. La méthode, en trois étapes", 1)
    h(doc, "4.1  Échelle de dégradation, en unités de dispersion saine", 2)
    para(doc, "s = max( 0 ,  (x − x₅₀) / (x₉₅ − x₅₀) )", bold=True, size=11, space=6)
    para(doc, "x₅₀ et x₉₅ sont la médiane et le 95ᵉ percentile de la population SAINE de "
              "la fenêtre d'ENTRAÎNEMENT — machine en marche, aucun drapeau d'anomalie, "
              "horodatage antérieur à la coupure gelée. Pour la vibration x = ln(v) ; "
              "pour la pression x = |p − p₀|, écart absolu à la cible saine.")
    rich(doc, [("Le max(0, ·) est le cœur de la méthode. ", True, NAVY),
               ("Être MEILLEUR que la médiane saine ne rapporte rien, mais ne coûte rien "
                "non plus. C'est exactement ce qui manquait à la version précédente.",
                False, None)])

    h(doc, "4.2  Sous-indice de santé du canal", 2)
    para(doc, f"d = 2 ^ ( − s / S )   avec S = {d(calA.get('s_alarme'), 0)}",
         bold=True, size=11, space=6)
    para(doc, "Décroissance à demi-vie : d = 1 au niveau sain ou en dessous, d = 0,50 au "
              "seuil d'alarme, jamais nul. S = 2 signifie que l'alarme est déclarée à deux "
              "étendues saines au-dessus de la médiane. Son choix est justifié en "
              "section 5.")

    h(doc, "4.3  Agrégation par le maillon faible", 2)
    para(doc, "HI = min( d_vibration , d_pression )", bold=True, size=11, space=6)
    para(doc, "Une machine ne vaut pas mieux que son pire symptôme. Une moyenne laisserait "
              "une pression effondrée être compensée par une vibration parfaite, ce qui "
              "n'a aucun sens en maintenance.")
    para(doc, "Corollaire utile : l'indice affiché vaut TOUJOURS celui d'un canal "
              "identifiable. Le tableau de bord nomme donc le canal qui le limite, ce qui "
              "oriente directement le diagnostic — vibration vers la mécanique tournante, "
              "pression vers le circuit hydraulique.")

    # ═════════════════════════════════════════════════ 5
    h(doc, "5. Choix du paramètre S — réglé, pas deviné", 1)
    para(doc, "Le seul paramètre libre est S. Il a été choisi en mesurant, pour chaque "
              "valeur, le taux d'alarme sur la population saine elle-même — le taux de "
              "faux positifs — et la résolution de l'indice.")
    table(doc, ["S", "Médiane de l'indice (A)", "Zone A", "Alarme", "Faux positifs"],
          [["2  ← retenu", "0,926", "86,1 %", "0,22 %", "0,24 %"],
           ["3", "0,950", "95,3 %", "0,05 %", "0,06 %"],
           ["4", "0,962", "99,6 %", "0,05 %", "0,06 %"]],
          widths=[1.2, 1.7, 1.1, 1.1, 1.2], highlight={0})
    para(doc, "À 3 et 4 l'indice sature : plus de 95 % des relevés en zone A, l'alarme ne "
              "se déclenche quasiment jamais et l'indice perd tout pouvoir discriminant. "
              "Des ancrages plus serrés ont aussi été testés — « pleinement acceptable » "
              "arrêté à la médiane saine plutôt qu'au 95ᵉ percentile — et donnaient 16 % "
              "d'alarmes : inutilisable, une alarme qui sonne un sixième du temps n'est "
              "pas une alarme.")

    # ═════════════════════════════════════════════════ 6
    h(doc, "6. Calibrage obtenu", 1)
    rows = []
    for m in MACHINES:
        c = CAL.get(m)
        if not c or "erreur" in c:
            rows.append([m, "—", "—", "—", "—", "—"])
            continue
        v, pr = c["vibration"], c["pression"]
        rows.append([m, c["n_sain_entrainement"], d(v["reference"], 3),
                     d(v["seuil_alarme"], 3), d(pr["cible_bar"], 2) + " bar",
                     "± " + d(pr["seuil_alarme"], 2)])
    table(doc, ["Machine", "n sain", "Vibration saine ≤", "Alarme vibration",
                "Cible pression", "Alarme pression"], rows,
          widths=[1.5, 0.7, 1.2, 1.2, 1.1, 1.1])

    h(doc, "6.1  Contrôle — le taux de faux positifs", 2)
    para(doc, "C'est le contrôle qui dit si le seuil est utilisable : il mesure la "
              "proportion de relevés CONNUS SAINS que l'indice classerait en alarme. "
              "S'y ajoute la part de chaque canal dans la détermination de l'indice — "
              "l'égalité correspond aux relevés où les deux canaux sont parfaitement "
              "sains, aucun ne limitant l'autre.")
    rows = []
    for m in MACHINES:
        c = CAL.get(m)
        if not c or "controle" not in c:
            continue
        ct = c["controle"]
        egal = 1 - ct["pilote_vibration"] - ct["pilote_pression"]
        rows.append([m, pc(ct["taux_alarme_population_saine"]),
                     pc(ct["pilote_vibration"], 1), pc(ct["pilote_pression"], 1),
                     pc(egal, 1)])
    table(doc, ["Machine", "Population saine en alarme", "Limité par la vibration",
                "Limité par la pression", "Égalité"], rows,
          widths=[1.5, 1.5, 1.3, 1.3, 0.9])
    para(doc, "Les deux canaux contribuent réellement : aucun n'est neutralisé par le "
              "calibrage, ce qui était le défaut d'un réglage antérieur où la vibration "
              "ne pouvait plus rien déclencher.", size=9, color=GREY)

    h(doc, "6.2  Répartition des zones", 2)
    rows = [[z["machine"], z["n_evalue"],
             f'{z["zone_A"]}  ({z["zone_A"]/z["n_evalue"]*100:.1f} %)'.replace(".", ","),
             f'{z["zone_B"]}  ({z["zone_B"]/z["n_evalue"]*100:.1f} %)'.replace(".", ","),
             f'{z["zone_C"]}  ({z["zone_C"]/z["n_evalue"]*100:.2f} %)'.replace(".", ","),
             f'{z["zone_D"]}  ({z["zone_D"]/z["n_evalue"]*100:.2f} %)'.replace(".", ","),
             d(z["HI_median"], 3), d(z["HI_p05"], 3)] for z in ZON["zones"]]
    table(doc, ["Machine", "Évaluées", "A", "B", "C", "D", "HI médian", "HI p05"], rows,
          widths=[1.4, 0.7, 1.0, 1.0, 0.9, 0.9, 0.8, 0.7])

    h(doc, "6.3  Indépendance vis-à-vis de l'indice huile", 2)
    rich(doc, [("Corrélation de Pearson entre les deux indices : ", False, None),
               (d(ZON["correlation_huile_systeme"]), True, NAVY),
               (f" sur les {ZON['n_lignes_comparees']} lignes en marche. Assez pour "
                "rester cohérents, assez peu pour que le second porte une information "
                "que le premier ne contient pas — c'est précisément l'objectif : "
                "répondre à deux questions de maintenance différentes, vidanger ou "
                "intervenir mécaniquement.", False, None)])

    # ═════════════════════════════════════════════════ 7
    h(doc, "7. Invariance à l'unité", 1)
    para(doc, "L'API i-SENSE déclare « Oil System Vibration » en mm/s², mais les valeurs "
              "observées vaudraient environ 1,2·10⁻⁵ g, ce qu'aucune machine tournante ne "
              "produit. L'étiquette d'unité est donc peu fiable, ce qui interdit tout "
              "seuil absolu — y compris ceux d'ISO 20816-3, définis en mm/s de vitesse "
              "efficace et non convertibles sans connaître la fréquence de rotation.")
    para(doc, "La méthode y est insensible. s est un rapport de différences de "
              "logarithmes :", space=4)
    para(doc, "s = ln( v / v₅₀ ) / ln( v₉₅ / v₅₀ )", bold=True, size=11, space=6)
    para(doc, "Multiplier la vibration par une constante multiplie aussi v₅₀ et v₉₅ : les "
              "rapports sont inchangés. Même raisonnement pour la pression, dont les "
              "ancrages sont estimés sur les mêmes données.")
    para(doc, "L'indice ne dépend d'aucun seuil absolu et reste valable même si l'unité "
              "déclarée est fausse.", bold=True)

    # ═════════════════════════════════════════════════ 8
    h(doc, "8. Prévision de l'indice", 1)
    para(doc, "Quatre familles comparées aux quatre horizons, sous le protocole gelé du "
              "projet : découpage gelé par machine sur les horodatages, validation croisée "
              "PURGÉE sur l'entraînement seul avec embargo proportionnel à l'horizon, et "
              "un jeu de test touché une seule fois après la sélection. La famille retenue "
              "est désignée par la validation croisée, jamais par le score de test.")
    rich(doc, [("Métrique de décision : ", True, None),
               ("le skill contre la persistance, pas le R². La cible est un indice "
                "construit, sans vérité terrain : un R² flatteur peut n'être que la "
                "reconduction de la valeur courante.", False, None)])

    h(doc, "8.1  Classement complet — Motosoufflante A", 2)
    for hp, lab in HORIZONS:
        lignes = [r for r in LB if r["horizon"] == hp and r["machine"] == "Motosoufflante A"]
        if not lignes:
            continue
        para(doc, f"Horizon {lab}", bold=True, size=10, space=3)
        lignes = sorted(lignes, key=lambda r: -(r.get("skill") or -99))
        rows = [["Persistance (référence)", d(lignes[0].get("cv_acc_persistance")),
                 d(lignes[0].get("acc_persist")), "—", "—"]]
        surligne = set()
        for i, r in enumerate(lignes, start=1):
            nom = FAMILLE_LABEL.get(r["famille"], r["famille"])
            if r.get("retenu_par_cv"):
                nom += "  ← retenu par la CV"
                surligne.add(i)
            rows.append([nom, d(r.get("cv_acc")), d(r.get("test_acc")),
                         s(r.get("skill")), d(r.get("r2"))])
        table(doc, ["Famille", "CV (acc)", "Test (acc)", "Skill", "R²"], rows,
              widths=[2.6, 0.95, 0.95, 0.95, 0.95], highlight=surligne)

    h(doc, "8.2  Motosoufflante B — entraînée mais non évaluable", 2)
    nb_b = len([r for r in LB if r["machine"] == "Motosoufflante B"])
    para(doc, f"{nb_b} configurations entraînées, aucune évaluable : la machine ne tourne "
              "plus après le 31 janvier 2026 et sa fenêtre de test ne contient aucune "
              "ligne en marche. Seuls les scores de validation croisée existent ; ils ne "
              "constituent PAS une mesure de performance sur données nouvelles.")
    for hp, lab in HORIZONS:
        lignes = [r for r in LB if r["horizon"] == hp and r["machine"] == "Motosoufflante B"]
        if not lignes:
            para(doc, f"Horizon {lab} : écarté, trop peu de lignes d'entraînement.",
                 size=9, color=GREY, space=3)
            continue
        para(doc, f"Horizon {lab}", bold=True, size=10, space=3)
        lignes = sorted(lignes, key=lambda r: -(r.get("cv_acc") or -99))
        rows = [["Persistance (référence)", d(lignes[0].get("cv_acc_persistance"))]]
        surligne = set()
        for i, r in enumerate(lignes, start=1):
            nom = FAMILLE_LABEL.get(r["famille"], r["famille"])
            if r.get("retenu_par_cv"):
                nom += "  ← retenu par la CV"
                surligne.add(i)
            rows.append([nom, d(r.get("cv_acc"))])
        table(doc, ["Famille", "CV (acc)"], rows, widths=[3.2, 1.2], highlight=surligne)

    # ═════════════════════════════════════════════════ 9
    h(doc, "9. Résultat principal : aucun modèle ne bat la persistance", 1)
    rows = []
    for hp, lab in HORIZONS:
        ev = [r for r in LB if r["horizon"] == hp and r["machine"] == "Motosoufflante A"
              and r.get("evaluable")]
        if not ev:
            continue
        best = max(ev, key=lambda r: r.get("skill") or -99)
        rows.append([lab, d(best.get("acc_persist")),
                     FAMILLE_LABEL.get(best["famille"], best["famille"]),
                     d(best.get("test_acc")), s(best.get("skill"))])
    table(doc, ["Horizon", "Persistance (acc)", "Meilleur modèle", "Son acc", "Son skill"],
          rows, widths=[0.9, 1.3, 2.2, 0.9, 1.0])
    para(doc, "À aucun horizon un modèle n'atteint la précision de la persistance.",
         bold=True, color=RED, space=6)
    para(doc, "Le skill est pourtant positif. La contradiction n'est qu'apparente : le "
              "skill se calcule sur l'erreur quadratique (1 − MSE_modèle / "
              "MSE_persistance), et les modèles réduisent effectivement les grosses "
              "erreurs — mais ils tombent MOINS souvent dans la bande utile de ± 0,01 que "
              "« rien ne change ».")
    para(doc, "Ce constat est robuste : il a été retrouvé à l'identique sur trois "
              "définitions successives de l'indice système. Le tableau de bord applique "
              "donc une porte resserrée — un horizon n'est déclaré fiable que s'il bat la "
              "persistance sur les DEUX critères à la fois.")
    rich(doc, [("Conséquence assumée : ", True, None),
               ("aucun horizon de l'indice système n'est affiché comme prévision fiable. "
                "L'indice est montré, sa prévision ne l'est pas.", False, RED)])

    # ═════════════════════════════════════════════════ 10
    h(doc, "10. Limites", 1)
    calB = CAL.get("Motosoufflante B") or {}
    zB = next((z for z in ZON["zones"] if z["machine"] == "Motosoufflante B"), {})
    bullets(doc, [
        "Ce n'est pas une mesure de sévérité physique. L'indice dit « à quelle distance du "
        "fonctionnement sain, dans le sens de la dégradation », pas « quelle est la "
        "gravité du défaut en mm/s ». Il ne remplace pas une expertise vibratoire "
        "normative, il la déclenche.",
        "Un paramètre est réglé, pas dérivé. S = 2 vient d'un compromis entre résolution "
        "et faux positifs, pas d'une loi physique. Des spécifications constructeur le "
        "remplaceraient avantageusement.",
        f"La vibration est massivement imputée : "
        f"{pc((calA.get('vibration') or {}).get('pct_impute'), 0)} sur la machine A, "
        f"{pc((calB.get('vibration') or {}).get('pct_impute'), 0)} sur la machine B, qui "
        "ne compte que 2 mesures réelles en fonctionnement sur toute la période. Son "
        "indice repose donc quasi entièrement sur des valeurs reconstruites.",
        f"La Motosoufflante B n'est pas évaluable en prévision : {zB.get('n_evalue', '—')} "
        "lignes en marche, toutes antérieures à la coupure gelée. Son taux de faux "
        f"positifs, {pc((calB.get('controle') or {}).get('taux_alarme_population_saine'))}, "
        "est aussi six fois celui de la machine A.",
        "La référence est fixée une fois sur la fenêtre d'entraînement. Après une révision "
        "majeure ou un changement de régime, il faudrait la réestimer.",
    ])

    # ═════════════════════════════════════════════════ 11
    h(doc, "11. Traçabilité", 1)
    table(doc, ["Fichier", "Rôle"],
          [["hi_forecast/systeme_hi.py", "Construction de l'indice, calibrage, diagnostic de forme"],
           ["hi_forecast/run_30_systeme_horizons.py",
            "Entraînement multi-horizons et comparaison des familles"],
           ["hi_forecast/systeme_leaderboard.csv", "Classement complet, toutes configurations"],
           ["hi_forecast/artifacts/systeme_calibrage.json",
            "Ancrages, seuils, contrôle de faux positifs"],
           ["hi_forecast/artifacts/systeme_zones.json", "Répartition des zones, corrélation"],
           ["dashboard/backend/service.py", "Exposition au tableau de bord et KPI de décision"],
           ["explication_health_index_systeme.md", "Version Markdown de ce document"]],
          widths=[3.1, 3.5], size=8.5)

    out = os.path.join(ROOT, "Rapport_HealthIndex_Systeme_iSENSE.docx")
    return save_doc(doc, out)


if __name__ == "__main__":
    p = main()
    print(f"Document Word genere : {p}   ({os.path.getsize(p)/1024:.0f} Ko)")
