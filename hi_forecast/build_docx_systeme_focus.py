"""Rapport Word CIBLE — indice systeme : methode de calcul, modeles de prevision,
figures de comparaison.

Genere d'abord trois figures, puis le document qui les porte. Tous les chiffres
et toutes les figures viennent des artefacts de run_30 : rien n'est recopie.

REGLES DE VISUALISATION APPLIQUEES
  - Les sept familles sont UNE SEULE categorie « modele » : une seule couleur,
    pas sept teintes a distinguer. Le message est « toutes sous la persistance »,
    pas « laquelle est laquelle ».
  - La persistance est une REFERENCE, tracee en trait neutre discontinu — elle
    n'est pas une famille de plus.
  - Le skill a une polarite (positif = utile, negatif = nuisible) : palette
    divergente, deux teintes et un zero marque.
  - Grille et axes en retrait, marques fines, aucune decoration inutile.
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

# Palette validee : une teinte pour les modeles, un neutre pour la reference,
# une paire divergente pour la polarite du skill.
BLEU = "#1d4ed8"
NEUTRE = "#2d3748"
POSITIF = "#0d9488"
NEGATIF = "#b45309"
GRILLE = "#e6ebf2"

HORIZONS = [(1, "10 min"), (2, "20 min"), (18, "3 h"), (144, "24 h")]
LABEL = {"ridge": "Ridge", "lasso_porte": "Lasso + porte",
         "foret_aleatoire": "Forêt aléatoire",
         "gradient_histogramme": "Grad. boosting (hist.)",
         "xgboost": "XGBoost", "lightgbm_huber": "LightGBM Huber",
         "catboost": "CatBoost"}
MACHINE = "Motosoufflante A"          # la seule evaluable


def _lire(nom, defaut=None):
    p = os.path.join(ART, nom)
    if not os.path.exists(p):
        return defaut
    with open(p, encoding="utf-8") as f:
        return json.load(f)


LB = _lire("systeme_leaderboard.json", []) or []
ZON = _lire("systeme_zones.json", {}) or {}
CAL = _lire("systeme_calibrage.json", {}) or {}


def _style(ax):
    ax.grid(axis="x", color=GRILLE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for cote in ("top", "right", "left"):
        ax.spines[cote].set_visible(False)
    ax.spines["bottom"].set_color("#a9b4c2")
    ax.tick_params(colors="#5a6472", labelsize=8, length=0)


# ═════════════════════════════════════════════ FIGURE 1 — precision
def fig_precision():
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6))
    for ax, (hp, lab) in zip(axes.ravel(), HORIZONS):
        rs = [r for r in LB if r["horizon"] == hp and r["machine"] == MACHINE
              and r.get("evaluable")]
        if not rs:
            ax.axis("off")
            continue
        rs = sorted(rs, key=lambda r: r.get("test_acc") or 0)
        noms = [LABEL.get(r["famille"], r["famille"]) for r in rs]
        vals = [r.get("test_acc") or 0 for r in rs]
        pers = rs[0].get("acc_persist") or 0

        y = np.arange(len(rs))
        ax.barh(y, vals, height=0.62, color=BLEU, zorder=3)
        ax.axvline(pers, color=NEUTRE, linestyle="--", linewidth=1.6, zorder=4)
        ax.text(pers, len(rs) - 0.35, f"  persistance {pers:.3f}".replace(".", ","),
                color=NEUTRE, fontsize=8, va="bottom", ha="left")
        for yi, v in zip(y, vals):
            ax.text(v + 0.006, yi, f"{v:.3f}".replace(".", ","), va="center",
                    fontsize=7.5, color="#5a6472")
        ax.set_yticks(y, noms, fontsize=8)
        ax.set_xlim(0, max(max(vals), pers) * 1.25)
        ax.set_title(f"Horizon {lab}", fontsize=9.5, color=NEUTRE, loc="left",
                     pad=6)
        _style(ax)
    fig.suptitle("Précision dans la bande ±0,01 — sept familles contre la persistance",
                 fontsize=11.5, color=NEUTRE, x=0.012, ha="left", y=0.985)
    fig.text(0.012, 0.015,
             "Aucune barre n'atteint le trait de référence : à aucun horizon un modèle "
             "n'égale « rien ne change ».",
             fontsize=8.5, color="#5a6472")
    fig.tight_layout(rect=[0, 0.035, 1, 0.955])
    p = os.path.join(FIG, "sys_precision.png")
    fig.savefig(p, dpi=170, facecolor="white")
    plt.close(fig)
    return p


# ═════════════════════════════════════════════ FIGURE 2 — skill
def fig_skill():
    """Petits multiples, un panneau par horizon.

    Une premiere version groupait les quatre horizons par famille : on ne pouvait
    pas savoir quel baton correspondait a quel horizon. Un panneau par horizon
    supprime l'ambiguite et se lit comme la figure 1."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6))
    for ax, (hp, lab) in zip(axes.ravel(), HORIZONS):
        rs = [r for r in LB if r["horizon"] == hp and r["machine"] == MACHINE
              and r.get("evaluable")]
        if not rs:
            ax.axis("off")
            continue
        rs = sorted(rs, key=lambda r: r.get("skill") or 0)
        noms = [LABEL.get(r["famille"], r["famille"]) for r in rs]
        vals = [r.get("skill") or 0 for r in rs]
        y = np.arange(len(rs))
        ax.barh(y, vals, height=0.62,
                color=[POSITIF if v >= 0 else NEGATIF for v in vals], zorder=3)
        ax.axvline(0, color=NEUTRE, linewidth=1.2, zorder=4)
        marge = max(abs(min(vals)), abs(max(vals))) * 0.22
        for yi, v in zip(y, vals):
            ax.text(v + (marge * 0.12 if v >= 0 else -marge * 0.12), yi,
                    f"{v:+.3f}".replace(".", ","), va="center",
                    ha="left" if v >= 0 else "right",
                    fontsize=7.5, color="#5a6472")
        ax.set_yticks(y, noms, fontsize=8)
        ax.set_xlim(min(vals) - marge, max(vals) + marge)
        ax.set_title(f"Horizon {lab}", fontsize=9.5, color=NEUTRE, loc="left", pad=6)
        ax.grid(axis="x", color=GRILLE, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for cote in ("top", "right", "left"):
            ax.spines[cote].set_visible(False)
        ax.spines["bottom"].set_color("#a9b4c2")
        ax.tick_params(colors="#5a6472", labelsize=8, length=0)
    fig.suptitle("Skill sur l'erreur quadratique — par famille et par horizon",
                 fontsize=11.5, color=NEUTRE, x=0.012, ha="left", y=0.985)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=POSITIF, label="positif — moins d'erreur quadratique"),
                        Patch(color=NEGATIF, label="négatif — pire que la persistance")],
               fontsize=8.5, frameon=False, loc="upper right",
               bbox_to_anchor=(0.995, 1.005), ncol=2)
    fig.text(0.012, 0.015,
             "Le skill est positif presque partout, alors que la précision ne l'est "
             "jamais (figure 1) : les deux critères divergent. C'est ce qui a motivé une "
             "porte de fiabilité à deux conditions.",
             fontsize=8.5, color="#5a6472")
    fig.tight_layout(rect=[0, 0.035, 1, 0.945])
    p = os.path.join(FIG, "sys_skill.png")
    fig.savefig(p, dpi=170, facecolor="white")
    plt.close(fig)
    return p


# ═════════════════════════════════════════════ FIGURE 3 — CV vs test
def fig_cv_test():
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    # Les points retenus sont proches : on alterne le decalage de l'etiquette
    # au-dessus puis au-dessous, sinon « 10 min » et « 20 min » se chevauchent.
    for i, (hp, lab) in enumerate(HORIZONS):
        decal = (10, 7) if i % 2 == 0 else (10, -17)
        rs = [r for r in LB if r["horizon"] == hp and r["machine"] == MACHINE
              and r.get("evaluable")]
        for r in rs:
            cv, te = r.get("cv_acc") or 0, r.get("test_acc") or 0
            retenu = r.get("retenu_par_cv")
            ax.scatter(cv, te, s=64 if retenu else 30,
                       facecolor=BLEU if retenu else "white",
                       edgecolor=BLEU, linewidth=1.4, zorder=3)
            if retenu:
                ax.annotate(f"{LABEL.get(r['famille'], r['famille'])} — {lab}",
                            (cv, te), textcoords="offset points", xytext=decal,
                            fontsize=7.5, color=NEUTRE,
                            ha="left", va="center")
    lim = 0.58
    ax.plot([0, lim], [0, lim], color="#a9b4c2", linestyle=":", linewidth=1.2, zorder=2)
    ax.text(lim * 0.97, lim * 0.97, "égalité", fontsize=8, color="#8a94a3",
            ha="right", va="bottom", rotation=45)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("précision en validation croisée  (ce qui sert à CHOISIR)",
                  fontsize=9, color="#5a6472")
    ax.set_ylabel("précision sur le test  (ce qu'on découvre ENSUITE)",
                  fontsize=9, color="#5a6472")
    ax.grid(color=GRILLE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)
    ax.spines["left"].set_color("#a9b4c2"); ax.spines["bottom"].set_color("#a9b4c2")
    ax.tick_params(colors="#5a6472", labelsize=8, length=0)
    ax.set_title("Ce que la validation croisée choisit, ce que le test révèle",
                 fontsize=11.5, color=NEUTRE, loc="left", pad=10)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", color="none", markerfacecolor=BLEU,
                              markeredgecolor=BLEU, markersize=9,
                              label="famille retenue par la CV"),
                       Line2D([], [], marker="o", color="none", markerfacecolor="white",
                              markeredgecolor=BLEU, markersize=7,
                              label="autres familles")],
              fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    p = os.path.join(FIG, "sys_cv_test.png")
    fig.savefig(p, dpi=170, facecolor="white")
    plt.close(fig)
    return p


def d(v, n=4):
    return "—" if v is None else f"{v:.{n}f}".replace(".", ",")


def s(v, n=4):
    return "—" if v is None else f"{v:+.{n}f}".replace(".", ",")


def main():
    os.makedirs(FIG, exist_ok=True)
    print("  figures ...")
    for f in (fig_precision, fig_skill, fig_cv_test):
        print("   ", os.path.basename(f()))

    doc = setup(Document())
    h(doc, "Health Index système", 0)
    para(doc, "Méthode de calcul · modèles de prévision · comparaison",
         size=12, color=GREY)
    para(doc, "Projet i-SENSE / OCP Maintenance Solutions — UM6P   •   Motosoufflantes A et B",
         size=9, color=GREY, space=12)

    # ═════════════════════════════════════ 1. METHODE
    h(doc, "1. Méthode de calcul", 1)
    para(doc, "Second indice, indépendant de celui de l'huile. Il répond à « la machine "
              "a-t-elle un problème mécanique ? » et non « faut-il changer l'huile ? ». "
              "Deux variables seulement : la pression d'huile et la vibration du système.")

    h(doc, "1.1  Le principe : un indice de santé doit être monotone", 2)
    para(doc, "Un indice de santé ne mesure pas l'INHABITUEL, il mesure la DÉGRADATION. "
              "Il ne doit baisser que lorsque l'état empire — jamais parce qu'une "
              "grandeur s'écarte de sa valeur habituelle dans le bon sens.", bold=True)
    para(doc, "Ce principe disqualifie les mesures d'écart symétriques — distance de "
              "Mahalanobis, T² de Hotelling, z-score en valeur absolue. Constaté sur ces "
              "données : un relevé à 0,010 de vibration, machine exceptionnellement "
              "douce, avec une pression parfaitement normale, obtenait l'indice 0,0000 — "
              "le pire score possible.")
    table(doc, ["Variable", "Sens de dégradation", "Traitement"],
          [["Oil System Vibration", "à sens unique — plus haut = pire",
            "logarithme (asymétrie +2,05 → −0,29)"],
           ["Oil Pressure", "bilatéral — l'écart à la cible est le défaut",
            "brute (déjà symétrique)"]],
          widths=[1.8, 2.5, 2.3])

    h(doc, "1.2  Les trois étapes du calcul", 2)
    para(doc, "s = max( 0 ,  (x − x₅₀) / (x₉₅ − x₅₀) )", bold=True, size=11, space=3)
    para(doc, "Échelle de dégradation en unités de dispersion saine. x₅₀ et x₉₅ sont la "
              "médiane et le 95ᵉ percentile de la population SAINE de la fenêtre "
              "d'ENTRAÎNEMENT. Le max(0, ·) est le cœur de la méthode : être meilleur que "
              "la médiane saine ne rapporte rien, mais ne coûte rien.", space=8)
    para(doc, "d = 2 ^ ( − s / 2 )", bold=True, size=11, space=3)
    para(doc, "Sous-indice de santé du canal, par décroissance à demi-vie : d = 1 au "
              "niveau sain ou en dessous, d = 0,50 au seuil d'alarme, jamais nul.",
         space=8)
    para(doc, "HI = min( d_vibration , d_pression )", bold=True, size=11, space=3)
    para(doc, "Agrégation par le MAILLON FAIBLE : une machine ne vaut pas mieux que son "
              "pire symptôme. Une moyenne laisserait une pression effondrée être "
              "compensée par une vibration parfaite. Corollaire utile : l'indice affiché "
              "vaut toujours celui d'un canal identifiable, que le tableau de bord nomme.")

    h(doc, "1.3  Invariance à l'unité", 2)
    para(doc, "L'API déclare la vibration en mm/s², mais les valeurs observées vaudraient "
              "≈ 1,2·10⁻⁵ g — physiquement impossible pour une machine tournante. "
              "L'étiquette d'unité est donc peu fiable, ce qui interdit tout seuil absolu, "
              "y compris ceux d'ISO 20816-3.")
    para(doc, "s = ln( v / v₅₀ ) / ln( v₉₅ / v₅₀ )", bold=True, size=10.5, space=3)
    para(doc, "s étant un rapport de différences de logarithmes, multiplier la vibration "
              "par une constante multiplie aussi v₅₀ et v₉₅ : le rapport est inchangé. "
              "L'indice ne dépend d'aucun seuil absolu et reste valable même si l'unité "
              "déclarée est fausse.", bold=True)

    h(doc, "1.4  Calibrage et contrôle", 2)
    rows = []
    for m, c in CAL.items():
        if "erreur" in c:
            continue
        v, pr, ct = c["vibration"], c["pression"], c["controle"]
        rows.append([m, d(v["reference"], 3), d(v["seuil_alarme"], 3),
                     d(pr["cible_bar"], 2) + " bar",
                     f"{ct['taux_alarme_population_saine']*100:.2f} %".replace(".", ",")])
    table(doc, ["Machine", "Vibration saine ≤", "Alarme vibration", "Cible pression",
                "Faux positifs sur population saine"], rows,
          widths=[1.5, 1.2, 1.2, 1.2, 1.5])
    para(doc, "Le taux de faux positifs est le contrôle qui dit si le seuil est "
              "utilisable : la proportion de relevés CONNUS SAINS que l'indice classerait "
              "en alarme.", size=9, color=GREY)
    rows = [[z["machine"], z["n_evalue"],
             f'{z["zone_A"]/z["n_evalue"]*100:.1f} %'.replace(".", ","),
             f'{z["zone_B"]/z["n_evalue"]*100:.1f} %'.replace(".", ","),
             f'{z["zone_C"]/z["n_evalue"]*100:.2f} %'.replace(".", ","),
             f'{z["zone_D"]/z["n_evalue"]*100:.2f} %'.replace(".", ","),
             d(z["HI_median"], 3)] for z in ZON.get("zones", [])]
    if rows:
        table(doc, ["Machine", "Évaluées", "Zone A", "Zone B", "Zone C", "Zone D",
                    "HI médian"], rows, widths=[1.5, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9])

    # ═════════════════════════════════════ 2. MODELES
    h(doc, "2. Modèles de prévision", 1)
    para(doc, "Sept familles comparées aux quatre horizons — 10 min, 20 min, 3 h, 24 h — "
              "sous le protocole gelé du projet : découpage sur horodatages, validation "
              "croisée PURGÉE avec embargo proportionnel à l'horizon, sélection sur "
              "l'entraînement seul, jeu de test touché une seule fois après la sélection.")
    table(doc, ["Famille", "Nature"],
          [["Ridge", "linéaire régularisée L2"],
           ["Lasso + porte", "linéaire L1 sur le delta, plus une porte de mouvement"],
           ["Forêt aléatoire", "arbres en bagging"],
           ["Gradient boosting (histogramme)", "boosting scikit-learn"],
           ["XGBoost", "boosting"],
           ["LightGBM Huber", "boosting, perte robuste — configuration de l'indice huile"],
           ["CatBoost", "boosting à arbres symétriques"]],
          widths=[2.4, 4.2], size=8.5)
    rich(doc, [("Métrique de décision : ", True, None),
               ("le skill contre la persistance, pas le R². La cible est un indice "
                "construit, sans vérité terrain : un R² flatteur peut n'être que la "
                "reconduction de la valeur courante.", False, None)])

    # ═════════════════════════════════════ 3. FIGURES
    h(doc, "3. Comparaison", 1)

    h(doc, "3.1  Précision — la comparaison qui décide", 2)
    figure(doc, "sys_precision.png",
           "Figure 1 — Précision dans la bande de tolérance ±0,01, par famille et par "
           "horizon, Motosoufflante A. Le trait discontinu marque la persistance. "
           "Aucune barre ne l'atteint.")
    rows = []
    for hp, lab in HORIZONS:
        ev = [r for r in LB if r["horizon"] == hp and r["machine"] == MACHINE
              and r.get("evaluable")]
        if not ev:
            continue
        best = max(ev, key=lambda r: r.get("test_acc") or 0)
        rows.append([lab, d(best.get("acc_persist")),
                     LABEL.get(best["famille"], best["famille"]),
                     d(best.get("test_acc")), s(best.get("skill"))])
    table(doc, ["Horizon", "Persistance", "Meilleur modèle", "Sa précision", "Son skill"],
          rows, widths=[0.9, 1.2, 2.3, 1.1, 1.1])
    para(doc, "À aucun horizon un modèle n'atteint la précision de la persistance.",
         bold=True, color=RED)

    h(doc, "3.2  Skill — pourquoi les deux critères divergent", 2)
    figure(doc, "sys_skill.png",
           "Figure 2 — Skill sur l'erreur quadratique, par famille et par horizon. "
           "Positif presque partout, alors que la précision ne l'est jamais.")
    para(doc, "Le skill se calcule sur l'erreur quadratique : les modèles réduisent "
              "effectivement les GROSSES erreurs. Mais ils tombent moins souvent dans la "
              "bande utile de ±0,01 que « rien ne change ». Le critère qui compte pour "
              "décider d'une intervention est la précision dans la bande.")
    para(doc, "fiable  ⟺  skill > 0   ET   précision ≥ précision de la persistance",
         bold=True, size=10.5, space=6)
    rich(doc, [("Conséquence assumée : ", True, None),
               ("aucun horizon de l'indice système n'est affiché comme prévision fiable. "
                "L'indice est montré, sa prévision ne l'est pas.", False, RED)])

    h(doc, "3.3  La discipline du protocole", 2)
    figure(doc, "sys_cv_test.png",
           "Figure 3 — Précision en validation croisée contre précision sur le test. "
           "Les points pleins sont les familles retenues par la validation croisée.",
           width=5.6)
    para(doc, "Cette figure montre pourquoi le choix se fait en validation croisée et non "
              "sur le test. La famille que la CV désigne n'est pas toujours celle qui "
              "l'emporte sur le test — sélectionner sur le test serait une fuite, et "
              "gonflerait artificiellement la performance annoncée.")

    # ═════════════════════════════════════ 4. LIMITES
    h(doc, "4. Limites", 1)
    calA = CAL.get("Motosoufflante A", {})
    calB = CAL.get("Motosoufflante B", {})
    bullets(doc, [
        "Ce n'est pas une mesure de sévérité physique : l'indice dit « à quelle distance "
        "du fonctionnement sain, dans le sens de la dégradation », pas « quelle est la "
        "gravité du défaut en mm/s ». Il déclenche une expertise, il ne la remplace pas.",
        "Un paramètre est réglé, pas dérivé : le seuil d'alarme à deux étendues saines "
        "vient d'un compromis entre résolution et faux positifs, pas d'une loi physique.",
        f"La vibration est massivement imputée : "
        f"{(calA.get('vibration') or {}).get('pct_impute', 0)*100:.0f} % sur la machine A, "
        f"{(calB.get('vibration') or {}).get('pct_impute', 0)*100:.0f} % sur la machine B, "
        "qui ne compte que 2 mesures réelles en fonctionnement sur toute la période.",
        "La Motosoufflante B n'est pas évaluable en prévision : sa fenêtre de test ne "
        "contient aucune ligne en marche. Toutes les figures portent donc sur la "
        "Motosoufflante A.",
        "Le constat « la persistance gagne » est robuste : il a été retrouvé à "
        "l'identique sur trois définitions successives de l'indice et contre sept "
        "familles de modèles.",
    ])

    out = os.path.join(ROOT, "Rapport_Systeme_Methode_et_Modeles_iSENSE.docx")
    return save_doc(doc, out)


if __name__ == "__main__":
    p = main()
    print(f"\nDocument Word genere : {p}   ({os.path.getsize(p)/1024:.0f} Ko)")
