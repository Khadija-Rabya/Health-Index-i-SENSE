"""
Phase 1 du plan de construction du Health Index (plan_construction_health_index.md) :
HI a base de regles metier — REVISION du 2026-09-01 : utilise desormais les
SEUILS OFFICIELS OCP (Cadrage des seuils_Systeme d'huile.pdf,
data_quality_isense.py::OFFICIAL_THRESHOLDS) sur 12 variables, plutot que les
4 variables a seuils empiriques (percentile) de la version precedente. La
justification originale de la Phase 1 ("s'appuyer sur des seuils deja etablis")
s'applique desormais a (presque) toutes les variables sante disponibles.

Decisions methodologiques explicites (validees) :
- DC : sens "eleve = critique" retenu tel quel (tableau de seuils officiel),
  malgre une contradiction avec les diapositives i-SENSE (DC elevee = bon etat)
  - point ouvert documente, cf. Questions_Reunion_Equipe_iSENSE.docx.
- Oil Conductivity : seuils appliques sur Oil Conductivity_nSm_filled (conversion
  x10 determinee par analyse, cf. clean_isense_data.py), pas la colonne brute.
- Reference TD46 (Shell Turbo T46, ISO VG 46) appliquee TELLE QUELLE aux deux
  machines, y compris Motosoufflante B qui en sort massivement pour la
  viscosite/densite - une decision assumee : voir cette sortie de plage comme
  un signal reel plutot que de la masquer par un seuil recalibre par machine.

Formule de distance par variable (0 = sain, 1 = seuil "surveillance" atteint,
au-dela clippe a 1) :
- Unidirectionnelle "haut = pire" (temperature, H2O saturation, DC, ISO 4/6/14,
  H2O ppm) : distance = clip((valeur - normal) / (surveillance - normal), 0, 1)
- Bidirectionnelle (Oil Conductivity_nSm) : distance = 0 si dans la bande normale,
  sinon proportionnelle a l'ecart au bord de bande le plus proche
- Ecart relatif a une reference (Density, Viscosity at 40°C : reference TD46
  absolue ; Kinematic/Dynamic Viscosity : reference = mediane par machine, le
  document ne donnant qu'une regle relative "±5%/±10%" sans valeur absolue) :
  distance = clip((|ecart%| - normal%) / (surveillance% - normal%), 0, 1)

Entree : isense_oil_data_features.csv
Sortie : isense_oil_data_health_index.csv (colonnes HI_regles, health_state_regles)
         + eda_output/health_index/01_hi_regles_timeline.png
         + rapport_health_index_regles.md
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_quality_isense import OFFICIAL_THRESHOLDS

warnings.filterwarnings("ignore")

INPUT_FILE = "isense_oil_data_features.csv"
OUTPUT_DIR = "eda_output/health_index"

ASSET_COLS = {"Motosoufflante A": "asset_Motosoufflante A", "Motosoufflante B": "asset_Motosoufflante B"}

# variable -> colonne reelle du dataset (versions _filled issues de fill_sensor_gap.py)
VARIABLE_COLUMNS = {
    "Oil Temperature": "Oil Temperature_filled",
    "Oil H2O Saturation": "Oil H2O Saturation_filled",
    "DC": "DC_filled",
    "Oil Conductivity_nSm": "Oil Conductivity_nSm_filled",
    "ISO 14": "ISO 14",
    "ISO 6": "ISO 6",
    "ISO 4": "ISO 4_filled",
    "Oil H2O ppm": "Oil H2O ppm_filled",
    "Density": "Density_filled",
    "Kinematic Viscosity": "Kinematic Viscosity_filled",
    "Dynamic Viscosity": "Dynamic Viscosity_filled",
    "Viscosity at 40°C": "Viscosity at 40°C_filled",
}
WEIGHT = 1 / len(VARIABLE_COLUMNS)  # poids egal, faute de ponderation officielle connue

SURVEILLANCE_PCTL = 0.95
ALARME_PCTL = 0.99


def distance_unidirectional(values, normal, surveillance):
    return ((values - normal) / (surveillance - normal)).clip(0, 1)


def distance_bidirectional_conductivity(values, spec):
    below = (spec["normal_low"] - values) / (spec["normal_low"] - spec["surveillance_low"])
    above = (values - spec["normal_high"]) / (spec["surveillance_high"] - spec["normal_high"])
    dist = pd.concat([below, above], axis=1).max(axis=1).clip(lower=0)
    return dist.clip(0, 1)


def distance_pct_reference(values, reference, normal_pct, surveillance_pct):
    pct_dev = (values - reference).abs() / reference * 100
    return ((pct_dev - normal_pct) / (surveillance_pct - normal_pct)).clip(0, 1)


def compute_distances(sub):
    """Calcule les 12 distances normalisees [0,1] pour un sous-dataframe (une machine)."""
    distances = pd.DataFrame(index=sub.index)

    for var, spec in OFFICIAL_THRESHOLDS.items():
        col = VARIABLE_COLUMNS[var]
        values = sub[col]

        if spec["sens"] == "high":
            distances[var] = distance_unidirectional(values, spec["normal"], spec["surveillance"])
        elif spec["sens"] == "bidirectionnel":
            distances[var] = distance_bidirectional_conductivity(values, spec)
        elif spec["sens"] == "ecart_pct":
            distances[var] = distance_pct_reference(values, spec["reference"], spec["normal_pct"], spec["surveillance_pct"])
        elif spec["sens"] == "ecart_pct_ref_machine":
            # reference = mediane de la machine (le document ne donne pas de valeur absolue,
            # seulement une regle relative "±5%/±10% autour de la reference")
            reference = values.median()
            distances[var] = distance_pct_reference(values, reference, spec["normal_pct"], spec["surveillance_pct"])
        else:
            raise ValueError(f"Sens inconnu pour {var} : {spec['sens']}")

    return distances


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_FILE, parse_dates=["created_at"])

    df["HI_regles"] = np.nan
    for var in OFFICIAL_THRESHOLDS:
        df[f"dist_{var}"] = np.nan

    report = ["# Rapport — Health Index Phase 1 (seuils officiels OCP)\n"]
    report.append(f"- Lignes totales : {len(df)}")
    report.append(f"- Variables utilisées ({len(OFFICIAL_THRESHOLDS)}) : "
                   f"{', '.join(OFFICIAL_THRESHOLDS.keys())} (poids égaux, {WEIGHT:.3f} chacune)")
    report.append(
        "- **Seuils** : issus de `Cadrage des seuils_Système d'huile.pdf` (OCP Maintenance "
        "Solutions), appliqués **tels quels aux deux machines** — y compris Motosoufflante B, "
        "dont plusieurs variables (viscosité, densité) sortent largement de la référence "
        "d'huile TD46 (Shell Turbo T46 / ISO VG 46), une sortie de plage traitée comme un "
        "signal réel plutôt que masquée par un recalibrage par machine.\n"
        "- **Point ouvert** : le sens de dégradation de `DC` est contradictoire entre le "
        "tableau de seuils (DC élevée = critique, retenu ici) et les diapositives i-SENSE "
        "(DC élevée = bon état) — à trancher avec l'équipe, cf. "
        "`Questions_Reunion_Equipe_iSENSE.docx`.\n"
        "- **Oil Conductivity** : seuils appliqués sur `Oil Conductivity_nSm_filled` "
        "(conversion ×10 déterminée par analyse, cf. `clean_isense_data.py`), pas la colonne "
        "brute.\n"
    )

    report.append("| Machine | HI moyen | HI min | % Surveillance | % Alarme | Seuil Surveillance | Seuil Alarme |")
    report.append("|---|---|---|---|---|---|---|")

    thresholds_hi = {}
    for asset, col in ASSET_COLS.items():
        mask = df[col] == 1
        sub = df.loc[mask]
        distances = compute_distances(sub)
        for var in OFFICIAL_THRESHOLDS:
            df.loc[mask, f"dist_{var}"] = distances[var]

        hi = (1 - WEIGHT * distances.sum(axis=1)).clip(0, 1)
        df.loc[mask, "HI_regles"] = hi

        seuil_surveillance = hi.quantile(1 - SURVEILLANCE_PCTL)
        seuil_alarme = hi.quantile(1 - ALARME_PCTL)
        thresholds_hi[asset] = (seuil_surveillance, seuil_alarme)

        n_surv = int(((hi <= seuil_surveillance) & (hi > seuil_alarme)).sum())
        n_alarm = int((hi <= seuil_alarme).sum())
        report.append(
            f"| {asset} | {hi.mean():.3f} | {hi.min():.3f} | {n_surv / len(sub):.1%} | "
            f"{n_alarm / len(sub):.1%} | {seuil_surveillance:.3f} | {seuil_alarme:.3f} |"
        )

    df["health_state_regles"] = "Normal"
    for asset, col in ASSET_COLS.items():
        mask = df[col] == 1
        seuil_surveillance, seuil_alarme = thresholds_hi[asset]
        df.loc[mask & (df["HI_regles"] <= seuil_surveillance) & (df["HI_regles"] > seuil_alarme), "health_state_regles"] = "Surveillance"
        df.loc[mask & (df["HI_regles"] <= seuil_alarme), "health_state_regles"] = "Alarme"

    report.append("\n## Contribution moyenne de chaque distance, par machine (diagnostic)\n")
    report.append("| Variable | Distance moy. A | Distance moy. B |")
    report.append("|---|---|---|")
    for var in OFFICIAL_THRESHOLDS:
        mean_a = df.loc[df[ASSET_COLS["Motosoufflante A"]] == 1, f"dist_{var}"].mean()
        mean_b = df.loc[df[ASSET_COLS["Motosoufflante B"]] == 1, f"dist_{var}"].mean()
        report.append(f"| {var} | {mean_a:.3f} | {mean_b:.3f} |")

    report.append(
        "\n⚠️ Conséquence directe du choix « DC élevée = critique, appliqué tel quel » : "
        "une DC anormalement **basse** (cas de Motosoufflante B, ~1,03 en moyenne, sous le "
        "seuil « Impossible » officiel de 1,8) ne pénalise PAS le score — la formule "
        "unidirectionnelle du tableau de seuils ne définit aucune sévérité côté bas. "
        "distance DC = 0,000 pour les deux machines ci-dessus le confirme. Si l'équipe "
        "i-SENSE confirme l'interprétation des diapositives (DC basse = dégradation), ce "
        "point devra être corrigé — actuellement l'anomalie DC de B n'est donc visible que "
        "via le Contrôle 5 de la Phase 0 (`rapport_data_quality.md`), pas via ce Health "
        "Index à seuils officiels.\n"
        "\n⚠️ Limite similaire sur `Kinematic Viscosity`/`Dynamic Viscosity` : faute de "
        "référence absolue dans le document officiel (règle relative « ±5%/±10% autour de "
        "la référence » seulement), la référence a été approximée par la médiane de chaque "
        "machine — ce qui capture surtout le bruit de mesure autour de cette médiane "
        "(distances élevées, 0,71-0,86, pour les DEUX machines symétriquement) plutôt qu'une "
        "vraie dégradation. `Viscosity at 40°C`, qui dispose d'une référence absolue TD46 "
        "(46,0 cSt), reste beaucoup plus discriminante (0,040 pour A vs 0,945 pour B).\n"
    )

    report.append("\n## Répartition finale des états\n")
    report.append("| État | Lignes |")
    report.append("|---|---|")
    for state, count in df["health_state_regles"].value_counts().items():
        report.append(f"| {state} | {count} |")

    # --- Graphique : trajectoire temporelle par machine ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    colors = {"Normal": "#38a169", "Surveillance": "#dd6b20", "Alarme": "#e53e3e"}
    for ax, (asset, col) in zip(axes, ASSET_COLS.items()):
        mask = df[col] == 1
        sub = df.loc[mask].sort_values("created_at")
        seuil_surveillance, seuil_alarme = thresholds_hi[asset]
        ax.plot(sub["created_at"], sub["HI_regles"], color="#2b6cb0", linewidth=0.6, alpha=0.7)
        ax.axhline(seuil_surveillance, color=colors["Surveillance"], linestyle="--", linewidth=1, label="Seuil Surveillance")
        ax.axhline(seuil_alarme, color=colors["Alarme"], linestyle="--", linewidth=1, label="Seuil Alarme")
        ax.set_title(f"HI_regles (seuils officiels OCP) — {asset}")
        ax.set_ylabel("HI (1 = sain)")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/01_hi_regles_timeline.png", dpi=120)
    plt.close(fig)

    # --- Graphique : contribution moyenne par variable, par machine ---
    fig, ax = plt.subplots(figsize=(10, 6))
    variables = list(OFFICIAL_THRESHOLDS.keys())
    means_a = [df.loc[df[ASSET_COLS["Motosoufflante A"]] == 1, f"dist_{v}"].mean() for v in variables]
    means_b = [df.loc[df[ASSET_COLS["Motosoufflante B"]] == 1, f"dist_{v}"].mean() for v in variables]
    y = np.arange(len(variables))
    width = 0.35
    ax.barh(y - width / 2, means_a, height=width, label="Motosoufflante A", color="#2b6cb0")
    ax.barh(y + width / 2, means_b, height=width, label="Motosoufflante B", color="#dd6b20")
    ax.set_yticks(y)
    ax.set_yticklabels(variables)
    ax.set_xlabel("Distance moyenne (0=sain, 1=seuil surveillance atteint)")
    ax.set_title("Contribution moyenne de chaque variable au Health Index — seuils officiels OCP")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/00_distances_par_variable.png", dpi=120)
    plt.close(fig)

    df.to_csv("isense_oil_data_health_index.csv", index=False, encoding="utf-8-sig")

    with open("rapport_health_index_regles.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    print("\n".join(report))
    print(f"\nFichiers générés : isense_oil_data_health_index.csv, rapport_health_index_regles.md, "
          f"{OUTPUT_DIR}/01_hi_regles_timeline.png, {OUTPUT_DIR}/00_distances_par_variable.png")


if __name__ == "__main__":
    main()
