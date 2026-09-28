"""
Nettoyage du dataset i-SENSE (qualite huile de lubrification)
selon le plan d'action de l'audit de qualite (i-SENSE_Dataset.pdf, section 3.1),
REVISE suite a l'audit de Data Quality physico-chimique (Phase 0,
data_quality_isense.py / plan_refonte_pipeline.md) :

  - Kinematic Viscosity et Density ne sont PLUS supprimees pour cause de
    correlation avec Viscosity at 40C / Oil Temperature : la Phase 0 a montre
    que ces correlations sont des lois physiques (mu = nu x rho, densite
    normalisee ASTM D1298), pas des redondances. Les deux variables sont
    conservees telles quelles ici ; leur transformation en indices metier
    (density_15C, vi_proxy) est faite en aval (feature_engineering.py), une
    fois les valeurs manquantes comblees.

Entree : isense_oil_data_combined_wide.csv (format large genere par Api_to_excel.py)
Sortie : isense_oil_data_cleaned.csv / .xlsx + rapport de nettoyage (Markdown)
"""

import pandas as pd
import numpy as np

INPUT_FILE = "isense_oil_data_combined_wide.csv"

# Variables affectees par le sentinel -99.99 / -9999 (toutes sauf Oil Pressure
# et Oil System Vibration, qui suivent une logique differente - cf. audit 2.2 et 2.4/2.5)
SENTINEL_COLUMNS = [
    "DC", "Density", "Dynamic Viscosity",
    "ISO 14", "ISO 4", "ISO 6",
    "Kinematic Viscosity", "Oil Conductivity",
    "Oil H2O Saturation", "Oil H2O ppm",
    "Oil Temperature", "Viscosity at 40°C",
]

# Variables a forward-fill (desynchronisation export API, cf. audit 2.1)
FFILL_COLUMNS = ["ISO 6", "ISO 14"]

OIL_PRESSURE_ON_THRESHOLD = 0.1  # Bar - au-dessus = machine active (audit 2.4)
GAP_SESSION_THRESHOLD_HOURS = 1  # au-dela = nouvelle session (audit 2.6)


def load_data():
    df = pd.read_csv(INPUT_FILE, parse_dates=["created_at"])
    df = df.sort_values(["asset_name", "created_at"]).reset_index(drop=True)
    return df


def report_line(report, text):
    print(text)
    report.append(text)


def clean(df):
    report = []
    n_start = len(df)
    report_line(report, f"## Rapport de nettoyage — dataset i-SENSE\n")
    report_line(report, f"- Lignes en entree : **{n_start}**")

    # --- Etape 1 & 2 : valeurs sentinelles -99.99 / -9999 -> NaN ---------------
    sentinel_mask = pd.Series(False, index=df.index)
    for col in SENTINEL_COLUMNS:
        col_mask = df[col] < -9
        sentinel_mask |= col_mask
        df.loc[col_mask, col] = np.nan
    n_sentinel_rows = int(sentinel_mask.sum())
    report_line(
        report,
        f"- Etape 1-2 : **{n_sentinel_rows}** lignes contenaient des valeurs "
        f"sentinelles (-99.99 / -9999) sur {len(SENTINEL_COLUMNS)} variables -> remplacees par NaN.",
    )

    # --- Suppression des lignes totalement vides (artefact de jonction) -------
    # A faire AVANT le forward-fill (etape 4) ET avant la correction Oil Conductivity
    # ci-dessous (etape 1 bis) : a ce stade, la sentinelle non standard -0.09999 de
    # Oil Conductivity n'est pas encore convertie en NaN, ce qui permet de distinguer
    # le vrai artefact de jonction (1 ligne, TOUT est NaN y compris Oil Conductivity)
    # du trou capteur synchrone de 994 lignes (Oil Conductivity y vaut -0.09999,
    # donc "presente" a ce stade -> ne declenche pas ce masque). Inverser cet ordre
    # supprimerait a tort les 994 lignes du trou synchrone (deja verifie : elles ont
    # toutes un Oil Pressure valide, donc ne sont pas des artefacts de jonction).
    fully_empty_mask = df[SENTINEL_COLUMNS].isna().all(axis=1)
    n_empty = int(fully_empty_mask.sum())
    df = df[~fully_empty_mask].reset_index(drop=True)
    report_line(
        report,
        f"- Etape preliminaire : **{n_empty}** ligne(s) totalement vide(s) "
        f"(artefact de jonction, toutes variables NaN sauf Oil Pressure) supprimee(s).",
    )

    # --- Etape 1 bis (NOUVEAU) : sentinelle non detectee sur Oil Conductivity ---
    # Decouverte en croisant les seuils officiels OCP (Cadrage des seuils_Systeme
    # d'huile.pdf) avec les donnees reelles : Oil Conductivity contient une
    # sentinelle -0.09999 (et non -99.99/-9999 comme les autres colonnes), donc
    # le test generique `< -9` ci-dessus ne la detectait PAS - elle etait traitee
    # comme une vraie mesure depuis le debut du pipeline. Elle touche EXACTEMENT
    # les 994 memes lignes que le trou capteur synchrone deja documente
    # (fill_sensor_gap.py) - confirme qu'il s'agit du meme evenement de coupure.
    oil_cond_sentinel_mask = df["Oil Conductivity"].round(5) == -0.09999
    n_oil_cond_sentinel = int(oil_cond_sentinel_mask.sum())
    df.loc[oil_cond_sentinel_mask, "Oil Conductivity"] = np.nan
    sentinel_mask |= oil_cond_sentinel_mask.reindex(df.index, fill_value=False)
    report_line(
        report,
        f"- Etape 1 bis (nouveau) : **{n_oil_cond_sentinel}** lignes de `Oil Conductivity` "
        "contenaient une sentinelle non standard (-0.09999, non couverte par le test "
        "générique < -9) -> remplacées par NaN. Comblées ensuite par forward-fill "
        "(fill_sensor_gap.py), comme les 9 autres variables du même trou synchrone.",
    )

    # --- Etape 1 ter (NOUVEAU) : conversion d'unite Oil Conductivity -> nS/m ---
    # Les seuils officiels OCP sont exprimes en nS/m (normal 0.4-2.0). Les valeurs
    # reelles (hors sentinelle) du dataset se regroupent etroitement entre 0.062
    # et 0.072 - hors de toute plage documentee. Analyse : un facteur x10 les fait
    # tomber integralement dans la plage "Normal" (0.622-0.720, percentiles 1-99%
    # a 0.637-0.685) - facteur retenu, documente ici pour tracabilite plutot que
    # suppose sans verification.
    df["Oil Conductivity_nSm"] = df["Oil Conductivity"] * 10
    report_line(
        report,
        "- Etape 1 ter (nouveau) : colonne `Oil Conductivity_nSm` ajoutée = "
        "`Oil Conductivity` × 10, facteur déterminé par analyse (les valeurs réelles, "
        "0.062-0.072, tombent alors dans la plage officielle « Normal » 0.4-2.0 nS/m) — "
        "à confirmer avec l'équipe i-SENSE, cf. `Questions_Reunion_Equipe_iSENSE.docx`.",
    )

    # --- Etape 3 : flag machine ON/OFF a partir de Oil Pressure ---------------
    df["machine_state"] = np.where(
        df["Oil Pressure"] > OIL_PRESSURE_ON_THRESHOLD, "ON", "OFF"
    )
    off_counts = df.groupby("asset_name")["machine_state"].apply(
        lambda s: (s == "OFF").sum()
    )
    for asset, n_off in off_counts.items():
        total = (df["asset_name"] == asset).sum()
        report_line(
            report,
            f"- Etape 3 : {asset} — {n_off}/{total} lignes en etat OFF/veille "
            f"({n_off/total:.1%}), flag `machine_state` cree.",
        )

    # --- Etape 4 : forward-fill ISO 6 / ISO 14 (desync export API) -----------
    for col in FFILL_COLUMNS:
        n_before = df[col].isna().sum()
        df[col] = df.groupby("asset_name")[col].ffill()
        n_after = df[col].isna().sum()
        report_line(
            report,
            f"- Etape 4 : `{col}` — {n_before} NaN avant forward-fill, "
            f"{n_after} restants (debut de serie).",
        )

    # --- Etape 5 : vibration - conserver + flag disponibilite ----------------
    df["vibration_available"] = df["Oil System Vibration"].notna()
    n_vib_available = int(df["vibration_available"].sum())
    report_line(
        report,
        f"- Etape 5 : `Oil System Vibration` conservee ; "
        f"{n_vib_available}/{len(df)} mesures disponibles "
        f"({n_vib_available/len(df):.1%}), flag `vibration_available` cree.",
    )

    # --- Etape 6 : decoupage en sessions continues (gaps > 1h) ---------------
    def assign_sessions(group):
        gap_hours = group["created_at"].diff().dt.total_seconds() / 3600
        new_session = (gap_hours > GAP_SESSION_THRESHOLD_HOURS).fillna(False)
        return new_session.cumsum()

    df["session_id"] = (
        df.groupby("asset_name", group_keys=False)
        .apply(assign_sessions, include_groups=False)
    )
    df["session_id"] = df["asset_name"] + "_S" + df["session_id"].astype(str)
    n_sessions = df["session_id"].nunique()
    report_line(
        report,
        f"- Etape 6 : donnees decoupees en **{n_sessions}** sessions continues "
        f"(nouvelle session des qu'un ecart > {GAP_SESSION_THRESHOLD_HOURS}h est detecte).",
    )

    # --- Etape 7 (REVISEE) : Kinematic Viscosity et Density sont CONSERVEES ---
    # L'audit initial avait supprime Kinematic Viscosity (r=0.86 avec Viscosity at 40C)
    # et exclu Density (r=-1.00 avec Oil Temperature) pour "redondance". La Phase 0
    # (data_quality_isense.py) a montre que ce sont des lois physiques verifiees
    # (mu = nu x rho a 0.01% d'erreur ; densite normalisee ASTM D1298 quasi constante
    # a 869 kg/m3), pas des doublons : les deux variables portent une information
    # utile (proxy d'indice de viscosite, composition de l'huile) qui serait perdue
    # en les supprimant. Voir plan_refonte_pipeline.md, Partie 1.2.
    corr_visc = df[["Kinematic Viscosity", "Viscosity at 40°C"]].corr().iloc[0, 1]
    corr_dens = df[["Density", "Oil Temperature"]].corr().iloc[0, 1]
    report_line(
        report,
        f"- Etape 7 (révisée) : `Kinematic Viscosity` / `Viscosity at 40°C` corrélées "
        f"(r={corr_visc:.4f}) et `Density` / `Oil Temperature` corrélées (r={corr_dens:.4f}), "
        f"mais **aucune des deux n'est supprimée** — la Phase 0 (audit physico-chimique) a "
        f"montré qu'il s'agit de lois physiques (μ=ν×ρ, densité ASTM D1298), pas de "
        f"redondances. Les deux variables sont conservées pour permettre le calcul "
        f"d'indices métier dérivés (density_15C, proxy d'indice de viscosité) en aval.",
    )
    report_line(
        report,
        "- ⚠️ Anomalie détectée en Phase 0 (voir `rapport_data_quality.md`) : "
        "la plupart des mesures `DC` de Motosoufflante B sont physiquement impossibles "
        "au sens de la borne universelle (< 1, alors qu'une constante diélectrique "
        "relative ne peut être < 1) — et encore plus au sens de la borne officielle OCP "
        "resserrée (< 1.8, cf. `Cadrage des seuils_Système d'huile.pdf`, valeurs "
        "normales 2.1-2.4 pour une huile hydrocarbure). Probable biais de calibration "
        "capteur — **non corrigé ici**, à confirmer avec l'équipe i-SENSE avant toute "
        "correction (cf. `Questions_Reunion_Equipe_iSENSE.docx`).",
    )

    n_end = len(df)
    report_line(report, f"\n- Lignes en sortie : **{n_end}** (delta : {n_end - n_start})")

    return df, report


def main():
    df = load_data()
    df_clean, report = clean(df)

    df_clean.to_csv("isense_oil_data_cleaned.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter("isense_oil_data_cleaned.xlsx", engine="openpyxl") as writer:
        df_clean.to_excel(writer, sheet_name="cleaned_combined", index=False)
        for asset in df_clean["asset_name"].unique():
            sub = df_clean[df_clean["asset_name"] == asset]
            sheet = "A_cleaned" if "A" in asset else "B_cleaned"
            sub.to_excel(writer, sheet_name=sheet, index=False)

    with open("rapport_nettoyage.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    print("\nFichiers generes :")
    print("- isense_oil_data_cleaned.csv")
    print("- isense_oil_data_cleaned.xlsx")
    print("- rapport_nettoyage.md")


if __name__ == "__main__":
    main()
