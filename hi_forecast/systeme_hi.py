"""État du système — DEUX indicateurs INDEPENDANTS, pression et vibration.

Ce module ne produit plus d'indice systeme agrege. La version precedente
combinait les deux canaux par un maillon faible, HI = min(pression, vibration) :
un seul chiffre, dont on ne savait pas d'ou il venait sans lire un champ annexe.

CE QUI A CHANGE, ET POURQUOI
Les deux grandeurs decrivent des organes differents et appellent des
interventions differentes :

    Oil Pressure           -> circuit hydraulique : pompe, colmatage, fuite
    Oil System Vibration   -> mecanique tournante : balourd, desalignement,
                              usure de roulement

Les agreger produisait un chiffre sans destinataire : personne n'intervient sur
« le systeme », on intervient sur la pompe OU sur les paliers. Les deux
indicateurs sont donc maintenant calcules, affiches et lus SEPAREMENT.

MEME LOGIQUE DE CALCUL POUR LES DEUX — seul le sens de degradation change.

  1. ECHELLE DE DEGRADATION, en unites de dispersion saine :

         s = max( 0 ,  (x - x50) / (x95 - x50) )

     x50 et x95 : mediane et 95e percentile de la population SAINE de la fenetre
     d'ENTRAINEMENT. Le max(0, .) est le coeur de la methode — un indicateur de
     sante doit etre MONOTONE : il ne baisse que lorsque l'etat empire, jamais
     parce qu'une grandeur s'ecarte dans le bon sens.

  2. INDICATEUR, decroissance a demi-vie :

         indice = 2 ^ ( -s / S_ALARME )      avec S_ALARME = 2

     1 au niveau sain ou en dessous, 0,50 au seuil d'alarme, jamais nul.

CE QUI DIFFERE ENTRE LES DEUX CANAUX

    PRESSION    bilaterale  : x = |p - cible saine|. Trop basse signale une
                              fuite ou une usure de pompe, trop haute un
                              colmatage. Elle reste en echelle brute, sa
                              distribution etant deja symetrique.
    VIBRATION   sens unique : x = log(v). Plus haut = pire, plus bas = mieux.
                              Le logarithme parce qu'elle varie en ordre de
                              grandeur (asymetrie +2,05 brute, -0,29 en log).

INVARIANCE D'ECHELLE — l'unite declaree de `Oil System Vibration` (mm/s2) est
douteuse : les valeurs observees vaudraient ~1e-5 g. s etant un RAPPORT de
differences de logarithmes, multiplier la vibration par une constante ne change
rien. Aucun seuil absolu n'est invoque, ISO 20816-3 compris.

MACHINE A L'ARRET — aucun des deux indicateurs n'est evalue : la pression tombe
a zero et la vibration ne porte aucun signal mecanique.

FUITE — tous les ancrages sont estimes sur la fenetre d'ENTRAINEMENT seule,
puis appliques a toutes les lignes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from protocol import ASSET_COLS, FLAG_COLS

VIB_COL = "Oil System Vibration_filled"
PRESS_COL = "Oil Pressure"

# Degradation, en unites (mediane saine -> 95e percentile sain), a laquelle on
# declare l'alarme.
S_ALARME = 2.0
P_REF, P_SPAN = 0.50, 0.95

# Frontieres de zone, communes aux deux indicateurs.
ZONES = {"A": 0.75, "B": 0.50, "C": 0.25}
ZONE_ETAT = {"A": "Normal", "B": "Surveillance", "C": "Alarme", "D": "Alarme"}
NON_EVALUABLE = "Non évaluable"

VIB_MIN = 1e-3          # plancher avant passage au logarithme

# Les deux canaux, decrits une fois pour toutes.
CANAUX = {
    "pression": {"colonne": PRESS_COL, "sens": "bilatéral", "echelle": "brute",
                 "organe": "circuit hydraulique — pompe, colmatage, fuite"},
    "vibration": {"colonne": VIB_COL, "sens": "unique", "echelle": "logarithme",
                  "organe": "mécanique tournante — balourd, désalignement, roulement"},
}


def _zone_de(indice: np.ndarray) -> np.ndarray:
    return np.select([indice <= ZONES["C"], indice <= ZONES["B"], indice <= ZONES["A"]],
                     ["D", "C", "B"], default="A")


def _masque_sain(df: pd.DataFrame, train_mask: pd.Series, col_machine: str) -> np.ndarray:
    """Lignes saines : machine EN MARCHE, dans la fenetre d'entrainement, aucun
    drapeau d'anomalie leve."""
    on = (df["state_OFF"] == 0).to_numpy()
    flags = df[[c for c in FLAG_COLS if c in df.columns]].any(axis=1).to_numpy()
    return (df[col_machine] == 1).to_numpy() & train_mask.to_numpy() & on & ~flags


def _indicateur(x: pd.Series, sain: np.ndarray) -> tuple[pd.Series, dict]:
    """Un canal : echelle calee sur la population saine, puis indicateur."""
    x50 = float(x[sain].quantile(P_REF))
    x95 = float(x[sain].quantile(P_SPAN))
    etendue = x95 - x50
    if not np.isfinite(etendue) or etendue <= 0:
        # Population saine degeneree : aucune dispersion exploitable.
        return pd.Series(1.0, index=x.index), {"degenere": True, "ref": x50, "p95": x95}
    s = ((x - x50) / etendue).clip(lower=0)
    return np.power(2.0, -s / S_ALARME), {
        "degenere": False, "ref": x50, "p95": x95, "etendue": etendue,
        "seuil_alarme": x50 + S_ALARME * etendue}


def _valeurs_canal(df: pd.DataFrame, canal: str, p0: float | None = None) -> pd.Series:
    """Grandeur sur laquelle porte la degradation, selon le canal."""
    if canal == "vibration":
        return np.log(df[VIB_COL].clip(lower=VIB_MIN))
    return (df[PRESS_COL] - p0).abs()


def build_systeme_hi(df: pd.DataFrame, train_mask: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Ajoute les DEUX indicateurs independants et renvoie (df, calibrage).

    Colonnes ajoutees, par canal (pression, vibration) :
        indice_<canal>_lf    indicateur dans ]0, 1], 1 = sain, 0,50 = alarme
        zone_<canal>_lf      A / B / C / D
        etat_<canal>_lf      Normal / Surveillance / Alarme / Non évaluable
    plus :
        systeme_evaluable    True si la machine tourne
    """
    for canal in CANAUX:
        df[f"indice_{canal}_lf"] = np.nan
        df[f"zone_{canal}_lf"] = ""
        df[f"etat_{canal}_lf"] = NON_EVALUABLE
    on_all = (df["state_OFF"] == 0).to_numpy()
    df["systeme_evaluable"] = on_all

    calibrage = {}
    for asset, col in ASSET_COLS.items():
        m_eval = (df[col] == 1).to_numpy() & on_all
        if m_eval.sum() == 0:
            continue
        sain = _masque_sain(df, train_mask, col)
        if sain.sum() < 100:
            calibrage[asset] = {"erreur": f"seulement {int(sain.sum())} lignes saines"}
            continue

        cible = float(df.loc[sain, PRESS_COL].median())
        fiche = {"n_sain_entrainement": int(sain.sum()), "n_evalue": int(m_eval.sum()),
                 "s_alarme": S_ALARME, "canaux": {}}

        for canal, spec in CANAUX.items():
            x = _valeurs_canal(df, canal, cible)
            ind, anc = _indicateur(x, sain)
            zone = _zone_de(ind.to_numpy())
            sel = m_eval & ind.notna().to_numpy()
            df.loc[sel, f"indice_{canal}_lf"] = ind[sel]
            df.loc[sel, f"zone_{canal}_lf"] = zone[sel]
            df.loc[sel, f"etat_{canal}_lf"] = [ZONE_ETAT[z] for z in zone[sel]]

            # Les ancrages de la vibration sont stockes en unites physiques.
            en_clair = (lambda v: float(np.exp(v))) if canal == "vibration" else float
            fiche["canaux"][canal] = {
                **{k: v for k, v in spec.items()},
                "reference": en_clair(anc["ref"]), "p95": en_clair(anc["p95"]),
                "seuil_alarme": (None if anc["degenere"]
                                 else en_clair(anc["seuil_alarme"])),
                "degenere": anc["degenere"],
                # Taux de fausse alarme sur la population SAINE elle-meme : le
                # controle qui dit si le seuil est utilisable.
                "taux_alarme_population_saine": float((ind[sain] <= ZONES["B"]).mean()),
            }
            if canal == "pression":
                fiche["canaux"][canal]["cible_bar"] = cible
            if canal == "vibration" and "vibration_confidence_score" in df.columns:
                fiche["canaux"][canal]["pct_impute"] = float(
                    df.loc[m_eval, "vibration_confidence_score"].gt(0).mean())
        calibrage[asset] = fiche
    return df, calibrage


def appliquer_systeme_hi(df: pd.DataFrame, calibrage: dict) -> pd.DataFrame:
    """Applique un calibrage DEJA AJUSTE a de nouvelles lignes.

    Pendant « service » de build_systeme_hi : celui-ci ESTIME les ancrages,
    celui-la les APPLIQUE. Un flux temps reel ne doit jamais reestimer — sinon
    les seuils derivent avec les donnees qu'ils jugent."""
    for canal in CANAUX:
        df[f"indice_{canal}_lf"] = np.nan
        df[f"zone_{canal}_lf"] = ""
        df[f"etat_{canal}_lf"] = NON_EVALUABLE
    on_all = (df["state_OFF"] == 0).to_numpy()
    df["systeme_evaluable"] = on_all

    for asset, col in ASSET_COLS.items():
        cal = calibrage.get(asset)
        if not cal or "erreur" in cal:
            continue
        if "canaux" not in cal:
            # Calibrage d'une version anterieure — l'artefact fige n'a pas ete
            # regenere depuis le passage aux deux canaux independants. On laisse
            # les indicateurs a NaN plutot que de faire tomber le service : le
            # tableau de bord affichera « non evaluable », ce qui est exact.
            continue
        m_eval = (df[col] == 1).to_numpy() & on_all
        if m_eval.sum() == 0:
            continue
        s_alarme = float(cal.get("s_alarme", S_ALARME))
        cible = (cal["canaux"].get("pression") or {}).get("cible_bar")

        for canal, anc in cal["canaux"].items():
            x = _valeurs_canal(df, canal, cible)
            ref, p95 = anc["reference"], anc["p95"]
            if canal == "vibration":          # ancrages stockes en clair
                ref, p95 = np.log(max(ref, VIB_MIN)), np.log(max(p95, VIB_MIN))
            etendue = p95 - ref
            if not np.isfinite(etendue) or etendue <= 0:
                ind = pd.Series(1.0, index=df.index)
            else:
                ind = np.power(2.0, -((x - ref) / etendue).clip(lower=0) / s_alarme)
            zone = _zone_de(ind.to_numpy())
            sel = m_eval & ind.notna().to_numpy()
            df.loc[sel, f"indice_{canal}_lf"] = ind[sel]
            df.loc[sel, f"zone_{canal}_lf"] = zone[sel]
            df.loc[sel, f"etat_{canal}_lf"] = [ZONE_ETAT[z] for z in zone[sel]]
    return df


def diagnostic_formes(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """Asymetrie des deux variables sur la population saine, brute et apres
    logarithme. C'est ce tableau qui justifie de transformer la vibration et de
    laisser la pression en l'etat."""
    lignes = []
    for asset, col in ASSET_COLS.items():
        sain = _masque_sain(df, train_mask, col)
        if sain.sum() < 100:
            continue
        for c, libelle, sens in [(PRESS_COL, "Oil Pressure", "bilatéral"),
                                 (VIB_COL, "Oil System Vibration", "unique")]:
            v = df.loc[sain, c].dropna()
            if v.empty:
                continue
            lv = np.log(v.clip(lower=VIB_MIN))
            lignes.append({"machine": asset, "variable": libelle, "n": int(len(v)),
                           "asymetrie": float(stats.skew(v)),
                           "asymetrie_log": float(stats.skew(lv)),
                           "sens": sens, "transformee": c == VIB_COL})
    return pd.DataFrame(lignes)


def resume_zones(df: pd.DataFrame) -> pd.DataFrame:
    """Repartition des zones, PAR CANAL et par machine."""
    lignes = []
    for asset, col in ASSET_COLS.items():
        for canal in CANAUX:
            m = ((df[col] == 1) & df["systeme_evaluable"]
                 & df[f"indice_{canal}_lf"].notna())
            if m.sum() == 0:
                continue
            vc = df.loc[m, f"zone_{canal}_lf"].value_counts()
            lignes.append({"machine": asset, "canal": canal, "n_evalue": int(m.sum()),
                           **{f"zone_{z}": int(vc.get(z, 0)) for z in "ABCD"},
                           "indice_median": float(df.loc[m, f"indice_{canal}_lf"].median()),
                           "indice_p05": float(df.loc[m, f"indice_{canal}_lf"].quantile(.05))})
    return pd.DataFrame(lignes)
