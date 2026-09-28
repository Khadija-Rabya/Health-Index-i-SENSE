"""Couche metier du tableau de bord Health Index i-SENSE.

Trois familles de sorties :
  PREDICTION  — DEUX health index par machine : HUILE (etat physico-chimique du
                lubrifiant) et SYSTEME (etat mecanique et hydraulique, construit
                sur Oil Pressure + Oil System Vibration, cf. systeme_hi.py).
                Chacun calcule a t, predit a 4 horizons (10 min / 20 min / 3 h /
                24 h), compare a la persistance, plus les KPI de decision.
  QUALITE     — etat machine, densite a 15 C, coherence H2O, distribution DC,
                coherence physique mu = nu x rho, monotonie des codes ISO.
  DIAGNOSTIC  — imputation de la vibration, synchronisme des capteurs, profils
                calendaires, sessions, scores de confiance.

Le chemin de code est identique pour le rejeu et pour un futur flux temps reel :
le service ne recoit qu'un DataFrame de lignes « deja arrivees ».
"""
from __future__ import annotations

import json
import os
import sys
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
HI = os.path.join(ROOT, "hi_forecast")
for p in (ROOT, HI, "C:\\pylib"):
    if p not in sys.path:
        sys.path.insert(0, p)

from protocol import ASSET_COLS, TOLERANCE  # noqa: E402
from systeme_hi import NON_EVALUABLE, ZONES  # noqa: E402

STEP_MIN = 10
# Peremption d'une mesure. La constante vaut pour la CADENCE HISTORIQUE — une
# mesure toutes les 10 minutes — dont elle tolere trois manquees. Elle sert de
# plancher ; en direct, le seuil est recalcule sur la cadence REELLE de la
# source (voir _seuil_fraicheur), qui est bien plus lente.
STALE_AFTER_MIN = 30
# Au-dela, un flux est mort quelle que soit sa cadence nominale : le seuil
# adaptatif ne doit jamais normaliser une acquisition arretee.
STALE_PLAFOND_MIN = 360

# Seuil qui definit l'etat machine, repris de clean_isense_data.py : au-dessus,
# le circuit est en pression, la machine tourne. C'est la SEULE variable qui
# distingue marche et arret — la vibration ne le fait pas (mediane 0,13 en
# marche contre 0,11 a l'arret).
PRESSION_ON_BAR = 0.1

# Quatre horizons affiches par le tableau de bord, en pas de 10 min.
HORIZONS = (1, 2, 18, 144)

# Les deux indices atteignent 0,50 a leur seuil d'alarme, par deux chemins
# mathematiques differents : HI = 1/(1+severite) avec severite = 1 au seuil pour
# l'indice HUILE, HI = 2^(-s/s_alarme) sur le canal le plus degrade pour l'indice
# SYSTEME. Les marges affichees restent donc comparables.
SEUIL_ALARME = float(ZONES["B"])          # 0,50
SEUIL_SURVEILLANCE = float(ZONES["A"])    # 0,75

# ── Seuils de l'ETAT DU SYSTEME, ETABLIS PENDANT LE TRAITEMENT DE DONNEES ──
# Aucun n'est invente ici : chacun porte le fichier ou il a ete fixe. Ils sont
# exprimes DANS L'UNITE DU CAPTEUR, pas en indice normalise. L'etat du systeme
# se lit donc directement sur la mesure, sans emprunter le seuil d'alarme du
# Health Index huile (0,50) — qui vaut pour un indice sans unite et n'a aucun
# sens pour des bar ou des mm/s2.
SEUIL_PRESSION_MARCHE = PRESSION_ON_BAR   # 0,1 bar  · OIL_PRESSURE_ON_THRESHOLD
SEUIL_VIBRATION_ALARME = 1.5              # flag_high_vibration · feature_engineering.py
BORNE_VIBRATION_PHYSIQUE = 9.8            # PHYSICAL_RANGES · data_quality_isense.py
# Le document officiel OCP (« Cadrage des seuils_Systeme d'huile.pdf ») ne fixe
# de seuil NI pour la pression NI pour la vibration : OFFICIAL_THRESHOLDS couvre
# temperature, eau, DC, ISO 4406, densite et viscosites. Le 1,5 est qualifie de
# « seuil indicatif » dans feature_engineering.py. Il est retenu parce que c'est
# celui que le pipeline applique, et parce qu'il discrimine reellement : il tombe
# juste au-dessus du p99 en marche de la Motosoufflante A (1,42) et declenche sur
# 0,80 % de ses mesures. Ce n'est pas une norme constructeur.
VISC_COL = "Viscosity at 40°C_filled"
VISC_MIN = 20.0                      # cSt : sous ce seuil, impossible pour une ISO VG 46
DENSITY_BASELINE = 869.0             # kg/m3, ligne de base Phase 0

SENSORS = ["Oil Temperature_filled", "Oil H2O ppm_filled", "Oil H2O Saturation_filled",
           VISC_COL, "Kinematic Viscosity_filled", "Dynamic Viscosity_filled",
           "Density_filled", "ISO 4_filled", "ISO 6", "ISO 14", "DC_filled",
           "Oil Conductivity_nSm_filled", "Oil System Vibration_filled", "Oil Pressure"]


# ------------------------------------------------------------------ artefacts
def _lire_json(nom):
    """Artefact optionnel : renvoie None s'il n'a pas encore ete produit, pour que
    le tableau de bord reste affichable pendant un reentrainement."""
    p = os.path.join(HI, "artifacts", nom)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_artifacts():
    models = joblib.load(os.path.join(HI, "artifacts", "modeles_multi_horizon.joblib"))
    with open(os.path.join(HI, "artifacts", "multi_horizon_metriques.json"),
              encoding="utf-8") as f:
        metrics = json.load(f)
    shap = pd.read_csv(os.path.join(HI, "importance_shap_final.csv")).head(20)
    idx = {(m["horizon"], m["machine"]): m for m in metrics}

    # --- artefacts du Health Index SYSTEME (produits par run_30) ---
    p_sys = os.path.join(HI, "artifacts", "modeles_systeme_multi_horizon.joblib")
    sys_models = joblib.load(p_sys) if os.path.exists(p_sys) else {}
    sys_metrics = _lire_json("systeme_metriques.json") or []
    sys_idx = {(m["horizon"], m["machine"]): m for m in sys_metrics}

    return {"models": models, "metrics": metrics, "metrics_idx": idx, "shap": shap,
            "sys_models": sys_models, "sys_metrics": sys_metrics, "sys_idx": sys_idx,
            "sys_calibrage": _lire_json("systeme_calibrage.json") or {},
            "sys_leaderboard": _lire_json("systeme_leaderboard.json") or []}


@lru_cache(maxsize=1)
def load_dataset():
    """Source du rejeu. Les variables `_lf` (sans fuite) sont produites par
    protocol.prepare() : servir le CSV brut ferait echouer l'inference — c'est le
    skew entrainement/service documente dans ARCHITECTURE_PIPELINE.md."""
    from protocol import prepare
    from systeme_hi import build_systeme_hi
    df, train_mask, _, _ = prepare(verbose=False, mask_off_viscosity=False)
    # Le Health Index SYSTEME est construit ici, avec le meme masque
    # d'entrainement que l'indice huile : ses frontieres de zone sont calibrees
    # sur la fenetre d'entrainement seule, jamais sur le futur.
    df, _ = build_systeme_hi(df, train_mask)
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(np.int8)
    df["health_index"] = df["health_index_lf"]
    df["health_state"] = df["health_state_lf"]
    # Deux indicateurs INDEPENDANTS, plus d'agregat : voir systeme_hi.py.
    for canal in ("pression", "vibration"):
        df[f"indice_{canal}"] = df[f"indice_{canal}_lf"]
    return df.sort_values(["session_id", "created_at"]).reset_index(drop=True)


def asset_of(row) -> str:
    return ("Motosoufflante A" if row[ASSET_COLS["Motosoufflante A"]] == 1
            else "Motosoufflante B")


# ------------------------------------------------------------ porte de qualite
def quality_gate(df: pd.DataFrame) -> pd.DataFrame:
    q = pd.DataFrame(index=df.index)
    off = df["state_OFF"] == 1 if "state_OFF" in df else pd.Series(False, index=df.index)
    visc = df[VISC_COL] if VISC_COL in df else pd.Series(np.nan, index=df.index)
    q["visc_impossible"] = visc < VISC_MIN
    q["visc_invalide_off"] = q["visc_impossible"] & off
    q["visc_anomalie_on"] = q["visc_impossible"] & ~off
    present = [c for c in SENSORS if c in df.columns]
    q["n_manquants"] = df[present].isna().sum(axis=1)
    q["trop_de_manquants"] = q["n_manquants"] > len(present) * 0.5
    q["statut"] = "valide"
    q.loc[q["visc_invalide_off"], "statut"] = "degradee"
    q.loc[q["trop_de_manquants"] | q["visc_anomalie_on"], "statut"] = "rejetee"
    q["motif"] = np.where(q["visc_anomalie_on"], "viscosité impossible en marche",
                 np.where(q["trop_de_manquants"], "trop de capteurs manquants",
                 np.where(q["visc_invalide_off"], "viscosité invalide à l'arrêt", "")))
    return q


# ------------------------------------------------------- indice systeme (helpers)
def _fiable(m: dict) -> bool:
    """Un horizon n'est declare fiable que s'il bat la persistance sur les DEUX
    criteres :
      - skill = 1 - MSE_modele / MSE_persistance  > 0   (moins de grosses erreurs)
      - accuracy dans la bande de tolerance >= celle de la persistance

    Les deux ne vont pas toujours ensemble : sur l'indice systeme a 10 min, le
    modele reduit l'erreur quadratique (skill +0,11) tout en tombant MOINS
    souvent dans la bande ±0,01 que « rien ne change » (0,41 contre 0,50).
    Afficher « fiable » sur le seul skill ferait passer pour une prevision
    utilisable quelque chose qui degrade la precision utile a la decision."""
    if not m:
        return False
    if (m.get("skill") or 0) <= 0:
        return False
    acc, acc_p = m.get("test_acc"), m.get("acc_persist")
    if acc is None or acc_p is None:
        return False
    return acc >= acc_p


# ═══════════════════════════════════════════ NOTIFICATIONS ET ACTIONS
# Chaque regle produit une notification ET les actions correctives qui vont avec.
# Les deux sont ecrites ensemble : une alerte sans action a faire est une alerte
# qu'on apprend a ignorer.
NIVEAUX = {"critique": 3, "alerte": 2, "info": 1}


def _seuil_fraicheur(sub_tri) -> float:
    """Seuil de peremption, CALIBRE SUR LA CADENCE REELLE de la source.

    Une constante de 30 minutes suppose une mesure toutes les 10 minutes. C'est
    la cadence de l'historique, pas celle de l'API : en direct, l'intervalle
    median observe est de 100 min sur la Motosoufflante A et 80 min sur la B.
    Avec 30 minutes, 76 % et 71 % des intervalles declenchent l'alerte — une
    alerte qui se declenche trois fois sur quatre n'informe plus, elle apprend
    a etre ignoree.

    Le seuil retenu est le 90e PERCENTILE des intervalles observes : le retard
    que la source elle-meme depasse une fois sur dix. Mesure sur les deux
    machines, il produit un taux d'alerte de 9,6 % et 9,5 % — stable d'une
    machine a l'autre, sans multiplicateur arbitraire a justifier.

    Il est borne des deux cotes : jamais sous STALE_AFTER_MIN, pour ne pas
    devenir plus permissif que la cadence historique ne le justifie ; jamais
    au-dessus de STALE_PLAFOND_MIN, pour qu'un flux mort finisse par alerter
    meme si son historique recent est lacunaire."""
    if "created_at" not in sub_tri or len(sub_tri) < 6:
        return float(STALE_AFTER_MIN)
    t = sub_tri["created_at"].sort_values()
    dts = t.diff().dt.total_seconds().dropna() / 60.0
    dts = dts[dts > 0]
    if len(dts) < 5:
        return float(STALE_AFTER_MIN)
    return float(min(max(dts.quantile(0.90), STALE_AFTER_MIN), STALE_PLAFOND_MIN))


def _notifications(m: dict) -> list[dict]:
    """Problemes detectes sur une machine, et ce qu'on peut y faire.

    Les regles ne lisent que la charge utile deja construite : elles ne
    recalculent rien et ne peuvent donc pas diverger de ce qui est affiche."""
    n = []
    mach = m["machine"]

    def add(ident, niveau, titre, message, actions):
        n.append({"id": f"{ident}", "machine": mach, "niveau": niveau,
                  "titre": titre, "message": message, "actions": actions})

    # --- 1. porte de qualite : la donnee elle-meme est ecartee ---
    # Le motif designe le capteur en cause ; on le NOMME dans l'action plutot que
    # de dire « verifier le capteur », qui laisse l'operateur chercher lequel.
    motif = (m.get("qualite_motif") or "").lower()
    if "viscosit" in motif:
        capteur = f"le VISCOSIMÈTRE de la {mach}"
        preuve = ("Comparer « Viscosity at 40 °C » à « Kinematic Viscosity » au même "
                  "instant : à 40 °C elles décrivent la même grandeur. Si l'une "
                  "s'effondre pendant que l'autre reste normale, le capteur se "
                  "contredit lui-même — c'est lui qu'il faut changer, pas l'huile.")
    elif "capteurs manquants" in motif:
        capteur = f"la CHAÎNE D'ACQUISITION de la {mach}"
        preuve = ("Un trou SYNCHRONE — plusieurs capteurs muets sur les mêmes lignes "
                  "— désigne une coupure d'acquisition ou une carte électronique, "
                  "pas une panne de capteur isolée.")
    else:
        capteur = f"le capteur mis en cause sur la {mach}"
        preuve = "Confronter la mesure suspecte aux variables voisines."

    if m.get("qualite") == "rejetee":
        add("qualite_rejetee", "critique",
            "Donnée rejetée",
            f"{m.get('qualite_motif') or 'contrôle de qualité négatif'}. L'indice "
            "affiché repose sur une mesure que le contrôle de qualité écarte : "
            "à vérifier avant toute décision d'intervention.",
            [f"VÉRIFIER {capteur} — c'est l'action prioritaire.",
             "Ne déclencher aucune intervention mécanique ni vidange sur la seule "
             "foi de cet indice.",
             preuve,
             "Demander un prélèvement en laboratoire pour trancher entre défaut "
             "capteur et dégradation réelle.",
             "Signaler l'anomalie à l'équipe i-SENSE avec l'horodatage et la valeur."])
    elif m.get("qualite") == "degradee":
        add("qualite_degradee", "alerte",
            "Donnée dégradée",
            f"{m.get('qualite_motif') or 'qualité partielle'}. L'indice reste "
            "calculé mais sur une base incomplète.",
            [f"VÉRIFIER {capteur}.",
             "Traiter l'indice comme indicatif jusqu'au retour d'une donnée valide.",
             preuve])

    # --- 2. fraicheur : on affiche un passe qui n'est plus l'etat courant ---
    if m.get("perimee"):
        seuil = m.get("seuil_fraicheur_min") or STALE_AFTER_MIN
        add("perimee", "alerte",
            "Donnée périmée",
            f"Dernière mesure il y a {m.get('fraicheur_min', 0):.0f} minutes, "
            f"au-delà du seuil de {seuil:.0f} minutes. Ce seuil est le 90e "
            "percentile des intervalles réellement observés sur cette machine : "
            "le retard n'est pas seulement long, il est inhabituel POUR CETTE "
            "SOURCE. L'écran montre un état passé, pas l'état courant.",
            ["Vérifier que l'acquisition i-SENSE fonctionne pour cette machine.",
             "Contrôler l'état du flux : point d'entrée /ingestion.",
             "Si la machine est à l'arrêt, l'espacement des mesures est normal."])

    # --- 3. etat de l'huile ---
    if m.get("etat") == "Alarme":
        add("huile_alarme", "critique",
            "Health Index huile en alarme",
            f"Indice à {m.get('health_index', 0):.4f}, sous le seuil de "
            f"{SEUIL_ALARME:.2f}.",
            ["Vérifier d'abord la qualité de la donnée : une alarme sur une mesure "
             "rejetée est un défaut de capteur, pas une dégradation d'huile.",
             "Examiner la tendance : une chute brutale oriente vers le capteur, "
             "une dérive lente vers une vraie dégradation.",
             "Programmer une analyse d'huile en laboratoire."])
    elif m.get("etat") == "Surveillance":
        add("huile_surveillance", "alerte",
            "Health Index huile en surveillance",
            f"Indice à {m.get('health_index', 0):.4f}, entre surveillance et alarme.",
            ["Suivre la tendance sur les prochains relevés.",
             "Pas d'intervention immédiate justifiée à ce stade."])

    # --- 4. etat du systeme, canal par canal, contre les seuils du pipeline ---
    # Les zones A/B/C/D ont disparu avec l'indice normalise : on compare
    # maintenant la MESURE a son seuil, et le message porte les deux nombres.
    for c in (m.get("systeme") or {}).get("canaux", []):
        et, nom = c.get("etat"), c.get("nom")
        if et == "Mesure impossible":
            add(f"systeme_{nom}_impossible", "critique",
                f"{c['libelle']} — mesure hors borne physique",
                f"{c.get('motif', '')}. Une valeur au-delà de la borne physique ne "
                "décrit plus la machine, elle décrit le capteur.",
                [f"VÉRIFIER le capteur de vibration de la {mach} — action prioritaire.",
                 "Ne déclencher aucune intervention mécanique sur la foi de cette valeur.",
                 f"Borne physique {c.get('borne_physique')} {c.get('unite', '')}, "
                 "fixée dans data_quality_isense.py (PHYSICAL_RANGES).",
                 "Confronter la mesure au relevé vibratoire portatif du service maintenance."])
        elif et == "Alarme" and nom == "vibration":
            add("systeme_vibration", "critique",
                f"{c['libelle']} — seuil franchi",
                f"{c.get('motif', '')}. Organe concerné : {c.get('organe', '')}.",
                ["Inspecter la mécanique tournante : balourd, alignement, "
                 "état des roulements.",
                 "Vérifier que la mesure est réelle et non reconstruite par "
                 "imputation avant de programmer une intervention.",
                 f"Règle appliquée : {c.get('regle', '')} "
                 f"(seuil indicatif du traitement de données, pas une norme constructeur).",
                 "Comparer à l'autre motosoufflante : un dépassement sur les deux "
                 "oriente vers la mesure, sur une seule vers la machine."])
        elif et == "Alarme":
            add(f"systeme_{nom}", "critique",
                f"{c['libelle']} — seuil franchi",
                f"{c.get('motif', '')}. Organe concerné : {c.get('organe', '')}.",
                ["Inspecter le circuit hydraulique : pompe, filtre, fuite éventuelle.",
                 f"Règle appliquée : {c.get('regle', '')}."])

    # --- 5. machine a l'arret : rien d'anormal, mais l'ecran doit le dire ---
    if m.get("etat_machine") == "OFF":
        add("machine_arret", "info",
            "Machine à l'arrêt",
            "Les indicateurs système ne sont pas évalués : sans pression ni "
            "vibration, ils mesureraient l'arrêt et non l'état de la machine.",
            ["Aucune action. L'état système redeviendra évaluable au redémarrage."])

    # --- 6. prevision indisponible ---
    k = m.get("kpis") or {}
    if k.get("n_horizons_fiables") == 0:
        add("aucune_prevision", "info",
            "Aucune prévision fiable",
            "Aucun horizon ne bat la persistance sur les deux critères retenus. "
            "Les valeurs affichées sont celles de la persistance.",
            ["S'appuyer sur la tendance mesurée plutôt que sur la prévision.",
             "Un réentraînement sur données récentes peut restaurer la fiabilité."])

    return sorted(n, key=lambda x: -NIVEAUX.get(x["niveau"], 0))


def _etat_machine(sub_tri, last):
    """Bloc « etat machine », derive de la PRESSION D'HUILE.

    Renvoie l'etat courant, la pression qui le justifie, le seuil applique, la
    duree passee dans cet etat et le nombre de basculements. Exposer la pression
    ET le seuil permet de verifier la decision au lieu de la croire."""
    pression = _val(last, "Oil Pressure")
    off = int(last.get("state_OFF", pd.Series([0])).iloc[0]) == 1
    etat = "OFF" if off else "ON"

    n_bascule = None
    depuis_min = None
    if "state_OFF" in sub_tri and len(sub_tri):
        marche = (sub_tri["state_OFF"] == 0).to_numpy()
        n_bascule = int((sub_tri["state_OFF"].diff().abs() == 1).sum())
        # Remonter tant que l'etat ne change pas : duree dans l'etat courant.
        courant = marche[-1]
        n_meme = 1
        while n_meme < len(marche) and marche[-1 - n_meme] == courant:
            n_meme += 1
        t = sub_tri["created_at"]
        depuis_min = float((t.iloc[-1] - t.iloc[-n_meme]).total_seconds() / 60)

    return {
        "etat": etat,
        "pression_bar": pression,
        "seuil_bar": PRESSION_ON_BAR,
        "depuis_min": depuis_min,
        "n_basculements": n_bascule,
    }


def _val(last, col):
    """Valeur scalaire d'une colonne sur la derniere ligne, None si absente."""
    if col not in last or pd.isna(last[col].iloc[0]):
        return None
    return float(last[col].iloc[0])


def _pente_par_jour(sub, col, heures=24):
    """Pente de `col` sur les dernieres `heures`, en points d'indice par jour.
    Renvoie None si la fenetre est trop courte ou la serie trop creuse."""
    if col not in sub or sub.empty:
        return None
    fin = sub["created_at"].max()
    w = sub[sub["created_at"] >= fin - pd.Timedelta(hours=heures)]
    v = w[["created_at", col]].dropna()
    if len(v) < 10:
        return None
    x = (v["created_at"] - v["created_at"].min()).dt.total_seconds().to_numpy() / 86400.0
    y = v[col].to_numpy(dtype=float)
    if x.max() <= 0:
        return None
    return float(np.polyfit(x, y, 1)[0])


def _kpis(sub, hi, hs, pente_hi, pente_sys, horizons, off_col):
    """Indicateurs d'aide a la decision.

    Tous rapportes au MEME seuil : HI = 0,5. Les deux indices valent
    1 / (1 + severite) et la severite vaut 1 au seuil d'alarme, donc 0,5 est la
    frontiere d'alarme des deux — la marge est directement comparable."""
    k = {"seuil_alarme": SEUIL_ALARME,
         "marge_alarme_huile": float(hi - SEUIL_ALARME),
         "marge_alarme_systeme": (None if hs is None else float(hs - SEUIL_ALARME)),
         "pente_huile_par_jour": pente_hi, "pente_systeme_par_jour": pente_sys}

    # Heures avant de toucher le seuil, en prolongeant la pente observee.
    # Extrapolation lineaire : une aide a la priorisation, pas une prevision.
    #   None  = pas de derive vers le seuil (indice stable ou en amelioration)
    #   0.0   = seuil DEJA franchi — sinon la formule renverrait un delai negatif,
    #           qu'on afficherait comme « -8 h avant alarme ».
    def _heures(v, pente):
        if v is None:
            return None
        if v <= SEUIL_ALARME:
            return 0.0
        if pente is None or pente >= 0:
            return None
        return float((v - SEUIL_ALARME) / (-pente) * 24.0)
    k["heures_avant_alarme_huile"] = _heures(hi, pente_hi)
    k["heures_avant_alarme_systeme"] = _heures(hs, pente_sys)

    # Ce que le modele apporte vraiment : le plus grand ecart a la persistance
    # parmi les horizons juges fiables.
    ecarts = [abs(h["delta"]) for h in horizons if h.get("fiable")]
    k["apport_modele"] = float(max(ecarts)) if ecarts else 0.0
    fiables = [h for h in horizons if h.get("fiable")]
    k["horizon_fiable_max"] = (max(fiables, key=lambda h: h["horizon_pas"])["label"]
                               if fiables else None)
    k["n_horizons_fiables"] = len(fiables)

    # Contexte d'exploitation sur la fenetre.
    if off_col in sub and len(sub):
        k["disponibilite"] = float((sub[off_col] == 0).mean())
    else:
        k["disponibilite"] = None
    if "vibration_confidence_score" in sub and len(sub):
        k["vibration_imputee"] = float((sub["vibration_confidence_score"] > 0).mean())
    else:
        k["vibration_imputee"] = None
    return k


# ------------------------------------------------------------------ PREDICTION
def predict_latest(df: pd.DataFrame, maintenant=None) -> dict:
    """Health index HUILE et etat du systeme, par machine, avec la prevision de
    l'huile aux quatre horizons.

    `maintenant` : instant de reference pour la FRAICHEUR des mesures.

    En REJEU on le laisse a None : le temps est celui des donnees rejouees, et
    l'heure murale n'a aucun rapport avec elles.

    EN DIRECT il faut le passer, et c'est essentiel. Le prendre dans les donnees
    — df["created_at"].max() — rend l'indicateur de peremption AVEUGLE A LA
    PANNE QU'IL DOIT DETECTER : si toute l'acquisition s'arrete, la reference se
    fige avec elle et chaque machine annonce une fraicheur de zero minute. Le
    cas s'est produit, avec des mesures vieilles de plus de vingt-quatre heures
    affichees comme instantanees."""
    art = load_artifacts()
    models, idx = art["models"], art["metrics_idx"]
    q = quality_gate(df)
    now = pd.Timestamp(maintenant) if maintenant is not None else df["created_at"].max()
    # Les horodatages du jeu sont en UTC naif : aligner pour que la soustraction
    # ait un sens.
    if getattr(now, "tzinfo", None) is not None:
        now = now.tz_convert("UTC").tz_localize(None)
    out = []
    for asset, col in ASSET_COLS.items():
        sub = df[df[col] == 1]
        if sub.empty:
            continue
        last = sub.iloc[[-1]]
        i = last.index[0]
        hi = float(last["health_index"].iloc[0])
        t = last["created_at"].iloc[0]
        horizons = []
        for h in HORIZONS:
            key = f"{h}|{asset}"
            if key not in models:
                continue
            b = models[key]
            X = last[b["colonnes"]]
            delta = float(b["regresseur"].predict(X)[0])
            prob = float(b["porte"].predict_proba(X)[:, 1][0])
            gate = prob > b["seuil"]
            pred = float(np.clip(hi + b["alpha"] * delta * gate, 0, 1))
            m = idx.get((h, asset), {})
            fiable = _fiable(m)
            horizons.append({
                "horizon_pas": h, "label": b["horizon_label"],
                "cible_horodatage": (t + pd.Timedelta(minutes=STEP_MIN * h)).isoformat(),
                "predit": pred if fiable else hi,
                "persistance": hi,
                "delta": (pred - hi) if fiable else 0.0,
                "proba_mouvement": prob, "porte_ouverte": bool(gate),
                "fiable": bool(fiable),
                "skill_historique": m.get("skill"), "r2_historique": m.get("r2"),
                "acc_historique": m.get("test_acc"),
                "acc_persistance": m.get("acc_persist"),
                "modele": "Lasso + porte" if fiable else "aucune prévision fiable",
            })
        staleness = float((now - t).total_seconds() / 60)
        a_l_arret = int(last.get("state_OFF", pd.Series([0])).iloc[0]) == 1

        # -------------------------------------------- etat du systeme, par canal
        # DEUX indicateurs independants, lus DIRECTEMENT SUR LA MESURE contre les
        # seuils fixes pendant le traitement de donnees. L'indice normalise et ses
        # zones A/B/C/D ont ete abandonnes : ils empruntaient le seuil d'alarme du
        # Health Index huile, un nombre sans unite, pour juger des bar et des
        # mm/s2. Un operateur lit maintenant « 5,40 bar contre un seuil de marche
        # a 0,10 » au lieu de « indice 0,83, zone A » — la meme information, mais
        # verifiable sur le capteur.
        p = _val(last, "Oil Pressure")
        v = _val(last, "Oil System Vibration_filled")

        if p is None:
            etat_p, motif_p = NON_EVALUABLE, "pression non mesurée"
        elif p > SEUIL_PRESSION_MARCHE:
            etat_p = "Normal"
            motif_p = (f"circuit en pression — {p:.2f} bar au-dessus du seuil de "
                       f"marche {SEUIL_PRESSION_MARCHE:.2f} bar")
        else:
            etat_p = "À l'arrêt"
            motif_p = (f"pression quasi nulle — {p:.2f} bar sous le seuil de marche "
                       f"{SEUIL_PRESSION_MARCHE:.2f} bar")

        if a_l_arret:
            etat_v = NON_EVALUABLE
            motif_v = "machine à l'arrêt — la vibration ne porte aucun signal mécanique"
        elif v is None:
            etat_v, motif_v = NON_EVALUABLE, "vibration non mesurée"
        elif v > BORNE_VIBRATION_PHYSIQUE:
            etat_v = "Mesure impossible"
            motif_v = (f"{v:.3f} au-delà de la borne physique "
                       f"{BORNE_VIBRATION_PHYSIQUE:.1f} — la valeur décrit le capteur, "
                       "plus la machine")
        elif v > SEUIL_VIBRATION_ALARME:
            etat_v = "Alarme"
            motif_v = (f"{v:.3f} au-dessus du seuil de vibration élevée "
                       f"{SEUIL_VIBRATION_ALARME:.1f}")
        else:
            etat_v = "Normal"
            motif_v = (f"{v:.3f} sous le seuil de vibration élevée "
                       f"{SEUIL_VIBRATION_ALARME:.1f}")

        canaux = [
            {"nom": "pression", "libelle": "Pression d'huile",
             "valeur": p, "unite": "bar",
             "seuil": SEUIL_PRESSION_MARCHE, "sens": "bas", "borne_physique": None,
             "etat": etat_p, "motif": motif_p,
             "regle": f"machine en marche si Oil Pressure > {SEUIL_PRESSION_MARCHE:.2f} bar",
             "source": "clean_isense_data.py · data_quality_isense.py",
             "organe": "circuit hydraulique — pompe, colmatage, fuite"},
            {"nom": "vibration", "libelle": "Vibration du système d'huile",
             "valeur": v, "unite": "mm/s²",
             "seuil": SEUIL_VIBRATION_ALARME, "sens": "haut",
             "borne_physique": BORNE_VIBRATION_PHYSIQUE,
             "etat": etat_v, "motif": motif_v,
             "regle": f"alarme si Oil System Vibration > {SEUIL_VIBRATION_ALARME:.1f} mm/s²",
             "source": "feature_engineering.py · flag_high_vibration",
             "organe": "mécanique tournante — balourd, désalignement, roulement"},
        ]
        systeme = {
            "evaluable": not a_l_arret,
            "canaux": canaux,
            "motif": ("machine à l'arrêt — seule la pression reste lisible ; la "
                      "vibration d'une machine immobile ne mesure pas son état"
                      if a_l_arret else ""),
        }

        sub_tri = sub.sort_values("created_at")
        seuil_frais = _seuil_fraicheur(sub_tri)
        kpis = _kpis(sub_tri, hi, None,
                     _pente_par_jour(sub_tri, "health_index"), None,
                     horizons, "state_OFF")
        kpis["qualite_valide"] = float((quality_gate(sub_tri)["statut"] == "valide").mean())

        out.append({
            "machine": asset, "horodatage": t.isoformat(),
            "health_index": hi, "etat": str(last["health_state"].iloc[0]),
            "etat_machine": "OFF" if a_l_arret else "ON",
            "fraicheur_min": staleness, "perimee": staleness > seuil_frais,
            "seuil_fraicheur_min": seuil_frais,
            "qualite": str(q.loc[i, "statut"]), "qualite_motif": str(q.loc[i, "motif"]),
            "horizons": horizons,
            "systeme": systeme,
            "etat_machine_detail": _etat_machine(sub_tri, last),
            "kpis": kpis,
        })
        out[-1]["notifications"] = _notifications(out[-1])
    return {"maintenant": now.isoformat(), "machines": out, "tolerance": TOLERANCE}


def history_series(df: pd.DataFrame, heures: int = 48, n_points: int = 400,
                   jours_systeme: int | None = None) -> dict:
    """Serie OBSERVEE recente, par machine — les deux index calcules.

    Le tableau de bord est un ecran de supervision : il montre ce qui vient de se
    passer, pas sept mois d'archives. On ne renvoie donc que les `heures`
    dernieres heures de mesure de CHAQUE machine (chacune a sa propre derniere
    mesure : elles n'acquierent pas en meme temps).

    DEUX FENETRES, parce que les deux index ne vivent pas au meme rythme :
      - index HUILE   : les `heures` dernieres heures de mesure. La chimie de
                        l'huile evolue meme machine arretee, la serie est donc
                        toujours renseignee.
      - index SYSTEME : TOUT l'historique de fonctionnement de la machine
                        (`jours_systeme=None`), sous-echantillonne a `n_points`.
                        L'indice n'existe qu'en marche ; sur une machine arretee
                        depuis des semaines, une fenetre calendaire de 48 h ne
                        contiendrait aucun point. On parcourt donc ses periodes
                        de marche en ignorant les arrets intercales, ce qui donne
                        une courbe continue sur tous les mois d'exploitation.

    La PREVISION n'est pas dans cette serie. Elle est portee par /predict, qui
    donne pour chaque horizon l'horodatage cible et la valeur predite : le
    tableau de bord prolonge la courbe vers le futur a partir de la. Tracer une
    prevision a l'instant ou elle a ete faite, comme le faisait la version
    precedente, melangeait passe et futur sur le meme point.
    """
    series = {}
    for asset, col in ASSET_COLS.items():
        sub = df[df[col] == 1].sort_values("created_at")
        if len(sub) < 2:
            series[asset] = []
            continue
        fin_t = sub["created_at"].max()
        w = sub[sub["created_at"] >= fin_t - pd.Timedelta(hours=heures)]
        if len(w) < 10:                       # machine muette : on garde le dernier bout
            w = sub.tail(min(len(sub), n_points))
        if len(w) > n_points:                 # densite raisonnable pour le trace
            w = w.iloc[:: max(1, len(w) // n_points)]
        hi = w["health_index"].to_numpy(dtype=float)

        series[asset] = [
            {"t": a.isoformat(), "calcule": float(v), "persistance": float(v),
             "etat": str(e), "marche": int(o) == 0}
            for a, v, e, o in zip(w["created_at"], hi, w["health_state"],
                                  w.get("state_OFF", pd.Series(0, index=w.index)))]

    # --- fenetre propre a l'ETAT DU SYSTEME : uniquement les lignes en marche ---
    # Deux series independantes par machine, une par canal, EN UNITES REELLES :
    # la pression en bar, la vibration en mm/s2. L'indice normalise 0-1 a ete
    # abandonne — il forcait deux grandeurs physiques sur une echelle sans unite,
    # ou l'on ne pouvait plus rien verifier sur un capteur.
    series_sys = {}
    for asset, col in ASSET_COLS.items():
        # En marche : c'est le seul critere qui vaille maintenant que l'indice ne
        # decide plus rien. Une machine arretee a une pression nulle et une
        # vibration muette, les tracer reviendrait a tracer l'arret.
        off = df.get("state_OFF", pd.Series(0, index=df.index)) == 1
        marche = df[(df[col] == 1) & (~off)].sort_values("created_at")
        if len(marche) < 2:
            series_sys[asset] = []
            continue
        if jours_systeme is None:
            w = marche                       # tout l'historique de fonctionnement
        else:
            fin_t = marche["created_at"].max()
            w = marche[marche["created_at"] >= fin_t - pd.Timedelta(days=jours_systeme)]
        if len(w) > n_points:
            w = w.iloc[:: max(1, len(w) // n_points)]
        def _col(nom, defaut=np.nan):
            return w[nom] if nom in w else pd.Series(defaut, index=w.index)

        series_sys[asset] = [
            {"t": a.isoformat(),
             "pression": (None if not np.isfinite(pp) else float(pp)),
             "vibration": (None if not np.isfinite(vv) else float(vv))}
            for a, pp, vv in zip(
                w["created_at"],
                pd.to_numeric(_col("Oil Pressure"), errors="coerce"),
                pd.to_numeric(_col("Oil System Vibration_filled"), errors="coerce"))]

    # --- VISCOSITE : valeurs MESUREES, meme fenetre que l'index huile ---
    # La viscosite a 40 °C est la grandeur directement comparable au grade
    # ISO VG 46 ; la cinematique et la dynamique l'accompagnent pour que l'on
    # puisse verifier sa coherence au survol (une Vis_40 a 19 cSt a cote d'une
    # cinematique a 71 cSt a 32 °C est un defaut de capteur, pas une huile).
    # On trace la mesure brute, y compris les valeurs impossibles : les masquer
    # cacherait precisement l'anomalie que la courbe doit montrer.
    series_visc = {}
    for asset, col in ASSET_COLS.items():
        sub = df[df[col] == 1].sort_values("created_at")
        if len(sub) < 2:
            series_visc[asset] = []
            continue
        fin_t = sub["created_at"].max()
        w = sub[sub["created_at"] >= fin_t - pd.Timedelta(hours=heures)]
        if len(w) < 10:
            w = sub.tail(min(len(sub), n_points))
        if len(w) > n_points:
            w = w.iloc[:: max(1, len(w) // n_points)]

        def _num(nom):
            s = w[nom] if nom in w else pd.Series(np.nan, index=w.index)
            return pd.to_numeric(s, errors="coerce")

        def _f(x):
            return None if not np.isfinite(x) else round(float(x), 2)

        series_visc[asset] = [
            {"t": a.isoformat(), "vis40": _f(v4), "cinematique": _f(k),
             "dynamique": _f(d), "marche": int(o) == 0}
            for a, v4, k, d, o in zip(
                w["created_at"], _num(VISC_COL), _num("Kinematic Viscosity_filled"),
                _num("Dynamic Viscosity_filled"),
                w.get("state_OFF", pd.Series(0, index=w.index)))]

    return {"series": series, "fenetre_heures": heures,
            "series_viscosite": series_visc,
            # Seuils de Vis_40 : Cadrage des seuils OCP, section 12 (reference
            # TD46 46,0 cSt). Le plancher de 20 cSt n'est PAS dans le cadrage :
            # c'est la regle « mesure impossible » du projet (VISC_MIN).
            "seuils_viscosite": {"reference": 46.0,
                                 "normal": [43.7, 48.3],
                                 "surveillance": [41.4, 50.6],
                                 "impossible_projet": VISC_MIN,
                                 "unite": "cSt",
                                 "source": "Cadrage des seuils OCP, section 12 (Vis_40)"},
            "series_systeme": series_sys, "fenetre_systeme_jours": jours_systeme,
            # Unites servies au trace : chaque canal a la sienne, donc chaque
            # canal a son propre axe. Les superposer sur un axe commun ferait
            # lire « 5,4 » et « 0,15 » comme deux points d'une meme grandeur.
            "unites_systeme": {"pression": "bar", "vibration": "mm/s²"},
            # Seuils traces sur les courbes. Servis par le backend et non codes
            # dans le frontend : une valeur ecrite a deux endroits finit par
            # differer, et c'est l'affichage qui ment alors sur la regle.
            "seuils_systeme": {"pression": SEUIL_PRESSION_MARCHE,
                               "vibration": SEUIL_VIBRATION_ALARME},
            "libelles_seuils": {"pression": "seuil de marche",
                                "vibration": "seuil d'alarme"}}
            # fenetre_systeme_jours = None -> tout l'historique de fonctionnement


# --------------------------------------------------------------------- QUALITE
def quality_report(df: pd.DataFrame) -> dict:
    q = quality_gate(df)
    present = [c for c in SENSORS if c in df.columns]
    off = df["state_OFF"] == 1 if "state_OFF" in df else pd.Series(False, index=df.index)
    out = {"n_lignes": int(len(df))}

    # --- 1. etat machine : arret / marche ---
    etats = []
    for asset, col in ASSET_COLS.items():
        m = df[col] == 1
        if m.sum() == 0:
            continue
        etats.append({"machine": asset, "n": int(m.sum()),
                      "marche": int((m & ~off).sum()), "arret": int((m & off).sum()),
                      "pct_marche": float((m & ~off).sum() / m.sum())})
    # basculements dans le temps
    d2 = df.sort_values("created_at")
    bascule = int((d2["state_OFF"].diff().abs() == 1).sum()) if "state_OFF" in d2 else 0
    out["etat_machine"] = {"par_machine": etats, "n_basculements": bascule}

    # --- 2. densite a 15 C : doit etre quasi constante ---
    dens = []
    if "density_15C" in df:
        for asset, col in ASSET_COLS.items():
            v = df.loc[df[col] == 1, "density_15C"].dropna()
            if len(v) < 5:
                continue
            dens.append({"machine": asset, "n": int(len(v)), "moyenne": float(v.mean()),
                         "ecart_type": float(v.std()), "min": float(v.min()),
                         "max": float(v.max()),
                         "ecart_baseline_pct": float((v.mean() - DENSITY_BASELINE)
                                                     / DENSITY_BASELINE * 100)})
    out["densite_15C"] = {"baseline": DENSITY_BASELINE, "par_machine": dens,
                          "nuage": [{"t": a.isoformat(), "v": float(b),
                                     "machine": asset_of(r)}
                                    for (_, r), a, b in zip(
                                        df.iloc[::max(1, len(df)//400)].iterrows(),
                                        df["created_at"].iloc[::max(1, len(df)//400)],
                                        df["density_15C"].iloc[::max(1, len(df)//400)])
                                    if np.isfinite(b)]}

    # --- 3. coherence H2O ppm vs saturation ---
    h2o = []
    if "Oil H2O ppm_filled" in df and "Oil H2O Saturation_filled" in df:
        for asset, col in ASSET_COLS.items():
            s = df[df[col] == 1][["Oil H2O ppm_filled", "Oil H2O Saturation_filled"]].dropna()
            if len(s) < 10:
                continue
            r = float(s.corr().iloc[0, 1])
            h2o.append({"machine": asset, "n": int(len(s)), "correlation": r,
                        "coherent": bool(r > 0.5)})
        st = max(1, len(df) // 500)
        out["h2o"] = {"par_machine": h2o,
                      "nuage": [{"ppm": float(a), "sat": float(b), "machine": asset_of(r)}
                                for (_, r), a, b in zip(
                                    df.iloc[::st].iterrows(),
                                    df["Oil H2O ppm_filled"].iloc[::st],
                                    df["Oil H2O Saturation_filled"].iloc[::st])
                                if np.isfinite(a) and np.isfinite(b)]}

    # --- 4. distribution de DC par machine ---
    dc = []
    if "DC_filled" in df:
        for asset, col in ASSET_COLS.items():
            v = df.loc[df[col] == 1, "DC_filled"].dropna()
            if len(v) < 5:
                continue
            hist, edges = np.histogram(v, bins=24)
            dc.append({"machine": asset, "n": int(len(v)), "mediane": float(v.median()),
                       "p05": float(v.quantile(.05)), "p95": float(v.quantile(.95)),
                       "hist": [{"x": float((edges[i] + edges[i+1]) / 2),
                                 "n": int(hist[i])} for i in range(len(hist))]})
    out["dc"] = {"par_machine": dc,
                 "note": "Sens de dégradation contradictoire entre le tableau de seuils "
                         "(DC élevée = critique) et les diapositives i-SENSE — point ouvert."}

    # --- 5. coherence physique mu = nu x rho (ASTM) ---
    coh = []
    need = ["Dynamic Viscosity_filled", "Kinematic Viscosity_filled", "Density_filled"]
    if all(c in df for c in need):
        attendu = df["Kinematic Viscosity_filled"] * df["Density_filled"] / 1000.0
        err = (df["Dynamic Viscosity_filled"] - attendu)
        rel = (err.abs() / attendu.replace(0, np.nan)) * 100
        for asset, col in ASSET_COLS.items():
            for st_, lab in [(0, "ON"), (1, "OFF")]:
                m = (df[col] == 1) & (off == bool(st_))
                if m.sum() < 10:
                    continue
                coh.append({"machine": asset, "etat": lab, "n": int(m.sum()),
                            "erreur_relative_mediane": float(rel[m].median()),
                            "pct_hors_5pct": float((rel[m] > 5).mean())})
        st = max(1, len(df) // 500)
        out["coherence_viscosite"] = {
            "formule": "μ [mPa·s] = ν [cSt] × ρ [kg/m³] / 1000",
            "par_groupe": coh,
            "nuage": [{"attendu": float(a), "mesure": float(b), "machine": asset_of(r),
                       "etat": "OFF" if int(o) == 1 else "ON"}
                      for (_, r), a, b, o in zip(
                          df.iloc[::st].iterrows(), attendu.iloc[::st],
                          df["Dynamic Viscosity_filled"].iloc[::st],
                          df.get("state_OFF", pd.Series(0, index=df.index)).iloc[::st])
                      if np.isfinite(a) and np.isfinite(b)]}

    # --- 6. monotonie des codes ISO 4406 : ISO4 >= ISO6 >= ISO14 ---
    iso = []
    if all(c in df for c in ["ISO 4_filled", "ISO 6", "ISO 14"]):
        ok46 = df["ISO 4_filled"] >= df["ISO 6"]
        ok614 = df["ISO 6"] >= df["ISO 14"]
        for asset, col in ASSET_COLS.items():
            m = df[col] == 1
            if m.sum() == 0:
                continue
            iso.append({"machine": asset, "n": int(m.sum()),
                        "pct_ISO4_ge_ISO6": float(ok46[m].mean()),
                        "pct_ISO6_ge_ISO14": float(ok614[m].mean()),
                        "pct_monotone": float((ok46 & ok614)[m].mean()),
                        "med_ISO4": float(df.loc[m, "ISO 4_filled"].median()),
                        "med_ISO6": float(df.loc[m, "ISO 6"].median()),
                        "med_ISO14": float(df.loc[m, "ISO 14"].median())})
    out["iso_monotonie"] = {"regle": "ISO 4 ≥ ISO 6 ≥ ISO 14 (plus la particule est grosse, "
                                     "moins elle est fréquente)", "par_machine": iso}

    # --- porte de qualite + manquants + viscosite ---
    out["manquants"] = sorted([{"capteur": c, "taux": float(df[c].isna().mean()),
                                "n": int(df[c].isna().sum())} for c in present],
                              key=lambda x: -x["taux"])
    grp = []
    for asset, col in ASSET_COLS.items():
        for st_, lab in [(0, "ON"), (1, "OFF")]:
            m = (df[col] == 1) & (off == bool(st_))
            if m.sum() == 0:
                continue
            grp.append({"machine": asset, "etat": lab, "n": int(m.sum()),
                        "impossible": int((q["visc_impossible"] & m).sum()),
                        "pct": float((q["visc_impossible"] & m).mean()),
                        "mediane": float(df.loc[m, VISC_COL].median())})
    out["viscosite"] = {"seuil": VISC_MIN, "reference": 46.0, "par_groupe": grp,
                        "n_impossible": int(q["visc_impossible"].sum())}
    out["porte"] = {"valide": int((q["statut"] == "valide").sum()),
                    "degradee": int((q["statut"] == "degradee").sum()),
                    "rejetee": int((q["statut"] == "rejetee").sum()),
                    "motifs": q.loc[q["motif"] != "", "motif"].value_counts().to_dict()}
    return out


# ------------------------------------------------------------------ DIAGNOSTIC
def diagnostics_report(df: pd.DataFrame) -> dict:
    out = {}
    off = df["state_OFF"] == 1 if "state_OFF" in df else pd.Series(False, index=df.index)

    # --- 1. imputation de la vibration ---
    vib = {}
    if "Oil System Vibration" in df and "Oil System Vibration_filled" in df:
        brut = df["Oil System Vibration"]
        vib["taux_manquant_brut"] = float(brut.isna().mean())
        vib["n_mesure"] = int(brut.notna().sum())
        vib["n_impute"] = int(brut.isna().sum())
        src = []
        for c, lab in [("vibsource_mesuré", "mesuré"),
                       ("vibsource_imputé_zero_off", "imputé 0 (arrêt)"),
                       ("vibsource_imputé_rf_on", "imputé RF (marche)")]:
            if c in df:
                src.append({"source": lab, "n": int(df[c].sum()),
                            "pct": float(df[c].mean())})
        vib["sources"] = src
        cmp_ = []
        for asset, col in ASSET_COLS.items():
            for st_, lab in [(0, "ON"), (1, "OFF")]:
                m = (df[col] == 1) & (off == bool(st_))
                if m.sum() < 5:
                    continue
                cmp_.append({"machine": asset, "etat": lab, "n": int(m.sum()),
                             "mediane_remplie": float(df.loc[m, "Oil System Vibration_filled"].median()),
                             "pct_impute": float(brut[m].isna().mean())})
        vib["par_groupe"] = cmp_
        st = max(1, len(df) // 500)
        vib["serie"] = [{"t": a.isoformat(), "v": float(b), "impute": bool(pd.isna(c))}
                        for a, b, c in zip(df["created_at"].iloc[::st],
                                           df["Oil System Vibration_filled"].iloc[::st],
                                           brut.iloc[::st]) if np.isfinite(b)]
    out["vibration"] = vib

    # --- 2. synchronisme des capteurs ---
    present = [c for c in SENSORS if c in df.columns]
    miss = df[present].isna()
    n_miss_par_ligne = miss.sum(axis=1)
    out["synchronisme"] = {
        "capteurs": len(present),
        "lignes_completes": int((n_miss_par_ligne == 0).sum()),
        "lignes_partielles": int(((n_miss_par_ligne > 0) & (n_miss_par_ligne < len(present))).sum()),
        "lignes_vides": int((n_miss_par_ligne == len(present)).sum()),
        "distribution": [{"n_manquants": int(k), "n_lignes": int(v)}
                         for k, v in n_miss_par_ligne.value_counts().sort_index().items()],
        "note": "Un trou SYNCHRONE (tous les capteurs manquants sur les mêmes lignes) "
                "indique une coupure d'acquisition, pas une panne de capteur isolée.",
    }

    # --- 3. profils calendaires ---
    cal = {}
    if "hour" in df:
        h = df.groupby("hour").agg(n=("health_index", "size"),
                                   hi=("health_index", "mean")).reset_index()
        cal["par_heure"] = [{"heure": int(r["hour"]), "n": int(r["n"]),
                             "health_index": float(r["hi"])} for _, r in h.iterrows()]
    if "day_of_week" in df:
        JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
        j = df.groupby("day_of_week").agg(n=("health_index", "size"),
                                          hi=("health_index", "mean")).reset_index()
        cal["par_jour"] = [{"jour": JOURS[int(r["day_of_week"])], "n": int(r["n"]),
                            "health_index": float(r["hi"])} for _, r in j.iterrows()]
    out["calendaire"] = cal

    # --- 4. sessions ---
    g = df.groupby("session_id")
    taille = g.size()
    duree = (g["created_at"].max() - g["created_at"].min()).dt.total_seconds() / 3600
    out["sessions"] = {
        "n": int(taille.size), "mediane_lignes": float(taille.median()),
        "p75": float(taille.quantile(.75)), "p95": float(taille.quantile(.95)),
        "max": int(taille.max()), "duree_mediane_h": float(duree.median()),
        "duree_max_h": float(duree.max()),
        "distribution": [{"tranche": str(b), "n": int(v)} for b, v in
                         taille.groupby(pd.cut(taille, [0, 1, 5, 20, 100, 500, 10**6],
                                               labels=["1", "2-5", "6-20", "21-100",
                                                       "101-500", "500+"]),
                                        observed=True).size().items()],
        "note": "126 sessions sur 352 ne comptent qu'une seule ligne et ne contribuent "
                "à aucune fenêtre temporelle.",
    }

    # --- 5 et 6. scores de confiance ---
    conf = {}
    if "vibration_confidence_score" in df:
        vc = df["vibration_confidence_score"].value_counts().sort_index()
        LAB = {0: "mesuré", 1: "imputé 0 (arrêt)", 2: "imputé RF (marche)"}
        conf["vibration"] = [{"niveau": LAB.get(int(k), str(k)), "score": int(k),
                              "n": int(v), "pct": float(v / len(df))}
                             for k, v in vc.items()]
    if "sensor_gap_filled" in df:
        n = int(df["sensor_gap_filled"].sum())
        conf["capteur"] = {"comble": n, "mesure": int(len(df) - n),
                           "pct_comble": float(n / max(len(df), 1)),
                           "note": "DC, ISO 4, H2O, température et viscosité partagent le "
                                   "MÊME trou de 994 lignes — comblé par forward-fill puis "
                                   "Random Forest."}
    if "vibration_gap_distance" in df:
        vg = df["vibration_gap_distance"]
        conf["distance_trou_vibration"] = {
            "mediane": float(vg.median()), "p95": float(vg.quantile(.95)),
            "max": float(vg.max()),
            "note": "Distance à la mesure réelle la plus proche : plus elle est grande, "
                    "moins l'imputation est fiable."}
    out["confiance"] = conf

    out["comparaison_modeles"] = comparaison_modeles()
    return out


# ------------------------------------------------------ COMPARAISON DE MODELES
def comparaison_modeles() -> dict:
    """Classement des familles de modeles pour l'indice SYSTEME (run_30).

    La famille est choisie par validation croisee purgee sur l'entrainement ; le
    test n'est touche qu'une fois. On publie le classement COMPLET pour que le
    choix soit verifiable plutot qu'affirme — y compris les cas ou la famille
    gagnante en CV n'est pas celle qui gagne sur le test.

    Sortie legere (quelques kilo-octets) : elle a son propre point d'entree pour
    que le tableau de bord n'ait pas a charger tout le rapport de diagnostic.
    """
    art = load_artifacts()
    return {
        "leaderboard": art["sys_leaderboard"],
        "calibrage": art["sys_calibrage"],
        "note": "Familles comparées sous le même protocole gelé. La famille retenue "
                "est celle qu'a désignée la validation croisée, jamais le score de "
                "test. Un modèle n'est déclaré fiable que s'il bat la persistance "
                "à la fois en erreur quadratique (skill > 0) et en précision dans "
                "la bande de tolérance.",
    }
