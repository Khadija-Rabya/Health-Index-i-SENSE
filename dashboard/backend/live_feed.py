"""Connecteur TEMPS REEL i-SENSE — la classe `LiveFeed` annoncee comme « a ecrire »
dans ARCHITECTURE_PIPELINE.md, section 7.

Elle expose exactement la MEME INTERFACE que `ReplayFeed` :

    window(n_rows) -> DataFrame     « les lignes traitees dont on dispose maintenant »
    status()       -> dict

Aucun etage en aval ne change : `service.predict_latest`, `history_series`,
l'API FastAPI et le tableau de bord ne savent pas d'ou viennent les lignes.
C'est la propriete qui rend la bascule rejeu -> temps reel sure.

CHAINE PARCOURUE A CHAQUE RELEVE
    1. RELEVE       appel API, format long (une mesure = une ligne)
    2. PIVOT        long -> large : une ligne = un horodatage x une machine
    3. NETTOYAGE    sentinelles -> NaN, Oil Conductivity x10 -> nS/m,
                    etat machine depuis Oil Pressure, rattachement de session
    4. IMPUTATION   vibration : 0 a l'arret, foret FIGEE en marche
    5. INGESTION    FluxIncremental : variables recalculees sur le TAMPON seul,
                    transformations a parametres FIGES
    6. INDICES      health_index et HI_systeme, prets pour la prevision

Les etapes 3 a 6 n'utilisent que des parametres geles dans
`artifacts/pretraitement.joblib`. Rien n'est reajuste au fil de l'eau : c'est la
condition pour que deux lignes identiques recoivent le meme indice, quel que
soit le moment ou elles arrivent.

IDENTIFIANTS — ils se lisent dans l'ENVIRONNEMENT, jamais dans le code :

    $env:ISENSE_EMAIL    = "..."
    $env:ISENSE_PASSWORD = "..."

`Api_to_excel.py` les porte encore en clair (lignes 9-10) : a retirer et a faire
tourner, un mot de passe dans un fichier suivi en version est compromis.

TESTABLE SANS RESEAU — `SimulateurAPI` rejoue le CSV nettoye en se faisant passer
pour l'API. Le chemin complet est donc exercable hors ligne, ce qui permet de
verifier le connecteur sans dependre de la disponibilite du serveur.
"""
from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, ROOT, os.path.join(ROOT, "hi_forecast"), "C:\\pylib"):
    if p not in sys.path:
        sys.path.insert(0, p)

from ingestion import (FluxIncremental, appliquer_imputation_capteurs,    # noqa: E402
                       appliquer_imputation_vibration, charger_pretraitement)

API_BASE_URL = os.environ.get("ISENSE_API",
                              "https://v3back-demo.i-sense.io/api/i-sense-v3")
PERIODE_S = 600                      # cadence d'acquisition i-SENSE : 10 min
OIL_PRESSURE_ON = 0.1                # bar — au-dessus, machine active
GAP_SESSION_H = 1                    # au-dela, nouvelle session
SENTINELLES = (-99.99, -9999.0, -0.09999)


def normaliser_temps(serie: pd.Series) -> pd.Series:
    """Ramene une colonne d'horodatages a des instants NAIFS en UTC, quel que
    soit son point de depart : texte ISO, datetime naif, ou datetime avec fuseau.

    `np.issubdtype` ne sait pas interpreter 'datetime64[us, UTC]' et leve une
    TypeError : on passe donc par l'API pandas, qui connait les types a fuseau."""
    if not pd.api.types.is_datetime64_any_dtype(serie):
        return horodatage(serie)
    if getattr(serie.dt, "tz", None) is not None:
        return serie.dt.tz_localize(None)
    return serie


def horodatage(serie) -> pd.Series:
    """Convertit les horodatages de l'API en instants NAIFS, en UTC.

    L'API renvoie de l'ISO 8601 suffixe Z — « 2026-09-12T08:59:00.000000Z ».
    Pandas 3 refuse de deviner ce format et leve une ValueError. On l'impose
    donc, puis on RETIRE le fuseau : le jeu historique local porte des instants
    naifs, et comparer un instant avec fuseau a un instant sans echoue. Les deux
    viennent de la meme source, tous deux en UTC : retirer l'etiquette ne
    decale rien."""
    t = pd.to_datetime(serie, format="ISO8601", utc=True)
    return t.dt.tz_localize(None) if hasattr(t, "dt") else t.tz_localize(None)


# ═════════════════════════════════════════════════════ 1. SOURCES
class SourceAPI:
    """Appelle reellement l'API i-SENSE. Identifiants pris dans l'environnement."""

    def __init__(self, jours_initial: int | None = None):
        """`jours_initial` : profondeur demandee quand on ne sait pas encore
        jusqu'ou on a lu. Volontairement COURTE — voir relever()."""
        # .strip() : un espace ou un retour chariot colle en fin de valeur — cas
        # courant d'un copier-coller — fait echouer la validation cote serveur
        # avec un 422 sans rapport apparent avec les identifiants.
        self.email = (os.environ.get("ISENSE_EMAIL") or "").strip()
        self.password = (os.environ.get("ISENSE_PASSWORD") or "").strip()
        # ISENSE_JOURS_AMORCE : profondeur du premier releve quand le serveur
        # demarre sans historique local (hebergement). 3 jours suffisent aux
        # variables temporelles ; bien plus fait tomber l'API de demonstration.
        self.jours_initial = int(jours_initial if jours_initial is not None
                                 else os.environ.get("ISENSE_JOURS_AMORCE", 3))
        self._token = None
        self._muettes: list[str] = []      # variables tombees en erreur au relevé
        # Diagnostic ajoute pour trancher entre panne reseau et absence structurelle :
        # un appel qui REUSSIT mais renvoie 0 ligne n'est pas une erreur (pas d'exception,
        # donc jamais dans `_muettes`) et passait jusqu'ici inapercu. S'il se repete a
        # 100% sur des dizaines de releves pour la meme variable, ce n'est plus du bruit
        # reseau : c'est que ce flux temps reel ne diffuse tout simplement pas cette
        # grandeur, contrairement a l'export historique en masse.
        self._appels: dict[str, int] = {}     # tentatives par "machine/variable"
        self._vides: dict[str, int] = {}      # parmi elles, celles revenues a 0 ligne
        # Echantillon des dernieres valeurs BRUTES recues pour les variables suspectes
        # (celles vues manquantes de facon systematique) : sert a rejouer un point
        # anormal (ex. Motosoufflante B soudain "en marche") sans devoir le deviner.
        self._echantillon: dict[str, list[dict]] = {}
        if not self.email or not self.password:
            raise RuntimeError(
                "Identifiants absents. Definir ISENSE_EMAIL et ISENSE_PASSWORD "
                "dans l'environnement — jamais en dur dans le code.")

    def connecter(self) -> None:
        import requests
        r = requests.post(f"{API_BASE_URL}/auth/login",
                          json={"email": self.email, "password": self.password},
                          headers={"Accept": "application/json",
                                   "Content-Type": "application/json"}, timeout=60)
        if r.status_code == 422:
            # 422 n'est PAS « mauvais mot de passe » (ce serait 401) : le serveur
            # juge le corps de la requete invalide. En pratique c'est presque
            # toujours une valeur tronquee ou vide — le piege classique etant une
            # chaine PowerShell entre guillemets DOUBLES, ou $ declenche une
            # substitution de variable et ampute le mot de passe.
            vide = lambda v: " — VIDE" if not v else ""
            raise RuntimeError("\n".join([
                "Connexion i-SENSE refusee (422 : corps de requete invalide).",
                f"  ISENSE_EMAIL    : {len(self.email)} caracteres{vide(self.email)}",
                f"  ISENSE_PASSWORD : {len(self.password)} caracteres{vide(self.password)}",
                "  422 signifie corps invalide, pas identifiants refuses (401).",
                "  En PowerShell, utiliser Read-Host plutot qu'une affectation :",
                "     $env:ISENSE_PASSWORD = Read-Host 'Mot de passe i-SENSE'",
                "  Entre guillemets DOUBLES, un $ dans le mot de passe est",
                "  interprete comme une variable et la valeur part tronquee.",
                f"  Reponse du serveur : {r.text[:300]}",
            ]))
        r.raise_for_status()
        self._token = r.json()["token"]

    def relever(self, depuis: pd.Timestamp | None) -> pd.DataFrame:
        """Mesures posterieures a `depuis`, au format LONG.

        `fetch_trend` renvoie une LISTE DE DICTIONNAIRES — une mesure par entree,
        avec `variable`, `value` et `created_at` — et c'est l'appelant qui y
        ajoute le nom de la machine. On reprend exactement la boucle de
        Api_to_excel.main() plutot que de la reecrire.

        FENETRE COURTE, ET C'EST ESSENTIEL. La configuration d'ASSETS porte un
        `startDate` au 29 decembre 2025 : elle est faite pour un export massif en
        une fois. Demander neuf mois de mesures a la minute pour chaque variable
        fait TOMBER LE SERVEUR EN 504 (constate). Un flux n'a pas besoin de tout
        cet historique — les tampons sont amorces depuis le jeu local — il ne
        demande que ce qui a pu arriver depuis sa derniere lecture."""
        from datetime import timedelta

        import requests

        import Api_to_excel as api
        if self._token is None:
            self.connecter()

        debut = ((depuis - pd.Timedelta(hours=2)) if depuis is not None
                 else pd.Timestamp.now() - pd.Timedelta(days=self.jours_initial))
        session = requests.Session()
        lignes = []
        # Variables vues manquantes de facon systematique (section 6, diagnostic
        # ajoute pour comprendre le trou synchrone de 4 capteurs sur 14 constate
        # sur le flux temps reel) : on garde un echantillon brut de celles-ci,
        # quelle que soit la machine, pour pouvoir examiner une valeur anormale
        # sans reproduire l'incident.
        VARIABLES_SUIVIES = ("Oil Pressure", "Oil System Vibration", "ISO 6", "ISO 14")
        for nom, cfg in api.ASSETS.items():
            cfg = {**cfg, "startDate": debut.date().isoformat()}
            for feature in api.FEATURES:
                cle = f"{nom}/{feature.get('variable') or feature.get('api_name')}"
                self._appels[cle] = self._appels.get(cle, 0) + 1
                try:
                    rows = api.fetch_trend(session, self._token, cfg, feature)
                except Exception as e:      # une variable muette ne doit pas
                    self._muettes.append(   # interrompre le releve des autres
                        f"{cle}: {type(e).__name__}")
                    continue
                if not rows:
                    # Appel REUSSI (pas d'exception) mais 0 point renvoye : distinct
                    # d'une panne, potentiellement une absence structurelle du flux.
                    self._vides[cle] = self._vides.get(cle, 0) + 1
                for r in rows:
                    r["asset_name"] = nom
                    r["asset_id"] = cfg["asset_id"]
                    r["oil_device"] = cfg["oil_device"]
                    if r.get("variable") in VARIABLES_SUIVIES:
                        ech = self._echantillon.setdefault(nom, [])
                        ech.append({"variable": r.get("variable"),
                                   "value": r.get("value"),
                                   "created_at": r.get("created_at")})
                        del ech[:-20]     # les 20 dernieres, toutes variables suivies confondues
                lignes.extend(rows)
        if not lignes:
            return pd.DataFrame()
        brut = pd.DataFrame(lignes)
        if "created_at" in brut:
            brut["created_at"] = horodatage(brut["created_at"])
            if depuis is not None:
                brut = brut[brut["created_at"] > depuis]
        return brut


class SimulateurAPI:
    """Rejoue le CSV nettoye en se faisant passer pour l'API.

    Sert a exercer le chemin temps reel SANS reseau : meme interface, memes
    donnees, meme traitement en aval. Indispensable pour tester un connecteur
    dont le serveur n'est pas toujours joignable."""

    def __init__(self, chemin: str | None = None, depart: float = 0.90):
        chemin = chemin or os.path.join(ROOT, "isense_oil_data_cleaned.csv")
        self._df = pd.read_csv(chemin, parse_dates=["created_at"]).sort_values("created_at")
        self._i = int(len(self._df) * depart)
        self.deja_large = True          # le CSV est deja pivote

    def connecter(self) -> None:
        pass

    def relever(self, depuis: pd.Timestamp | None) -> pd.DataFrame:
        """Renvoie la prochaine ligne, comme le ferait un relevé de l'API."""
        if self._i >= len(self._df):
            return pd.DataFrame()
        ligne = self._df.iloc[[self._i]]
        self._i += 1
        return ligne

    def amorce(self) -> pd.DataFrame:
        """Historique anterieur au point de depart, pour remplir les tampons."""
        return self._df.iloc[:self._i]


# ═════════════════════════════════════════════════════ 2. NETTOYAGE
def pivoter(brut: pd.DataFrame) -> pd.DataFrame:
    """Format LONG (une mesure par ligne) -> LARGE (un horodatage x une machine).

    Si la source livre deja du large — cas du simulateur — on la laisse telle."""
    if brut.empty or {"variable", "value"} - set(brut.columns):
        return brut
    large = brut.pivot_table(index=["created_at", "asset_name"],
                            columns="variable", values="value", aggfunc="last")
    return large.reset_index()


def nettoyer(df: pd.DataFrame, dernier_par_machine: dict) -> pd.DataFrame:
    """Sentinelles, unite de conductivite, etat machine, rattachement de session.

    Reproduit clean_isense_data.py pour un LOT QUI ARRIVE, en continuant les
    sessions ouvertes au lieu de tout redecouper."""
    if df.empty:
        return df
    df = df.copy()
    df["created_at"] = normaliser_temps(df["created_at"])

    # --- sentinelles -> NaN ---
    num = df.select_dtypes(include=[np.number]).columns
    for s in SENTINELLES:
        df[num] = df[num].mask(df[num].round(5) == s)

    # --- unite : l'API livre des nS/m/10 ---
    if "Oil Conductivity" in df and "Oil Conductivity_nSm" not in df:
        df["Oil Conductivity_nSm"] = df["Oil Conductivity"] * 10

    # --- etat machine, depuis la pression ---
    if "Oil Pressure" in df:
        df["machine_state"] = np.where(df["Oil Pressure"] > OIL_PRESSURE_ON, "ON", "OFF")

    # --- session : on CONTINUE celle qui est ouverte si l'ecart est court ---
    # La colonne doit etre de type OBJET : les identifiants de session sont du
    # texte. Creee avec np.nan elle serait en float64, et pandas 3 refuse d'y
    # ecrire une chaine (TypeError: Invalid value ... for dtype 'float64').
    if "session_id" not in df.columns:
        df["session_id"] = pd.Series([None] * len(df), index=df.index, dtype=object)
    else:
        df["session_id"] = df["session_id"].astype(object)
    for asset, part in df.groupby("asset_name"):
        precedent = dernier_par_machine.get(asset)
        idx = part.sort_values("created_at").index
        sid = precedent["session_id"] if precedent else f"{asset}-live-0"
        t_prec = precedent["created_at"] if precedent else None
        for i in idx:
            t = df.at[i, "created_at"]
            if t_prec is not None and (t - t_prec).total_seconds() / 3600 > GAP_SESSION_H:
                sid = f"{asset}-live-{t:%Y%m%d%H%M}"
            df.at[i, "session_id"] = sid
            t_prec = t
        dernier_par_machine[asset] = {"session_id": sid, "created_at": t_prec}
    return df


# ═════════════════════════════════════════════════════ 3. LE FLUX
class LiveFeed:
    """Flux temps reel. Interface identique a ReplayFeed."""

    def __init__(self, source=None, art: dict | None = None,
                 periode_s: int = PERIODE_S, taille_tampon: int = 600):
        self.art = art if art is not None else charger_pretraitement()
        self.source = source if source is not None else SourceAPI()
        self.periode_s = int(periode_s)
        self.flux = FluxIncremental(self.art, taille_tampon=taille_tampon)
        self._dernier: pd.Timestamp | None = None
        self._sessions: dict = {}
        self._n_relevees = 0
        self._n_erreurs = 0
        self._derniere_erreur: str | None = None
        self._bilan: dict | None = None     # dernier releve, etage par etage
        self._cumul: dict = {}              # somme depuis le demarrage
        self._dernier_tick: float = 0.0     # horloge monotone du dernier releve
        self._boucle: threading.Thread | None = None
        self._arret = threading.Event()

    # ------------------------------------------------------------------ amorce
    def amorcer(self, historique: pd.DataFrame | None = None,
                n_lignes: int = 4000) -> None:
        """Remplit les tampons a froid. Sans amorce, les 18 premieres lignes
        sortent avec des variables temporelles incompletes.

        L'entree attendue est le jeu NETTOYE (isense_oil_data_cleaned.csv), pas
        le jeu deja traite : la chaine reconstruit variables et indices a partir
        des capteurs. A defaut d'argument, on lit la fin de ce fichier."""
        if historique is None and hasattr(self.source, "amorce"):
            historique = self.source.amorce()
        if historique is None:
            chemin = os.path.join(ROOT, "isense_oil_data_cleaned.csv")
            if os.path.exists(chemin):
                historique = (pd.read_csv(chemin, parse_dates=["created_at"])
                              .sort_values("created_at").tail(n_lignes))
        if historique is None or historique.empty:
            return
        h = nettoyer(historique, self._sessions)
        h = appliquer_imputation_capteurs(h, self.art)
        h = appliquer_imputation_vibration(h, self.art)
        self.flux.amorcer(h)
        self._dernier = h["created_at"].max()

    # -------------------------------------------------------------------- tick
    def tick(self) -> int:
        """Un cycle : relever, nettoyer, imputer, ingerer. Renvoie le nombre de
        lignes nouvellement traitees. Ne leve jamais : une panne de reseau ne
        doit pas eteindre le tableau de bord, elle doit se voir dans status()."""
        self._dernier_tick = time.monotonic()
        try:
            brut = self.source.relever(self._dernier)
            if brut is None or brut.empty:
                self._bilan = {"brut": 0, "pivot": 0, "nettoye": 0, "ingere": 0}
                self._cumul["releves"] = self._cumul.get("releves", 0) + 1
                self._cumul["releves_vides"] = self._cumul.get("releves_vides", 0) + 1
                return 0
            # BILAN DE PERTE, etage par etage. Les mesures recues tombent a
            # 98 % sur une grille de 10 minutes, mais on n'en garde qu'environ
            # 7 % : la perte est donc reelle, et il faut savoir OU elle se
            # produit — a la source, au pivot, ou au nettoyage. Sans ce compte,
            # on ne peut que supposer.
            n_brut = len(brut)
            n_horod = int(brut["created_at"].nunique()) if "created_at" in brut else 0
            lignes = pivoter(brut)
            n_pivot = len(lignes)
            lignes = nettoyer(lignes, self._sessions)
            n_net = len(lignes)
            lignes = appliquer_imputation_capteurs(lignes, self.art)
            lignes = appliquer_imputation_vibration(lignes, self.art)
            self.flux.ingerer(lignes)
            self._bilan = {"brut": n_brut, "horodatages_distincts": n_horod,
                           "pivot": n_pivot, "nettoye": n_net, "ingere": len(lignes)}
            # CUMUL sur toute la vie du connecteur. Un releve isole ne prouve
            # rien quand la source est lente : c'est la somme qui dira si les
            # points manquants manquent DEJA a l'arrivee, ou si nous les
            # perdons en chemin.
            self._cumul["releves"] = self._cumul.get("releves", 0) + 1
            for cle, val in (("brut", n_brut), ("horodatages", n_horod),
                             ("pivot", n_pivot), ("nettoye", n_net),
                             ("ingere", len(lignes))):
                self._cumul[cle] = self._cumul.get(cle, 0) + val
            self._dernier = max(self._dernier or lignes["created_at"].min(),
                                lignes["created_at"].max())
            self._n_relevees += len(lignes)
            return len(lignes)
        except Exception as e:
            self._n_erreurs += 1
            self._derniere_erreur = f"{type(e).__name__}: {e}"
            return 0

    def rafraichir(self, age_max_s: float = 45.0) -> bool:
        """Releve immediat, SI le dernier date de plus de `age_max_s`.

        Sert au rafraichissement manuel : recharger la page doit aller chercher
        les mesures nouvelles, et non se contenter du tampon que la boucle de
        fond met a jour toutes les dix minutes. Sans cela, un utilisateur qui
        recharge a 10 h 09 voit encore l'etat de 10 h 00.

        Le garde-fou d'age est la pour ne pas marteler l'API : recharger cinq
        fois d'affilee ne declenche qu'un seul releve. `time.monotonic` et non
        `time.time` — un ajustement d'horloge ne doit pas ouvrir la porte.

        Renvoie True si un releve a eu lieu."""
        if time.monotonic() - self._dernier_tick < age_max_s:
            return False
        self.tick()
        return True

    # ----------------------------------------------------------- boucle de fond
    def demarrer(self) -> None:
        """Lance la boucle de releve en tache de fond, a la cadence des capteurs."""
        if self._boucle is not None and self._boucle.is_alive():
            return

        def _tourner():
            while not self._arret.wait(0):
                self.tick()
                if self._arret.wait(self.periode_s):
                    break

        self._arret.clear()
        self._boucle = threading.Thread(target=_tourner, daemon=True,
                                        name="isense-live")
        self._boucle.start()

    def arreter(self) -> None:
        self._arret.set()

    # ------------------------------------------------- interface de ReplayFeed
    def window(self, n_rows: int = 4000) -> pd.DataFrame:
        return self.flux.window(n_rows)

    def status(self) -> dict:
        return {
            "mode": "temps réel",
            "source": type(self.source).__name__,
            "periode_s": self.periode_s,
            "derniere_mesure": (self._dernier.isoformat() if self._dernier is not None
                                else None),
            "lignes_relevees": self._n_relevees,
            "erreurs": self._n_erreurs,
            "derniere_erreur": self._derniere_erreur,
            "boucle_active": bool(self._boucle and self._boucle.is_alive()),
            "age_dernier_releve_s": (round(time.monotonic() - self._dernier_tick, 1)
                                     if self._dernier_tick else None),
            # Perte etage par etage au dernier releve : brut -> pivot ->
            # nettoye -> ingere. Permet de dire si les points manquants le sont
            # deja a la source ou si c'est nous qui les perdons.
            "dernier_releve": getattr(self, "_bilan", None),
            "cumul_releves": dict(getattr(self, "_cumul", {})),
            "variables_muettes": list(getattr(self.source, "_muettes", []))[-10:],
            # Panne (exception) vs absence structurelle (appel reussi, 0 ligne) :
            # une variable a 0/0 vide n'a jamais ete rappelee correctement, une
            # variable a N/N vide (N = appels_tentes) est absente du flux temps
            # reel a 100% des essais, ce qui n'est plus imputable au reseau.
            "appels_tentes": dict(sorted(getattr(self.source, "_appels", {}).items())),
            "appels_vides": dict(sorted(getattr(self.source, "_vides", {}).items())),
            "echantillon_valeurs_suspectes": dict(getattr(self.source, "_echantillon", {})),
            **self.flux.status(),
        }
