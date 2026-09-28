"""API FastAPI du tableau de bord Health Index i-SENSE.

    uvicorn main:app --reload --port 8000     (depuis dashboard/backend)

Points d'entree :
    GET  /health          etat du service et des artefacts charges
    POST /predict         prediction ; corps optionnel = lignes a INGERER
                          (chemin incremental, parametres de pretraitement figes)
    GET  /ingestion       etat du flux incremental (tampons, artefact)
    GET  /quality         indicateurs de qualite de donnees
    GET  /history         serie observee recente (fenetre en heures)
    GET  /diagnostics     residus, SHAP top-20, viscosite par machine x etat
    GET  /comparaison     classement des familles de modeles (indice systeme)
    GET  /replay/status   etat du rejeu
    POST /replay/control  play / pause / vitesse / position
    WS   /ws              flux pousse (1 message toutes les 10 min)

TRANSPORT : WebSocket pour le direct, avec repli par polling REST. Le front
ouvre /ws ; si la connexion echoue il interroge /predict toutes les 10 minutes,
cadence d'acquisition des capteurs.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# .env local AVANT tout autre import : Api_to_excel, auth et MODE_LIVE lisent
# l'environnement au chargement.
import env_local  # noqa: E402
_charges = env_local.charger()
if _charges:
    print(f"[env] .env charge : {', '.join(_charges)}", file=sys.stderr)

from ingestion import (FluxIncremental, appliquer_imputation_capteurs,  # noqa: E402
                       appliquer_imputation_vibration)
from replay import ReplayFeed          # noqa: E402
from service import (STEP_MIN, asset_of, comparaison_modeles,  # noqa: E402
                     diagnostics_report,
                     history_series, load_artifacts, load_dataset, predict_latest,
                     quality_gate, quality_report)

import auth  # noqa: E402
from fastapi import HTTPException, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

# Documentation interactive fermee quand la connexion est active : elle
# decrirait publiquement toutes les routes protegees.
app = FastAPI(title="Health Index i-SENSE", version="1.0",
              docs_url=None if auth.ACTIVE else "/docs",
              redoc_url=None, openapi_url=None if auth.ACTIVE else "/openapi.json")

# Routes accessibles SANS connexion. Tout le reste exige un jeton valide.
ROUTES_PUBLIQUES = {"/", "/healthz", "/auth/login", "/docs", "/openapi.json"}


@app.middleware("http")
async def exiger_connexion(request: Request, call_next):
    """Refuse toute route de donnees sans jeton valide. Ajoute AVANT le CORS
    pour que le CORS l'enveloppe : un 401 doit porter les en-tetes CORS, sinon
    le navigateur le masque en « Failed to fetch » et l'ecran ne sait pas qu'il
    doit afficher la page de connexion."""
    if request.method == "OPTIONS" or request.url.path in ROUTES_PUBLIQUES:
        return await call_next(request)
    entete = request.headers.get("authorization", "")
    jeton = entete[7:] if entete.lower().startswith("bearer ") else None
    if not auth.jeton_valide(jeton):
        return JSONResponse({"detail": "connexion requise"}, status_code=401)
    return await call_next(request)


# CORS_ORIGINS : adresses du frontend autorisees, separees par des virgules
# (ex. https://mon-dashboard.vercel.app). « * » par defaut, pour le local.
_ORIGINES = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_ORIGINES, allow_methods=["*"],
                   allow_headers=["*"])


@app.post("/auth/login")
def connexion(corps: dict = Body(...)):
    """Echange identifiant + mot de passe contre un jeton de session."""
    if not auth.ACTIVE:
        jeton, exp = auth.emettre_jeton("local")
        return {"token": jeton, "expire_a": exp, "auth_active": False}
    u = str(corps.get("utilisateur", ""))
    m = str(corps.get("mot_de_passe", ""))
    if not auth.verifier_identifiants(u, m):
        time.sleep(1.0)                    # freine les essais en rafale
        raise HTTPException(status_code=401, detail="identifiant ou mot de passe incorrect")
    jeton, exp = auth.emettre_jeton(u)
    return {"token": jeton, "expire_a": exp, "auth_active": True}


@app.get("/healthz")
def healthz():
    """Sonde de vie pour l'hebergeur : publique, ne revele rien."""
    return {"ok": True}

# Bascule REJEU <-> TEMPS REEL. Par defaut le rejeu, pour qu'un lancement sans
# identifiants continue de fonctionner. ISENSE_LIVE=1 branche le connecteur.
MODE_LIVE = os.environ.get("ISENSE_LIVE", "").strip().lower() in ("1", "true", "oui")


if MODE_LIVE and not (os.environ.get("ISENSE_EMAIL", "").strip()
                      and os.environ.get("ISENSE_PASSWORD", "").strip()):
    # Sans ce controle, l'absence d'identifiants ne se voit qu'a la premiere
    # requete, sous la forme d'un 500 opaque repete indefiniment : le serveur a
    # l'air demarre, le tableau de bord affiche « Failed to fetch », et rien ne
    # dit pourquoi. Mieux vaut refuser de demarrer en le disant.
    barre = "=" * 70
    etats = [f"    {nom:16} "
             + ("defini" if os.environ.get(nom, "").strip() else "ABSENT")
             for nom in ("ISENSE_EMAIL", "ISENSE_PASSWORD")]
    print("\n".join([
        "", barre,
        "  ISENSE_LIVE=1 mais les identifiants ne sont pas definis.",
        barre, *etats,
        "",
        "  Dans CE terminal, avant de relancer :",
        "      $env:ISENSE_EMAIL    = Read-Host 'Adresse i-SENSE'",
        "      $env:ISENSE_PASSWORD = Read-Host 'Mot de passe i-SENSE'",
        "",
        "  Les variables ne survivent ni a la fermeture du terminal ni a",
        "  l'ouverture d'un nouvel onglet : elles sont a redefinir a chaque fois.",
        "",
        "  Pour demarrer sans API, en rejeu du jeu local :",
        '      $env:ISENSE_LIVE = ""',
        barre, "",
    ]), file=sys.stderr)
    raise SystemExit(2)


def _maintenant():
    """Instant de reference pour la fraicheur des mesures.

    EN DIRECT : l'heure reelle, en UTC naif comme les horodatages du jeu. Sans
    cela la fraicheur se mesure par rapport a la derniere mesure RECUE, et un
    flux totalement arrete affiche « 0 min » sur toutes les machines.

    EN REJEU : None — le temps est celui des donnees rejouees."""
    if not MODE_LIVE:
        return None
    # now("UTC").tz_localize(None) et non utcnow() : ce dernier est deprecie et
    # disparait en pandas 4.
    return pd.Timestamp.now("UTC").tz_localize(None)


FEED = None
FLUX: FluxIncremental | None = None
# Verrou de CREATION de la source. Sans lui, le planificateur et la premiere
# requete arrivee au demarrage creaient chacun leur connecteur : le
# planificateur rafraichissait le sien, les calculs lisaient l'autre, fige sur
# le premier releve — l'ecran ne suivait plus i-SENSE (constate le 28/09).
_FEED_LOCK = threading.Lock()


def flux() -> FluxIncremental:
    """Chemin d'INGESTION : tampon par machine + transformations figees.

    EN TEMPS REEL, c'est celui du connecteur — surtout pas un second. Deux
    tampons paralleles divergeraient et doubleraient le calcul.

    L'amorce se fait sur le jeu NETTOYE, pas sur load_dataset() : ce dernier
    renvoie des lignes DEJA traitees, sur lesquelles rejouer l'ingenierie de
    variables echoue. C'etait la cause de l'erreur 500 sur /ingestion."""
    global FLUX
    if MODE_LIVE:
        return feed().flux
    if FLUX is None:
        FLUX = FluxIncremental()
        chemin = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                              "isense_oil_data_cleaned.csv")
        if os.path.exists(chemin):
            h = (pd.read_csv(chemin, parse_dates=["created_at"])
                 .sort_values("created_at").tail(4000))
            # Le jeu NETTOYE n'a pas les colonnes `_filled` : l'imputation est un
            # etage a part, que le connecteur applique aussi. Sans elle,
            # l'ingenierie de variables echoue sur Density_filled.
            h = appliquer_imputation_capteurs(h, FLUX.art)
            h = appliquer_imputation_vibration(h, FLUX.art)
            FLUX.amorcer(h)
    return FLUX


def feed():
    """Source des lignes. `ReplayFeed` et `LiveFeed` exposent la meme interface :
    window(n) renvoie « les lignes dont on dispose maintenant ». Aucun etage en
    aval ne sait laquelle des deux tourne."""
    global FEED
    if FEED is not None:
        return FEED
    with _FEED_LOCK:
        if FEED is not None:              # cree par un autre fil entre-temps
            return FEED
        return _creer_feed()


def _creer_feed():
    global FEED
    if FEED is None and MODE_LIVE:
        from live_feed import LiveFeed
        lf = LiveFeed()
        # Amorce depuis l'historique local : les variables temporelles (lags,
        # pente 3 h, EWMA) exigent du passe. Puis premier releve immediat, puis
        # boucle de fond a la cadence des capteurs.
        lf.amorcer()
        lf.tick()
        # Pas de lf.demarrer() : c'est le PLANIFICATEUR ci-dessous qui cadence a
        # la fois le releve ET le calcul. Deux boucles independantes (releve
        # d'un cote, prediction a chaque requete de l'autre) n'etaient pas
        # synchronisees : l'ecran pouvait montrer un calcul fait avant le
        # dernier releve.
        FEED = lf
    if FEED is None:
        # Vitesse 1 = temps reel : une nouvelle mesure toutes les 10 minutes, la
        # cadence d'acquisition i-SENSE. Le tableau de bord se comporte alors
        # comme un ecran de supervision et non comme un film accelere.
        # REPLAY_SPEED=1800 rejoue 7 mois en quelques minutes, pour les tests.
        FEED = ReplayFeed(load_dataset(), speed=float(os.environ.get("REPLAY_SPEED", 1)))
    return FEED


def _clean(o):
    """numpy -> types JSON."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o



# ================================================================ PLANIFICATEUR
# Un seul cycle, toutes les PERIODE_CALCUL_S secondes (10 min, cadence des
# capteurs) : releve i-SENSE -> prediction -> mise en cache. Les routes /predict
# et /history ne CALCULENT plus rien : elles servent le dernier resultat. Un
# rechargement de page est donc instantane, et deux navigateurs ouverts voient
# exactement le meme calcul, horodate.
PERIODE_CALCUL_S = int(os.environ.get("PERIODE_CALCUL_S", STEP_MIN * 60))

_CACHE: dict = {"prediction": None, "history": None, "calcule_a": None,
                "prochain_calcul": None, "duree_calcul_s": None,
                "n_calculs": 0, "erreur": None}
_CACHE_LOCK = threading.Lock()
_ARRET = threading.Event()


def _horloge():
    """Horodatage du calcul, sur la meme horloge que les mesures."""
    t = _maintenant()
    return (t if t is not None else pd.Timestamp.now("UTC").tz_localize(None))


def calculer() -> None:
    """Un calcul complet sur le tampon courant, publie d'un bloc dans le cache."""
    t0 = time.monotonic()
    df = feed().window(4000)
    pred = predict_latest(df, _maintenant())
    hist = history_series(df, 48)
    hist["maintenant"] = df["created_at"].max().isoformat()
    fin = _horloge()
    meta = {"calcule_a": fin.isoformat(timespec="seconds"),
            "prochain_calcul": (fin + pd.Timedelta(seconds=PERIODE_CALCUL_S))
                               .isoformat(timespec="seconds"),
            "periode_calcul_s": PERIODE_CALCUL_S}
    pred.update(meta)
    with _CACHE_LOCK:
        _CACHE.update({"prediction": pred, "history": hist, "erreur": None,
                       "duree_calcul_s": round(time.monotonic() - t0, 2),
                       "n_calculs": _CACHE["n_calculs"] + 1, **meta})


def _planificateur() -> None:
    feed()                                       # amorce + premier releve
    while not _ARRET.is_set():
        try:
            calculer()
        except Exception as e:                   # le cache garde le dernier bon calcul
            with _CACHE_LOCK:
                _CACHE["erreur"] = f"{type(e).__name__}: {e}"
            print(f"[planificateur] calcul en echec : {e}", file=sys.stderr)
        if _ARRET.wait(PERIODE_CALCUL_S):
            break
        if MODE_LIVE:
            # feed() et non une reference gardee : toujours LA source partagee,
            # celle que lisent les calculs et les routes.
            feed().tick()                        # releve, puis calcul au tour suivant


@app.on_event("startup")
def _demarrer_planificateur() -> None:
    threading.Thread(target=_planificateur, daemon=True, name="planificateur").start()


@app.on_event("shutdown")
def _arreter_planificateur() -> None:
    _ARRET.set()


def _dernier_calcul() -> dict:
    """Le dernier calcul publie ; calcule a la demande s'il n'existe pas encore
    (premiere requete arrivee avant la fin du premier cycle)."""
    with _CACHE_LOCK:
        pret = _CACHE["prediction"] is not None
    if not pret:
        calculer()
    with _CACHE_LOCK:
        return dict(_CACHE)

# ------------------------------------------------------------------------- /
@app.get("/")
def racine():
    """La racine n'avait aucune route et renvoyait un {"detail":"Not Found"} nu.

    C'est l'adresse qu'on tape spontanement en voulant « ouvrir le tableau de
    bord », et le message ne disait pas que le tableau de bord est ailleurs."""
    return {
        "service": "API Health Index i-SENSE",
        "ceci_n_est_pas_le_tableau_de_bord": True,
        "tableau_de_bord": os.environ.get("DASHBOARD_URL", "http://localhost:5173"),
        "mode": "temps réel" if MODE_LIVE else "rejeu",
        "points_d_entree": ["/health", "/predict", "/history", "/ingestion",
                            "/quality", "/diagnostics", "/comparaison", "/ws"],
    }


# --------------------------------------------------------------------- /health
@app.get("/health")
def health():
    art = load_artifacts()
    f = feed()
    return _clean({
        "statut": "ok",
        "modele": {
            "type": "Lasso + porte, un modèle par machine et par horizon",
            "horizons": [1, 2, 18, 144],
            "n_modeles": len(art["models"]),
            "tolerance": 0.01,
        },
        "metriques_gelees": art["metrics"],
        "mode": "temps réel" if MODE_LIVE else "rejeu",
        "source": f.status(),
    })


# -------------------------------------------------------------------- /predict
@app.post("/predict")
def predict(rows: list[dict] | None = Body(default=None),
            n_rows: int = Query(4000, ge=100, le=20000)):
    """Sans corps : predit sur la fenetre courante du rejeu.

    AVEC un corps (lignes nettoyees/imputees venant du connecteur i-SENSE) :
    elles passent par le FLUX INCREMENTAL. Les variables sont recalculees sur le
    seul tampon de la machine concernee, avec des parametres de pretraitement
    FIGES — jamais reajustes sur le jeu complet. C'est le chemin de production ;
    sa conformite au calcul hors ligne est verifiee par test_ingestion.py."""
    if rows:
        df = flux().ingerer(pd.DataFrame(rows))
        if df is None or not len(df):
            return _clean({"erreur": "aucune ligne exploitable apres ingestion"})
        return _clean(predict_latest(df, _maintenant()))
    # Sans corps : le DERNIER CALCUL du planificateur. Aucun appel a i-SENSE,
    # aucun recalcul — un rechargement de page lit le resultat publie.
    return _clean(_dernier_calcul()["prediction"])


@app.get("/predict")
def predict_get():
    return _clean(_dernier_calcul()["prediction"])


# -------------------------------------------------------------------- /quality
@app.get("/quality")
def quality(n_rows: int = Query(8000, ge=100, le=50000)):
    return _clean(quality_report(feed().window(n_rows)))


# -------------------------------------------------------------------- /history
@app.get("/history")
def history(n_rows: int = Query(4000, ge=100, le=20000),
            heures: int = Query(48, ge=1, le=720)):
    """Serie observee des `heures` dernieres heures. La prevision n'est pas ici :
    elle vient de /predict, qui porte les horodatages cibles."""
    if heures == 48:
        return _clean(_dernier_calcul()["history"])
    df = feed().window(n_rows)
    r = history_series(df, heures)
    r["maintenant"] = df["created_at"].max().isoformat()
    return _clean(r)


# ------------------------------------------------------------------- /calcul
@app.get("/calcul")
def calcul_status():
    """Etat du planificateur : heure du dernier calcul, du prochain, duree."""
    with _CACHE_LOCK:
        c = {k: v for k, v in _CACHE.items() if k not in ("prediction", "history")}
    c["periode_calcul_s"] = PERIODE_CALCUL_S
    return _clean(c)


# ---------------------------------------------------------------- /diagnostics
@app.get("/diagnostics")
def diagnostics(n_rows: int = Query(6000, ge=200, le=30000)):
    """Imputation de la vibration, synchronisme des capteurs, profils calendaires,
    sessions, scores de confiance — plus les residus et l'importance SHAP."""
    art = load_artifacts()
    df = feed().window(n_rows)
    out = diagnostics_report(df)

    # residus a t+3h, sur les lignes dont la cible est deja arrivee
    d = df.sort_values(["session_id", "created_at"]).reset_index(drop=True)
    y_next = d.groupby("session_id", sort=False)["health_index"].shift(-18)
    ok = y_next.notna().to_numpy()
    resid = []
    if ok.sum() > 20:
        sub = d.loc[ok]
        hi = sub["health_index"].to_numpy(dtype=float)
        yv = y_next[ok].to_numpy(dtype=float)
        step = max(1, len(sub) // 800)
        for asset, col in [("Motosoufflante A", "asset_Motosoufflante A"),
                           ("Motosoufflante B", "asset_Motosoufflante B")]:
            key = f"18|{asset}"
            m = (sub[col] == 1).to_numpy()
            if key not in art["models"] or m.sum() < 20:
                continue
            b = art["models"][key]
            s2 = sub.loc[m]
            pred = np.clip(hi[m] + b["alpha"] * b["regresseur"].predict(s2[b["colonnes"]])
                           * (b["porte"].predict_proba(s2[b["colonnes"]])[:, 1] > b["seuil"]),
                           0, 1)
            resid += [{"predit": float(p), "residu": float(t - p), "machine": asset}
                      for p, t in zip(pred[::step], yv[m][::step])]
    out["residus"] = resid
    out["shap"] = [{"variable": r["feature"], "importance": float(r["mean_abs_shap"])}
                   for _, r in art["shap"].iterrows()]
    return _clean(out)


# ------------------------------------------------------------------ /ingestion
@app.get("/ingestion")
def ingestion_status():
    """Etat du chemin d'ingestion : tampons, artefact, et en temps reel le
    compteur de releves et la derniere erreur du connecteur."""
    if MODE_LIVE:
        return _clean(feed().status())
    return _clean(flux().status())


# ---------------------------------------------------------------- /comparaison
@app.get("/comparaison")
def comparaison():
    """Classement des familles de modeles pour l'indice systeme. Charge utile
    legere, appelee une fois par le tableau de bord."""
    return _clean(comparaison_modeles())


# ---------------------------------------------------------------------- rejeu
@app.get("/replay/status")
def replay_status():
    return _clean(feed().status())


@app.post("/replay/control")
def replay_control(action: str = Query(...), value: float = Query(0.0)):
    f = feed()
    if action == "play":
        f.play(True)
    elif action == "pause":
        f.play(False)
    elif action == "speed":
        f.set_speed(value)
    elif action == "seek":
        f.seek_fraction(value)
    elif action == "reset":
        f.reset()
    return _clean(f.status())


# ------------------------------------------------------------------ WebSocket
@app.websocket("/ws")
async def ws(sock: WebSocket):
    # Le navigateur ne peut pas poser d'en-tete sur un WebSocket : le jeton
    # passe en parametre d'adresse.
    if not auth.jeton_valide(sock.query_params.get("token")):
        await sock.close(code=4401)
        return
    await sock.accept()
    try:
        vu = None
        while True:
            # Pousse le dernier calcul DES QU'IL CHANGE : l'ecran suit le
            # planificateur au lieu d'avoir sa propre horloge, decalee de
            # l'instant ou la page a ete ouverte.
            c = await asyncio.to_thread(_dernier_calcul)
            if c["calcule_a"] != vu:
                vu = c["calcule_a"]
                await sock.send_text(json.dumps(_clean({
                    "type": "tick", "prediction": c["prediction"]})))
            await asyncio.sleep(15)
    except WebSocketDisconnect:
        return
    except asyncio.CancelledError:
        # Arret du serveur (Ctrl+C) pendant le sleep. CancelledError herite de
        # BaseException, pas de Exception : le `except Exception` ci-dessous ne
        # l'attrapait pas, et uvicorn affichait une trace de plusieurs dizaines
        # de lignes a chaque arret. Rien n'etait casse, mais l'ecran donnait a
        # penser le contraire — et une trace qu'on apprend a ignorer est une
        # trace qui cachera la vraie panne le jour venu.
        return
    except Exception:
        return


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
