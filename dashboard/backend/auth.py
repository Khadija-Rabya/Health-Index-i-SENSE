"""Connexion au tableau de bord : un compte unique, defini dans l'environnement.

    DASHBOARD_USER       identifiant de connexion
    DASHBOARD_PASSWORD   mot de passe
    AUTH_SECRET          cle de signature des jetons (fortement recommandee en
                         production ; a defaut, une cle aleatoire est tiree a
                         chaque demarrage et les sessions ne survivent pas a un
                         redemarrage du serveur)
    AUTH_TTL_H           duree de validite d'une session, en heures (12 par defaut)

Si DASHBOARD_PASSWORD n'est pas defini, l'authentification est DESACTIVEE : c'est
le cas du developpement local. Le serveur le signale au demarrage.

Le jeton est un couple « charge utile . signature HMAC-SHA256 », sans etat cote
serveur et sans dependance externe. Aucun mot de passe n'est jamais journalise
ni renvoye.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sys
import time

UTILISATEUR = os.environ.get("DASHBOARD_USER", "").strip()
MOT_DE_PASSE = os.environ.get("DASHBOARD_PASSWORD", "")
ACTIVE = bool(MOT_DE_PASSE)
TTL_S = int(float(os.environ.get("AUTH_TTL_H", "12")) * 3600)
_SECRET = (os.environ.get("AUTH_SECRET") or secrets.token_hex(32)).encode()

if ACTIVE and not UTILISATEUR:
    print("[auth] DASHBOARD_PASSWORD est defini mais pas DASHBOARD_USER.",
          file=sys.stderr)
    raise SystemExit(2)
if ACTIVE and not os.environ.get("AUTH_SECRET"):
    print("[auth] AUTH_SECRET absent : cle aleatoire, les sessions seront "
          "perdues au prochain redemarrage.", file=sys.stderr)
if not ACTIVE:
    print("[auth] DASHBOARD_PASSWORD non defini : connexion DESACTIVEE "
          "(acceptable en local uniquement).", file=sys.stderr)


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def verifier_identifiants(utilisateur: str, mot_de_passe: str) -> bool:
    """Comparaison a temps constant : la duree de la reponse ne doit pas
    renseigner sur le nombre de caracteres justes."""
    ok_u = hmac.compare_digest(utilisateur.encode(), UTILISATEUR.encode())
    ok_p = hmac.compare_digest(mot_de_passe.encode(), MOT_DE_PASSE.encode())
    return ACTIVE and ok_u and ok_p


def emettre_jeton(utilisateur: str) -> tuple[str, int]:
    exp = int(time.time()) + TTL_S
    charge = _b64(json.dumps({"u": utilisateur, "exp": exp}).encode())
    sig = _b64(hmac.new(_SECRET, charge.encode(), hashlib.sha256).digest())
    return f"{charge}.{sig}", exp


def jeton_valide(jeton: str | None) -> bool:
    if not ACTIVE:
        return True
    if not jeton or "." not in jeton:
        return False
    charge, sig = jeton.rsplit(".", 1)
    attendu = _b64(hmac.new(_SECRET, charge.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(sig, attendu):
        return False
    try:
        return int(json.loads(_unb64(charge))["exp"]) > time.time()
    except Exception:
        return False
