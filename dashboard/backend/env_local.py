"""Charge dashboard/backend/.env au demarrage, s'il existe.

Format : une affectation NOM=valeur par ligne ; les lignes vides et celles qui
commencent par # sont ignorees ; les guillemets autour de la valeur sont retires.

Les variables DEJA definies dans l'environnement gardent la priorite : chez un
hebergeur (Render...), ce sont ses variables qui font foi, et un .env oublie ne
peut pas les ecraser. Aucune dependance externe (pas de python-dotenv).

Le fichier .env contient des secrets : il est exclu du depot par .gitignore.
"""
from __future__ import annotations

import os

CHEMIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def charger(chemin: str = CHEMIN) -> list[str]:
    """Renvoie la liste des NOMS charges (jamais les valeurs)."""
    if not os.path.exists(chemin):
        return []
    charges = []
    with open(chemin, encoding="utf-8-sig") as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne or ligne.startswith("#") or "=" not in ligne:
                continue
            nom, valeur = ligne.split("=", 1)
            nom, valeur = nom.strip(), valeur.strip()
            if len(valeur) >= 2 and valeur[0] == valeur[-1] and valeur[0] in "\"'":
                valeur = valeur[1:-1]
            if nom and nom not in os.environ:
                os.environ[nom] = valeur
                charges.append(nom)
    return charges
