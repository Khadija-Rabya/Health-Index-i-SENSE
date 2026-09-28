"""PHASE 3 — iterations 1 a 5 : qualite des donnees, formulation, features.

Selection sur la CV uniquement (cv_acc_tol_mean). Le score de test est
enregistre mais n'oriente aucune decision.
"""
import os
import sys
import warnings

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from harness import evaluate, show_log
from protocol import SEED
from transforms import (ClippedRegressor, ShrunkRegressor, add_dynamics_features,
                        add_regime_features, chain, drop_constant_and_dupes,
                        prune_correlated)


def rf(n=300, depth=14, mf=1.0):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("rf", RandomForestRegressor(n_estimators=n, max_depth=depth,
                                                  max_features=mf, random_state=SEED,
                                                  n_jobs=-1))])


CLEAN = chain(drop_constant_and_dupes, prune_correlated(0.995))

# ---------------------------------------------------------------- iteration 1
evaluate(1, "RF + nettoyage colonnes",
         "Retrait des colonnes constantes, des doublons structurels (DC/DC_filled, "
         "state_ON, asset_B, ...) et elagage des paires |r|>0.995",
         lambda: rf(), transform=CLEAN, delta=True,
         notes="levier 1 : qualite des donnees")

# ---------------------------------------------------------------- iteration 2
evaluate(2, "RF max_features=0.3",
         "max_features 1.0 -> 0.3 : decorrele les arbres (163 features tres "
         "redondantes) et divise le temps d'entrainement par ~4",
         lambda: rf(mf=0.3), transform=CLEAN, delta=True,
         notes="etablit le modele-sonde des iterations suivantes")

evaluate(3, "RF max_features=sqrt",
         "max_features='sqrt' : decorrelation maximale des arbres",
         lambda: rf(mf="sqrt"), transform=CLEAN, delta=True)

# ---------------------------------------------------------------- iteration 4
evaluate(4, "Formulation directe (sans delta)",
         "Apprend health_index[t+18] directement au lieu du delta — verifie "
         "l'affirmation du rapport d'origine selon laquelle le delta est meilleur",
         lambda: rf(mf=0.3), transform=CLEAN, delta=False,
         notes="test de la formulation")

# ---------------------------------------------------------------- iteration 5
evaluate(5, "Delta borne (clipping q=0.999)",
         "Borne le delta predit aux quantiles 0.1%/99.9% du delta d'entrainement : "
         "garde-fou contre les plis catastrophiques (R² -61 au diagnostic D6)",
         lambda: Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("m", ClippedRegressor(
                               RandomForestRegressor(n_estimators=300, max_depth=14,
                                                     max_features=0.3, random_state=SEED,
                                                     n_jobs=-1), q=0.999))]),
         transform=CLEAN, delta=True)

print("\n" + "=" * 118)
print("JOURNAL (trie par CV acc — le critere de selection)")
print("=" * 118)
show_log(sort_by="cv_acc_tol_mean")
