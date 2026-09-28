# Dashboard temps réel — prédiction du `health_index` à t+n

Prédit `health_index` en direct depuis l'API i-SENSE, pour chaque horizon benchmarqué
(10 min / 1h / 3h / 24h / 3j / 1 semaine) et chaque machine, en utilisant — par
(horizon, machine) — le modèle réellement le plus précis d'après
[`rapport_prediction_health_index_comparaison_finale.md`](../rapport_prediction_health_index_comparaison_finale.md) :
Random Forest à 10 min (A et B) et à 3h (B) ; Persistance (« pas de changement »)
partout ailleurs — XGBoost et les 3 variantes autoencodeur ne gagnent jamais dans le
rapport, donc ne sont pas utilisés en production.

## Installation

```
pip install -r requirements_realtime.txt
```

## Utilisation

```
python realtime/artifacts.py       # une fois (ou pour rafraîchir les modèles/seuils
                                    # si les CSV historiques sont régénérés)
streamlit run realtime/dashboard.py
```

Le dashboard se rafraîchit automatiquement toutes les 2 minutes (bouton « Rafraîchir
maintenant » disponible aussi) — chaque cycle interroge l'API i-SENSE sur les 3 derniers
jours, reconstruit les features (mêmes règles que le pipeline batch du projet), recalcule
`health_index` via les modèles PCA/Isolation Forest sauvegardés, et prédit les 6 horizons.

## Fichiers

- `artifacts.py` — entraîne et persiste (`artifacts/`) tout ce que le pipeline batch
  calculait à la volée : PCA + Isolation Forest par machine, l'imputer de vibration, les
  statistiques gelées (z-score, régression `vi_proxy`, seuils `HI_regles`, médianes de
  repli), et les 3 modèles Random Forest là où ils battent réellement la persistance.
- `api_client.py` — récupère une fenêtre récente (3 jours) via l'API i-SENSE (réutilise
  `Api_to_excel.py`).
- `feature_pipeline.py` — rejoue le pipeline batch (`clean_isense_data.py` →
  imputations → `feature_engineering.py` → `health_index_*.py`) sur cette fenêtre.
- `predict.py` — applique, par horizon et par machine, le modèle gagnant du manifest.
- `dashboard.py` — l'app Streamlit.

## Simplifications assumées par rapport au pipeline batch

- Le trou capteur synchrone (`fill_sensor_gap.py`) est traité par simple forward-fill
  (avec repli sur une médiane historique gelée) au lieu du Random Forest dédié à `ISO 4`
  dans le batch — ce modèle ciblait un épisode de coupure historique précis (994 lignes),
  pas un pattern attendu en continu. L'imputation de vibration (OFF→0 / ON→Random Forest),
  elle, est conservée telle quelle (modèle réentraîné et persisté).
- Le z-score par machine et la régression `vi_proxy` utilisent des statistiques **gelées**
  à l'entraînement plutôt que recalculées sur le petit buffer live (recalculer sur
  quelques dizaines de lignes serait instable).
- L'état (Normal/Surveillance/Alarme) d'une valeur **prédite** à t+n est estimé à partir
  de cette seule valeur de `health_index` (seuils gelés) — on ne peut pas recalculer
  T²/SPE/Isolation Forest pour un point futur non encore mesuré. L'état **actuel**, lui,
  reste calculé avec la méthode officielle du projet (T²/SPE/Isolation Forest). Les deux
  ne sont donc pas directement comparables terme à terme — le dashboard l'indique en
  légende.
