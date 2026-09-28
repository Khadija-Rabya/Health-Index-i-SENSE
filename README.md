# Health Index i-SENSE — surveillance de l'état de l'huile des motosoufflantes

Projet de fin d'études, OCP / UM6P. Construction d'un indice de santé (*health index*)
à partir des mesures du capteur i-SENSE, puis prévision de cet indice à 20 min, 3 h et
24 h, avec un tableau de bord de restitution.

**Périmètre** : 2 motosoufflantes (A et B), 42 141 relevés du 29/12/2025 au 06/08/2026,
pas de mesure ≈ 10 min.

---

> ## ⚠ Confidentialité — ce dépôt public ne contient que du code
>
> Les **données d'exploitation OCP** (mesures horodatées, états d'alarme,
> calendrier de marche), les rapports et les exports du tableau de bord **ne sont
> pas versionnés** : `.gitignore` les exclut. Seuls le code et les modèles
> utilisés par le serveur (paramètres ajustés, sans mesure brute) sont publiés.
>
> - **Aucun secret dans le dépôt** : identifiants i-SENSE et compte de connexion
>   sont des variables d'environnement (`dashboard/backend/.env` en local, jamais
>   commité ; interface de l'hébergeur en production).
> - Le tableau de bord déployé est protégé par une **page de connexion**.
> - Déploiement : voir [DEPLOIEMENT.md](DEPLOIEMENT.md).

---

## Le résultat, sans enjolivure

L'indice de santé est **construit**, pas mesuré : il n'existe aucune vérité terrain
d'usure d'huile dans ce jeu de données. Un R² élevé sur une série aussi inerte ne
prouve donc rien — un modèle qui recopie la dernière valeur (« persistance ») obtient
déjà R² ≈ 0,95. La seule question qui compte est : **le modèle fait-il mieux que
« rien ne change » ?** On la mesure par le *skill* :

> skill = 1 − MSE(modèle) / MSE(persistance)  ·  positif = utile, négatif = nuisible

Métriques sur le **test gelé** (20 % final, jamais utilisé pour choisir quoi que ce soit) :

| Machine | Horizon | Exactitude ±0,01 | Persistance | Skill | R² |
|---|---|---:|---:|---:|---:|
| A | 20 min | 0,980 | 0,976 | **+0,046** | 0,973 |
| A | 3 h | 0,941 | 0,926 | **+0,051** | 0,890 |
| A | 24 h | 0,868 | 0,903 | **−0,156** | 0,804 |
| B | 20 min | 0,824 | 0,769 | **+0,215** | 0,318 |
| B | 3 h | 0,774 | 0,726 | **+0,273** | 0,076 |
| B | 24 h | 0,638 | 0,638 | +0,026 | −0,338 |

À retenir :

- **5 couples sur 6 battent la persistance**, mais de peu sur la Motosoufflante A.
- **La Motosoufflante A à 24 h est en skill négatif** : le tableau de bord l'affiche
  « aucune prévision fiable » et retombe sur la persistance, plutôt que de présenter
  une prévision qui dégrade la décision.
- **La Motosoufflante B à 24 h** a un skill à peine positif mais un R² négatif, et sa
  porte reste fermée presque tout le temps (α = 0,05, seuil = 0,8) : son exactitude est
  identique à celle de la persistance. C'est un quasi-nul, pas un succès.
- Le gain réel se situe sur la **Motosoufflante B à court et moyen terme** (+0,215 et
  +0,273 de skill).

### Deux réserves méthodologiques importantes

1. **Le viscosimètre de la Motosoufflante B est faux à l'arrêt.** 16 839 lignes
   (39,96 % du jeu) portent une viscosité physiquement impossible, presque toutes sur
   B à l'arrêt. Comme la qualité des données est aujourd'hui évaluée *après* la
   construction de l'indice, ces valeurs contaminent l'étiquette elle-même. En les
   masquant, la Motosoufflante A passe d'un skill de −0,119 à +0,086 et le bénéfice
   global devient statistiquement significatif. **Ce masquage n'est pas encore activé
   en production** (`mask_off_viscosity=False`) — c'est le premier chantier à reprendre.
2. **La taille d'échantillon effective est bien inférieure au nombre de lignes.** Les
   relevés sont fortement autocorrélés au sein d'une session ; un bootstrap par blocs
   ramène ~3 700 lignes de test à quelques centaines d'observations indépendantes. Les
   intervalles de confiance du rapport sont calculés sur cette base, pas sur le nombre
   de lignes brut.

### Sur les alertes

Seuiller l'indice *prédit* ne détecte aucune transition d'état (rappel 0,000) : l'indice
bouge trop peu et trop tard. Un classifieur dédié aux transitions atteint un rappel de
0,677 pour une précision de 0,277, là où la persistance est structurellement à 0 —
elle ne peut annoncer un changement par construction. Ce classifieur n'est pas encore
branché sur le tableau de bord.

---

## Le modèle retenu

Un couple **Lasso + porte LightGBM**, un par machine et par horizon (6 au total) :

```
ŷ = hi[t] + α · δ̂ · 1[P(mouvement) > seuil]
```

On part de la persistance, et on ne s'en écarte que lorsqu'un classifieur estime
qu'un mouvement est probable. Ce garde-fou vient d'un constat de la validation croisée :
sur 13 familles de modèles sur 14, le α optimal était **0**, autrement dit « ne rien
corriger ». La porte permet de corriger là où c'est utile sans dégrader le reste.

**Protocole gelé une seule fois** (`hi_forecast/protocol.py`, source unique de vérité) :
test final 20 %, validation croisée temporelle à 5 plis purgés, graine 42, tolérance
±0,01. La sélection s'est faite **uniquement en validation croisée** ; le test gelé n'a
été touché qu'une fois, à la fin.

---

## Organisation du dépôt

| Dossier | Contenu |
|---|---|
| `hi_forecast/` | Protocole gelé, entraînements, autoencodeurs, comparaisons, scripts de rapport |
| `hi_forecast/artifacts/` | Modèles sérialisés (`modeles_multi_horizon.joblib` sert le tableau de bord) |
| `realtime/` | Pipeline temps réel antérieur (imputation vibration, indice de santé v1) |
| `dashboard/backend/` | API FastAPI : prédiction, qualité, diagnostics, rejeu |
| `dashboard/frontend/` | Tableau de bord React + Vite + Recharts |
| `*.docx` | Rapports rédigés (audit, comparaison des modèles, autoencodeurs, par machine) |
| `ARCHITECTURE_PIPELINE.md` | Architecture détaillée, budget de latence, skew entraînement/service |

**Les données brutes ne sont pas versionnées** (401 Mo, dont un CSV de 71 Mo). Le dépôt
contient le code, les modèles et les résultats agrégés.

---

## Faire tourner le tableau de bord

### Mode statique (aucun serveur)

Le tableau de bord fonctionne sans backend : les prédictions, la qualité et les
diagnostics ont été précalculés par les modèles entraînés puis figés en JSON dans
`dashboard/frontend/public/data/` (1,4 Mo). Le rejeu devient une animation dans le
navigateur.

```bash
cd dashboard/frontend
npm ci
npm run dev          # http://localhost:5173
```

Pour régénérer ces JSON après un réentraînement (nécessite les données locales) :

```bash
python hi_forecast/run_29_export_statique.py
```

### Mode live (avec l'API)

Nécessite les données brutes en local.

```powershell
.\dashboard\start_backend.ps1     # http://localhost:8000
.\dashboard\start_frontend.ps1
```

Le frontend bascule automatiquement en mode live dès que la variable
`VITE_API` est définie (voir `dashboard/frontend/.env.example`).

---

## Les trois onglets

- **Prédiction** — indice calculé, prédit à 20 min / 3 h / 24 h, et persistance en
  regard. L'écart entre les deux est *tout* ce qu'apporte le modèle. Un horizon en
  skill négatif est signalé, pas tracé.
- **Data quality** — état marche/arrêt, densité à 15 °C, H₂O ppm vs saturation,
  distribution de la conductivité par machine, cohérence μ = ν·ρ, monotonie ISO.
- **Diagnostics** — imputation de la vibration, synchronisme des capteurs, effets
  calendaires, sessions, indices de confiance.
