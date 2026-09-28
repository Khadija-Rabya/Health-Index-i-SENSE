# Déploiement — Health Index i-SENSE

Deux services :

| Rôle | Hébergeur | Dossier |
|---|---|---|
| **Backend** FastAPI : relevé i-SENSE, prévision toutes les 10 min, cache, connexion | Render | `dashboard/backend` |
| **Frontend** React : page de connexion et tableau de bord | Vercel | `dashboard/frontend` |

Ce dépôt est **public** : il ne contient que le code et les modèles utilisés par le
serveur (paramètres ajustés, sans mesure brute). Les données, les rapports et tous
les secrets en sont exclus par `.gitignore`.

---

## 1. Backend sur Render

1. <https://render.com> → **New** → **Blueprint** → choisir ce dépôt.
   Render lit `render.yaml`.
2. Renseigner les variables secrètes demandées :

   | Variable | Valeur |
   |---|---|
   | `ISENSE_EMAIL` | adresse du compte i-SENSE |
   | `ISENSE_PASSWORD` | mot de passe i-SENSE |
   | `DASHBOARD_USER` | identifiant de connexion au tableau de bord |
   | `DASHBOARD_PASSWORD` | mot de passe de connexion au tableau de bord |
   | `CORS_ORIGINS` | adresse du frontend Vercel (étape 2), ex. `https://health-index-isense.vercel.app` |
   | `DASHBOARD_URL` | la même adresse |

   `AUTH_SECRET` est générée automatiquement par Render.
3. Déployer. Vérifier : `https://<service>.onrender.com/healthz` → `{"ok": true}`.

Au démarrage, sans historique local, le serveur récupère les
`ISENSE_JOURS_AMORCE` derniers jours depuis l'API (3 par défaut), puis
recalcule toutes les `PERIODE_CALCUL_S` secondes (600 = 10 min).

> **Offre gratuite Render** : le service s'endort après 15 min sans visite, et le
> calcul automatique s'arrête pendant le sommeil. Le réveil prend ~1 min. Pour
> qu'il reste éveillé, configurer une sonde gratuite (UptimeRobot, cron-job.org)
> sur `/healthz` toutes les 5 minutes, ou passer à une offre payante.

## 2. Frontend sur Vercel

1. <https://vercel.com> → **Add New** → **Project** → importer ce dépôt.
2. **Root Directory** : `dashboard/frontend` (le reste est lu dans `vercel.json`).
3. **Environment Variables** : `VITE_API` = adresse du backend Render,
   ex. `https://health-index-isense-api.onrender.com` (sans `/` final).
4. Déployer, puis reporter l'adresse Vercel obtenue dans `CORS_ORIGINS` et
   `DASHBOARD_URL` côté Render.

## 3. Test en local

```powershell
copy dashboard\backend\.env.example dashboard\backend\.env   # puis le compléter
cd dashboard\backend
python -m uvicorn main:app --port 8000
```

Le backend lit `dashboard/backend/.env` au démarrage (les variables déjà
définies dans l'environnement gardent la priorité). Le frontend local lit
`VITE_API` dans `dashboard/frontend/.env.local`.

## Variables d'environnement

| Variable | Défaut | Rôle |
|---|---|---|
| `ISENSE_LIVE` | vide | `1` = données en direct ; vide = rejeu d'un jeu local |
| `ISENSE_API_URL` | API de démonstration i-SENSE | adresse de l'API |
| `ISENSE_EMAIL`, `ISENSE_PASSWORD` | — | compte i-SENSE (obligatoires en direct) |
| `ISENSE_JOURS_AMORCE` | 3 | profondeur du premier relevé |
| `DASHBOARD_USER`, `DASHBOARD_PASSWORD` | — | compte de connexion ; sans mot de passe, la connexion est **désactivée** |
| `AUTH_SECRET` | aléatoire | clé de signature des sessions |
| `AUTH_TTL_H` | 12 | durée d'une session (heures) |
| `PERIODE_CALCUL_S` | 600 | cadence du calcul automatique |
| `CORS_ORIGINS` | `*` | origines autorisées |
| `DASHBOARD_URL` | `http://localhost:5173` | adresse affichée par la racine de l'API |
| `VITE_API` (frontend) | — | adresse du backend ; absente = mode statique |
