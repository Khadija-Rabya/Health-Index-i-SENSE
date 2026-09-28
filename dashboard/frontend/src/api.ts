/**
 * Deux modes de fonctionnement, choisis automatiquement :
 *
 *  LIVE     si VITE_API est defini  -> appelle l'API FastAPI (developpement local)
 *  STATIQUE sinon                   -> lit les JSON precalcules dans /data
 *                                      (deploiement Vercel, sans backend)
 *
 * Le tableau de bord se comporte comme un ecran de supervision : il affiche le
 * DERNIER instant de mesure disponible et le reinterroge toutes les 10 minutes,
 * qui est la cadence d'acquisition i-SENSE. Il n'y a plus d'animation acceleree.
 */

/** Cadence d'acquisition i-SENSE : une mesure toutes les 10 minutes. */
export const PERIODE_MS = 10 * 60 * 1000;
const API = import.meta.env.VITE_API as string | undefined;
export const MODE: "live" | "statique" = API ? "live" : "statique";
const BASE = API ?? "";
const DATA = `${import.meta.env.BASE_URL}data`;

export type Horizon = {
  horizon_pas: number; label: string; cible_horodatage: string;
  predit: number; persistance: number; delta: number;
  proba_mouvement: number; porte_ouverte: boolean; fiable: boolean;
  skill_historique: number | null; r2_historique: number | null;
  acc_historique: number | null; acc_persistance: number | null; modele: string;
};

/** Un canal de l'état système. Les deux — pression et vibration — sont
 *  INDÉPENDANTS : ils décrivent des organes différents et appellent des
 *  interventions différentes. Chacun est jugé sur SA PROPRE MESURE, contre le
 *  seuil fixé pendant le traitement de données et exprimé dans l'unité du
 *  capteur : 0,10 bar pour la marche, 1,5 mm/s² pour la vibration élevée.
 *  L'indice normalisé et ses zones A/B/C/D ont disparu — ils empruntaient le
 *  seuil d'alarme du Health Index huile, un nombre sans unité. */
export type CanalSysteme = {
  nom: "pression" | "vibration"; libelle: string;
  valeur: number | null; unite: string;
  seuil: number; sens: "haut" | "bas"; borne_physique: number | null;
  etat: string; motif: string;
  regle: string; source: string; organe: string;
};

export type Systeme = {
  evaluable: boolean;
  canaux: CanalSysteme[]; motif: string;
};

/** Indicateurs d'aide à la décision. La marge est comptée par rapport à
 *  HI = 0,5, seuil d'alarme des DEUX indices par construction. */
export type Kpis = {
  seuil_alarme: number;
  marge_alarme_huile: number; marge_alarme_systeme: number | null;
  pente_huile_par_jour: number | null; pente_systeme_par_jour: number | null;
  heures_avant_alarme_huile: number | null; heures_avant_alarme_systeme: number | null;
  apport_modele: number; horizon_fiable_max: string | null; n_horizons_fiables: number;
  disponibilite: number | null; vibration_imputee: number | null;
  qualite_valide: number;
};

/** État machine, dérivé de la PRESSION D'HUILE — la seule variable qui
 *  distingue marche et arrêt. La pression et le seuil sont exposés pour que la
 *  décision soit vérifiable, pas seulement affirmée. */
export type EtatMachineDetail = {
  etat: "ON" | "OFF";
  pression_bar: number | null;
  seuil_bar: number;
  depuis_min: number | null;
  n_basculements: number | null;
};

/** Un problème détecté, et les actions correctives qui vont avec.
 *  Les deux sont écrits ensemble : une alerte sans action à faire est une alerte
 *  qu'on apprend à ignorer. */
export type Notification = {
  id: string; machine: string;
  niveau: "critique" | "alerte" | "info";
  titre: string; message: string; actions: string[];
};

export type Machine = {
  machine: string; horodatage: string; health_index: number;
  etat: "Normal" | "Surveillance" | "Alarme";
  etat_machine: "ON" | "OFF"; fraicheur_min: number; perimee: boolean;
  /** Seuil de péremption CALIBRÉ SUR LA CADENCE RÉELLE de la source : 90e
   *  percentile des intervalles observés sur cette machine. Une constante de
   *  30 min supposerait la cadence de l'historique — en direct, l'intervalle
   *  médian est de 80 à 100 min et l'alerte se déclencherait sept fois sur dix. */
  seuil_fraicheur_min?: number;
  qualite: "valide" | "degradee" | "rejetee"; qualite_motif: string;
  horizons: Horizon[];
  systeme: Systeme;
  etat_machine_detail: EtatMachineDetail;
  kpis: Kpis;
  notifications?: Notification[];
};

export type Prediction = {
  /** Horodatage du calcul publie par le planificateur du backend, et du suivant. */
  calcule_a?: string; prochain_calcul?: string; periode_calcul_s?: number;
  maintenant: string; machines: Machine[]; tolerance: number;
  progress?: number; cursor?: number;
};


/** Point de la série OBSERVÉE. La prévision n'est pas ici : elle vient de
 *  /predict, qui porte les horodatages cibles de chaque horizon. */
export type HistPoint = {
  t: string; calcule: number; persistance: number; etat: string; marche: boolean;
};

/** Point de la série SYSTÈME, EN UNITÉS RÉELLES : la pression en bar, la
 *  vibration en mm/s². Elle a sa propre fenêtre — ces grandeurs n'ont de sens
 *  qu'en marche, donc on remonte le long des périodes de fonctionnement au lieu
 *  de prendre les dernières heures calendaires. */
export type PointSysteme = {
  t: string;
  pression: number | null; vibration: number | null;
};

/** Un point de viscosité MESURÉE (cSt / cP), valeurs brutes du capteur —
 *  y compris les valeurs impossibles, que la courbe doit justement montrer. */
export type PointViscosite = {
  t: string;
  vis40: number | null; cinematique: number | null; dynamique: number | null;
  marche: boolean;
};

/** Seuils de Vis_40, servis par le backend : Cadrage des seuils OCP, section 12.
 *  `impossible_projet` (20 cSt) est une règle du projet, pas du cadrage. */
export type SeuilsViscosite = {
  reference: number;
  normal: [number, number]; surveillance: [number, number];
  impossible_projet: number; unite: string; source: string;
};

export type Meta = {
  mode: string; genere_le: string; n_lignes: number; debut: string; fin: string;
  n_frames: number; horizons: number[]; tolerance: number; note: string;
  /** Conservés dans meta.json pour la traçabilité du site déployé ; le détail
   *  de la comparaison de modèles est publié dans le rapport Word. */
  systeme_leaderboard?: unknown[];
  systeme_calibrage?: Record<string, unknown>;
};

/* ─────────────────────────────── connexion ─────────────────────────────── */
/** Jeton de session délivré par POST /auth/login. Conservé dans le navigateur
 *  jusqu'à expiration ou déconnexion. Inutile en mode statique (pas de backend). */
const CLE_JETON = "hi_isense_jeton";

/** Levée quand le backend répond 401 : l'écran doit afficher la connexion. */
export class ConnexionRequise extends Error {
  constructor() { super("connexion requise"); }
}

export const session = {
  jeton: (): string | null => { try { return localStorage.getItem(CLE_JETON); } catch { return null; } },
  enregistrer: (j: string) => { try { localStorage.setItem(CLE_JETON, j); } catch { /* navigation privée */ } },
  effacer: () => { try { localStorage.removeItem(CLE_JETON); } catch { /* idem */ } },
};

function entetes(): Record<string, string> {
  const j = session.jeton();
  return j ? { Authorization: `Bearer ${j}` } : {};
}

async function json<T>(url: string): Promise<T> {
  const r = await fetch(url, { headers: MODE === "live" ? entetes() : {} });
  if (r.status === 401) { session.effacer(); throw new ConnexionRequise(); }
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return r.json();
}

/* ─────────────────────────── état du mode statique ─────────────────────────── */
let _frames: Prediction[] | null = null;
let _hist: { series: Record<string, HistPoint[]>; fenetre_heures?: number;
             series_systeme?: Record<string, PointSysteme[]>;
             fenetre_systeme_jours?: number;
             series_viscosite?: Record<string, PointViscosite[]>;
             seuils_viscosite?: SeuilsViscosite } | null = null;
let _meta: Meta | null = null;

/** Index de l'instantane affiche : TOUJOURS le dernier disponible. */
let _idx = 0;

async function ensureStatic() {
  if (!_frames) {
    _frames = await json<Prediction[]>(`${DATA}/frames.json`);
    _idx = _frames.length - 1;      // dernier instant mesure, pas le premier
  }
  if (!_meta) _meta = await json<Meta>(`${DATA}/meta.json`);
  if (!_hist) _hist = await json<typeof _hist>(`${DATA}/history.json`) as any;
}

/* ──────────────────────────────── API publique ──────────────────────────────── */
export const api = {
  mode: MODE,

  meta: async (): Promise<Meta | null> => {
    if (MODE === "live") return null;
    await ensureStatic();
    return _meta;
  },

  predict: async (): Promise<Prediction> => {
    if (MODE === "live") {
      const r = await fetch(`${BASE}/predict`, { method: "POST", headers: entetes() });
      if (r.status === 401) { session.effacer(); throw new ConnexionRequise(); }
      if (!r.ok) throw new Error(`/predict → ${r.status}`);
      return r.json();
    }
    await ensureStatic();
    return _frames![_idx];
  },

  /** Série observée récente. La prévision est portée par predict(). */
  history: async () => {
    if (MODE === "live") return json<any>(`${BASE}/history?heures=48`);
    await ensureStatic();
    return { series: _hist!.series, fenetre_heures: _hist!.fenetre_heures ?? 48,
             series_systeme: _hist!.series_systeme ?? {},
             fenetre_systeme_jours: _hist!.fenetre_systeme_jours ?? 60,
             series_viscosite: _hist!.series_viscosite ?? {},
             seuils_viscosite: _hist!.seuils_viscosite };
  },

  wsUrl: () => (MODE === "live"
    ? `${BASE.replace(/^http/, "ws")}/ws?token=${encodeURIComponent(session.jeton() ?? "")}`
    : ""),

  /** Connexion : renvoie true si le compte est accepté. */
  connexion: async (utilisateur: string, mot_de_passe: string): Promise<boolean> => {
    const r = await fetch(`${BASE}/auth/login`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ utilisateur, mot_de_passe }),
    });
    if (r.status === 401) return false;
    if (!r.ok) throw new Error(`/auth/login → ${r.status}`);
    const d = await r.json();
    session.enregistrer(d.token);
    return true;
  },

  deconnexion: () => session.effacer(),
};
