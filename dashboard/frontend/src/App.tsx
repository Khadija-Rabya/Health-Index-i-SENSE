import { Component, useEffect, useRef, useState, type ReactNode } from "react";
import {
  CartesianGrid, ComposedChart, Legend, Line, ReferenceArea, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api, ConnexionRequise, MODE, session, type EtatMachineDetail, type HistPoint, type Horizon,
         type CanalSysteme, type Machine, type Notification, type PointSysteme,
         type PointViscosite, type SeuilsViscosite,
         type Prediction, type Systeme } from "./api";
import "./App.css";

/* Statuts — réservés aux états, jamais réutilisés pour identifier une série. */
const ETAT = { Normal: "#38a169", Surveillance: "#dd6b20", Alarme: "#e53e3e" } as Record<string, string>;

/* Pastille d'identité des quatre horizons, sur une rampe ordonnée : le temps est
   une grandeur, pas une catégorie. */
const HCOL = { 1: "#93c5fd", 2: "#60a5fa", 18: "#2563eb", 144: "#1e3a8a" } as Record<number, string>;
/* HEURES. Le backend sert des horodatages SANS fuseau, dans l'horloge des
 * mesures (celle du PC et du serveur). On les lit toujours comme UTC et on les
 * affiche en UTC : l'heure affichee est alors exactement celle servie, partout.
 * Auparavant l'en-tete les lisait en heure locale du navigateur et les graphes
 * en UTC : avec un navigateur qui place le Maroc a +1 h et Windows a +0 h, une
 * meme mesure apparaissait a deux heures differentes selon l'endroit. */
const epoch = (s: string) =>
  new Date(/[zZ]$|[+-]\d\d:\d\d$/.test(s) ? s : s + "Z").getTime();
const hhmm = (s: string) => new Date(epoch(s)).toLocaleString("fr-FR",
  { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit", timeZone: "UTC" });

/* ═══════════════════════════════════════════════════ PRÉDICTION */

const NIVEAU = {
  critique: { couleur: "#e53e3e", libelle: "CRITIQUE" },
  alerte: { couleur: "#dd6b20", libelle: "ALERTE" },
  info: { couleur: "#718096", libelle: "INFO" },
} as Record<string, { couleur: string; libelle: string }>;


/** Emplacement pour la photo de la machine.
 *
 *  Dépose une image dans `public/machines/` nommée d'après la machine —
 *  `motosoufflante-a` ou `motosoufflante-b` — avec l'une des extensions
 *  acceptées. Elle remplace le repère automatiquement, aucun code à modifier.
 *  Les extensions sont essayées dans l'ordre : si le .jpg n'existe pas, le
 *  navigateur tente le .png, et ainsi de suite jusqu'au repère en pointillés. */
const EXTENSIONS = ["jpg", "jpeg", "png", "webp", "svg"];

function MachineImage({ machine }: { machine: string }) {
  const [essai, setEssai] = useState(0);
  const slug = machine.toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "")
    .replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");

  if (essai < EXTENSIONS.length) {
    return (
      <img className="machine-img" alt={machine} onError={() => setEssai(essai + 1)}
           src={`${import.meta.env.BASE_URL}machines/${slug}.${EXTENSIONS[essai]}`} />
    );
  }
  return (
    <div className="machine-img vide" role="img" aria-label={`${machine}, photo absente`}>
      <svg viewBox="0 0 48 48" width="34" height="34" aria-hidden="true">
        <rect x="5" y="16" width="28" height="18" rx="2" fill="none"
              stroke="currentColor" strokeWidth="2" />
        <circle cx="19" cy="25" r="5.5" fill="none" stroke="currentColor" strokeWidth="2" />
        <path d="M33 21h6l4 4v5h-10z" fill="none" stroke="currentColor" strokeWidth="2" />
        <path d="M9 34v4M29 34v4" stroke="currentColor" strokeWidth="2" />
      </svg>
      <span className="tiny">machines/{slug}.jpg</span>
    </div>
  );
}

/** Tableau des horizons de l'index d'huile : valeur prédite, persistance
 *  et écart entre les deux. La pastille colorée porte l'identité de l'horizon ;
 *  le texte reste en encre neutre. */
function HorizonTable({ horizons }: { horizons: Horizon[] }) {
  return (
    <table className="tbl hz">
      <thead>
        <tr><th>Horizon</th><th>Prédit</th><th>Persistance</th><th>Écart</th></tr>
      </thead>
      <tbody>
        {horizons.map((h) => (
          <tr key={h.horizon_pas} className={h.fiable ? "" : "unreliable"}>
            <td>
              <span className="puce" style={{ background: HCOL[h.horizon_pas] }} aria-hidden="true" />
              <b>{h.label}</b>
              <div className="tiny">{hhmm(h.cible_horodatage)}</div>
            </td>
            <td><b>{h.predit.toFixed(4)}</b></td>
            <td>{h.persistance.toFixed(4)}</td>
            <td style={{ color: Math.abs(h.delta) > 0.01 ? "#b45309" : "#718096" }}>
              {h.delta >= 0 ? "+" : ""}{h.delta.toFixed(5)}
            </td>
          </tr>
        ))}
        {horizons.length === 0 && (
          <tr><td colSpan={4} className="tiny">aucun modèle disponible</td></tr>
        )}
      </tbody>
    </table>
  );
}

const fmtDuree = (min: number | null) => {
  if (min === null || min === undefined) return "—";
  if (min < 60) return `${Math.round(min)} min`;
  const h = Math.floor(min / 60);
  if (h < 48) return `${h} h ${String(Math.round(min % 60)).padStart(2, "0")}`;
  return `${Math.floor(h / 24)} j ${h % 24} h`;
};

/** État machine, lu sur la PRESSION D'HUILE.
 *
 *  C'est la seule variable qui distingue marche et arrêt : la vibration ne le
 *  fait pas — médiane 0,13 en marche contre 0,11 à l'arrêt. On affiche la
 *  pression ET le seuil pour que la décision soit vérifiable. */
function EtatMachineBloc({ e }: { e?: EtatMachineDetail }) {
  // Le backend peut servir une charge utile anterieure a ce bloc — typiquement
  // s'il n'a pas ete relance apres une mise a jour. On n'affiche alors rien
  // plutot que de faire tomber toute la page.
  if (!e) return null;
  const on = e.etat === "ON";
  const teinte = on ? "#2b6cb0" : "#4a5568";
  // Jauge bornée à 8 bar : couvre la plage observée sans écraser le seuil.
  const largeur = (v: number) => `${Math.min(100, Math.max(0, (v / 8) * 100))}%`;
  return (
    <div className="etat-bloc">
      <div className="sys-head">
        <h4>État machine</h4>
        <span className="badge" style={{ background: teinte }}>
          {on ? "EN MARCHE" : "À L'ARRÊT"}
        </span>
      </div>

      {/* La pression est la MESURE qui decide de l'etat : elle se lit en premier
          et en grand, comme les canaux du systeme. Le seuil reste a cote pour
          que la decision soit verifiable, pas seulement affirmee. */}
      <div className="etat-mesure">
        <b style={{ color: teinte }}>
          {e.pression_bar === null ? "—" : e.pression_bar.toFixed(2)}
        </b>
        <span className="unite">bar</span>
        <span className="tiny seuil-txt">
          seuil de marche {e.seuil_bar.toFixed(2)} bar
        </span>
      </div>

      <div className="jauge-pression">
        {e.pression_bar !== null && (
          <div className="remplissage" style={{ width: largeur(e.pression_bar),
                                                background: on ? "#2b6cb0" : "#a0aec0" }} />
        )}
        <div className="marque-seuil" style={{ left: largeur(e.seuil_bar) }} />
      </div>
      <p className="tiny etat-regle">
        au-dessus du seuil, le circuit est en pression : la machine tourne
      </p>

      {/* Deux chiffres de contexte, en tuiles plutot qu'en lignes de tableau :
          « depuis 5 j 11 h » et « 0 basculement » racontent le regime de
          fonctionnement, ils meritent d'etre lus d'un coup d'oeil. */}
      <div className="etat-stats">
        <div className="etat-stat">
          <label>Dans cet état depuis</label>
          <b>{fmtDuree(e.depuis_min)}</b>
        </div>
        <div className="etat-stat">
          <label>Basculements marche / arrêt</label>
          <b>{e.n_basculements ?? "—"}</b>
          <span className="tiny">sur la fenêtre affichée</span>
        </div>
      </div>
    </div>
  );
}

/** État du système : DEUX indicateurs INDÉPENDANTS.
 *
 *  Pression et vibration décrivent des organes différents — circuit hydraulique
 *  d'un côté, mécanique tournante de l'autre — et appellent des interventions
 *  différentes. Les agréger produisait un chiffre sans destinataire : personne
 *  n'intervient sur « le système », on intervient sur la pompe ou sur les
 *  paliers. Ils sont donc lus séparément. */
/** Couleur d'etat d'un canal. Les statuts gardent leurs couleurs reservees ;
 *  ce qui n'est pas evaluable reste gris — l'absence de jugement n'est ni une
 *  bonne ni une mauvaise nouvelle. */
const ETAT_CANAL: Record<string, string> = {
  "Normal": "#38a169",
  "Alarme": "#e53e3e",
  "Mesure impossible": "#e53e3e",
  "À l'arrêt": "#718096",
  "Non évaluable": "#718096",
};

function CanalBloc({ c }: { c: CanalSysteme }) {
  const mesure = c.valeur;
  const decimales = c.unite === "bar" ? 2 : 3;
  const franchi = c.etat === "Alarme" || c.etat === "Mesure impossible";
  const couleur = ETAT_CANAL[c.etat] ?? "#718096";

  // Echelle de la jauge. Pour la vibration on prend deux fois le seuil : sur la
  // borne physique (9,8) une mesure courante de 0,15 serait invisible, et c'est
  // le franchissement du seuil qu'il faut voir venir, pas l'impossible.
  const echelle = c.nom === "pression" ? 8 : c.seuil * 2;
  const largeur = (v: number) => `${Math.min(100, Math.max(0, (v / echelle) * 100))}%`;

  return (
    <div className="canal">
      <div className="canal-tete">
        <h5>{c.libelle}</h5>
        <span className="badge" style={{ background: couleur }}>{c.etat}</span>
      </div>

      <div className="canal-mesure">
        <b style={{ color: franchi ? couleur : "inherit" }}>
          {mesure === null ? "—" : mesure.toFixed(decimales)}
        </b>
        <span className="unite">{c.unite}</span>
        <span className="tiny seuil-txt">
          seuil {c.sens === "haut" ? "d'alarme" : "de marche"} {c.seuil.toFixed(decimales)} {c.unite}
        </span>
      </div>

      {/* La mesure et son seuil sur la meme reglette : on voit la marge, pas
          seulement le verdict. */}
      <div className="jauge-canal">
        {mesure !== null && (
          <div className="remplissage" style={{ width: largeur(mesure), background: couleur }} />
        )}
        <div className="marque-seuil" style={{ left: largeur(c.seuil) }} />
      </div>

      <p className="tiny canal-motif">{c.motif}</p>
      <p className="tiny canal-organe">{c.organe}</p>
      {/* La condition seule. Le fichier qui fixe le seuil reste dans la charge
          utile (`c.source`) et dans le rapport, pour la traçabilité — mais un
          nom de script Python n'a rien à dire à qui surveille une machine. */}
      <p className="tiny canal-regle">{c.regle}</p>
    </div>
  );
}

function SystemeBloc({ s }: { s?: Systeme }) {
  if (!s) return null;
  return (
    <div className="sys-bloc">
      <div className="sys-head">
        <h4>État du système</h4>
      </div>
      {!s.evaluable && s.motif && <p className="hint">{s.motif}</p>}
      <div className="canaux">
        {s.canaux.map((c) => <CanalBloc key={c.nom} c={c} />)}
      </div>
    </div>
  );
}

/** Nom de machine -> nom de fichier : « Motosoufflante A » -> motosoufflante-a */
function slugMachine(machine: string) {
  return machine.toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "")
    .replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
}

/** Schéma de l'équipement, affiché en tête du détail. Dépose une image dans
 *  public/machines/schema.png — ou schema-motosoufflante-a.png pour en avoir un
 *  par machine, il est essayé en premier. */
function SchemaEquipement({ machine }: { machine: string }) {
  const [essai, setEssai] = useState(0);
  const noms = [`schema-${slugMachine(machine)}`, "schema"];
  const exts = ["png", "jpg", "jpeg", "webp", "svg"];
  const candidats = noms.flatMap((n) => exts.map((e) => `${n}.${e}`));
  if (essai >= candidats.length) {
    return (
      <>
        <h4 className="titre-schema">Schéma de l'équipement</h4>
        <div className="schema vide">
          <span className="tiny">Schéma absent — déposer <code>public/machines/schema.png</code></span>
        </div>
      </>
    );
  }
  return (
    <>
      <h4 className="titre-schema">Schéma de l'équipement</h4>
      <div className="schema">
        <img alt={`Schéma ${machine}`} onError={() => setEssai(essai + 1)}
             src={`${import.meta.env.BASE_URL}machines/${candidats[essai]}`} />
      </div>
    </>
  );
}

/** Case de la grille d'accueil : la photo de la machine en fond, son état, et
 *  une barre par indicateur. Cliquable — elle ouvre le détail. */
function MachineTuile({ m, onClic }: { m: Machine; onClic: () => void }) {
  const slug = slugMachine(m.machine);
  const barre = (libelle: string, etat: string, texte: string) => (
    <div className="tuile-barre">
      <span>{libelle}</span>
      <div className="jauge" style={{ background: ETAT_CANAL[etat] ?? ETAT[etat] ?? "#4a5568" }} />
      <b>{texte}</b>
    </div>
  );
  return (
    <button className="tuile" onClick={onClic}
            style={{ backgroundImage:
              `url(${import.meta.env.BASE_URL}machines/${slug}.jpg),
               url(${import.meta.env.BASE_URL}machines/${slug}.png)` }}>
      <div className="tuile-voile" />
      <div className="tuile-contenu">
        <div className="tuile-tete">
          <h3>{m.machine}</h3>
          <span className="badge" style={{ background: ETAT[m.etat] ?? "#718096" }}>
            {m.etat}
          </span>
        </div>
        {barre("HUILE", m.etat, m.health_index.toFixed(3))}
        {m.systeme?.canaux?.map((c) => (
          <div key={c.nom}>
            {barre(c.nom.toUpperCase(), c.etat,
                   c.valeur === null ? "—"
                     : `${c.valeur.toFixed(c.unite === "bar" ? 2 : 3)} ${c.unite}`)}
          </div>
        ))}
        <div className="tuile-pied">
          <span className={`badge ${m.etat_machine === "ON" ? "on" : "off"}`}>
            {m.etat_machine === "ON" ? "EN MARCHE" : "À L'ARRÊT"}
          </span>
          <span className="voir">Voir le détail →</span>
        </div>
      </div>
    </button>
  );
}

function MachineCard({ m }: { m: Machine }) {
  const notes = m.notifications ?? [];
  // Le badge prend la couleur du problème le PLUS GRAVE : un critique ne doit
  // pas se cacher derrière deux info.
  const rang = { critique: 3, alerte: 2, info: 1 } as Record<string, number>;
  const pire = notes.reduce(
    (acc, n) => ((rang[n.niveau] ?? 0) > (rang[acc] ?? 0) ? n.niveau : acc), "info");
  const [ouvert, setOuvert] = useState(false);

  return (
    <div className="card">
      <div className="kpi-head">
        <div className="titre-machine">
          <MachineImage machine={m.machine} />
          <h3>{m.machine}</h3>
        </div>
        <div>
          <span className="badge" style={{ background: ETAT[m.etat] ?? "#718096" }}>{m.etat}</span>
          <span className="badge" style={{ background: m.etat_machine === "ON" ? "#2b6cb0" : "#718096",
                                           marginLeft: 6 }}>
            {m.etat_machine === "ON" ? "EN MARCHE" : "À L'ARRÊT"}
          </span>
          {notes.length > 0 && (
            <button className="badge bouton-notif" aria-expanded={ouvert}
                    style={{ background: (NIVEAU[pire] ?? NIVEAU.info).couleur, marginLeft: 6 }}
                    title={`${notes.length} message(s) — cliquer pour voir les actions correctives`}
                    onClick={() => setOuvert(!ouvert)}>
              {notes.length}&nbsp;{pire === "info"
                ? (notes.length > 1 ? "MESSAGES" : "MESSAGE")
                : (notes.length > 1 ? "ALERTES" : "ALERTE")}
              <span className="chevron" aria-hidden="true">{ouvert ? "▲" : "▼"}</span>
            </button>
          )}
        </div>
      </div>

      {ouvert && (
        <div className="panneau-notif">
          <ZoneNotifications notes={notes} />
          <ZoneActions notes={notes} />
        </div>
      )}

      {/* Le schema vient EN PREMIER : avant tout chiffre, on montre de quel
          equipement on parle et d'ou viennent les mesures. */}
      <SchemaEquipement machine={m.machine} />

      <div className="hi-now">
        <label>Health Index huile calculé</label>
        <b>{m.health_index.toFixed(4)}</b>
        <span className="ts">à {hhmm(m.horodatage)}
          {m.perimee && (
            <em className="stale" title={m.seuil_fraicheur_min
              ? `Seuil : ${m.seuil_fraicheur_min.toFixed(0)} min — 90e percentile des `
                + `intervalles réellement observés sur cette machine.`
              : undefined}>
              {" "}· donnée périmée ({m.fraicheur_min.toFixed(0)} min
              {m.seuil_fraicheur_min ? ` > ${m.seuil_fraicheur_min.toFixed(0)}` : ""})
            </em>
          )}
        </span>
      </div>
      <HorizonTable horizons={m.horizons} />

      <EtatMachineBloc e={m.etat_machine_detail} />

      <SystemeBloc s={m.systeme} />

      {/* Notifications et actions sont dépliées par le bouton de l'en-tête,
          au-dessus — elles n'ont pas à être répétées ici. */}
    </div>
  );
}

/** Courbe de supervision : à gauche ce qui vient d'être mesuré, à droite la
 *  prévision qui part de l'instant courant vers les quatre horizons.
 *
 *  L'axe est un VRAI axe de temps (numérique), pas une suite de catégories :
 *  sans cela les échéances 10 min, 20 min, 3 h et 24 h seraient espacées à
 *  intervalles égaux et le graphique mentirait sur la distance au futur. */
function CourbeIndice({ titre, hint, series, pred, champ, horizonsDe, valeurDe }:
    { titre: string; hint: React.ReactNode;
      series: Record<string, Array<{ t: string; calcule?: number; systeme?: number | null }>>;
      pred: Prediction | null; champ: "calcule" | "systeme";
      horizonsDe: (m: Machine) => Horizon[];
      valeurDe: (m: Machine) => number | null }) {
  const machines = Object.keys(series);
  return (
    <div className="card">
      <h3>{titre}</h3>
      <p className="hint">{hint}</p>
      {machines.map((machine) => {
        const pts = series[machine] ?? [];
        const m = pred?.machines.find((x) => x.machine === machine) ?? null;
        const courant = m ? valeurDe(m) : null;
        const horizons = m ? horizonsDe(m) : [];

        const data: any[] = pts.map((p) => ({
          ts: epoch(p.t),
          observe: (champ === "calcule" ? p.calcule : p.systeme) ?? null,
        }));
        // L'instant courant porte les trois séries : c'est le point de raccord
        // entre la mesure et la prévision.
        if (data.length && courant !== null) {
          data[data.length - 1] = { ...data[data.length - 1],
                                    predit: courant, persistance: courant };
        }
        horizons.forEach((h) => data.push({
          ts: epoch(h.cible_horodatage),
          observe: null, predit: h.predit, persistance: h.persistance,
        }));
        const maintenant = data.length && courant !== null
          ? data[Math.max(0, pts.length - 1)].ts : null;
        const aucuneFiable = horizons.length > 0 && horizons.every((h) => !h.fiable);

        return (
          <div key={machine} className="chart-block">
            <h4>{machine}</h4>
            {!data.some((d) => d.observe !== null) && (
              <p className="hint bad-hint">
                Aucune mesure exploitable sur la fenêtre — machine à l'arrêt.
              </p>
            )}
            <ResponsiveContainer width="100%" height={250}>
              <ComposedChart data={data} margin={{ top: 6, right: 14, bottom: 4, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#edf2f7" />
                <XAxis dataKey="ts" type="number" scale="time"
                       domain={["dataMin", "dataMax"]} fontSize={10} minTickGap={50}
                       stroke="#a0aec0"
                       tickFormatter={(v: any) => hhmm(new Date(v).toISOString())} />
                <YAxis domain={["auto", "auto"]} fontSize={10} stroke="#a0aec0"
                       tickFormatter={(v: any) => Number(v).toFixed(3)} />
                <Tooltip labelFormatter={(l: any) => hhmm(new Date(l).toISOString())}
                         formatter={(v: any, n: any) =>
                           [v === null ? "—" : Number(v).toFixed(5), n]} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <ReferenceLine y={0.5} stroke="#e53e3e" strokeDasharray="2 4"
                               label={{ value: "alarme", position: "insideBottomRight",
                                        fontSize: 10, fill: "#e53e3e" }} />
                {maintenant !== null && (
                  <ReferenceLine x={maintenant} stroke="#2d3748" strokeDasharray="4 3"
                                 label={{ value: "maintenant", position: "top",
                                          fontSize: 10, fill: "#2d3748" }} />
                )}
                <Line type="monotone" dataKey="persistance" name="Persistance"
                      stroke="#a0aec0" dot={false} strokeWidth={1.5}
                      strokeDasharray="5 4" connectNulls />
                <Line type="monotone" dataKey="observe" name="Mesuré"
                      stroke="#1d4ed8" dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="predit" name="Prévision"
                      stroke="#0d9488" strokeWidth={2} connectNulls
                      dot={{ r: 3, fill: "#0d9488" }} />
              </ComposedChart>
            </ResponsiveContainer>
            {aucuneFiable && (
              <p className="hint">
                Aucun horizon n'est jugé fiable sur cette machine : la prévision affichée
                est la persistance elle-même, et les deux courbes se superposent.
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}

/** Les deux indicateurs du système, tracés côte à côte sur la même échelle.
 *
 *  Deux séries, pas d'agrégat : on veut voir lequel des deux canaux bouge. Une
 *  seule échelle verticale — les deux indicateurs sont dans ]0, 1] et partagent
 *  le même seuil d'alarme, donc un axe commun est légitime. */
/** Les deux canaux du systeme, EN PETITS MULTIPLES.
 *
 *  Un graphe par variable, chacun avec SON axe et SON unite. La version
 *  precedente superposait les deux sur un axe 0-1 sans unite : « 0,76 » et
 *  « 0,62 » se lisaient comme deux points d'une meme grandeur, alors que l'une
 *  est une pression et l'autre une vibration. Un axe commun n'a de sens que pour
 *  des series comparables — ce n'est pas le cas ici. */
/*  `zero` : l'axe part-il de zero ?
 *  VIBRATION oui — elle descend reellement jusqu'a 0,01, et zero veut dire
 *    « immobile ». L'echelle complete est lisible.
 *  PRESSION non — elle vit entre 4 et 6,5 bar. Ancrer a zero ecraserait toute
 *    la variation dans le cinquieme haut du graphe : une chute de 6,30 a 5,10,
 *    soit 19 %, deviendrait invisible. C'est precisement ce qu'on cherche a
 *    voir. L'axe est donc cadre sur les donnees, et le sous-titre le dit. */
const CANAL_TRACE = [
  { cle: "pression" as const, titre: "Pression d'huile",
    couleur: "#1d4ed8", dec: 2, zero: false },
  { cle: "vibration" as const, titre: "Vibration du système d'huile",
    couleur: "#0d9488", dec: 3, zero: true },
];

function CourbeSysteme({ series, fenetreJours, unites, seuils, libelles }:
    { series: Record<string, PointSysteme[]>;
      fenetreJours?: number | null;
      unites?: Record<string, string>;
      seuils?: Record<string, number>;
      libelles?: Record<string, string> }) {
  const unite = (c: string) => unites?.[c] ?? (c === "pression" ? "bar" : "mm/s²");
  const seuil = (c: string) => seuils?.[c];
  const nomSeuil = (c: string) =>
    libelles?.[c] ?? (c === "pression" ? "seuil de marche" : "seuil d'alarme");
  return (
    <div className="card">
      <h3>État du système — pression et vibration</h3>
      <p className="hint">
        Deux grandeurs <b>indépendantes</b>, jamais agrégées : la pression décrit le
        circuit hydraulique, la vibration la mécanique tournante. Chacune est tracée
        dans <b>son unité de mesure</b>, sur son propre axe. Cette courbe a sa propre
        fenêtre — ces grandeurs n'ont de sens qu'en marche, on montre donc
        {fenetreJours ? ` les ${fenetreJours} derniers jours` : " tout l'historique"} de
        <b> fonctionnement</b>, arrêts intercalés ignorés.
      </p>
      {Object.entries(series).map(([machine, pts]) => {
        const data = (pts ?? []).map((p) => ({
          ts: epoch(p.t),
          pression: p.pression, vibration: p.vibration,
        }));
        return (
          <div key={machine} className="chart-block">
            <h4>{machine}</h4>
            {data.length === 0 && (
              <p className="hint bad-hint">
                Aucune période de fonctionnement sur cette machine.
              </p>
            )}
            {data.length > 0 && (
              <div className="petits-multiples">
                {CANAL_TRACE.map((c) => (
                  <div key={c.cle} className="multiple">
                    <h5>{c.titre} <span className="unite-axe">({unite(c.cle)})</span>
                      {!c.zero && seuil(c.cle) === undefined &&
                        <span className="unite-axe"> · axe non ancré à zéro</span>}
                    </h5>
                    <ResponsiveContainer width="100%" height={210}>
                      <ComposedChart data={data}
                                     margin={{ top: 6, right: 12, bottom: 4, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#edf2f7" />
                        <XAxis dataKey="ts" type="number" scale="time"
                               domain={["dataMin", "dataMax"]} fontSize={10}
                               minTickGap={50} stroke="#a0aec0"
                               tickFormatter={(v: any) => hhmm(new Date(v).toISOString())} />
                        {/* Axe propre au canal. Voir CANAL_TRACE pour le choix du
                            zero : il depend de la grandeur, pas d'une regle unique. */}
                        <YAxis domain={(c.zero || seuil(c.cle) !== undefined)
                                 ? [0, "auto"] : ["auto", "auto"]}
                               fontSize={10} stroke="#a0aec0"
                               width={52}
                               tickFormatter={(v: any) => Number(v).toFixed(c.dec)}
                               label={{ value: unite(c.cle), angle: -90, position: "insideLeft",
                                        fontSize: 10, fill: "#a0aec0" }} />
                        <Tooltip labelFormatter={(l: any) => hhmm(new Date(l).toISOString())}
                                 formatter={(v: any) =>
                                   [v === null ? "—" : `${Number(v).toFixed(c.dec)} ${unite(c.cle)}`,
                                    c.titre]} />
                        {/* Le seuil est un REPERE, pas une serie : trait fin,
                            discontinu, en encre neutre, annote de sa valeur. */}
                        {seuil(c.cle) !== undefined && (
                          <ReferenceLine y={seuil(c.cle)} stroke="#2d3748"
                                         strokeDasharray="4 3" strokeWidth={1.2}
                                         label={{
                                           value: `${nomSeuil(c.cle)} ${seuil(c.cle)} ${unite(c.cle)}`,
                                           position: "insideTopLeft",
                                           fontSize: 9.5, fill: "#2d3748" }} />
                        )}
                        <Line type="monotone" dataKey={c.cle} name={c.titre}
                              stroke={c.couleur} dot={false} strokeWidth={2}
                              connectNulls={false} />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}


/** Seuils par défaut si le backend ne les sert pas (export statique ancien).
 *  Mêmes valeurs que le Cadrage des seuils OCP, section 12. */
const SEUILS_VISC_DEFAUT: SeuilsViscosite = {
  reference: 46.0, normal: [43.7, 48.3], surveillance: [41.4, 50.6],
  impossible_projet: 20, unite: "cSt",
  source: "Cadrage des seuils OCP, section 12 (Vis_40)",
};

/** Classe d'une valeur de Vis_40 selon le cadrage OCP (+ plancher du projet). */
function classeVisc(v: number | null, s: SeuilsViscosite) {
  if (v === null) return { libelle: "non mesurée", couleur: "#718096" };
  if (v < s.impossible_projet) return { libelle: "impossible — défaut capteur probable", couleur: ETAT.Alarme };
  if (v >= s.normal[0] && v <= s.normal[1]) return { libelle: "normale", couleur: ETAT.Normal };
  if (v >= s.surveillance[0] && v <= s.surveillance[1]) return { libelle: "surveillance", couleur: ETAT.Surveillance };
  return { libelle: "critique", couleur: ETAT.Alarme };
}

/** Viscosité à 40 °C MESURÉE, avec les bandes du Cadrage OCP.
 *
 *  Les bandes sont des REPÈRES d'état (vert normal, orange surveillance) ; la
 *  série elle-même reste en bleu « mesuré ». Le plancher de 20 cSt est tracé à
 *  part et nommé comme règle du projet, pour ne pas le faire passer pour une
 *  valeur du cadrage. La cinématique et la dynamique sont dans l'infobulle :
 *  c'est ce qui permet de voir, au survol, qu'une Vis_40 à 19 cSt à côté d'une
 *  cinématique à 71 cSt vient du capteur et non de l'huile. */
function CourbeViscosite({ series, seuils }:
    { series: Record<string, PointViscosite[]>; seuils?: SeuilsViscosite }) {
  const s = seuils ?? SEUILS_VISC_DEFAUT;
  return (
    <div className="card">
      <h3>Viscosité de l'huile à 40 °C — valeurs mesurées</h3>
      <p className="hint">
        Valeurs brutes du viscosimètre sur les 48 dernières heures. Bande <b>verte</b> :
        plage normale du Cadrage OCP ({s.normal[0]}–{s.normal[1]} {s.unite}) ; bandes
        <b> orange</b> : surveillance ({s.surveillance[0]}–{s.normal[0]} et {s.normal[1]}–{s.surveillance[1]} {s.unite}) ;
        au-delà : critique. La ligne rouge ({s.impossible_projet} {s.unite}) est la règle
        « mesure impossible » du projet — elle n'est pas dans le cadrage. Survolez un
        point pour voir aussi la viscosité cinématique et dynamique.
      </p>
      {Object.entries(series).map(([machine, pts]) => {
        const data = (pts ?? []).map((p) => ({
          ts: epoch(p.t), vis40: p.vis40,
          cinematique: p.cinematique, dynamique: p.dynamique, marche: p.marche,
        }));
        const dernier = [...(pts ?? [])].reverse().find((p) => p.vis40 !== null) ?? null;
        const cl = classeVisc(dernier?.vis40 ?? null, s);
        const maxV = Math.max(s.surveillance[1] + 5,
          ...data.map((d) => d.vis40 ?? 0));
        return (
          <div key={machine} className="chart-block">
            <h4>{machine}</h4>
            {data.length === 0 && <p className="hint bad-hint">Aucune mesure de viscosité.</p>}
            {dernier && (
              <p className="hint">
                Dernière valeur : <b style={{ color: cl.couleur }}>
                  {dernier.vis40!.toFixed(2)} {s.unite} — {cl.libelle}</b>
                {dernier.cinematique !== null && <> · cinématique {dernier.cinematique.toFixed(2)} cSt</>}
                {dernier.dynamique !== null && <> · dynamique {dernier.dynamique.toFixed(2)} cP</>}
              </p>
            )}
            {data.length > 0 && (
              <ResponsiveContainer width="100%" height={240}>
                <ComposedChart data={data} margin={{ top: 6, right: 12, bottom: 4, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#edf2f7" />
                  <XAxis dataKey="ts" type="number" scale="time" domain={["dataMin", "dataMax"]}
                         fontSize={10} minTickGap={50} stroke="#a0aec0"
                         tickFormatter={(v: any) => hhmm(new Date(v).toISOString())} />
                  <YAxis domain={[0, Math.ceil(maxV / 5) * 5]} fontSize={10} stroke="#a0aec0" width={52}
                         label={{ value: s.unite, angle: -90, position: "insideLeft",
                                  fontSize: 10, fill: "#a0aec0" }} />
                  <ReferenceArea y1={s.surveillance[0]} y2={s.normal[0]}
                                 fill={ETAT.Surveillance} fillOpacity={0.12} />
                  <ReferenceArea y1={s.normal[0]} y2={s.normal[1]}
                                 fill={ETAT.Normal} fillOpacity={0.15} />
                  <ReferenceArea y1={s.normal[1]} y2={s.surveillance[1]}
                                 fill={ETAT.Surveillance} fillOpacity={0.12} />
                  <ReferenceLine y={s.reference} stroke="#2d3748" strokeDasharray="4 3"
                                 strokeWidth={1}
                                 label={{ value: `référence ISO VG 46 : ${s.reference} ${s.unite}`,
                                          position: "insideTopLeft", fontSize: 9.5, fill: "#2d3748" }} />
                  <ReferenceLine y={s.impossible_projet} stroke={ETAT.Alarme} strokeDasharray="4 3"
                                 strokeWidth={1.2}
                                 label={{ value: `impossible (règle projet) < ${s.impossible_projet} ${s.unite}`,
                                          position: "insideBottomLeft", fontSize: 9.5, fill: ETAT.Alarme }} />
                  <Tooltip labelFormatter={(l: any) => hhmm(new Date(l).toISOString())}
                           formatter={(v: any, nom: any) => {
                             if (v === null || v === undefined) return ["—", nom];
                             const u = nom === "Dynamique" ? "cP" : "cSt";
                             return [`${Number(v).toFixed(2)} ${u}`, nom];
                           }} />
                  <Line type="monotone" dataKey="vis40" name="Viscosité à 40 °C"
                        stroke="#1d4ed8" strokeWidth={2} connectNulls={false}
                        dot={(props: any) => {
                          const v = props.payload?.vis40;
                          if (v === null || v === undefined || v >= s.impossible_projet)
                            return <g key={props.key} />;
                          return <circle key={props.key} cx={props.cx} cy={props.cy} r={3}
                                         fill={ETAT.Alarme} />;
                        }} />
                  {/* Séries d'infobulle seulement : non tracées (trait transparent). */}
                  <Line dataKey="cinematique" name="Cinématique" stroke="transparent"
                        dot={false} activeDot={false} legendType="none" />
                  <Line dataKey="dynamique" name="Dynamique" stroke="transparent"
                        dot={false} activeDot={false} legendType="none" />
                </ComposedChart>
              </ResponsiveContainer>
            )}
          </div>
        );
      })}
      <p className="tiny">Source des seuils : {s.source}.</p>
    </div>
  );
}

/** Zone de NOTIFICATIONS : tout ce que le système a détecté, sur toutes les
 *  machines, du plus grave au moins grave. */
function ZoneNotifications({ notes }: { notes: Notification[] }) {
  return (
    <div className="zone-bloc">
      <div className="sys-head">
        <h4>Notifications</h4>
        <span className="compteur">{notes.length}</span>
      </div>
      {notes.length === 0 && (
        <p className="hint">Aucun problème détecté sur les machines suivies.</p>
      )}
      <ul className="notifs">
        {notes.map((n) => (
          <li key={n.id} className={`notif ${n.niveau}`}>
            <div className="notif-tete">
              <span className="badge" style={{ background: NIVEAU[n.niveau].couleur }}>
                {NIVEAU[n.niveau].libelle}
              </span>
              <b>{n.titre}</b>
            </div>
            <p>{n.message}</p>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Zone d'ACTIONS CORRECTIVES : ce qu'on peut faire, rattaché au problème qui
 *  le motive. Une action orpheline ne dit pas pourquoi on la ferait. */
function ZoneActions({ notes }: { notes: Notification[] }) {
  const avecActions = notes.filter((n) => n.actions?.length);
  return (
    <div className="zone-bloc">
      <div className="sys-head"><h4>Actions correctives</h4></div>
      {avecActions.length === 0 && (
        <p className="hint">Aucune action requise.</p>
      )}
      {avecActions.map((n) => (
        <div key={n.id} className="bloc-actions">
          <div className="notif-tete">
            <span className="puce" style={{ background: NIVEAU[n.niveau].couleur }}
                  aria-hidden="true" />
            <b>{n.titre}</b>
          </div>
          <ol>
            {n.actions.map((a, i) => <li key={i}>{a}</li>)}
          </ol>
        </div>
      ))}
    </div>
  );
}

function PredictionPanel({ pred, series, seriesSys, unitesSys,
                           seuilsSys, libellesSeuils, seriesVisc, seuilsVisc }:
    { pred: Prediction | null; series: Record<string, HistPoint[]>;
      seriesSys: Record<string, PointSysteme[]>;
      seriesVisc: Record<string, PointViscosite[]>;
      seuilsVisc?: SeuilsViscosite;
      unitesSys?: Record<string, string>;
      seuilsSys?: Record<string, number>;
      libellesSeuils?: Record<string, string> }) {
  // `null` = grille d'accueil ; un nom de machine = vue de detail.
  const [choisie, setChoisie] = useState<string | null>(null);

  if (!pred) return <div className="card">Chargement…</div>;

  if (choisie === null) {
    return (
      <>
        <p className="hint">
          Deux motosoufflantes suivies. Cliquez sur une machine pour ouvrir son
          Health Index d'huile, son état machine, son état système et ses courbes.
        </p>
        <section className="grille-machines">
          {pred.machines.map((m) => (
            <MachineTuile key={m.machine} m={m} onClic={() => setChoisie(m.machine)} />
          ))}
        </section>
      </>
    );
  }

  const m = pred.machines.find((x) => x.machine === choisie);
  if (!m) return <div className="card">Machine introuvable.</div>;

  // On ne garde que la machine choisie : les courbes ne montrent qu'elle.
  const uneMachine = <T,>(src: Record<string, T>) =>
    choisie in src ? { [choisie]: src[choisie] } : {};
  const predUne: Prediction = { ...pred, machines: [m] };

  return (
    <>
      <button className="retour" onClick={() => setChoisie(null)}>
        ← Toutes les machines
      </button>

      <section className="kpis">
        <MachineCard m={m} />
      </section>

      <CourbeIndice
        titre="Health index HUILE — mesuré et prévu"
        hint={<>À gauche du repère <b>maintenant</b>, les 48 dernières heures mesurées.
              À droite, la prévision aux quatre horizons. La ligne <b>persistance</b> est
              ce que donnerait « rien ne change » : l'écart entre les deux courbes de
              droite est tout ce que le modèle apporte.</>}
        series={uneMachine(series)} pred={predUne} champ="calcule"
        horizonsDe={(x) => x.horizons} valeurDe={(x) => x.health_index} />

      <CourbeSysteme series={uneMachine(seriesSys)} unites={unitesSys}
                     seuils={seuilsSys} libelles={libellesSeuils} />

      <CourbeViscosite series={uneMachine(seriesVisc)} seuils={seuilsVisc} />
    </>
  );
}

/** Garde-fou de rendu.
 *
 *  Sans lui, la moindre erreur dans un composant vide TOUTE la page — l'ecran
 *  devient blanc, sans aucune indication. C'est arrive : le backend servait une
 *  charge utile anterieure a un bloc de l'interface, un champ manquait, et le
 *  tableau de bord disparaissait. Un ecran de supervision doit dire ce qui ne va
 *  pas, jamais s'effacer en silence. */
class GardeFou extends Component<{ children: ReactNode },
                                 { erreur: Error | null }> {
  state = { erreur: null as Error | null };

  static getDerivedStateFromError(erreur: Error) {
    return { erreur };
  }

  render() {
    if (!this.state.erreur) return this.props.children;
    return (
      <div className="err">
        <b>L'affichage a rencontré une erreur.</b> Les données reçues ne
        correspondent pas à ce que cette version du tableau de bord attend — le
        plus souvent, le backend n'a pas été relancé après une mise à jour.
        <div className="tiny" style={{ marginTop: 6 }}>
          {String(this.state.erreur?.message ?? this.state.erreur)}
        </div>
      </div>
    );
  }
}

/* ═══════════════════════════════════════════════════ APP */
/* ═══════════════════════════════════════════════════ CONNEXION */
function Connexion({ surConnexion }: { surConnexion: () => void }) {
  const [utilisateur, setUtilisateur] = useState("");
  const [motDePasse, setMotDePasse] = useState("");
  const [erreur, setErreur] = useState<string | null>(null);
  const [envoi, setEnvoi] = useState(false);

  const valider = async (e: React.FormEvent) => {
    e.preventDefault();
    setEnvoi(true); setErreur(null);
    try {
      if (await api.connexion(utilisateur.trim(), motDePasse)) surConnexion();
      else setErreur("Identifiant ou mot de passe incorrect.");
    } catch {
      setErreur("Serveur injoignable. Réessayez dans un instant : un serveur "
                + "gratuit peut mettre jusqu'à une minute à se réveiller.");
    } finally { setEnvoi(false); }
  };

  return (
    <div className="connexion">
      <form className="connexion-carte" onSubmit={valider}>
        <h1>Health Index i-SENSE</h1>
        <p className="hint">Motosoufflantes A &amp; B · accès réservé</p>
        <label>Identifiant
          <input value={utilisateur} onChange={(e) => setUtilisateur(e.target.value)}
                 autoComplete="username" autoFocus required />
        </label>
        <label>Mot de passe
          <input type="password" value={motDePasse}
                 onChange={(e) => setMotDePasse(e.target.value)}
                 autoComplete="current-password" required />
        </label>
        {erreur && <div className="err">{erreur}</div>}
        <button className="btn-actualiser" type="submit" disabled={envoi}>
          {envoi ? "Connexion…" : "Se connecter"}
        </button>
      </form>
    </div>
  );
}

/** En mode statique il n'y a pas de backend, donc pas de connexion : le site
 *  déployé en statique doit être protégé par l'hébergeur lui-même. */
export default function App() {
  const [connecte, setConnecte] = useState(MODE !== "live" || session.jeton() !== null);
  if (!connecte) return <Connexion surConnexion={() => setConnecte(true)} />;
  return <TableauDeBord surDeconnexion={() => { api.deconnexion(); setConnecte(false); }} />;
}

function TableauDeBord({ surDeconnexion }: { surDeconnexion: () => void }) {
  const [pred, setPred] = useState<Prediction | null>(null);
  const [series, setSeries] = useState<Record<string, HistPoint[]>>({});
  const [seriesSys, setSeriesSys] = useState<Record<string, PointSysteme[]>>({});
  // Unites servies par le backend : chaque canal trace dans la sienne.
  const [unitesSys, setUnitesSys] = useState<Record<string, string>>(
    { pression: "bar", vibration: "mm/s²" });
  // Seuils traces sur les courbes, servis par le backend.
  const [seuilsSys, setSeuilsSys] = useState<Record<string, number> | undefined>();
  const [libellesSeuils, setLibellesSeuils] =
    useState<Record<string, string> | undefined>();
  // Viscosite mesuree et seuils du cadrage OCP, servis par le backend.
  const [seriesVisc, setSeriesVisc] = useState<Record<string, PointViscosite[]>>({});
  const [seuilsVisc, setSeuilsVisc] = useState<SeuilsViscosite | undefined>();
  const [transport, setTransport] = useState<"websocket" | "polling" | "statique" | "…">("…");
  const [err, setErr] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  // Le bouton « Actualiser » appelle la MEME fonction de lecture que le
  // rafraichissement automatique : aucun second chemin de code a maintenir.
  const lireRef = useRef<(() => Promise<void>) | null>(null);
  const [actualisation, setActualisation] = useState(false);
  // Heure d'actualisation lue sur l'horloge du SERVEUR, la meme que celle des
  // mesures : l'horloge du navigateur peut appliquer un autre fuseau (constate :
  // +1 h pour le Maroc), ce qui faisait paraitre une mesure recente vieille d'une heure.
  const [derniereActu, setDerniereActu] = useState<string | null>(null);

  const actualiser = async () => {
    if (!lireRef.current || actualisation) return;
    setActualisation(true);
    try { await lireRef.current(); } finally { setActualisation(false); }
  };

  // Cadence UNIQUE du tableau de bord : celle des capteurs, une mesure toutes
  // les 10 minutes. Entre deux mesures l'écran ne bouge pas — c'est le
  // comportement attendu d'un écran de supervision, il n'y a rien de nouveau
  // à afficher tant que la mesure suivante n'est pas arrivée.
  useEffect(() => {
    let poll: number | undefined; let closed = false;
    const lire = async () => {
      try {
        const p = await api.predict();
        setPred(p);
        const h = await api.history();
        setSeries(h.series);
        setSeriesSys(h.series_systeme ?? {});
        if (h.unites_systeme) setUnitesSys(h.unites_systeme);
        if (h.seuils_systeme) setSeuilsSys(h.seuils_systeme);
        if (h.libelles_seuils) setLibellesSeuils(h.libelles_seuils);
        setSeriesVisc(h.series_viscosite ?? {});
        if (h.seuils_viscosite) setSeuilsVisc(h.seuils_viscosite);
        setDerniereActu(p.calcule_a ?? p.maintenant);
        setErr(null);
      } catch (e: any) {
        if (e instanceof ConnexionRequise) { surDeconnexion(); return; }
        setErr(String(e.message ?? e));
      }
    };
    lireRef.current = lire;
    const startPolling = () => {
      if (poll) return;
      setTransport(MODE === "statique" ? "statique" : "polling");
      lire();
      // Lecture du CACHE du backend : peu couteuse, on peut interroger souvent.
      // Un nouveau calcul (toutes les 10 min) apparait donc en moins d'une minute.
      poll = window.setInterval(lire, 60_000);
    };
    if (MODE === "statique") { startPolling(); return () => { if (poll) clearInterval(poll); }; }
    try {
      const ws = new WebSocket(api.wsUrl());
      wsRef.current = ws;
      ws.onopen = () => { if (!closed) { setTransport("websocket"); lire(); } };
      ws.onmessage = (ev) => {
        const m = JSON.parse(ev.data);
        // Nouveau calcul publie : on relit aussi les courbes, pour que tout
        // l'ecran corresponde au MEME calcul.
        if (m.type === "tick") { lire(); }
      };
      ws.onerror = () => startPolling();
      ws.onclose = () => { if (!closed) startPolling(); };
    } catch { startPolling(); }
    return () => { closed = true; wsRef.current?.close(); if (poll) clearInterval(poll); };
  }, []);

  return (
    <div className="app">
      <header>
        <div>
          <h1>Health Index i-SENSE</h1>
          <p>Motosoufflantes A &amp; B · index d'huile prévu à 10 min / 20 min / 3 h / 24 h
             · état machine et état du système</p>
        </div>
        <div className="transport">
          <span className={`dot ${transport}`} /> {transport}
          {pred && pred.machines.length > 0 && (() => {
            // La mesure la plus RECENTE recue du capteur, toutes machines
            // confondues — et non l'horloge du serveur (`maintenant`), que
            // l'ancien libelle affichait a tort sous le nom de « derniere mesure ».
            const recente = pred.machines.reduce((a, m) =>
              m.horodatage > a.horodatage ? m : a);
            const age = Math.round(recente.fraicheur_min);
            return (
              <span className="sim"> · dernière mesure reçue {hhmm(recente.horodatage)}
                {" "}({age < 1 ? "à l'instant" : age < 120 ? `il y a ${age} min`
                                                  : `il y a ${Math.round(age / 60)} h`})</span>
            );
          })()}
          <button className="btn-actualiser" onClick={actualiser} disabled={actualisation}
                  title="Interroger immédiatement le capteur i-SENSE">
            {actualisation ? "Actualisation…" : "⟳ Actualiser"}
          </button>
          {MODE === "live" && (
            <button className="btn-deconnexion" onClick={surDeconnexion}>Déconnexion</button>
          )}
          {derniereActu && (
            <div className="tiny">
              calculé à {derniereActu.slice(11, 16)}
              {pred?.prochain_calcul && <> · prochain calcul à {pred.prochain_calcul.slice(11, 16)}</>}
            </div>
          )}
        </div>
      </header>

      {err && <div className="err">Erreur : {err}</div>}

      <GardeFou>
        <PredictionPanel pred={pred} series={series} seriesSys={seriesSys}
                         unitesSys={unitesSys} seuilsSys={seuilsSys}
                         libellesSeuils={libellesSeuils}
                         seriesVisc={seriesVisc} seuilsVisc={seuilsVisc} />
      </GardeFou>

      <footer>
        Un index d'huile, un état machine et deux canaux système · quatre horizons de
        prévision, sur l'huile seule · tolérance ±0,01 · l'écran se réinterroge
        toutes les 10 minutes ; les capteurs livrent à leur propre rythme.
      </footer>
    </div>
  );
}
