"""Export STATIQUE du tableau de bord, pour un hebergement sans backend (Vercel).

Precalcule tout ce que servait l'API, sous forme de fichiers JSON embarques dans
le site :
  frames.json  : N instantanes de prediction le long de la chronologie
                 (le rejeu devient une animation cote navigateur)
  history.json : serie OBSERVEE des 48 dernieres heures de chaque machine, pour
                 les DEUX index. La prevision n'y est pas : elle est portee par
                 frames.json, qui donne les horodatages cibles.
  meta.json    : metriques gelees, horizons, bornes temporelles, et le
                 classement des familles de modeles de l'indice systeme

Le tableau de bord n'affiche plus que ce qui releve de la prevision : les
rapports de qualite et de diagnostic ne sont donc plus exportes.

Sortie : dashboard/frontend/public/data/
"""
import json
import os
import shutil
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, "C:\\pylib")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BE = os.path.join(ROOT, "dashboard", "backend")
sys.path.insert(0, HERE)
sys.path.insert(0, BE)
warnings.filterwarnings("ignore")

from protocol import ASSET_COLS  # noqa: E402
from service import (history_series, load_artifacts, load_dataset,  # noqa: E402
                     predict_latest)

OUT = os.path.join(ROOT, "dashboard", "frontend", "public", "data")
N_FRAMES = int(os.environ.get("N_FRAMES", 180))
WINDOW = 4000


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float):
        return None if not np.isfinite(o) else round(o, 6)
    return o


def dump(obj, name):
    p = os.path.join(OUT, name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(clean(obj), f, ensure_ascii=False, separators=(",", ":"))
    return os.path.getsize(p) / 1024


if os.path.isdir(OUT):
    shutil.rmtree(OUT)
os.makedirs(OUT, exist_ok=True)

print("=" * 92)
print("EXPORT STATIQUE DU TABLEAU DE BORD")
print("=" * 92)
df = load_dataset()
# load_dataset() trie par (session, temps) : les lignes de la Motosoufflante A
# occupent donc tout le debut du tableau et celles de B tout la fin. Un curseur
# qui avance par numero de ligne remonterait le temps a la bascule. On retrie par
# temps, exactement comme le fait le rejeu du backend (replay.py).
df = df.sort_values("created_at").reset_index(drop=True)
art = load_artifacts()
n = len(df)
print(f"  {n} lignes, {df['created_at'].min()} -> {df['created_at'].max()}")
print(f"  ordre chronologique : {df['created_at'].is_monotonic_increasing}")

# ------------------------------------------------------------------ frames
# La fenetre est constituee PAR MACHINE : les deux motosoufflantes ne mesurent pas
# en meme temps, une fenetre globale de 4000 lignes tombe le plus souvent dans la
# session d'une seule d'entre elles et l'autre disparait du tableau de bord. On
# prend donc, pour chaque machine, ses dernieres lignes anterieures a l'instant
# simule ; une machine muette depuis longtemps reste affichee et son indicateur
# de fraicheur le signale ("donnee perimee"), ce qui est l'information utile.
print(f"\n  frames de prediction ({N_FRAMES}) ...")
start = 600                                   # amorce : historique deja constitue
positions = np.linspace(start, n - 1, N_FRAMES).astype(int)
PAR_MACHINE = 1000
COLS_MACHINE = [c for c in ASSET_COLS.values() if c in df.columns]
frames = []
for k, cur in enumerate(positions):
    t_sim = df["created_at"].iloc[cur]
    passe = df[df["created_at"] <= t_sim]
    parts = [passe[passe[c] == 1].tail(PAR_MACHINE) for c in COLS_MACHINE]
    parts = [p for p in parts if not p.empty]
    w = (pd.concat(parts).sort_values("created_at") if parts
         else df.iloc[max(0, cur + 1 - WINDOW):cur + 1])
    try:
        p = predict_latest(w)
        p["progress"] = round(float(k / (N_FRAMES - 1)), 5)
        p["cursor"] = int(cur)
        frames.append(clean(p))
    except Exception as e:
        print(f"    frame {k} ignoree : {type(e).__name__}")
    if (k + 1) % 30 == 0:
        print(f"    {k+1}/{N_FRAMES}")
kb = dump(frames, "frames.json")
print(f"  frames.json : {len(frames)} instantanes, {kb:.0f} Ko")

# ------------------------------------------------------------------ history
print("\n  series historiques ...")
hist = history_series(df, heures=48)
kb = dump(hist, "history.json")
print(f"  history.json : {kb:.0f} Ko")

# ------------------------------------------------------------------- meta
meta = {
    "mode": "statique",
    "genere_le": pd.Timestamp.now().isoformat(),
    "n_lignes": int(n),
    "debut": df["created_at"].min().isoformat(),
    "fin": df["created_at"].max().isoformat(),
    "n_frames": len(frames),
    "horizons": [1, 2, 18, 144],
    "tolerance": 0.01,
    "metriques": art["metrics"],
    # --- Health Index SYSTEME ---
    "systeme_metriques": art["sys_metrics"],
    "systeme_calibrage": art["sys_calibrage"],
    "systeme_leaderboard": art["sys_leaderboard"],
    "note": "Donnees precalculees : le rejeu est une animation cote navigateur, "
            "il n'y a pas d'API vivante derriere ce site.",
}
kb = dump(meta, "meta.json")
print(f"\n  meta.json : {kb:.0f} Ko")

total = sum(os.path.getsize(os.path.join(OUT, f))
            for f in os.listdir(OUT)) / 1024 / 1024
print(f"\n  TOTAL : {total:.2f} Mo dans dashboard/frontend/public/data/")
for f in sorted(os.listdir(OUT)):
    print(f"    {f:<24} {os.path.getsize(os.path.join(OUT, f))/1024:>8.0f} Ko")
