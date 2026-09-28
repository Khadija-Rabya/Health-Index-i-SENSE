"""Moteur de REJEU — il n'existe pas de flux temps reel, donc on rejoue le CSV
historique a vitesse configurable, PAR LE MEME CHEMIN DE CODE qu'un flux live.

Le moteur ne connait rien au modele : il expose seulement « quelles lignes sont
arrivees jusqu'a maintenant ». C'est exactement ce qu'un connecteur API fournirait,
et c'est ce qui garantit que le passage au vrai temps reel ne changera qu'une
classe (la source), pas la chaine de traitement.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ReplayState:
    speed: float = 60.0          # 1 s reelle = `speed` s simulees
    playing: bool = True
    cursor: int = 0              # index de la derniere ligne « arrivee »
    t0_wall: float = field(default_factory=time.time)
    t0_sim: pd.Timestamp | None = None


class ReplayFeed:
    """Rejoue un DataFrame trie par horodatage comme s'il arrivait en direct."""

    def __init__(self, df: pd.DataFrame, speed: float = 60.0, warmup_rows: int = 600):
        self.df = df.sort_values("created_at").reset_index(drop=True)
        self.times = self.df["created_at"].to_numpy()
        self._lock = threading.Lock()
        self.state = ReplayState(speed=speed)
        # on demarre avec un historique deja constitue : sans cela les features
        # a etat (EWMA, lags, time_in_session_h) n'auraient rien a se mettre sous
        # la dent au premier tick — le meme probleme qu'un demarrage a froid en
        # production, traite ici de la meme facon.
        self.state.cursor = min(warmup_rows, len(self.df) - 1)
        self.state.t0_sim = pd.Timestamp(self.times[self.state.cursor])
        self.state.t0_wall = time.time()

    # ---------------------------------------------------------------- controle
    def set_speed(self, speed: float):
        with self._lock:
            self._sync()
            self.state.speed = max(0.1, float(speed))
            self.state.t0_wall = time.time()
            self.state.t0_sim = pd.Timestamp(self.times[self.state.cursor])

    def play(self, playing: bool):
        with self._lock:
            self._sync()
            self.state.playing = bool(playing)
            self.state.t0_wall = time.time()
            self.state.t0_sim = pd.Timestamp(self.times[self.state.cursor])

    def seek_fraction(self, frac: float):
        with self._lock:
            self.state.cursor = int(np.clip(frac, 0, 1) * (len(self.df) - 1))
            self.state.t0_wall = time.time()
            self.state.t0_sim = pd.Timestamp(self.times[self.state.cursor])

    def reset(self, warmup_rows: int = 600):
        with self._lock:
            self.state.cursor = min(warmup_rows, len(self.df) - 1)
            self.state.t0_wall = time.time()
            self.state.t0_sim = pd.Timestamp(self.times[self.state.cursor])

    # ---------------------------------------------------------------- horloge
    def _sync(self):
        """Avance le curseur au temps simule courant."""
        st = self.state
        if not st.playing or st.t0_sim is None:
            return
        elapsed = (time.time() - st.t0_wall) * st.speed
        target = st.t0_sim + pd.Timedelta(seconds=elapsed)
        idx = int(np.searchsorted(self.times, np.datetime64(target), side="right")) - 1
        st.cursor = int(np.clip(idx, 0, len(self.df) - 1))

    # ---------------------------------------------------------------- lecture
    def now(self) -> pd.Timestamp:
        with self._lock:
            self._sync()
            return pd.Timestamp(self.times[self.state.cursor])

    def window(self, n_rows: int = 4000) -> pd.DataFrame:
        """Toutes les lignes « deja arrivees », limitees aux n_rows dernieres."""
        with self._lock:
            self._sync()
            hi = self.state.cursor + 1
            lo = max(0, hi - n_rows)
            return self.df.iloc[lo:hi].copy()

    def full_history(self) -> pd.DataFrame:
        with self._lock:
            self._sync()
            return self.df.iloc[: self.state.cursor + 1].copy()

    def status(self) -> dict:
        with self._lock:
            self._sync()
            st = self.state
            return {
                "playing": st.playing,
                "speed": st.speed,
                "cursor": st.cursor,
                "total_rows": int(len(self.df)),
                "progress": float(st.cursor / max(len(self.df) - 1, 1)),
                "sim_time": pd.Timestamp(self.times[st.cursor]).isoformat(),
                "first_time": pd.Timestamp(self.times[0]).isoformat(),
                "last_time": pd.Timestamp(self.times[-1]).isoformat(),
            }
