# vio_frontend/core/vision/landmarks.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

@dataclass
class Landmark:
    p_w: np.ndarray  # (3,) world point
    born_t_ns: int

class LandmarkDB:
    def __init__(self):
        self._db: Dict[int, Landmark] = {}

    def has(self, track_id: int) -> bool:
        return track_id in self._db

    def get(self, track_id: int) -> Optional[Landmark]:
        return self._db.get(track_id)

    def set(self, track_id: int, p_w: np.ndarray, born_t_ns: int) -> None:
        self._db[track_id] = Landmark(p_w=p_w.astype(np.float64).copy(), born_t_ns=born_t_ns)

    def remove(self, track_id: int) -> None:
        self._db.pop(track_id, None)

    def prune_to_active(self, active_track_ids: set[int]) -> None:
        # drop landmarks whose tracks died
        dead = [tid for tid in self._db.keys() if tid not in active_track_ids]
        for tid in dead:
            self._db.pop(tid, None)
