# core data types foe stereo cam
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, List
import numpy as np

@dataclass
class StereoTrack:
    id: int
    age: int                 # how many frames tracked
    uL: float
    vL: float
    uR: float
    vR: float
    X_c0: Optional[np.ndarray] = None  # (3,) point in cam0 frame if triangulated
    active: bool = True

@dataclass
class StereoMeasurements:
    t_ns: int
    tracks: List[StereoTrack]
