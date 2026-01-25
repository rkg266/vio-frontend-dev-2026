'''
Triangulation (rectified)
This gives depth from disparity: fast + stable.
'''
from __future__ import annotations
import numpy as np

def triangulate_rectified(uL: float, vL: float, uR: float,
                          fx: float, fy: float, cx: float, cy: float,
                          baseline_m: float) -> np.ndarray | None:
    d = uL - uR
    if d <= 1e-6:
        return None

    Z = fx * baseline_m / d
    X = (uL - cx) * Z / fx
    Y = (vL - cy) * Z / fy
    return np.array([X, Y, Z], dtype=np.float64)
