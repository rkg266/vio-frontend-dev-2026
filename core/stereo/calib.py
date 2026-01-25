# vio_frontend/core/stereo/calib.py
from dataclasses import dataclass

@dataclass(frozen=True)
class StereoCalibRectified:
    fx: float
    fy: float
    cx: float
    cy: float
    baseline_m: float
