# vio_frontend/core/vision/stereo_model.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class RectifiedStereoParams:
    fx: float
    fy: float
    cx: float
    cy: float
    baseline_m: float  # baseline in meters

def project_rectified_stereo_from_cam0(p_c0: np.ndarray, P: RectifiedStereoParams) -> np.ndarray | None:
    """
    p_c0: (3,) point in cam0 rectified frame.
    returns z_hat = [uL, vL, uR] (3,) or None if behind camera.
    """
    X, Y, Z = float(p_c0[0]), float(p_c0[1]), float(p_c0[2])
    if Z <= 1e-6:
        return None

    uL = P.fx * (X / Z) + P.cx
    vL = P.fy * (Y / Z) + P.cy
    uR = uL - (P.fx * P.baseline_m) / Z
    return np.array([uL, vL, uR], dtype=np.float64)

def world_to_cam0(
    m_w: np.ndarray,
    R_wb: np.ndarray,
    p_w: np.ndarray,
    T_CB: np.ndarray,
) -> np.ndarray:
    """
    m_w: (3,) landmark in world
    R_wb: (3,3) body orientation in world (body->world)
    p_w: (3,) body position in world
    T_CB: (4,4) body->cam0 transform
    returns p_c0 (3,) in cam0 frame
    """
    # world -> body: p_b = R_bw (m_w - p_w)
    R_bw = R_wb.T
    p_b = R_bw @ (m_w - p_w)

    R_CB = T_CB[:3, :3]
    t_CB = T_CB[:3, 3]
    p_c0 = R_CB @ p_b + t_CB
    return p_c0
