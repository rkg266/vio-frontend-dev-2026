'''
Updates only pose (R_wb, p_w) using a 6D increment δ = [δθ, δp]
Uses numeric Jacobians for each observation
Includes simple outlier rejection on pixel residual norm
'''
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from vio_frontend.core.state import NominalState
from vio_frontend.core.stereo.stereo_types import StereoMeasurements, StereoTrack
from vio_frontend.core.vision.landmarks import LandmarkDB
from vio_frontend.core.vision.stereo_model import (
    RectifiedStereoParams,
    world_to_cam0,
    project_rectified_stereo_from_cam0,
)

def skew(w: np.ndarray) -> np.ndarray:
    wx, wy, wz = w
    return np.array([[0, -wz, wy],
                     [wz, 0, -wx],
                     [-wy, wx, 0]], dtype=np.float64)

def so3_exp(phi: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(phi)
    if theta < 1e-12:
        return np.eye(3) + skew(phi)
    K = skew(phi / theta)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

def apply_pose_update(R_wb: np.ndarray, p_w: np.ndarray, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    delta: (6,) = [dtheta(3), dp(3)].
    Right-multiply rotation: R <- R * Exp(dtheta)
    """
    dtheta = delta[:3]
    dp = delta[3:]
    R_new = R_wb @ so3_exp(dtheta)
    p_new = p_w + dp
    return R_new, p_new

def predict_z(R_wb: np.ndarray, p_w: np.ndarray, m_w: np.ndarray, T_CB: np.ndarray, P: RectifiedStereoParams):
    p_c0 = world_to_cam0(m_w, R_wb, p_w, T_CB)
    return project_rectified_stereo_from_cam0(p_c0, P)

def numeric_J_pose(
    R_wb: np.ndarray,
    p_w: np.ndarray,
    m_w: np.ndarray,
    T_CB: np.ndarray,
    P: RectifiedStereoParams,
    eps_rot: float = 1e-6,
    eps_pos: float = 1e-4,
) -> np.ndarray | None:
    """
    Returns J (3x6) for z=[uL,vL,uR] w.r.t pose delta=[dtheta, dp].
    Numeric finite differences around current pose.
    """
    z0 = predict_z(R_wb, p_w, m_w, T_CB, P)
    if z0 is None:
        return None

    J = np.zeros((3, 6), dtype=np.float64)

    # rotation perturbations
    for i in range(3):
        d = np.zeros(6)
        d[i] = eps_rot
        R1, p1 = apply_pose_update(R_wb, p_w, d)
        z1 = predict_z(R1, p1, m_w, T_CB, P)
        if z1 is None:
            return None
        J[:, i] = (z1 - z0) / eps_rot

    # translation perturbations
    for i in range(3):
        d = np.zeros(6)
        d[3 + i] = eps_pos
        R1, p1 = apply_pose_update(R_wb, p_w, d)
        z1 = predict_z(R1, p1, m_w, T_CB, P)
        if z1 is None:
            return None
        J[:, 3 + i] = (z1 - z0) / eps_pos

    return J

@dataclass
class GNParams:
    max_iters: int = 3
    huber_delta_px: float = 3.0      # robust-ish weighting
    outlier_thresh_px: float = 15.0  # reject if ||r|| > thresh
    damping: float = 1e-3            # LM-like damping

def huber_weight(r_norm: float, delta: float) -> float:
    if r_norm <= delta:
        return 1.0
    return delta / (r_norm + 1e-12)

def vision_update_pose_gn(
    x: NominalState,
    meas: StereoMeasurements,
    lmdb: LandmarkDB,
    T_BS_cam0: np.ndarray,           # cam0->body (from YAML)
    P: RectifiedStereoParams,        # rectified intrinsics + baseline
    params: GNParams = GNParams(),
) -> NominalState:
    """
    Pose-only GN update. Landmarks are kept in lmdb keyed by track_id.
    Uses measurement z=[uL,vL,uR] from rectified pixels.
    """
    T_CB = np.linalg.inv(T_BS_cam0)  # body->cam0

    # 0) Create landmarks for new tracks that have triangulated X_c0
    # Track.X_c0 is in cam0 frame at current time; lift to world using current pose.
    for tr in meas.tracks:
        if tr.X_c0 is None:
            continue
        if lmdb.has(tr.id):
            continue

        # cam0 -> body: use T_BS (cam->body)
        R_BC = T_BS_cam0[:3, :3]
        t_BC = T_BS_cam0[:3, 3]
        p_b = R_BC @ tr.X_c0 + t_BC

        # body -> world
        m_w = x.R_wb @ p_b + x.p_w
        lmdb.set(tr.id, m_w, born_t_ns=meas.t_ns)

    # 1) prune dead landmarks
    active_ids = {tr.id for tr in meas.tracks}
    lmdb.prune_to_active(active_ids)

    R = x.R_wb.copy()
    p = x.p_w.copy()

    # 2) GN iterations
    for it in range(params.max_iters):
        H = np.zeros((6, 6), dtype=np.float64)
        b = np.zeros((6,), dtype=np.float64)

        inliers = 0
        res_norms = []

        for tr in meas.tracks:
            lm = lmdb.get(tr.id)
            if lm is None:
                continue

            if lm.born_t_ns == meas.t_ns:
                continue  # don't use same-frame landmarks

            z = np.array([tr.uL, tr.vL, tr.uR], dtype=np.float64)
            zhat = predict_z(R, p, lm.p_w, T_CB, P)
            if zhat is None:
                continue

            r = z - zhat
            rn = float(np.linalg.norm(r))
            if rn > params.outlier_thresh_px:
                continue

            J = numeric_J_pose(R, p, lm.p_w, T_CB, P)
            if J is None:
                continue

            w = huber_weight(rn, params.huber_delta_px)
            W = w  # scalar weight

            H += W * (J.T @ J)
            b += W * (J.T @ r)

            inliers += 1
            res_norms.append(rn)

        if inliers < 8:
            # not enough constraints to update pose robustly
            break

        # Damped solve
        H_d = H + params.damping * np.eye(6)
        try:
            delta = np.linalg.solve(H_d, b)
        except np.linalg.LinAlgError:
            break

        # Apply update
        R, p = apply_pose_update(R, p, delta)

        if res_norms:
            print(f"[GN it {it}] inliers={inliers}  median|r|={np.median(res_norms):.2f}px  |delta|={np.linalg.norm(delta):.3e}")

        # if res_norms:
        #     print(
        #         f"[GN it {it}] inliers={inliers} "
        #         f"median|r|={np.median(res_norms):.2f}px "
        #         f"p95|r|={np.percentile(res_norms, 95):.2f}px "
        #         f"|delta|={np.linalg.norm(delta):.3e}"
        #     )


        # Small update => stop
        if float(np.linalg.norm(delta)) < 1e-6:
            break

    return NominalState(
        t_ns=x.t_ns,
        R_wb=R,
        p_w=p,
        v_w=x.v_w,
        b_g=x.b_g,
        b_a=x.b_a,
    )
