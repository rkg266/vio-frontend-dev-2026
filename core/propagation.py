# vio_frontend/core/propagation.py
import numpy as np
from vio_frontend.types import ImuSample
from vio_frontend.core.state import NominalState


def skew(w: np.ndarray) -> np.ndarray:
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]], dtype=np.float64)


def so3_exp(phi: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(phi)
    if theta < 1e-12:
        return np.eye(3) + skew(phi)
    K = skew(phi / theta)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def propagate_midpoint(
    x: NominalState,
    imu: list[ImuSample],
    t_target_ns: int,
    g_w: np.ndarray = np.array([0, 0, -9.81], dtype=np.float64),
) -> NominalState:
    if not imu:
        return NominalState(
            t_ns=t_target_ns,
            R_wb=x.R_wb, p_w=x.p_w, v_w=x.v_w, b_g=x.b_g, b_a=x.b_a
        )

    R = x.R_wb.copy()
    p = x.p_w.copy()
    v = x.v_w.copy()
    bg = x.b_g
    ba = x.b_a

    t_prev = x.t_ns

    for k in range(len(imu) - 1):
        s0, s1 = imu[k], imu[k + 1]
        t0 = max(s0.t_ns, t_prev)
        t1 = min(s1.t_ns, t_target_ns)
        if t1 <= t0:
            continue

        dt = (t1 - t0) * 1e-9

        w_mid = 0.5 * ((s0.gyro - bg) + (s1.gyro - bg))
        a_mid = 0.5 * ((s0.accel - ba) + (s1.accel - ba))

        dR = so3_exp(w_mid * dt)
        R_new = R @ dR

        R_mid = R @ so3_exp(w_mid * (0.5 * dt))  # better approximation of orientation at dt/2 midpoint instance
        tep = R_mid @ a_mid
        tep1 = np.linalg.norm(tep)
        a_w = R_mid @ a_mid + g_w

        v_new = v + a_w * dt
        p_new = p + v * dt + 0.5 * a_w * dt * dt

        R, v, p = R_new, v_new, p_new
        t_prev = t1

        if t_prev >= t_target_ns:
            break

    return NominalState(t_ns=t_target_ns, R_wb=R, p_w=p, v_w=v, b_g=bg, b_a=ba)
