from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GravityInitResult:
    R_wb0: np.ndarray          # (3,3) body->world
    g_w: np.ndarray            # (3,)
    b_g0: np.ndarray           # (3,)
    b_a0: np.ndarray           # (3,)
    a_avg: np.ndarray          # (3,)
    w_avg: np.ndarray          # (3,)


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0, -z,  y],
                     [z,  0, -x],
                     [-y, x,  0]], dtype=np.float64)


def R_from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Returns R such that R @ a = b for unit vectors a,b.
    Robust Rodrigues formula, handles near-parallel/anti-parallel.
    """
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)

    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))

    if s < 1e-10:
        # a and b are parallel or anti-parallel
        if c > 0.0:
            return np.eye(3, dtype=np.float64)
        # 180 deg: choose any axis orthogonal to a
        # pick an axis not collinear with a
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(a, tmp)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        K = _skew(axis)
        # rotation by pi: R = I + 2K^2
        return np.eye(3, dtype=np.float64) + 2.0 * (K @ K)

    K = _skew(v / s)
    # Rodrigues: R = I + sinθ K + (1-cosθ) K^2, with sinθ=s, cosθ=c for unit vectors
    R = np.eye(3, dtype=np.float64) + s * K + (1.0 - c) * (K @ K)
    return R


def gravity_align_init(
    accel_samples: np.ndarray,   # (N,3) m/s^2
    gyro_samples: np.ndarray,    # (N,3) rad/s (or consistent units)
    g_mag: float = 9.81,
    z_up: bool = True,
) -> GravityInitResult:
    """
    Gravity-aligned initialization:
    - Assumes platform is stationary during the sample window.
    - Estimates gyro bias as mean gyro.
    - Estimates accel bias along gravity magnitude (optional simple model).
    - Computes R_wb0 that aligns measured gravity direction to world -Z (if z_up).
      Yaw remains arbitrary (unobservable from gravity alone).

    Returns body->world rotation R_wb0 and initial biases.
    """
    assert accel_samples.ndim == 2 and accel_samples.shape[1] == 3
    assert gyro_samples.ndim == 2 and gyro_samples.shape[1] == 3
    N = accel_samples.shape[0]
    assert N >= 20, "Use at least ~20 IMU samples; 1-2 seconds is better."

    a_avg = accel_samples.mean(axis=0).astype(np.float64)
    w_avg = gyro_samples.mean(axis=0).astype(np.float64)

    # World gravity direction (unit)
    g_w_unit = np.array([0.0, 0.0, -1.0 if z_up else 1.0], dtype=np.float64)
    g_w = g_mag * g_w_unit

    # At rest: a_meas ≈ -R_bw g_w  ->  g in body frame ≈ -a_meas (up to bias)
    g_b_est = -a_avg
    g_b_unit = g_b_est / (np.linalg.norm(g_b_est) + 1e-12)

    # Find R_wb0 such that R_wb0 @ g_b_unit = g_w_unit
    R_wb0 = R_from_two_vectors(g_b_unit, g_w_unit)

    # Bias estimates (simple, good enough for startup)
    b_g0 = w_avg.copy()

    # Accel bias: treat as difference between measured average and expected gravity in body frame.
    # expected a_avg ≈ -R_bw g_w = -(R_wb^T) g_w
    expected_a_avg = -(R_wb0.T @ g_w)
    b_a0 = (a_avg - expected_a_avg)

    return GravityInitResult(
        R_wb0=R_wb0,
        g_w=g_w,
        b_g0=b_g0,
        b_a0=b_a0,
        a_avg=a_avg,
        w_avg=w_avg,
    )
