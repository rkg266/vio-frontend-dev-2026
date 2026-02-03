from pathlib import Path
import numpy as np

from vio_frontend.providers.euroc_provider import EuRoCProvider
from vio_frontend.frontend.vio_frontend import VIOFrontend
from vio_frontend.core.init.gravity_init import gravity_align_init
from vio_frontend.core.state import NominalState
from vio_frontend.core.stereo.rectify import StereoRectifier
from vio_frontend.core.stereo.euroc_calib import (
    K_from_intrinsics,
    D_from_radtan4,
    T_from_yaml_data,
    T_cam0_cam1_from_T_BS,
)
import matplotlib.pyplot as plt

import numpy as np

def Rz(yaw: float) -> np.ndarray:
    """Rotation about +Z axis."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[ c, -s, 0.0],
                     [ s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)

def best_yaw_and_translation(est_p: np.ndarray, gt_p: np.ndarray):
    """
    Find yaw ψ and translation t minimizing:
        sum || (Rz(ψ) * est_p[i] + t) - gt_p[i] ||^2

    Uses 2D Procrustes on XY (closed form).
    """
    assert est_p.shape == gt_p.shape and est_p.shape[1] == 3
    e = est_p[:, :2]   # XY
    g = gt_p[:, :2]    # XY

    # subtract centroids
    e_mu = e.mean(axis=0)
    g_mu = g.mean(axis=0)
    E = e - e_mu
    G = g - g_mu

    # 2D best rotation angle:
    # minimize || R*E - G ||^2 with R in SO(2)
    # angle = atan2( sum(E_x*G_y - E_y*G_x), sum(E_x*G_x + E_y*G_y) )
    num = np.sum(E[:, 0]*G[:, 1] - E[:, 1]*G[:, 0])
    den = np.sum(E[:, 0]*G[:, 0] + E[:, 1]*G[:, 1])
    yaw = np.arctan2(num, den)

    R = Rz(yaw)

    # translation in 3D using full centroids (keeps Z shift too)
    t = gt_p.mean(axis=0) - (R @ est_p.mean(axis=0))

    return yaw, R, t



def main():
    # ---- dataset ----
    seq_root = Path("./EURO_data_set/vicon_room1/V1_01_easy")

    provider = EuRoCProvider(seq_root)

    # load ground truth data
    gt_t, gt_p = EuRoCProvider.load_groundtruth(seq_root)
    print("Ground truth loaded:", gt_p.shape[0], "rows")

    # ---- calibration (same as sanity_rectify.py) ----
    K0 = K_from_intrinsics(458.654, 457.296, 367.215, 248.375)
    D0 = D_from_radtan4(-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05)

    K1 = K_from_intrinsics(457.587, 456.134, 379.999, 255.238)
    D1 = D_from_radtan4(-0.28368365, 0.07451284, -0.00010473, -3.55590700e-05)

    T_BS0 = T_from_yaml_data([
        0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
        0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
       -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
        0.0, 0.0, 0.0, 1.0
    ])

    T_BS1 = T_from_yaml_data([
        0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
        0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,
       -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038,
        0.0, 0.0, 0.0, 1.0
    ])

    T_cam0_cam1 = T_cam0_cam1_from_T_BS(T_BS0, T_BS1)

    rectifier = StereoRectifier.from_calib(
        K0, D0, K1, D1,
        T_cam0_cam1,
        image_size=(752, 480)
    )

    # ---- initial state ----
    # 1) pull first ~1-2 seconds of IMU before running VIO
    accel_buf = []
    gyro_buf = []
    # read until you have e.g. 200 samples (if 200 Hz, that's 1 sec)
    while provider.has_next() and len(accel_buf) < 200:
        ev = provider.next_event()
        if ev.type == "imu":
            accel_buf.append(ev.accel)
            gyro_buf.append(ev.gyro)
        elif ev.type == "stereo":
            # ignore images during init window (or break if you prefer)
            pass

    accel_samples = np.asarray(accel_buf, dtype=np.float64)
    gyro_samples  = np.asarray(gyro_buf, dtype=np.float64)

    init = gravity_align_init(accel_samples, gyro_samples, g_mag=9.81, z_up=True)

    # 2) build initial state using R_wb0 and biases
    t0_ns = 0  # or use the first IMU timestamp if you want
    init_state = NominalState(
        t_ns=t0_ns,
        R_wb=init.R_wb0,
        # p_w=np.zeros(3),
        p_w=gt_p[0, :],
        v_w=np.zeros(3),
        b_g=init.b_g0,
        b_a=init.b_a0,
    )

    print("init a_avg:", init.a_avg, "||a_avg||", np.linalg.norm(init.a_avg))
    print("init b_g0 :", init.b_g0)
    print("init b_a0 :", init.b_a0)

    vio = VIOFrontend(
        provider=provider,
        init_state=init_state,
        rectifier=rectifier,
        T_BS_cam0=T_BS0,
    )

    traj = []
    # ---- run loop ----
    for i in range(200):
        out = vio.step()
        if out is None:
            break

        tag = out[0]
        if tag == "stereo_updated":
            _, ev, state, meas = out
            print(f"[RUN] t={ev.t_ns}  pos={state.p_w}")

            traj.append((state.t_ns, float(state.p_w[0]), float(state.p_w[1]), float(state.p_w[2]), *state.R_wb.reshape(-1).tolist()))
    traj = np.array(traj, dtype=np.float64)
    dur = (traj[-1, 0] - traj[0, 0]) * 1e-9

    # to compare with ground truth
    est_t = traj[:, 0].astype(np.int64)
    est_p = traj[:, 1:4]
    gt_p_at_est = EuRoCProvider.interp_gt_pos(gt_t, gt_p, est_t)
    
    # perform yaw alignment - to match initial orientation with ground truth
    yaw, R_align, t_align = best_yaw_and_translation(est_p, gt_p_at_est)

    est_p_yaw_aligned = (R_align @ est_p.T).T + t_align

    print("estimated yaw offset (deg):", np.degrees(yaw))
    print("translation:", t_align)

    err = est_p_yaw_aligned - gt_p_at_est
    rmse = np.sqrt(np.mean(np.sum(err**2, axis=1)))
    print("Position RMSE (m):", rmse)

    e_norm = np.linalg.norm(err, axis=1)

    results_dir = Path("vio_frontend/results")
    results_dir.mkdir(exist_ok=True)

    plt.figure()
    plt.plot((est_t - est_t[0])*1e-9, e_norm)
    plt.title("Position error norm vs time")
    plt.ylabel("||p_est - p_gt|| (m)")
    plt.xlabel("time(s)")

    err_fig_path = results_dir / "position_error_vs_time.png"
    plt.savefig(err_fig_path, dpi=200, bbox_inches="tight")
    print(f"Saved error plot to {err_fig_path}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # ---- estimated trajectory ----
    ax.plot(
        est_p_yaw_aligned[:,0], est_p_yaw_aligned[:,1], est_p_yaw_aligned[:,2],
        label="VIO estimate",
        color="tab:blue"
    )
    ax.scatter(*est_p_yaw_aligned[0], marker="o", color="tab:blue")
    ax.scatter(*est_p_yaw_aligned[-1], marker="x", color="tab:blue")

    # ---- ground truth trajectory ----
    ax.plot(
        gt_p_at_est[:,0], gt_p_at_est[:,1], gt_p_at_est[:,2],
        label="Ground truth",
        color="tab:orange"
    )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(f"VIO vs Ground Truth (duration = {dur:.2f}s)")
    ax.legend()

    # equal axis scaling (VERY important for interpretation)
    mins = np.minimum(est_p.min(axis=0), gt_p_at_est.min(axis=0))
    maxs = np.maximum(est_p.max(axis=0), gt_p_at_est.max(axis=0))
    mid = (mins + maxs) / 2
    r = (maxs - mins).max() / 2

    ax.set_xlim(mid[0]-r, mid[0]+r)
    ax.set_ylim(mid[1]-r, mid[1]+r)
    ax.set_zlim(mid[2]-r, mid[2]+r)

    traj_fig_path = results_dir / "vio_vs_ground_truth.png"
    plt.savefig(traj_fig_path, dpi=200, bbox_inches="tight")
    print(f"Saved trajectory plot to {traj_fig_path}")

    plt.show()

if __name__ == "__main__":
    main()
    # python -m vio_frontend.scripts.run_vio_frontend