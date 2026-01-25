# vio_frontend/frontend/vio_frontend.py
#This is the event-driven orchestrator
# vio_frontend/frontend/vio_frontend.py
from __future__ import annotations

from typing import Optional

from vio_frontend.types import ImuSample
from vio_frontend.core.imu_buffer import ImuBuffer
from vio_frontend.core.state import NominalState
from vio_frontend.core.propagation import propagate_midpoint

# Step-4
from vio_frontend.core.stereo.tracker import StereoFrontendFASTLK, TrackerParams
from vio_frontend.core.stereo.rectify import StereoRectifier

# Step-5
from vio_frontend.core.vision.landmarks import LandmarkDB
from vio_frontend.core.vision.stereo_model import RectifiedStereoParams
from vio_frontend.core.vision.update_gn import vision_update_pose_gn, GNParams


class VIOFrontend:
    # This is the event-driven orchestrator
    def __init__(
        self,
        provider,
        init_state: NominalState,
        rectifier: StereoRectifier,
        T_BS_cam0,                           # np.ndarray (4,4) from YAML
        tracker_params: TrackerParams = TrackerParams(),
        gn_params: GNParams = GNParams(),
    ):
        self.provider = provider
        self.imu_buf = ImuBuffer()
        self.state = init_state
        self.last_frame_t_ns: Optional[int] = None

        # --- Step-4 (rectification happens inside this object) ---
        self.stereo_frontend = StereoFrontendFASTLK(rectifier=rectifier, params=tracker_params)

        # --- Step-5 ---
        self.lmdb = LandmarkDB()
        self.T_BS_cam0 = T_BS_cam0
        self.gn_params = gn_params

        # rectified intrinsics + baseline come from rectifier
        self.stereo_proj = RectifiedStereoParams(
            fx=float(rectifier.K_rect[0, 0]),
            fy=float(rectifier.K_rect[1, 1]),
            cx=float(rectifier.K_rect[0, 2]),
            cy=float(rectifier.K_rect[1, 2]),
            baseline_m=float(rectifier.baseline_m),
        )

    def step(self):
        while self.provider.has_next():
            ev = self.provider.next_event()

            if ev.type == "imu":
                self.imu_buf.push(ImuSample(ev.t_ns, ev.accel, ev.gyro))

            elif ev.type == "stereo":
                t_k = ev.t_ns

                # First stereo: initialize tracker, optionally initialize landmarks
                if self.last_frame_t_ns is None:
                    self.last_frame_t_ns = t_k

                    meas = self.stereo_frontend.process(t_k, ev.left, ev.right)

                    # Optional: run Step-5 once to create landmarks from X_c0 (if any)
                    self.state = vision_update_pose_gn(
                        self.state, meas, self.lmdb, self.T_BS_cam0, self.stereo_proj, self.gn_params
                    )

                    return ("stereo_first", ev, self.state, meas)

                # 1) IMU propagate to this stereo time
                seg = self.imu_buf.get_segment(self.last_frame_t_ns, t_k)
                x_pred = propagate_midpoint(self.state, seg, t_k)
                self.last_frame_t_ns = t_k

                # 2) Step-4: rectified stereo measurements
                meas = self.stereo_frontend.process(t_k, ev.left, ev.right)

                # 3) Step-5: pose update using reprojection residuals
                x_upd = vision_update_pose_gn(
                    x_pred, meas, self.lmdb, self.T_BS_cam0, self.stereo_proj, self.gn_params
                )
                self.state = x_upd

                return ("stereo_updated", ev, self.state, meas)

        return None

