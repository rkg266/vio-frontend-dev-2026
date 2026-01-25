from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2

@dataclass
class StereoRectifier:
    map1_x: np.ndarray
    map1_y: np.ndarray
    map2_x: np.ndarray
    map2_y: np.ndarray
    Q: np.ndarray
    K_rect: np.ndarray   # rectified intrinsics (3x3)
    baseline_m: float

    @staticmethod
    def from_calib(
            K0: np.ndarray, D0: np.ndarray,
            K1: np.ndarray, D1: np.ndarray,
            T_cam0_cam1: np.ndarray,      # 4x4 transform from cam0 to cam1
            image_size: tuple[int,int],   # (w,h)
        ) -> "StereoRectifier":
        
        w, h = image_size
        R = T_cam0_cam1[:3, :3].astype(np.float64)
        t = T_cam0_cam1[:3, 3].astype(np.float64).reshape(3, 1)

        # stereoRectify gives rectification rotations and new projection matrices
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K0, D0, K1, D1, (w, h), R, t,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0
        )

        # Rectification maps (compute once)
        map1_x, map1_y = cv2.initUndistortRectifyMap(K0, D0, R1, P1, (w, h), cv2.CV_32FC1)
        map2_x, map2_y = cv2.initUndistortRectifyMap(K1, D1, R2, P2, (w, h), cv2.CV_32FC1)

        # Rectified intrinsics are embedded in P1 (left projection)
        K_rect = P1[:3, :3].copy()

        # Baseline magnitude in rectified coordinates: b = -P2[0,3] / fx
        fx = K_rect[0, 0]
        baseline_m = abs(P2[0, 3]) / fx

        return StereoRectifier(map1_x, map1_y, map2_x, map2_y, Q, K_rect, baseline_m)

    def rectify_pair(self, img0: np.ndarray, img1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r0 = cv2.remap(img0, self.map1_x, self.map1_y, interpolation=cv2.INTER_LINEAR)
        r1 = cv2.remap(img1, self.map2_x, self.map2_y, interpolation=cv2.INTER_LINEAR)
        return r0, r1
