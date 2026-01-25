'''
Stereo right matching (LK constrained)
For rectified stereo, you can initialize right points as same (u,v) and let LK find match; then gate by |vR-vL|.
Later you can improve this by giving an initial u-shift guess based on typical disparity range, or by searching along epipolar line.
'''

from __future__ import annotations
from typing import Tuple
import numpy as np
import cv2

def match_right_with_lk_rectified(left, right, ptsL, params) -> tuple[np.ndarray, np.ndarray]:
    """
    ptsL: (N,1,2) float32 in left image at time k
    Returns:
      ptsR: (N,1,2)
      ok: (N,) bool
    """
    # Initial guess: same pixel location (works surprisingly well for rectified + small baseline)
    ptsR0 = ptsL.copy()

    ptsR, st, err = cv2.calcOpticalFlowPyrLK(
        left, right, ptsL, ptsR0,
        winSize=params.lk_win, maxLevel=params.lk_max_level, criteria=params.lk_criteria
    )

    ok = (st.reshape(-1) == 1)
    return ptsR, ok

def compute_disparity_sgbm(left_r: np.ndarray, right_r: np.ndarray) -> np.ndarray:
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16*12,  # 192
        blockSize=7,
        P1=8*3*7*7,
        P2=32*3*7*7,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    disp = stereo.compute(left_r, right_r).astype(np.float32) / 16.0
    return disp

def match_right_with_disp_init(
    left_r: np.ndarray,
    right_r: np.ndarray,
    ptsL: np.ndarray,           # (N,1,2) float32
    disp: np.ndarray,           # (H,W) float32
    lk_win=(21,21),
    lk_max_level=3,
) -> tuple[np.ndarray, np.ndarray]:
    H, W = disp.shape[:2]
    ptsR0 = ptsL.copy()

    for i in range(ptsL.shape[0]):
        u, v = ptsL[i,0]
        ui = int(round(u))
        vi = int(round(v))
        if 0 <= ui < W and 0 <= vi < H:
            d = disp[vi, ui]
            if d > 0:
                ptsR0[i,0,0] = u - d
                ptsR0[i,0,1] = v
            else:
                ptsR0[i,0,0] = u  # fallback
                ptsR0[i,0,1] = v
        else:
            ptsR0[i,0] = ptsL[i,0]

    ptsR, st, err = cv2.calcOpticalFlowPyrLK(
        left_r, right_r, ptsL, ptsR0,
        winSize=lk_win, maxLevel=lk_max_level,
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW
    )
    ok = (st.reshape(-1) == 1)
    return ptsR, ok

