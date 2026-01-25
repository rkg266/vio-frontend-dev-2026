'''
builds rectifier
prints baseline computed two ways
rectifies one pair and checks |vL - vR| on LK matches (the real sanity test)
'''
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import cv2

from vio_frontend.core.stereo.euroc_calib import (
    K_from_intrinsics, D_from_radtan4, T_from_yaml_data, T_cam0_cam1_from_T_BS
)
from vio_frontend.core.stereo.rectify import StereoRectifier

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="EuRoC sequence root, e.g. MH_01_easy")
    ap.add_argument("--n", type=int, default=200, help="num stereo frames to try")
    args = ap.parse_args()

    seq = Path(args.seq)

    # ----- paste your calibration numbers here -----
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

    T_01 = T_cam0_cam1_from_T_BS(T_BS0, T_BS1)
    t_01 = T_01[:3, 3]
    print("T_cam0_cam1 translation:", t_01)
    print("baseline (norm):", float(np.linalg.norm(t_01)))

    # Image size
    w, h = 752, 480
    rect = StereoRectifier.from_calib(K0, D0, K1, D1, T_01, (w, h))
    print("rectified fx:", rect.K_rect[0, 0])
    print("baseline from P2/fx:", rect.baseline_m)

    # Grab first stereo pair paths from EuRoC csv
    cam0_csv = seq / "mav0" / "cam0" / "data.csv"
    cam1_csv = seq / "mav0" / "cam1" / "data.csv"
    cam0_dir = seq / "mav0" / "cam0" / "data"
    cam1_dir = seq / "mav0" / "cam1" / "data"

    def load_cam_rows(csv_path):
        rows = []
        import csv
        with csv_path.open("r", newline="") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                rows.append((int(row[0]), row[1]))
        return rows

    L = load_cam_rows(cam0_csv)
    R = load_cam_rows(cam1_csv)
    # assume same timestamps; take first frame
    t = L[0][0]
    left_path = cam0_dir / L[0][1]
    right_path = cam1_dir / R[0][1]

    left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
    if left is None or right is None:
        raise FileNotFoundError("Could not read first stereo images")

    left_r, right_r = rect.rectify_pair(left, right)

    # Real sanity: match a bunch of points left->right and check v-difference
    pts = cv2.goodFeaturesToTrack(left_r, maxCorners=200, qualityLevel=0.01, minDistance=10)
    if pts is None:
        print("No features found for sanity check.")
        return

    # # Rough initial guess: shift u left by ~30 px (tune if needed)
    # ptsR0 = pts.copy()
    # ptsR0[:, 0, 0] = pts[:, 0, 0] - 80.0

    ptsR, st, err = cv2.calcOpticalFlowPyrLK(left_r, right_r, pts, None, winSize=(21,21), maxLevel=3)
    # ptsR, st, err = cv2.calcOpticalFlowPyrLK(left_r, right_r, pts, ptsR0, winSize=(31,31), maxLevel=4, 
    #                                          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01), flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    goodL = pts[st.reshape(-1) == 1].reshape(-1, 2)
    goodR = ptsR[st.reshape(-1) == 1].reshape(-1, 2)

    if len(goodL) == 0:
        print("No LK matches found.")
        return

    vdiff = np.abs(goodL[:, 1] - goodR[:, 1])
    disp = goodL[:, 0] - goodR[:, 0]

    # Gate to keep only valid rectified stereo matches
    mask = (vdiff < 2.0) & (disp > 0.5) & (disp < 220.0)

    print("Matches before gating:", len(goodL))
    print("Matches after  gating:", int(mask.sum()))
    if mask.sum() > 0:
        print("v-diff mean/95/max:", float(vdiff[mask].mean()),
            float(np.percentile(vdiff[mask], 95)),
            float(vdiff[mask].max()))
        print("disp   mean/95/max:", float(disp[mask].mean()),
            float(np.percentile(disp[mask], 95)),
            float(disp[mask].max()))
    print("Sanity expectation: v-diff ~ < 1-2 px for rectified stereo.")

    # Another better sanity check - Disparity stats
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16*12,  # 192 px range
        blockSize=7,
        P1=8*3*7*7,
        P2=32*3*7*7,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disp = stereo.compute(left_r, right_r).astype(np.float32) / 16.0

    valid = disp > 0
    print("SGBM disparity stats:")
    print("  mean:", float(disp[valid].mean()))
    print("  p95 :", float(np.percentile(disp[valid], 95)))
    print("  max :", float(disp[valid].max()))


if __name__ == "__main__":
    main()
