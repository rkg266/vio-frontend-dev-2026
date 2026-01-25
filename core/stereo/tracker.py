# Implementation: FAST + LK tracker
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2

from .rectify import StereoRectifier
from .stereo_match import compute_disparity_sgbm, match_right_with_disp_init
from .stereo_types import StereoTrack, StereoMeasurements
from .calib import StereoCalibRectified
from .triangulation import triangulate_rectified


# @dataclass
# class StereoCalibRectified:  # Moved to neural module - calib.py
#     fx: float
#     fy: float
#     cx: float
#     cy: float
#     baseline_m: float   # baseline in meters (or same unit you want for depth)

@dataclass
class TrackerParams:
    max_tracks: int = 300
    min_tracks: int = 200
    fast_threshold: int = 20
    min_dist: int = 15              # pixels
    lk_win: Tuple[int,int] = (21, 21)
    lk_max_level: int = 3
    lk_criteria: Tuple[int,int,float] = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    fb_thresh: float = 1.0          # forward-backward px error
    epi_v_thresh: float = 1.5       # |vR - vL| for rectified
    disp_min: float = 0.5
    disp_max: float = 200.0

class StereoFrontendFASTLK:
    def __init__(self, rectifier: StereoRectifier, params: TrackerParams = TrackerParams()):
        self.rectifier = rectifier
        self.p = params

        self._prev_left: Optional[np.ndarray] = None
        self._tracks: List[StereoTrack] = []
        self._next_id = 0

        self._fast = cv2.FastFeatureDetector_create(threshold=self.p.fast_threshold, nonmaxSuppression=True)

    def process(self, t_ns: int, left: np.ndarray, right: np.ndarray) -> StereoMeasurements:
        # Ensure grayscale
        if left.ndim == 3:
            left_g = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        else:
            left_g = left
        if right.ndim == 3:
            right_g = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        else:
            right_g = right

        # Rectify
        left_r, right_r = self.rectifier.rectify_pair(left_g, right_g)

        if self._prev_left is None:
            # Bootstrap: detect and stereo-match to initialize tracks
            self._tracks = self._spawn_new_tracks(left_r, right_r, needed=self.p.max_tracks)
            self._prev_left = left_r
            return StereoMeasurements(t_ns=t_ns, tracks=self._tracks.copy())

        # 1) temporal tracking in left image
        self._tracks = self._track_left_temporal(self._prev_left, left_r, self._tracks)

        # 2) stereo match in current frame
        self._tracks = self._stereo_match_current(left_r, right_r, self._tracks)

        # 3) triangulate (optional but useful)
        self._tracks = self._triangulate_tracks(self._tracks)

        # 4) replenish with new tracks if needed
        if len(self._tracks) < self.p.min_tracks:
            needed = self.p.max_tracks - len(self._tracks)
            new_tracks = self._spawn_new_tracks(left_r, right_r, needed=needed)
            self._tracks.extend(new_tracks)

        self._prev_left = left_r
        return StereoMeasurements(t_ns=t_ns, tracks=self._tracks.copy())

    def _track_left_temporal(self, prev_left: np.ndarray, cur_left: np.ndarray, tracks: List[StereoTrack]) -> List[StereoTrack]:
        if not tracks:
            return []

        pts0 = np.array([[tr.uL, tr.vL] for tr in tracks], dtype=np.float32).reshape(-1,1,2)

        pts1, st, err = cv2.calcOpticalFlowPyrLK(
            prev_left, cur_left, pts0, None,
            winSize=self.p.lk_win, maxLevel=self.p.lk_max_level, criteria=self.p.lk_criteria
        )

        # Forward-backward check
        pts0_back, st_back, err_back = cv2.calcOpticalFlowPyrLK(
            cur_left, prev_left, pts1, None,
            winSize=self.p.lk_win, maxLevel=self.p.lk_max_level, criteria=self.p.lk_criteria
        )

        good: List[StereoTrack] = []
        h, w = cur_left.shape[:2]

        for i, tr in enumerate(tracks):
            if st[i,0] == 0 or st_back[i,0] == 0:
                continue

            p1 = pts1[i,0]
            p0b = pts0_back[i,0]
            fb = float(np.linalg.norm(p0b - pts0[i,0]))

            if fb > self.p.fb_thresh:
                continue

            u, v = float(p1[0]), float(p1[1])
            if not (0 <= u < w and 0 <= v < h):
                continue

            good.append(StereoTrack(
                id=tr.id, age=tr.age + 1,
                uL=u, vL=v, uR=tr.uR, vR=tr.vR,
                X_c0=None, active=True
            ))

        # Optional: enforce min distance (deduplicate)
        return self._enforce_min_distance(good, self.p.min_dist)

    def _stereo_match_current(self, left: np.ndarray, right: np.ndarray, tracks: list[StereoTrack]) -> list[StereoTrack]:
        if not tracks:
            return []

        # Compute disparity map ONCE for this frame (you can later optimize to run only when spawning)
        disp = compute_disparity_sgbm(left, right)

        ptsL = np.array([[tr.uL, tr.vL] for tr in tracks], dtype=np.float32).reshape(-1,1,2)

        ptsR, ok = match_right_with_disp_init(
            left, right, ptsL, disp,
            lk_win=self.p.lk_win,
            lk_max_level=self.p.lk_max_level
        )

        matched: list[StereoTrack] = []
        for tr, pR, good in zip(tracks, ptsR.reshape(-1,2), ok):
            if not good:
                continue
            uR, vR = float(pR[0]), float(pR[1])

            disp_lr = tr.uL - uR
            if abs(vR - tr.vL) > self.p.epi_v_thresh:
                continue
            if disp_lr <= self.p.disp_min or disp_lr >= self.p.disp_max:
                continue

            matched.append(StereoTrack(
                id=tr.id, age=tr.age,
                uL=tr.uL, vL=tr.vL, uR=uR, vR=vR,
                X_c0=None, active=True
            ))

        # ---- SANITY PRINT (temporary) ----
        if matched:
            print(
                f"[Stereo] tracks={len(matched)} "
                f"med_disp={np.median([t.uL - t.uR for t in matched]):.1f} "
                f"med_|v|={np.median([abs(t.vL - t.vR) for t in matched]):.2f}"
            )
        # ---------------------------------

        return matched


    def _triangulate_tracks(self, tracks: List[StereoTrack]) -> List[StereoTrack]:
        fx = float(self.rectifier.K_rect[0, 0])
        fy = float(self.rectifier.K_rect[1, 1])
        cx = float(self.rectifier.K_rect[0, 2])
        cy = float(self.rectifier.K_rect[1, 2])
        b  = float(self.rectifier.baseline_m)

        out: List[StereoTrack] = []
        for tr in tracks:
            X = triangulate_rectified(tr.uL, tr.vL, tr.uR, fx, fy, cx, cy, b)
            out.append(StereoTrack(**{**tr.__dict__, "X_c0": X}))
        return out

    def _spawn_new_tracks(self, left: np.ndarray, right: np.ndarray, needed: int) -> List[StereoTrack]:
        if needed <= 0:
            return []

        mask = np.ones(left.shape[:2], dtype=np.uint8) * 255
        # block around existing tracks
        for tr in self._tracks:
            cv2.circle(mask, (int(tr.uL), int(tr.vL)), self.p.min_dist, 0, -1)

        kps = self._fast.detect(left, mask=mask)
        if not kps:
            return []

        # take best by response
        kps = sorted(kps, key=lambda k: k.response, reverse=True)[:needed*3]
        ptsL = np.array([[kp.pt[0], kp.pt[1]] for kp in kps], dtype=np.float32).reshape(-1,1,2)

        # 2) Compute disparity map ONCE
        disp = compute_disparity_sgbm(left, right)

        # 3) Stereo match with disparity-initialized LK
        ptsR, ok = match_right_with_disp_init(
            left, right, ptsL, disp,
            lk_win=self.p.lk_win,
            lk_max_level=self.p.lk_max_level
        )

        new_tracks: List[StereoTrack] = []
        for pL, pR, good in zip(ptsL.reshape(-1,2), ptsR.reshape(-1,2), ok):
            if len(new_tracks) >= needed:
                break
            if not good:
                continue
            uL, vL = float(pL[0]), float(pL[1])
            uR, vR = float(pR[0]), float(pR[1])
            disp = uL - uR
            if disp <= self.p.disp_min or disp >= self.p.disp_max:
                continue
            if abs(vR - vL) > self.p.epi_v_thresh:
                continue

            tr = StereoTrack(id=self._next_id, age=1, uL=uL, vL=vL, uR=uR, vR=vR, X_c0=None, active=True)
            self._next_id += 1
            new_tracks.append(tr)

        return self._enforce_min_distance(new_tracks, self.p.min_dist)

    def _enforce_min_distance(self, tracks: List[StereoTrack], min_dist: float) -> List[StereoTrack]:
        # greedy keep; good enough for now
        kept: List[StereoTrack] = []
        for tr in tracks:
            ok = True
            for kt in kept:
                if (tr.uL - kt.uL)**2 + (tr.vL - kt.vL)**2 < (min_dist**2):
                    ok = False
                    break
            if ok:
                kept.append(tr)
        return kept
