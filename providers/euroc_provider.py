from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import csv
import numpy as np
import cv2

from vio_frontend.types import (
    IDataProvider,
    RawEvent,
    RawImuEvent,
    RawStereoEvent,
    assert_non_decreasing,
)

class EuRoCProvider(IDataProvider):
    def __init__(self, seq_root: str | Path, stereo_tol_ns: int = 1_000_000):
        self.seq_root = Path(seq_root)

        imu_csv = self.seq_root / "mav0" / "imu0" / "data.csv"
        cam0_csv = self.seq_root / "mav0" / "cam0" / "data.csv"
        cam1_csv = self.seq_root / "mav0" / "cam1" / "data.csv"
        cam0_dir = self.seq_root / "mav0" / "cam0" / "data"
        cam1_dir = self.seq_root / "mav0" / "cam1" / "data"

        self._imu = self._load_imu_index(imu_csv)                 # [(t, accel, gyro)]
        cam0 = self._load_cam_index(cam0_csv, cam0_dir)           # [(t, path)]
        cam1 = self._load_cam_index(cam1_csv, cam1_dir)           # [(t, path)]
        self._stereo = self._pair_stereo(cam0, cam1, stereo_tol_ns)  # [(t, lp, rp)]

        self._i = 0
        self._j = 0
        self._last_t: Optional[int] = None

    def has_next(self) -> bool:
        return self._i < len(self._imu) or self._j < len(self._stereo)

    def next_event(self) -> RawEvent:
        if not self.has_next():
            raise StopIteration

        next_imu_t = self._imu[self._i][0] if self._i < len(self._imu) else None
        next_st_t  = self._stereo[self._j][0] if self._j < len(self._stereo) else None

        take_imu = False
        if next_imu_t is not None and next_st_t is not None:
            take_imu = next_imu_t <= next_st_t
        elif next_imu_t is not None:
            take_imu = True

        if take_imu:
            t, accel, gyro = self._imu[self._i]
            self._i += 1
            ev: RawEvent = RawImuEvent(type="imu", t_ns=t, accel=accel, gyro=gyro)
        else:
            t, lp, rp = self._stereo[self._j]
            self._j += 1
            left = self._read_gray(lp)
            right = self._read_gray(rp)
            ev = RawStereoEvent(type="stereo", t_ns=t, left=left, right=right)

        self._last_t = assert_non_decreasing(self._last_t, ev.t_ns, "EuRoCProvider")
        return ev

    # ---------- helpers ----------

    def _load_imu_index(self, imu_csv: Path) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        out = []
        with imu_csv.open("r", newline="") as f:
            r = csv.reader(f)
            next(r, None)  # header
            for row in r:
                t = int(row[0])
                wx, wy, wz = map(float, row[1:4])
                ax, ay, az = map(float, row[4:7])
                accel = np.array([ax, ay, az], dtype=np.float64)
                gyro = np.array([wx, wy, wz], dtype=np.float64)
                out.append((t, accel, gyro))
        return out

    def _load_cam_index(self, cam_csv: Path, img_dir: Path) -> List[Tuple[int, Path]]:
        out = []
        with cam_csv.open("r", newline="") as f:
            r = csv.reader(f)
            next(r, None)  # header
            for row in r:
                t = int(row[0])
                out.append((t, img_dir / row[1]))
        return out

    def _pair_stereo(
        self,
        left: List[Tuple[int, Path]],
        right: List[Tuple[int, Path]],
        tol_ns: int,
    ) -> List[Tuple[int, Path, Path]]:
        pairs = []
        i = j = 0
        while i < len(left) and j < len(right):
            tl, pl = left[i]
            tr, pr = right[j]
            if tl == tr or abs(tl - tr) <= tol_ns:
                t = tl if tl <= tr else tr
                pairs.append((t, pl, pr))
                i += 1
                j += 1
            elif tl < tr:
                i += 1
            else:
                j += 1
        return pairs

    def _read_gray(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        return img
