'''
-runs EuRoCProvider for the first 2000 raw events
-checks global monotonic timestamps
-prints counts + estimated rates
-prints dt stats for IMU and stereo separately
-tells you if stereo pairing looks off
'''

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List

import numpy as np

from vio_frontend.providers.euroc_provider import EuRoCProvider
from vio_frontend.types import RawImuEvent, RawStereoEvent, assert_non_decreasing


def ns_to_s(x_ns: int) -> float:
    return float(x_ns) * 1e-9


def summarize_dts(name: str, dts_ns: List[int]) -> None:
    if not dts_ns:
        print(f"{name}: no dt samples")
        return
    arr = np.asarray(dts_ns, dtype=np.int64)
    arr_s = arr * 1e-9
    print(
        f"{name}: n={len(arr_s)}  "
        f"mean={arr_s.mean():.6f}s  std={arr_s.std():.6f}s  "
        f"min={arr_s.min():.6f}s  p50={np.percentile(arr_s, 50):.6f}s  "
        f"p95={np.percentile(arr_s, 95):.6f}s  max={arr_s.max():.6f}s"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, type=str, help="Path to EuRoC sequence root (e.g., MH_01_easy)")
    ap.add_argument("--n", default=2000, type=int, help="Number of raw events to consume")
    args = ap.parse_args()

    seq_root = Path(args.seq)
    if not seq_root.exists():
        raise FileNotFoundError(seq_root)

    dp = EuRoCProvider(seq_root)

    # Global monotonicity + overall span
    last_t: Optional[int] = None
    t_first: Optional[int] = None
    t_last: Optional[int] = None

    # Counts
    imu_count = 0
    stereo_count = 0

    # dt stats per sensor
    last_imu_t: Optional[int] = None
    last_st_t: Optional[int] = None
    imu_dts: List[int] = []
    st_dts: List[int] = []

    # For quick stereo timestamp sanity (how often it repeats / jumps)
    stereo_ts: List[int] = []

    n = args.n
    k = 0
    while k < n and dp.has_next():
        ev = dp.next_event()

        # global monotonicity
        last_t = assert_non_decreasing(last_t, ev.t_ns, "sanity/raw_stream")
        if t_first is None:
            t_first = ev.t_ns
        t_last = ev.t_ns

        if ev.type == "imu":
            imu_count += 1
            if last_imu_t is not None:
                imu_dts.append(ev.t_ns - last_imu_t)
            last_imu_t = ev.t_ns

        elif ev.type == "stereo":
            stereo_count += 1
            stereo_ts.append(ev.t_ns)
            if last_st_t is not None:
                st_dts.append(ev.t_ns - last_st_t)
            last_st_t = ev.t_ns

        else:
            raise RuntimeError(f"Unknown event type: {getattr(ev, 'type', None)}")

        k += 1

    if t_first is None or t_last is None:
        print("No events read.")
        return

    span_s = ns_to_s(t_last - t_first)

    print("\n=== EuRoCProvider sanity ===")
    print(f"Sequence: {seq_root}")
    print(f"Read events: {k} (requested {n})")
    print(f"Time span: {span_s:.3f}s  (t_first={t_first}, t_last={t_last})")
    print(f"Counts: IMU={imu_count}, Stereo={stereo_count}")

    if span_s > 0:
        print(f"Approx rates over span: IMU={imu_count/span_s:.2f} Hz, Stereo={stereo_count/span_s:.2f} Hz")

    print()
    summarize_dts("IMU dt", imu_dts)
    summarize_dts("Stereo dt", st_dts)

    # Extra stereo checks
    if stereo_ts:
        stereo_arr = np.asarray(stereo_ts, dtype=np.int64)
        uniq = np.unique(stereo_arr).size
        repeats = len(stereo_arr) - uniq
        print()
        print(f"Stereo timestamps: unique={uniq}/{len(stereo_ts)} (repeats={repeats})")
        if repeats > 0:
            print("WARNING: repeated stereo timestamps found (possible pairing issue).")

    print("\nOK: provider stream is monotonic and stats printed.")


if __name__ == "__main__":
    main()
