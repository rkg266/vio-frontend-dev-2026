from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Union, Literal, Any
import numpy as np


# -----------------------------
# Core sensor samples
# -----------------------------

@dataclass(frozen=True)
class ImuSample:
    t_ns: int
    accel: np.ndarray  # shape (3,), float32/float64
    gyro: np.ndarray   # shape (3,), float32/float64


@dataclass(frozen=True)
class StereoFrame:
    t_ns: int
    left: np.ndarray   # HxW (uint8) or HxWx3
    right: np.ndarray  # HxW (uint8) or HxWx3
    exposure_ns: Optional[int] = None  # optional, if you have it


# -----------------------------
# Raw events from dataset/provider
# -----------------------------

@dataclass(frozen=True)
class RawImuEvent:
    type: Literal["imu"]
    t_ns: int
    accel: np.ndarray
    gyro: np.ndarray


@dataclass(frozen=True)
class RawStereoEvent:
    type: Literal["stereo"]
    t_ns: int
    left: np.ndarray
    right: np.ndarray
    exposure_ns: Optional[int] = None


RawEvent = Union[RawImuEvent, RawStereoEvent]


# -----------------------------
# Provider interface (dataset-agnostic)
# -----------------------------

class IDataProvider(Protocol):
    """Yields RawEvent in non-decreasing timestamp order."""
    def has_next(self) -> bool: ...
    def next_event(self) -> RawEvent: ...


# -----------------------------
# Utility: strict timestamp check
# -----------------------------

class TimestampError(RuntimeError):
    pass


def assert_non_decreasing(prev_t_ns: Optional[int], new_t_ns: int, name: str) -> int:
    if prev_t_ns is not None and new_t_ns < prev_t_ns:
        raise TimestampError(f"{name}: timestamps decreased ({new_t_ns} < {prev_t_ns})")
    return new_t_ns
