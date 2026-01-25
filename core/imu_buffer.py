# vio_frontend/core/imu_buffer.py
#Stores IMU samples and returns time segments
from collections import deque
from typing import Deque, List, Optional
from vio_frontend.types import ImuSample

class ImuBuffer:
    def __init__(self, maxlen: int = 50000):
        self._buf: Deque[ImuSample] = deque(maxlen=maxlen)

    def push(self, s: ImuSample) -> None:
        self._buf.append(s)

    def get_segment(self, t0_ns: int, t1_ns: int) -> List[ImuSample]:
        return [s for s in self._buf if t0_ns <= s.t_ns <= t1_ns]

    def latest_time_ns(self) -> Optional[int]:
        return self._buf[-1].t_ns if self._buf else None
