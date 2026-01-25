# vio_frontend/core/state.py
#Defines what your system state is
from dataclasses import dataclass
import numpy as np

@dataclass
class NominalState:
    t_ns: int
    R_wb: np.ndarray   # (3,3) rotation matrix
    p_w: np.ndarray    # (3,)
    v_w: np.ndarray    # (3,)
    b_g: np.ndarray    # (3,)
    b_a: np.ndarray    # (3,)

    @staticmethod
    def identity(t_ns: int) -> "NominalState":
        return NominalState(
            t_ns=t_ns,
            R_wb=np.eye(3),
            p_w=np.zeros(3),
            v_w=np.zeros(3),
            b_g=np.zeros(3),
            b_a=np.zeros(3),
        )
