'''
This will turn your YAML numbers into K0,D0,K1,D1,T_cam0_cam1.
You can keep your constants (the numbers you pasted) either:
hardcoded for now (fast), or later parse YAML properly. 
For now, hardcoding is fine for sanity.
'''

from __future__ import annotations
import numpy as np

def K_from_intrinsics(fu: float, fv: float, cu: float, cv: float) -> np.ndarray:
    return np.array([[fu, 0.0, cu],
                     [0.0, fv, cv],
                     [0.0, 0.0, 1.0]], dtype=np.float64)

def D_from_radtan4(k1: float, k2: float, p1: float, p2: float) -> np.ndarray:
    return np.array([k1, k2, p1, p2, 0.0], dtype=np.float64)

def T_from_yaml_data(data_list: list[float]) -> np.ndarray:
    return np.array(data_list, dtype=np.float64).reshape(4, 4)

def T_cam0_cam1_from_T_BS(T_BS0: np.ndarray, T_BS1: np.ndarray) -> np.ndarray:
    # cam0 -> cam1
    return np.linalg.inv(T_BS1) @ T_BS0
