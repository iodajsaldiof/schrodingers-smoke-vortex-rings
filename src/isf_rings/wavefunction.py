"""二分量波函数的归一化、密度和连续速度重建。"""

from __future__ import annotations

import numpy as np

from .fft_ops import spectral_derivative
from .grid import PeriodicGrid


def _check_spinor(psi: np.ndarray) -> None:
    if psi.ndim != 4 or psi.shape[0] != 2:
        raise ValueError("psi 的形状必须为 (2, Nx, Ny, Nz)。")


def density(psi: np.ndarray) -> np.ndarray:
    """计算 rho=|psi_1|^2+|psi_2|^2。"""

    _check_spinor(psi)
    return np.sum(np.abs(psi) ** 2, axis=0)


def normalize_spinor(psi: np.ndarray, floor: float = 1.0e-14) -> np.ndarray:
    """逐点归一化波函数，对应论文算法 1 的第二步。"""

    local_norm = np.sqrt(density(psi))
    if np.any(local_norm < floor):
        raise ValueError("发现接近零的二分量波函数，无法安全归一化。")
    return psi / local_norm[np.newaxis, ...]


def max_density_error(psi: np.ndarray) -> float:
    """报告 max|rho-1|，用来检查归一化约束。"""

    return float(np.max(np.abs(density(psi) - 1.0)))


def velocity_from_spinor(
    psi: np.ndarray, grid: PeriodicGrid, hbar: float
) -> np.ndarray:
    """按论文第 1 页的连续公式由波函数重建速度场。

    这个连续速度主要用于画涡量、计算能量和输出诊断；压力投影本身
    使用的是论文附录 D 的离散边相位形式，见 projection.py。
    """

    _check_spinor(psi)
    components = []
    for wave_numbers in (grid.kx, grid.ky, grid.kz):
        derivative = spectral_derivative(psi, wave_numbers)
        component = hbar * np.real(np.sum(np.conj(derivative) * (1j * psi), axis=0))
        components.append(component)
    return np.stack(components, axis=0)
