"""论文算法 2 和算法 3 所需的 FFT 演化、微分和泊松求解。"""

from __future__ import annotations

import numpy as np

from .grid import PeriodicGrid


def _fft3(field: np.ndarray) -> np.ndarray:
    """对数组最后三个轴做三维 FFT；前面的分量轴保持不变。"""

    return np.fft.fftn(field, axes=(-3, -2, -1))


def _ifft3(field_hat: np.ndarray) -> np.ndarray:
    """对数组最后三个轴做三维逆 FFT。"""

    return np.fft.ifftn(field_hat, axes=(-3, -2, -1))


def free_schrodinger_step(
    psi: np.ndarray, grid: PeriodicGrid, hbar: float, dt: float
) -> np.ndarray:
    """执行一次自由 Schrödinger 演化。

    对应论文算法 2。傅里叶空间中拉普拉斯算子的本征值为 -|k|^2，
    因而每个频率分量只需乘以一个相位因子。
    """

    phase = np.exp(-0.5j * hbar * grid.k2 * dt)
    return _ifft3(_fft3(psi) * phase)


def spectral_derivative(
    field: np.ndarray, wave_numbers: np.ndarray
) -> np.ndarray:
    """以谱方法计算任意标量或分量场的一个方向导数。"""

    return _ifft3(1j * wave_numbers * _fft3(field))


def divergence(vector: np.ndarray, grid: PeriodicGrid) -> np.ndarray:
    """计算 shape 为 (3, Nx, Ny, Nz) 的向量场散度。"""

    if vector.shape[0] != 3:
        raise ValueError("向量场的第一个轴必须有 3 个分量。")
    return np.real_if_close(
        spectral_derivative(vector[0], grid.kx)
        + spectral_derivative(vector[1], grid.ky)
        + spectral_derivative(vector[2], grid.kz)
    ).real


def curl(vector: np.ndarray, grid: PeriodicGrid) -> np.ndarray:
    """用谱导数计算速度场旋度，得到涡量。"""

    if vector.shape[0] != 3:
        raise ValueError("向量场的第一个轴必须有 3 个分量。")
    dvz_dy = spectral_derivative(vector[2], grid.ky)
    dvy_dz = spectral_derivative(vector[1], grid.kz)
    dvx_dz = spectral_derivative(vector[0], grid.kz)
    dvz_dx = spectral_derivative(vector[2], grid.kx)
    dvy_dx = spectral_derivative(vector[1], grid.kx)
    dvx_dy = spectral_derivative(vector[0], grid.ky)
    return np.real_if_close(
        np.stack(
            (dvz_dy - dvy_dz, dvx_dz - dvz_dx, dvy_dx - dvx_dy), axis=0
        )
    ).real


def solve_periodic_poisson(rhs: np.ndarray, grid: PeriodicGrid) -> np.ndarray:
    """求解 Delta(q)=rhs，并把压力的零频率固定为零。

    周期边界下压力只确定到一个常数。将 q_hat[0, 0, 0] 设为零，
    既消除了这个不唯一性，也不会影响压力梯度和投影结果。
    """

    rhs_hat = _fft3(rhs)
    solution_hat = np.zeros_like(rhs_hat, dtype=complex)
    nonzero = grid.k2 > 0.0
    solution_hat[nonzero] = -rhs_hat[nonzero] / grid.k2[nonzero]
    return _ifft3(solution_hat).real


def solve_periodic_fd_poisson(rhs: np.ndarray, grid: PeriodicGrid) -> np.ndarray:
    """求解与正向边散度完全匹配的二阶差分泊松方程。

    压力投影中的速度来自相邻节点的相位差，而不是连续导数。因此
    这里不能复用连续谱拉普拉斯，必须使用对应差分拉普拉斯的 FFT
    本征值；否则投影只能近似降低散度，不能在离散意义上准确消除它。
    """

    eigenvalue = (
        -4.0 * np.sin(0.5 * grid.kx * grid.dx) ** 2 / grid.dx**2
        -4.0 * np.sin(0.5 * grid.ky * grid.dy) ** 2 / grid.dy**2
        -4.0 * np.sin(0.5 * grid.kz * grid.dz) ** 2 / grid.dz**2
    )
    rhs_hat = _fft3(rhs)
    solution_hat = np.zeros_like(rhs_hat, dtype=complex)
    nonzero = np.abs(eigenvalue) > 0.0
    solution_hat[nonzero] = rhs_hat[nonzero] / eigenvalue[nonzero]
    return _ifft3(solution_hat).real
