"""基于离散边相位的不可压压力投影。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fft_ops import solve_periodic_fd_poisson
from .grid import PeriodicGrid
from .wavefunction import normalize_spinor


@dataclass(frozen=True)
class ProjectionDiagnostics:
    """记录投影前后的离散散度，便于验证算法 3 是否起效。"""

    divergence_l2_before: float
    divergence_l2_after: float
    pressure: np.ndarray


def forward_edge_phases(psi: np.ndarray) -> np.ndarray:
    """计算每条正向网格边上的无量纲相位差。

    Chern et al. 的离散速度一形式为 hbar*arg(<psi_v, psi_w>)。
    这里先保存 arg(<psi_v, psi_w>)，在后续步骤中再乘 hbar/dx 得到
    边中点速度，避免在压力方程中混淆量纲。
    """

    if psi.ndim != 4 or psi.shape[0] != 2:
        raise ValueError("psi 的形状必须为 (2, Nx, Ny, Nz)。")
    phases = []
    for axis in (1, 2, 3):
        neighbour = np.roll(psi, shift=-1, axis=axis)
        link_inner_product = np.sum(np.conj(psi) * neighbour, axis=0)
        phases.append(np.angle(link_inner_product))
    return np.stack(phases, axis=0)


def edge_phase_velocity(
    phases: np.ndarray, grid: PeriodicGrid, hbar: float
) -> np.ndarray:
    """把边相位差除以边长，得到三个方向的边速度近似。"""

    spacings = (grid.dx, grid.dy, grid.dz)
    return np.stack(
        [hbar * phases[axis] / spacings[axis] for axis in range(3)], axis=0
    )


def edge_divergence(edge_velocity: np.ndarray, grid: PeriodicGrid) -> np.ndarray:
    """对正向边速度做有限体积散度，适合相位投影的离散形式。"""

    spacings = (grid.dx, grid.dy, grid.dz)
    result = np.zeros(grid.shape, dtype=float)
    for axis in range(3):
        result += (
            edge_velocity[axis]
            - np.roll(edge_velocity[axis], shift=1, axis=axis)
        ) / spacings[axis]
    return result


def _l2_norm(field: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(field) ** 2)))


def pressure_project(
    psi: np.ndarray, grid: PeriodicGrid, hbar: float
) -> tuple[np.ndarray, ProjectionDiagnostics]:
    """执行论文算法 3 的相位压力投影。

    若 psi 乘以 exp(-i*q)，边相位会减去 q 的离散梯度。因此先解
    Delta(q)=div(v_edge)/hbar，再施加该相位，可使离散边速度趋于无散。
    """

    phases_before = forward_edge_phases(psi)
    velocity_before = edge_phase_velocity(phases_before, grid, hbar)
    divergence_before = edge_divergence(velocity_before, grid)

    pressure = solve_periodic_fd_poisson(divergence_before / hbar, grid)
    projected = normalize_spinor(np.exp(-1j * pressure)[np.newaxis, ...] * psi)

    phases_after = forward_edge_phases(projected)
    velocity_after = edge_phase_velocity(phases_after, grid, hbar)
    divergence_after = edge_divergence(velocity_after, grid)

    return projected, ProjectionDiagnostics(
        divergence_l2_before=_l2_norm(divergence_before),
        divergence_l2_after=_l2_norm(divergence_after),
        pressure=pressure,
    )
