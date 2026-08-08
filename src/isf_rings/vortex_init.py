"""由闭合圆形涡丝构造二分量波函数初值。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimulationConfig
from .grid import PeriodicGrid
from .projection import ProjectionDiagnostics, pressure_project
from .wavefunction import normalize_spinor


@dataclass(frozen=True)
class RingGeometry:
    """一条圆形涡丝的几何信息；圆环平面始终垂直于 x 轴。"""

    center: tuple[float, float, float]
    radius: float
    circulation_sign: int = 1

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("圆环半径必须为正数。")
        if self.circulation_sign not in (-1, 1):
            raise ValueError("circulation_sign 只能是 +1 或 -1。")


def _ring_cross_section(
    grid: PeriodicGrid, ring: RingGeometry
) -> tuple[np.ndarray, np.ndarray]:
    """返回圆环法截面中的两个局部坐标。

    s=0 且 n=0 的交集恰好是一条圆形涡丝。绕这个点在 (s,n) 平面
    转一周时，复数 s+i*n 的相位也转一周，因此它可作为涡丝的相位核。
    """

    cx, cy, cz = ring.center
    dx = grid.periodic_displacement(0, cx)[:, np.newaxis, np.newaxis]
    dy = grid.periodic_displacement(1, cy)[np.newaxis, :, np.newaxis]
    dz = grid.periodic_displacement(2, cz)[np.newaxis, np.newaxis, :]
    radial_distance = np.sqrt(dy**2 + dz**2)
    signed_radial_distance = radial_distance - ring.radius
    return signed_radial_distance, dx


def spinor_for_ring(
    grid: PeriodicGrid, ring: RingGeometry, core_radius: float
) -> np.ndarray:
    """为一条圆环建立正则化二分量波函数。

    第一分量包含绕涡丝转一圈的相位；第二分量为一个小的正实数，
    使波函数在涡核处仍可归一化。这是论文 §3.1 中 (phi, epsilon)
    构造在圆环几何下的直接实现。
    """

    if core_radius <= 0.0:
        raise ValueError("core_radius 必须为正数。")
    signed_radial, normal_distance = _ring_cross_section(grid, ring)
    phi = signed_radial + 1j * ring.circulation_sign * normal_distance
    epsilon = np.full(grid.shape, core_radius, dtype=complex)
    return normalize_spinor(np.stack((phi, epsilon), axis=0))


def combine_rings(ring_spinors: list[np.ndarray]) -> np.ndarray:
    """按分量相乘组合多条涡丝，再整体归一化。

    这对应论文 §3.1 对多条涡丝的组合方式。两条涡环同号时，
    它们的相位绕转会叠加，而不是相互抵消。
    """

    if not ring_spinors:
        raise ValueError("至少需要一条涡丝。")
    combined = np.ones_like(ring_spinors[0], dtype=complex)
    for spinor in ring_spinors:
        if spinor.shape != combined.shape:
            raise ValueError("所有涡丝波函数必须定义在同一网格上。")
        combined *= spinor
    return normalize_spinor(combined)


def initialize_two_coaxial_rings(
    grid: PeriodicGrid, config: SimulationConfig
) -> tuple[np.ndarray, ProjectionDiagnostics]:
    """建立题目要求的两个同轴、同向圆形涡环并做初始压力投影。"""

    lx, ly, lz = grid.lengths
    left_center = (0.5 * lx - 0.5 * config.ring_separation, 0.5 * ly, 0.5 * lz)
    right_center = (0.5 * lx + 0.5 * config.ring_separation, 0.5 * ly, 0.5 * lz)
    rings = [
        RingGeometry(left_center, config.ring_radius, circulation_sign=1),
        RingGeometry(right_center, config.ring_radius, circulation_sign=1),
    ]
    initial = combine_rings(
        [spinor_for_ring(grid, ring, config.core_radius) for ring in rings]
    )
    # 初始几何相位不必天然满足无散约束，因此按论文流程先投影一次。
    return pressure_project(initial, grid, config.hbar)
