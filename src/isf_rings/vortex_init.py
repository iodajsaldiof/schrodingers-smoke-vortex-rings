"""按 Chern et al. §3.1 构造论文相位盘与解析零集双涡环初值。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimulationConfig
from .grid import PeriodicGrid
from .projection import ProjectionDiagnostics, pressure_project
from .wavefunction import normalize_spinor


@dataclass(frozen=True)
class RingGeometry:
    """法向为 x 轴的圆形涡环几何。"""

    center: tuple[float, float, float]
    radius: float
    circulation_sign: int = 1

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("涡环半径必须为正数。")
        if self.circulation_sign not in (-1, 1):
            raise ValueError("circulation_sign 只能是 +1 或 -1。")


def _ring_cross_section(grid: PeriodicGrid, ring: RingGeometry) -> tuple[np.ndarray, np.ndarray]:
    """返回到圆环的截面局部坐标：径向有符号距离和盘法向距离。"""

    center_x, center_y, center_z = ring.center
    normal_distance = grid.periodic_displacement(0, center_x)[:, np.newaxis, np.newaxis]
    dy = grid.periodic_displacement(1, center_y)[np.newaxis, :, np.newaxis]
    dz = grid.periodic_displacement(2, center_z)[np.newaxis, np.newaxis, :]
    signed_radial = np.sqrt(dy**2 + dz**2) - ring.radius
    return signed_radial, normal_distance


def spinor_for_ring(
    grid: PeriodicGrid, ring: RingGeometry, core_radius: float
) -> np.ndarray:
    """为单涡环建立尚未归一化的解析复零集旋量 ``(phi, epsilon)``。

    ``phi = signed_radial + i*normal_distance`` 的零集恰为指定圆环，绕其
    截面一周相位改变 ``2*pi``。这落实了 Chern et al. §3.1 中“先构造
    零集为 filament 的复函数 phi”的初值原则。第二分量 ``epsilon`` 取为
    有限网格可解析的 ``core_radius``，随后由压力投影得到无散初始速度。
    """

    if core_radius <= 0.0:
        raise ValueError("core_radius 必须为正数。")
    signed_radial, normal_distance = _ring_cross_section(grid, ring)
    phi = signed_radial + 1j * ring.circulation_sign * normal_distance
    epsilon = np.full(grid.shape, core_radius, dtype=complex)
    return np.stack((phi, epsilon), axis=0)


def combine_rings(ring_spinors: list[np.ndarray]) -> np.ndarray:
    """组合多个涡环，并在最后一次性归一化。

    作者公开的 ``example_leapfrog.m`` 连续对 ``psi1`` 施加每条涡环的相位，
    同时保持 ``psi2=epsilon`` 为单个常数，最后才 Normalize/PressureProject。
    因而这里累积第一分量的 ``phi``，而不把第二分量错误地乘成 ``epsilon^N``。
    """

    if not ring_spinors:
        raise ValueError("至少需要一条涡丝。")
    first_component = np.ones_like(ring_spinors[0][0], dtype=complex)
    second_component = np.array(ring_spinors[0][1], copy=True)
    for spinor in ring_spinors:
        if spinor.shape != ring_spinors[0].shape:
            raise ValueError("所有涡丝波函数必须定义在同一网格上。")
        if not np.allclose(spinor[1], second_component):
            raise ValueError("组合涡环时必须使用相同的 core_radius。")
        first_component *= spinor[0]
    return normalize_spinor(np.stack((first_component, second_component), axis=0))


def apply_paper_phase_disk(
    psi_1: np.ndarray,
    grid: PeriodicGrid,
    ring: RingGeometry,
    thickness: float,
) -> np.ndarray:
    """对 ``psi_1`` 施加作者 ``ISF.m/AddCircle`` 的圆盘相位增量。

    圆盘是涡环的 Seifert 面；穿过该面时相位连续旋转 ``2*pi``。下式与
    Chern et al. 公开 MATLAB 代码的第 109--120 行一一对应，唯一的改动是
    使用周期最短位移以适配本项目的周期计算域。
    """

    if thickness <= 0.0:
        raise ValueError("相位圆盘厚度必须为正数。")
    signed_radial, normal_distance = _ring_cross_section(grid, ring)
    # 作者示例中的 normal=[-1,0,0]。circulation_sign 在此充当圆盘法向，
    # 两环取同一符号即可保证环量方向相同。
    disk_coordinate = np.broadcast_to(
        ring.circulation_sign * normal_distance, grid.shape
    )
    in_cylinder = signed_radial < 0.0
    positive_layer = (
        (disk_coordinate > 0.0)
        & (disk_coordinate <= 0.5 * thickness)
        & in_cylinder
    )
    negative_layer = (
        (disk_coordinate <= 0.0)
        & (disk_coordinate >= -0.5 * thickness)
        & in_cylinder
    )
    alpha = np.zeros(grid.shape, dtype=float)
    alpha[positive_layer] = -np.pi * (
        2.0 * disk_coordinate[positive_layer] / thickness - 1.0
    )
    alpha[negative_layer] = -np.pi * (
        2.0 * disk_coordinate[negative_layer] / thickness + 1.0
    )
    return psi_1 * np.exp(1j * alpha)


def _paper_phase_disk_spinor(
    grid: PeriodicGrid,
    rings: list[RingGeometry],
    config: SimulationConfig,
) -> np.ndarray:
    """建立作者示例的 ``psi1=exp(i phase), psi2=0.01 exp(i phase)`` 初值。"""

    x, y, z = grid.coordinate_mesh()
    background = config.background_velocity
    phase = (
        background[0] * x + background[1] * y + background[2] * z
    ) / config.hbar
    psi_1 = np.exp(1j * phase)
    psi_2 = 0.01 * np.exp(1j * phase)
    thickness = 5.0 * grid.dx
    for ring in rings:
        psi_1 = apply_paper_phase_disk(psi_1, grid, ring, thickness)
    return normalize_spinor(np.stack((psi_1, psi_2), axis=0))


def initialize_two_coaxial_rings(
    grid: PeriodicGrid, config: SimulationConfig
) -> tuple[np.ndarray, ProjectionDiagnostics]:
    """建立两个同轴、同向涡环并执行 Algorithm 3 初始压力投影。"""

    lx, ly, lz = grid.lengths
    left_center = (0.5 * lx - 0.5 * config.ring_separation, 0.5 * ly, 0.5 * lz)
    right_center = (0.5 * lx + 0.5 * config.ring_separation, 0.5 * ly, 0.5 * lz)
    rings = [
        RingGeometry(left_center, config.ring_radii[0], circulation_sign=-1),
        RingGeometry(right_center, config.ring_radii[1], circulation_sign=-1),
    ]
    if config.initialization_mode == "paper_phase_disk":
        initial = _paper_phase_disk_spinor(grid, rings, config)
    else:
        initial = combine_rings(
            [spinor_for_ring(grid, ring, config.core_radius) for ring in rings]
        )
    # 初始几何相位不必天然满足离散无散约束，因此按 Algorithm 3 先投影一次。
    return pressure_project(initial, grid, config.hbar)
