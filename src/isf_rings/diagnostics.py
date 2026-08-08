"""从速度和涡量场提取双涡环的可量化观测量。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fft_ops import curl
from .grid import PeriodicGrid


@dataclass(frozen=True)
class RingObservation:
    """一条涡环在某时刻的几何观测结果。"""

    x: float
    y: float
    z: float
    radius: float
    enstrophy: float


def vorticity_from_velocity(velocity: np.ndarray, grid: PeriodicGrid) -> np.ndarray:
    """从速度场计算涡量 omega=curl(v)。"""

    return curl(velocity, grid)


def vorticity_magnitude(vorticity: np.ndarray) -> np.ndarray:
    """返回 |omega|，可直接用于等值面或二维切片。"""

    if vorticity.shape[0] != 3:
        raise ValueError("涡量场的第一个轴必须有 3 个分量。")
    return np.sqrt(np.sum(vorticity**2, axis=0))


def kinetic_energy(velocity: np.ndarray) -> float:
    """返回单位体积平均动能；用于观察演算是否出现异常漂移。"""

    return float(0.5 * np.mean(np.sum(velocity**2, axis=0)))


def _periodic_index_distance(a: int, b: int, count: int) -> int:
    return min(abs(a - b), count - abs(a - b))


def _pick_two_axial_peaks(profile: np.ndarray, min_distance: int) -> tuple[int, int]:
    """从周期轴向涡量分布中挑选两个彼此分开的峰值。"""

    count = profile.size
    local_peaks = [
        index
        for index in range(count)
        if profile[index] >= profile[(index - 1) % count]
        and profile[index] >= profile[(index + 1) % count]
    ]
    candidates = sorted(local_peaks or list(range(count)), key=profile.__getitem__, reverse=True)
    selected: list[int] = []
    for candidate in candidates:
        if all(
            _periodic_index_distance(candidate, previous, count) >= min_distance
            for previous in selected
        ):
            selected.append(candidate)
        if len(selected) == 2:
            return selected[0], selected[1]
    raise ValueError("两个涡量峰过于接近，当前时刻无法可靠地区分两条涡环。")


def _periodic_weighted_mean(
    coordinates: np.ndarray, weights: np.ndarray, length: float
) -> float:
    """在周期区间内计算加权平均位置，避免靠近边界时平均到错误位置。"""

    phase = 2.0 * np.pi * coordinates / length
    complex_mean = np.sum(weights * np.exp(1j * phase))
    if abs(complex_mean) < 1.0e-15:
        return float(np.average(coordinates, weights=weights))
    angle = np.angle(complex_mean) % (2.0 * np.pi)
    return float(length * angle / (2.0 * np.pi))


def _periodic_delta(values: np.ndarray, center: float, length: float) -> np.ndarray:
    return (values - center + 0.5 * length) % length - 0.5 * length


def extract_two_coaxial_rings(
    vorticity: np.ndarray, grid: PeriodicGrid, axial_window: float | None = None
) -> tuple[RingObservation, RingObservation]:
    """用涡量能量峰值估计两条同轴圆环的中心和半径。

    对相位存在尖锐跳变的早期状态，连续速度求导得到的涡量可能带有
    数值噪声。任务 2 的主轨迹统计应优先调用下方的波函数版本；本函数
    保留用于与涡量等值面相互印证，以及后期替换为文献 6 的涡丝提取。
    """

    return _extract_two_rings_from_weights(
        vorticity_magnitude(vorticity) ** 2, grid, axial_window
    )


def extract_two_coaxial_rings_from_wavefunction(
    psi: np.ndarray, grid: PeriodicGrid, axial_window: float | None = None
) -> tuple[RingObservation, RingObservation]:
    """直接由二分量波函数的涡核指示量提取两条圆环。

    在 (phi, epsilon) 的涡丝构造中，第一分量在涡核附近变小、第二分量
    相对变大。因此 |psi_2|^2 可作为稳定的涡核指示量。它避免了先对相位
    奇点求导再找峰值的放大误差，适合本题跟踪环心和半径。
    """

    if psi.ndim != 4 or psi.shape[0] != 2 or psi.shape[1:] != grid.shape:
        raise ValueError("psi 的形状必须为 (2, Nx, Ny, Nz)，并与 grid 一致。")
    return _extract_two_rings_from_weights(np.abs(psi[1]) ** 2, grid, axial_window)


def _extract_two_rings_from_weights(
    weights: np.ndarray, grid: PeriodicGrid, axial_window: float | None
) -> tuple[RingObservation, RingObservation]:
    """从任意非负涡核权重场中完成共同的峰值与几何拟合过程。

    诊断器先沿 x 方向找到两座峰，再在各自附近做加权平均以估计中心
    和半径。若两环真正重叠到无法分开，函数会明确报错，届时再考虑
    引入参考文献 6 的涡丝提取方法。
    """

    if weights.shape != grid.shape:
        raise ValueError("权重场形状必须与 grid 一致。")
    axial_profile = np.sum(weights, axis=(1, 2))
    # 题目给定的 0.6 m 环间距在开发网格上只有约 4 个节点，
    # 因而这里不能使用过大的峰间距阈值。
    min_distance = 2
    peak_indices = _pick_two_axial_peaks(axial_profile, min_distance)
    # 使用一个轴向网格间距的窗口：既能覆盖涡核，又不会把相邻的 0.6 m
    # 双环初值过度平均到中间位置。
    window = axial_window if axial_window is not None else grid.dx

    observations: list[RingObservation] = []
    for peak_index in peak_indices:
        x_peak = grid.x[peak_index]
        dx = _periodic_delta(grid.x, x_peak, grid.lengths[0])
        axial_weight = np.exp(-0.5 * (dx / window) ** 2)[:, np.newaxis, np.newaxis]
        local_weights = weights * axial_weight
        total_weight = float(np.sum(local_weights))
        if total_weight <= 1.0e-15:
            raise ValueError("涡量权重为零，无法提取涡环几何。")

        # 坐标必须广播到完整三维形状，才能与局部涡量权重逐点相乘。
        x_values = np.broadcast_to(grid.x[:, np.newaxis, np.newaxis], grid.shape)
        y_values = np.broadcast_to(grid.y[np.newaxis, :, np.newaxis], grid.shape)
        z_values = np.broadcast_to(grid.z[np.newaxis, np.newaxis, :], grid.shape)
        x_center = _periodic_weighted_mean(x_values.ravel(), local_weights.ravel(), grid.lengths[0])
        y_center = _periodic_weighted_mean(y_values.ravel(), local_weights.ravel(), grid.lengths[1])
        z_center = _periodic_weighted_mean(z_values.ravel(), local_weights.ravel(), grid.lengths[2])

        dy = _periodic_delta(grid.y, y_center, grid.lengths[1])[np.newaxis, :, np.newaxis]
        dz = _periodic_delta(grid.z, z_center, grid.lengths[2])[np.newaxis, np.newaxis, :]
        radius = float(np.sum(local_weights * np.sqrt(dy**2 + dz**2)) / total_weight)
        observations.append(
            RingObservation(
                x=x_center,
                y=y_center,
                z=z_center,
                radius=radius,
                enstrophy=total_weight,
            )
        )
    return observations[0], observations[1]


class RingTracker:
    """通过与上一时刻最接近的匹配保持两个涡环的编号连续。"""

    def __init__(self, grid: PeriodicGrid) -> None:
        self.grid = grid
        self._previous: tuple[RingObservation, RingObservation] | None = None

    def update(
        self, observations: tuple[RingObservation, RingObservation]
    ) -> tuple[RingObservation, RingObservation]:
        """返回与前一时刻编号一致的两个观测量。"""

        if self._previous is None:
            self._previous = tuple(sorted(observations, key=lambda item: item.x))
            return self._previous

        first, second = observations
        previous_first, previous_second = self._previous

        def cost(current: RingObservation, previous: RingObservation) -> float:
            dx = _periodic_delta(
                np.array([current.x]), previous.x, self.grid.lengths[0]
            )[0]
            return float(dx**2 + 0.25 * (current.radius - previous.radius) ** 2)

        keep_order = cost(first, previous_first) + cost(second, previous_second)
        swap_order = cost(second, previous_first) + cost(first, previous_second)
        self._previous = (first, second) if keep_order <= swap_order else (second, first)
        return self._previous
