"""双涡环实验的几何、拓扑与能量诊断。

这里的主追踪器利用同轴涡环的轴对称先验，在 ``(x, r)`` 平面内定位两个
涡核峰。它比只沿 x 方向找峰更适合 leapfrogging: 两环轴向重合时仍可能
因半径不同而被分开。该选择与 Weißmann--Pinkall 的“由三维场提取涡丝、
再追踪其几何”的目标一致，但不是对其磁 Schrödinger 特征向量算法的复现；
后者可作为任务 3 的通用非轴对称后处理升级方案。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fft_ops import curl, spectral_derivative
from .grid import PeriodicGrid
from .projection import edge_divergence, edge_phase_velocity, forward_edge_phases
from .wavefunction import velocity_from_spinor


@dataclass(frozen=True)
class RingObservation:
    """一条同轴涡环在某时刻的几何观测结果。"""

    x: float
    y: float
    z: float
    radius: float
    enstrophy: float
    core_strength: float = float("nan")
    fit_spread: float = float("nan")


@dataclass(frozen=True)
class EnergyDiagnostics:
    """Chern et al. §4.4--4.5 的单位体积 ISF 能量分解。"""

    kinetic: float
    landau_lifshitz: float
    total: float
    spinor_total: float
    decomposition_residual: float


@dataclass(frozen=True)
class CycleEvent:
    """两个已编号涡环发生一次可靠轴向换序的事件。"""

    time: float
    name: str
    crossing_count: int
    axial_separation: float


@dataclass(frozen=True)
class CycleStatus:
    """完整周期检测器在某一时刻的状态快照。"""

    crossing_count: int
    completed: bool
    cycle_time: float | None
    axial_separation: float


def vorticity_from_velocity(velocity: np.ndarray, grid: PeriodicGrid) -> np.ndarray:
    """从速度场计算涡量 ``omega = curl(v)``。"""

    return curl(velocity, grid)


def vorticity_magnitude(vorticity: np.ndarray) -> np.ndarray:
    """返回 ``|omega|``，可直接用于等值面或二维切片。"""

    if vorticity.shape[0] != 3:
        raise ValueError("涡量场的第一个轴必须有 3 个分量。")
    return np.sqrt(np.sum(vorticity**2, axis=0))


def kinetic_energy(velocity: np.ndarray) -> float:
    """返回单位体积平均动能 ``H_e = 1/2 <|v|^2>``。"""

    if velocity.shape[0] != 3:
        raise ValueError("速度场的第一个轴必须有 3 个分量。")
    return float(0.5 * np.mean(np.sum(velocity**2, axis=0)))


def clebsch_vector(psi: np.ndarray) -> np.ndarray:
    """由二分量单位旋量构造 Chern et al. 式 Clebsch 向量 ``s in S^2``.

    该 Hopf 映射的坐标方向约定不影响下方的 ``|ds|^2`` 与能量计算。
    """

    _check_spinor(psi)
    first, second = psi
    product = np.conj(first) * second
    return np.stack(
        (
            np.abs(first) ** 2 - np.abs(second) ** 2,
            2.0 * np.real(product),
            2.0 * np.imag(product),
        ),
        axis=0,
    )


def isf_energy(psi: np.ndarray, grid: PeriodicGrid, hbar: float) -> EnergyDiagnostics:
    """计算 ISF 哈密顿量及其动能/LL 能量分解。

    Chern et al. 的 Eq. (9) 给出
    ``H_ISF = hbar^2/2 ||d psi||^2 = H_e + hbar^2/8 ||d s||^2``。
    这里使用体积平均而非积分，因此不同网格之间可直接比较。分解残差
    同时检验连续谱速度后处理与旋量梯度能量的一致性。
    """

    _check_spinor(psi)
    velocity = velocity_from_spinor(psi, grid, hbar)
    kinetic = kinetic_energy(velocity)

    spinor_gradient_squared = 0.0
    for wave_numbers in (grid.kx, grid.ky, grid.kz):
        derivative = spectral_derivative(psi, wave_numbers)
        spinor_gradient_squared += np.sum(np.abs(derivative) ** 2, axis=0)
    spinor_total = float(0.5 * hbar**2 * np.mean(spinor_gradient_squared))

    spin = clebsch_vector(psi)
    spin_gradient_squared = 0.0
    for wave_numbers in (grid.kx, grid.ky, grid.kz):
        derivative = spectral_derivative(spin, wave_numbers)
        spin_gradient_squared += np.sum(np.abs(derivative) ** 2, axis=0)
    landau_lifshitz = float(0.125 * hbar**2 * np.mean(spin_gradient_squared))
    total = kinetic + landau_lifshitz
    return EnergyDiagnostics(
        kinetic=kinetic,
        landau_lifshitz=landau_lifshitz,
        total=total,
        spinor_total=spinor_total,
        decomposition_residual=total - spinor_total,
    )


def discrete_divergence_l2(psi: np.ndarray, grid: PeriodicGrid, hbar: float) -> float:
    """返回 Algorithm 3 使用的离散边相位速度散度 L2 范数。"""

    phases = forward_edge_phases(psi)
    edge_velocity = edge_phase_velocity(phases, grid, hbar)
    divergence = edge_divergence(edge_velocity, grid)
    return float(np.sqrt(np.mean(divergence**2)))


def phase_plaquette_circulation(psi: np.ndarray) -> np.ndarray:
    """计算三个坐标面上的离散相位环量。

    这是 Algorithm 3 边相位的一次离散外微分。围住一条单位涡丝的面元
    环量接近 ``2*pi``，因而不会依赖 ``psi_2`` 是否在网格节点恰好取大值。
    """

    phase_x, phase_y, phase_z = forward_edge_phases(psi)
    around_x = (
        phase_y
        + np.roll(phase_z, shift=-1, axis=1)
        - np.roll(phase_y, shift=-1, axis=2)
        - phase_z
    )
    around_y = (
        phase_z
        + np.roll(phase_x, shift=-1, axis=2)
        - np.roll(phase_z, shift=-1, axis=0)
        - phase_x
    )
    around_z = (
        phase_x
        + np.roll(phase_y, shift=-1, axis=0)
        - np.roll(phase_x, shift=-1, axis=1)
        - phase_y
    )
    return np.stack((around_x, around_y, around_z), axis=0)


def discrete_vorticity_from_spinor(
    psi: np.ndarray, grid: PeriodicGrid, hbar: float
) -> np.ndarray:
    """由面元相位环量得到与网格方向对应的离散涡量近似。"""

    circulation = phase_plaquette_circulation(psi)
    return np.stack(
        (
            hbar * circulation[0] / (grid.dy * grid.dz),
            hbar * circulation[1] / (grid.dz * grid.dx),
            hbar * circulation[2] / (grid.dx * grid.dy),
        ),
        axis=0,
    )


def discrete_vorticity_magnitude(
    psi: np.ndarray, grid: PeriodicGrid, hbar: float
) -> np.ndarray:
    """返回相位面元涡量模，作为论文初值下的涡核定位和等值面标量场。"""

    return vorticity_magnitude(discrete_vorticity_from_spinor(psi, grid, hbar))


def linked_ring_circulation(
    psi: np.ndarray,
    grid: PeriodicGrid,
    hbar: float,
    ring: RingObservation,
    loop_half_width: float,
) -> float:
    """在涡环一点的法截面上计算离散相位环量。

    对轴向为 x 的圆环，在 ``(x, y)`` 小矩形上积分边相位；该矩形在
    ``y = y_center + radius, z = z_center`` 处链接涡丝。理论目标是
    ``2*pi*hbar``（Chern et al. §4.5）。它是离散量，不能替代一般曲线
    环量积分，但很适合作为本题同轴圆环的网格收敛诊断。
    """

    if loop_half_width <= 0.0:
        raise ValueError("loop_half_width 必须为正数。")
    phases = forward_edge_phases(psi)
    nx, ny, nz = grid.shape
    x_center = _nearest_periodic_index(ring.x, grid.dx, nx)
    y_center = _nearest_periodic_index(ring.y + ring.radius, grid.dy, ny)
    z_center = _nearest_periodic_index(ring.z, grid.dz, nz)
    half_x = max(1, int(round(loop_half_width / grid.dx)))
    half_y = max(1, int(round(loop_half_width / grid.dy)))
    if 2 * half_x >= nx or 2 * half_y >= ny:
        raise ValueError("环量积分回路过大，接近了周期计算域边界。")

    x_low = (x_center - half_x) % nx
    x_high = (x_center + half_x) % nx
    y_low = (y_center - half_y) % ny
    y_high = (y_center + half_y) % ny
    phase_sum = 0.0
    for offset in range(2 * half_x):
        phase_sum += phases[0, (x_low + offset) % nx, y_low, z_center]
        phase_sum -= phases[0, (x_low + offset) % nx, y_high, z_center]
    for offset in range(2 * half_y):
        phase_sum += phases[1, x_high, (y_low + offset) % ny, z_center]
        phase_sum -= phases[1, x_low, (y_low + offset) % ny, z_center]
    return float(hbar * phase_sum)


def _check_spinor(psi: np.ndarray) -> None:
    if psi.ndim != 4 or psi.shape[0] != 2:
        raise ValueError("psi 的形状必须为 (2, Nx, Ny, Nz)。")


def _nearest_periodic_index(value: float, spacing: float, count: int) -> int:
    return int(np.floor(value / spacing + 0.5)) % count


def _periodic_index_distance(a: int, b: int, count: int) -> int:
    return min(abs(a - b), count - abs(a - b))


def _periodic_weighted_mean(
    coordinates: np.ndarray, weights: np.ndarray, length: float
) -> float:
    """在周期区间内计算加权平均位置，避免跨边界时平均到错误位置。"""

    phase = 2.0 * np.pi * coordinates / length
    complex_mean = np.sum(weights * np.exp(1j * phase))
    if abs(complex_mean) < 1.0e-15:
        return float(np.average(coordinates, weights=weights))
    angle = np.angle(complex_mean) % (2.0 * np.pi)
    return float(length * angle / (2.0 * np.pi))


def _periodic_delta(values: np.ndarray, center: float, length: float) -> np.ndarray:
    return (values - center + 0.5 * length) % length - 0.5 * length


def _transverse_center(weights: np.ndarray, grid: PeriodicGrid) -> tuple[float, float]:
    """用所有涡核权重估计双环共享的横向中心。"""

    transverse = np.sum(weights, axis=0)
    y_values = np.broadcast_to(grid.y[:, np.newaxis], transverse.shape)
    z_values = np.broadcast_to(grid.z[np.newaxis, :], transverse.shape)
    return (
        _periodic_weighted_mean(y_values.ravel(), transverse.ravel(), grid.lengths[1]),
        _periodic_weighted_mean(z_values.ravel(), transverse.ravel(), grid.lengths[2]),
    )


def _radial_core_profile(
    weights: np.ndarray, radius: np.ndarray, grid: PeriodicGrid
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把三维涡核权重压缩到轴对称 ``(x,r)`` 的平均核强度。"""

    radial_limit = 0.5 * min(grid.lengths[1], grid.lengths[2])
    radial_bins = max(16, min(64, max(grid.shape[1], grid.shape[2])))
    dr = radial_limit / radial_bins
    radial_indices = np.minimum((radius / dr).astype(int), radial_bins - 1)
    counts = np.bincount(radial_indices.ravel(), minlength=radial_bins).astype(float)
    profile = np.empty((grid.shape[0], radial_bins), dtype=float)
    for x_index in range(grid.shape[0]):
        totals = np.bincount(
            radial_indices.ravel(),
            weights[x_index].ravel(),
            minlength=radial_bins,
        )
        profile[x_index] = totals / np.maximum(counts, 1.0)
    radial_coordinates = (np.arange(radial_bins, dtype=float) + 0.5) * dr
    return profile, radial_coordinates, radius


def _pick_two_core_peaks(
    profile: np.ndarray, radial_coordinates: np.ndarray, grid: PeriodicGrid
) -> tuple[tuple[int, int], tuple[int, int]]:
    """在 ``(x,r)`` 平面挑选相互分离的两个涡核峰。"""

    candidates = np.argsort(profile.ravel())[::-1]
    radial_spacing = radial_coordinates[1] - radial_coordinates[0]
    min_distance = 3.0 * max(grid.dx, radial_spacing)
    selected: list[tuple[int, int]] = []
    for flat_index in candidates:
        x_index, r_index = np.unravel_index(flat_index, profile.shape)
        if profile[x_index, r_index] <= 0.0:
            break
        separated = True
        for previous_x, previous_r in selected:
            dx = _periodic_index_distance(x_index, previous_x, grid.shape[0]) * grid.dx
            dr = radial_coordinates[r_index] - radial_coordinates[previous_r]
            if float(np.hypot(dx, dr)) < min_distance:
                separated = False
                break
        if separated:
            selected.append((x_index, r_index))
        if len(selected) == 2:
            return selected[0], selected[1]
    raise ValueError("无法在轴向-半径平面中可靠地区分两条涡环。")


def extract_two_coaxial_rings(
    vorticity: np.ndarray, grid: PeriodicGrid, axial_window: float | None = None
) -> tuple[RingObservation, RingObservation]:
    """由涡量模平方提取双环，用于与波函数涡核提取交叉核验。"""

    return _extract_two_rings_from_weights(
        vorticity_magnitude(vorticity) ** 2, grid, axial_window
    )


def extract_two_coaxial_rings_from_wavefunction(
    psi: np.ndarray,
    grid: PeriodicGrid,
    hbar: float | None = None,
    axial_window: float | None = None,
    seeds: tuple[RingObservation, RingObservation] | None = None,
) -> tuple[RingObservation, RingObservation]:
    """由 ``|psi_2|^2`` 的轴对称峰提取解析零集初始化的同轴双环。

    该指标适用于 ``analytic_zero_set``：第一分量在涡核减小、第二分量相对
    增大。Chern 作者的 ``paper_phase_disk`` 初值不具有这个幅值核，应改用
    :func:`extract_two_coaxial_rings_from_phase_vorticity`。
    """

    _check_spinor(psi)
    del hbar
    weights = np.abs(psi[1]) ** 2
    return _extract_two_rings_from_weights(weights, grid, axial_window, seeds)


def extract_two_coaxial_rings_from_phase_vorticity(
    psi: np.ndarray,
    grid: PeriodicGrid,
    hbar: float,
    axial_window: float | None = None,
    seeds: tuple[RingObservation, RingObservation] | None = None,
) -> tuple[RingObservation, RingObservation]:
    """以 Algorithm 3 的离散面元相位涡量定位论文相位盘初值的双涡环。

    ``AddCircle`` 将拓扑信息编码在波函数相位而非 ``|psi_2|`` 振幅中。先取
    面元相位环量的模平方作为权重，再在同轴 ``(x,r)`` 平面提取两个局部核，
    因而与压力投影使用的离散速度一形式保持一致。
    """

    _check_spinor(psi)
    weights = discrete_vorticity_magnitude(psi, grid, hbar) ** 2
    return _extract_two_rings_from_weights(weights, grid, axial_window, seeds)


def _extract_two_rings_from_weights(
    weights: np.ndarray,
    grid: PeriodicGrid,
    axial_window: float | None,
    seeds: tuple[RingObservation, RingObservation] | None = None,
) -> tuple[RingObservation, RingObservation]:
    if weights.shape != grid.shape:
        raise ValueError("权重场形状必须与 grid 一致。")
    if float(np.max(weights)) <= 1.0e-15:
        raise ValueError("涡核权重为零，无法提取涡环几何。")

    y_center, z_center = _transverse_center(weights, grid)
    dy = _periodic_delta(grid.y, y_center, grid.lengths[1])[:, np.newaxis]
    dz = _periodic_delta(grid.z, z_center, grid.lengths[2])[np.newaxis, :]
    radial_distance = np.sqrt(dy**2 + dz**2)
    profile, radial_coordinates, radius_field = _radial_core_profile(
        weights, radial_distance, grid
    )
    x_values = np.broadcast_to(grid.x[:, np.newaxis, np.newaxis], grid.shape)
    radial_values = np.broadcast_to(radius_field[np.newaxis, :, :], grid.shape)
    window_x = axial_window if axial_window is not None else max(1.5 * grid.dx, 0.12)
    radial_spacing = radial_coordinates[1] - radial_coordinates[0]
    window_r = max(1.5 * radial_spacing, 0.12)
    exclusive_regions: tuple[np.ndarray, np.ndarray] | None = None
    if seeds is None:
        peak_indices = _pick_two_core_peaks(profile, radial_coordinates, grid)
        centers = [(grid.x[x_index], radial_coordinates[r_index]) for x_index, r_index in peak_indices]
    else:
        centers = [(seed.x, seed.radius) for seed in seeds]
        # 当两环在轴向相遇时，两个局部核会重叠。以两个上一帧中心的 (x,r)
        # Voronoi 划分分配权重，防止两个重心更新吸到同一条涡环上。
        seed_distances = []
        for seed_x, seed_radius in centers:
            seed_dx = _periodic_delta(
                grid.x, seed_x, grid.lengths[0]
            )[:, np.newaxis, np.newaxis]
            seed_dr = radial_values - seed_radius
            seed_distances.append((seed_dx / window_x) ** 2 + (seed_dr / window_r) ** 2)
        exclusive_regions = (
            seed_distances[0] <= seed_distances[1],
            seed_distances[1] < seed_distances[0],
        )

    observations: list[RingObservation] = []
    for observation_index, (x_peak, r_peak) in enumerate(centers):
        # 首帧可做两次收敛；连续帧保留上一帧的专属 Voronoi 区域，只做一次
        # 更新，避免两环靠近时反复迭代到同一个全局峰。
        iterations = 2 if exclusive_regions is None else 1
        for _ in range(iterations):
            local_x = _periodic_delta(grid.x, x_peak, grid.lengths[0])[:, np.newaxis, np.newaxis]
            local_r = radial_values - r_peak
            kernel = np.exp(-0.5 * ((local_x / window_x) ** 2 + (local_r / window_r) ** 2))
            local_weights = weights * kernel
            if exclusive_regions is not None:
                local_weights = local_weights * exclusive_regions[observation_index]
            total_weight = float(np.sum(local_weights))
            if total_weight <= 1.0e-15:
                raise ValueError("局部涡核权重过小，无法估计涡环几何。")
            x_peak = _periodic_weighted_mean(
                x_values.ravel(), local_weights.ravel(), grid.lengths[0]
            )
            r_peak = float(np.sum(local_weights * radial_values) / total_weight)

        x_center = x_peak
        radius = r_peak
        local_x_centered = _periodic_delta(grid.x, x_center, grid.lengths[0])[:, np.newaxis, np.newaxis]
        fit_spread = float(
            np.sqrt(
                np.sum(local_weights * (local_x_centered**2 + (radial_values - radius) ** 2))
                / total_weight
            )
        )
        observations.append(
            RingObservation(
                x=x_center,
                y=y_center,
                z=z_center,
                radius=radius,
                enstrophy=total_weight,
                core_strength=float(np.max(local_weights)),
                fit_spread=fit_spread,
            )
        )
    return observations[0], observations[1]


class RingTracker:
    """通过 ``(x,r)`` 最近匹配保持两个涡环身份和周期展开坐标连续。"""

    def __init__(self, grid: PeriodicGrid, core_field: str = "wavefunction") -> None:
        if core_field not in ("wavefunction", "phase_vorticity"):
            raise ValueError("core_field 只能是 wavefunction 或 phase_vorticity。")
        self.grid = grid
        self.core_field = core_field
        self._previous: tuple[RingObservation, RingObservation] | None = None
        self._unwrapped_x: tuple[float, float] | None = None
        self._velocity: tuple[tuple[float, float], tuple[float, float]] | None = None

    @property
    def unwrapped_x(self) -> tuple[float, float]:
        """返回最近一次更新后的连续轴向坐标。"""

        if self._unwrapped_x is None:
            raise RuntimeError("必须先调用 update 后才能读取连续坐标。")
        return self._unwrapped_x

    def update(
        self, observations: tuple[RingObservation, RingObservation]
    ) -> tuple[RingObservation, RingObservation]:
        """返回与前一时刻身份一致的两个观测量，并更新展开坐标。"""

        if self._previous is None:
            self._previous = tuple(sorted(observations, key=lambda item: item.x))
            self._unwrapped_x = (self._previous[0].x, self._previous[1].x)
            self._velocity = ((0.0, 0.0), (0.0, 0.0))
            return self._previous

        first, second = observations
        previous_first, previous_second = self._previous

        assert self._unwrapped_x is not None
        assert self._velocity is not None

        def cost(
            current: RingObservation,
            previous: RingObservation,
            previous_unwrapped: float,
            velocity: tuple[float, float],
        ) -> float:
            candidate_unwrapped = previous_unwrapped + _periodic_delta(
                np.array([current.x]), previous.x, self.grid.lengths[0]
            )[0]
            predicted_x = previous_unwrapped + velocity[0]
            predicted_radius = previous.radius + velocity[1]
            return float(
                (candidate_unwrapped - predicted_x) ** 2
                + (current.radius - predicted_radius) ** 2
            )

        keep_order = cost(
            first, previous_first, self._unwrapped_x[0], self._velocity[0]
        ) + cost(second, previous_second, self._unwrapped_x[1], self._velocity[1])
        swap_order = cost(
            second, previous_first, self._unwrapped_x[0], self._velocity[0]
        ) + cost(first, previous_second, self._unwrapped_x[1], self._velocity[1])
        current = (first, second) if keep_order <= swap_order else (second, first)
        previous_unwrapped = self._unwrapped_x
        next_unwrapped = tuple(
            previous_unwrapped
            + float(_periodic_delta(np.array([item.x]), previous.x, self.grid.lengths[0])[0])
            for item, previous, previous_unwrapped in zip(
                current, self._previous, self._unwrapped_x
            )
        )
        self._velocity = tuple(
            (current_item_x - previous_item_x, current_item.radius - previous.radius)
            for current_item_x, previous_item_x, current_item, previous in zip(
                next_unwrapped, previous_unwrapped, current, self._previous
            )
        )
        self._unwrapped_x = next_unwrapped
        self._previous = current
        return current

    def update_from_wavefunction(
        self, psi: np.ndarray, hbar: float
    ) -> tuple[RingObservation, RingObservation]:
        """从当前旋量提取双环，并用上一帧的局部核限制候选区域。

        首帧没有历史信息，必须做全局寻峰；此后将上一帧两个观测值作为种子，
        在它们的 Voronoi 专属区域内更新两个重心。这是针对同轴双环的连续
        追踪约束，避免环靠近时全局第二峰突然跳到无关的低密度区域。
        """

        extractor = (
            extract_two_coaxial_rings_from_wavefunction
            if self.core_field == "wavefunction"
            else extract_two_coaxial_rings_from_phase_vorticity
        )
        observations = extractor(psi, self.grid, hbar, seeds=self._previous)
        return self.update(observations)


class LeapfroggingCycleDetector:
    """以换序和半径回归共同判断一个双涡环 leapfrogging 周期。"""

    def __init__(
        self,
        separation_hysteresis: float,
        recurrence_radius_tolerance: float | None = None,
        min_exchange_interval: float = 0.0,
    ) -> None:
        if separation_hysteresis <= 0.0:
            raise ValueError("separation_hysteresis 必须为正数。")
        if recurrence_radius_tolerance is not None and recurrence_radius_tolerance <= 0.0:
            raise ValueError("recurrence_radius_tolerance 必须为正数或 None。")
        if min_exchange_interval < 0.0:
            raise ValueError("min_exchange_interval 不能为负数。")
        self.separation_hysteresis = separation_hysteresis
        self.recurrence_radius_tolerance = recurrence_radius_tolerance
        self.min_exchange_interval = min_exchange_interval
        self._initial_sign: int | None = None
        self._initial_radii: tuple[float, float] | None = None
        self._stable_sign: int | None = None
        self._events: list[CycleEvent] = []
        self._cycle_time: float | None = None
        self._last_exchange_time: float | None = None

    @property
    def events(self) -> tuple[CycleEvent, ...]:
        return tuple(self._events)

    def update(
        self,
        time: float,
        ring_1_x: float,
        ring_2_x: float,
        ring_1_radius: float | None = None,
        ring_2_radius: float | None = None,
    ) -> CycleStatus:
        """更新状态；换序后还须检验半径是否回归初始构型。

        仅有两次 x 曲线交叉不足以证明周期。局部峰合并或身份误配也会产生
        这种图形；因此正式计算传入两个半径后，第二次换序只有在两环半径均
        回到初始值附近时才宣布完整周期。单元测试可省略半径以独立验证滞回。
        """

        separation = ring_2_x - ring_1_x
        if separation > self.separation_hysteresis:
            sign = 1
        elif separation < -self.separation_hysteresis:
            sign = -1
        else:
            sign = 0

        if self._initial_sign is None and sign != 0:
            self._initial_sign = sign
            self._stable_sign = sign
            if ring_1_radius is not None and ring_2_radius is not None:
                self._initial_radii = (ring_1_radius, ring_2_radius)
        elif (
            sign != 0
            and self._stable_sign is not None
            and sign != self._stable_sign
            and (
                self._last_exchange_time is None
                or time - self._last_exchange_time >= self.min_exchange_interval
            )
        ):
            crossing_count = len([event for event in self._events if event.name.startswith("order_exchange")]) + 1
            self._events.append(
                CycleEvent(
                    time=time,
                    name=f"order_exchange_{crossing_count}",
                    crossing_count=crossing_count,
                    axial_separation=separation,
                )
            )
            self._stable_sign = sign
            self._last_exchange_time = time
            if crossing_count >= 2 and sign == self._initial_sign and self._cycle_time is None:
                radii_recurrent = True
                if self._initial_radii is not None and self.recurrence_radius_tolerance is not None:
                    if ring_1_radius is None or ring_2_radius is None:
                        radii_recurrent = False
                    else:
                        radii_recurrent = max(
                            abs(ring_1_radius - self._initial_radii[0]),
                            abs(ring_2_radius - self._initial_radii[1]),
                        ) <= self.recurrence_radius_tolerance
                if radii_recurrent:
                    self._cycle_time = time
                    self._events.append(
                        CycleEvent(
                            time=time,
                            name="complete_cycle_detected",
                            crossing_count=crossing_count,
                            axial_separation=separation,
                        )
                    )
                else:
                    self._events.append(
                        CycleEvent(
                            time=time,
                            name="cycle_recurrence_rejected",
                            crossing_count=crossing_count,
                            axial_separation=separation,
                        )
                    )

        return CycleStatus(
            crossing_count=len([event for event in self._events if event.name.startswith("order_exchange")]),
            completed=self._cycle_time is not None,
            cycle_time=self._cycle_time,
            axial_separation=separation,
        )
