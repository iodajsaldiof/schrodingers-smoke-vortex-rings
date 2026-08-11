"""集中保存 B 题任务 2 的网格、物理参数和可解析涡核参数。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GridSpec:
    """周期长方体网格的尺寸和物理长度。"""

    shape: tuple[int, int, int]
    lengths: tuple[float, float, float]

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or len(self.lengths) != 3:
            raise ValueError("shape 和 lengths 都必须包含 x、y、z 三个方向。")
        if any(count < 4 for count in self.shape):
            raise ValueError("每个方向至少需要 4 个网格点。")
        if any(length <= 0.0 for length in self.lengths):
            raise ValueError("计算域长度必须为正数。")

    @property
    def spacing(self) -> tuple[float, float, float]:
        """返回三个方向的均匀网格间距。"""

        return tuple(length / count for length, count in zip(self.lengths, self.shape))


@dataclass(frozen=True)
class SimulationConfig:
    """任务 2 的 ISF 参数与双涡环几何。

    ``core_radius`` 是解析零集旋量 ``(phi, epsilon)`` 的有限网格正则化宽度，
    需要至少由约 1--2 个网格间距解析。Chern et al. 的相位盘初始化中使用
    ``epsilon=0.01``；本实现采用等价的解析复零集，并将其宽度显式纳入网格
    收敛实验，避免把未解析的点状核误当作可信结果。
    """

    grid: GridSpec
    dt: float
    hbar: float
    ring_radius: float
    ring_separation: float
    core_radius: float
    second_ring_radius: float | None = None
    initialization_mode: str = "paper_phase_disk"
    background_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    output_interval: int = 12

    def __post_init__(self) -> None:
        if self.dt <= 0.0 or self.hbar <= 0.0:
            raise ValueError("dt 与 hbar 必须为正数。")
        if self.ring_radius <= 0.0 or self.ring_separation < 0.0:
            raise ValueError("涡环半径必须为正数，轴向间距不能为负数。")
        if self.core_radius <= 0.0:
            raise ValueError("涡核正则化半径必须为正数。")
        if self.second_ring_radius is not None and self.second_ring_radius <= 0.0:
            raise ValueError("第二个涡环半径必须为正数。")
        if self.initialization_mode not in ("paper_phase_disk", "analytic_zero_set"):
            raise ValueError(
                "initialization_mode 只能是 paper_phase_disk 或 analytic_zero_set。"
            )
        if len(self.background_velocity) != 3:
            raise ValueError("background_velocity 必须包含三个速度分量。")
        if self.output_interval < 1:
            raise ValueError("output_interval 至少为 1。")

    @property
    def ring_radii(self) -> tuple[float, float]:
        """返回两个涡环半径；未单独给定时二者相同。"""

        return (
            self.ring_radius,
            self.ring_radius
            if self.second_ring_radius is None
            else self.second_ring_radius,
        )

    @classmethod
    def development(cls) -> "SimulationConfig":
        """较小网格：用于单元测试、脚本接口和诊断流程检查。"""

        return cls(
            grid=GridSpec(shape=(64, 32, 32), lengths=(10.0, 5.0, 5.0)),
            dt=1.0 / 24.0,
            hbar=0.1,
            ring_radius=1.0,
            ring_separation=0.6,
            core_radius=0.16,
            initialization_mode="paper_phase_disk",
        )

    @classmethod
    def assignment(cls) -> "SimulationConfig":
        """科研实践题目要求的正式网格与建议双环几何。"""

        return cls(
            grid=GridSpec(shape=(128, 64, 64), lengths=(10.0, 5.0, 5.0)),
            dt=1.0 / 24.0,
            hbar=0.1,
            ring_radius=1.0,
            ring_separation=0.6,
            core_radius=0.12,
            initialization_mode="paper_phase_disk",
        )

    @classmethod
    def paper_reference(cls) -> "SimulationConfig":
        """逐项对应 Chern 作者 ``example_leapfrog.m`` 的参考复现实例。

        该设置与 B 题的网格、盒子、``dt`` 和 ``hbar`` 相同，但采用作者示例
        的同心异半径环及背景速度。它作为可复现基准；B 题建议的 ``R=1,d=0.6``
        则由 ``assignment`` 预设单独给出。
        """

        return cls(
            grid=GridSpec(shape=(128, 64, 64), lengths=(10.0, 5.0, 5.0)),
            dt=1.0 / 24.0,
            hbar=0.1,
            ring_radius=1.5,
            second_ring_radius=0.9,
            ring_separation=0.0,
            core_radius=0.12,
            initialization_mode="paper_phase_disk",
            background_velocity=(-0.2, 0.0, 0.0),
        )
