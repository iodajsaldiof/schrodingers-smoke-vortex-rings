"""集中保存题目参数，避免把物理尺度散落在各个脚本中。"""

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
    """任务 2 所需的流体参数和双涡环几何参数。"""

    grid: GridSpec
    dt: float
    hbar: float
    ring_radius: float
    ring_separation: float
    core_radius: float
    output_interval: int = 12

    def __post_init__(self) -> None:
        if self.dt <= 0.0 or self.hbar <= 0.0:
            raise ValueError("dt 与 hbar 必须为正数。")
        if self.ring_radius <= 0.0 or self.ring_separation <= 0.0:
            raise ValueError("涡环半径和轴向间距必须为正数。")
        if self.core_radius <= 0.0:
            raise ValueError("涡核正则化半径必须为正数。")
        if self.output_interval < 1:
            raise ValueError("output_interval 至少为 1。")

    @classmethod
    def development(cls) -> "SimulationConfig":
        """较小网格：先验证算法流程，避免一开始就进行长时间大计算。"""

        return cls(
            grid=GridSpec(shape=(64, 32, 32), lengths=(10.0, 5.0, 5.0)),
            dt=1.0 / 24.0,
            hbar=0.1,
            ring_radius=1.0,
            ring_separation=0.6,
            core_radius=0.16,
        )

    @classmethod
    def assignment(cls) -> "SimulationConfig":
        """科研实践题目给出的正式网格和物理参数。"""

        return cls(
            grid=GridSpec(shape=(128, 64, 64), lengths=(10.0, 5.0, 5.0)),
            dt=1.0 / 24.0,
            hbar=0.1,
            ring_radius=1.0,
            ring_separation=0.6,
            core_radius=0.12,
        )
