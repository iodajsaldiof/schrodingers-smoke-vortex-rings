"""论文算法 1 的完整时间推进器。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimulationConfig
from .fft_ops import free_schrodinger_step
from .grid import PeriodicGrid
from .projection import ProjectionDiagnostics, pressure_project
from .wavefunction import max_density_error, normalize_spinor


@dataclass(frozen=True)
class StepDiagnostics:
    """记录每一步的基本数值健康指标。"""

    time: float
    density_error: float
    projection: ProjectionDiagnostics


class ISFSolver:
    """依次执行自由演化、归一化和压力投影。"""

    def __init__(self, grid: PeriodicGrid, config: SimulationConfig) -> None:
        if grid.spec != config.grid:
            raise ValueError("grid 必须由 config.grid 创建，避免参数不一致。")
        self.grid = grid
        self.config = config

    def step(
        self, psi: np.ndarray, time: float
    ) -> tuple[np.ndarray, StepDiagnostics]:
        """推进一个 dt；顺序与论文算法 1 保持一致。"""

        evolved = free_schrodinger_step(
            psi, self.grid, self.config.hbar, self.config.dt
        )
        normalized = normalize_spinor(evolved)
        projected, projection = pressure_project(normalized, self.grid, self.config.hbar)
        diagnostics = StepDiagnostics(
            time=time + self.config.dt,
            density_error=max_density_error(projected),
            projection=projection,
        )
        return projected, diagnostics
