"""验证完整时间推进不会破坏一个静止常量态。"""

from __future__ import annotations

import numpy as np

from isf_rings.config import SimulationConfig
from isf_rings.grid import PeriodicGrid
from isf_rings.isf_solver import ISFSolver


def test_constant_state_remains_constant_after_one_step() -> None:
    config = SimulationConfig.development()
    grid = PeriodicGrid(config.grid)
    solver = ISFSolver(grid, config)
    psi = np.zeros((2, *grid.shape), dtype=complex)
    psi[0] = 1.0

    advanced, diagnostics = solver.step(psi, time=0.0)

    assert np.max(np.abs(advanced - psi)) < 1.0e-12
    assert diagnostics.density_error < 1.0e-12
    assert diagnostics.projection.divergence_l2_after < 1.0e-12
