"""验证相位压力投影能消除一个平滑势流的离散散度。"""

from __future__ import annotations

import numpy as np

from isf_rings.config import GridSpec
from isf_rings.grid import PeriodicGrid
from isf_rings.projection import pressure_project


def test_pressure_projection_reduces_phase_divergence() -> None:
    length = 2.0 * np.pi
    grid = PeriodicGrid(GridSpec((24, 12, 12), (length, length, length)))
    phase = 0.2 * np.sin(grid.x)[:, np.newaxis, np.newaxis] * np.ones(grid.shape)
    psi = np.zeros((2, *grid.shape), dtype=complex)
    psi[0] = np.exp(1j * phase)

    projected, diagnostics = pressure_project(psi, grid, hbar=0.1)

    assert diagnostics.divergence_l2_after < diagnostics.divergence_l2_before * 1.0e-10
    assert np.max(np.abs(projected[0] - 1.0)) < 1.0e-8
