"""验证谱微分和周期泊松方程，不运行双涡环长时间实验。"""

from __future__ import annotations

import numpy as np

from isf_rings.config import GridSpec
from isf_rings.fft_ops import solve_periodic_poisson, spectral_derivative
from isf_rings.grid import PeriodicGrid


def test_spectral_derivative_of_sine() -> None:
    grid = PeriodicGrid(GridSpec((16, 8, 8), (2.0 * np.pi, 2.0 * np.pi, 2.0 * np.pi)))
    field = np.sin(grid.x)[:, np.newaxis, np.newaxis] * np.ones(grid.shape)
    derivative = spectral_derivative(field, grid.kx).real
    expected = np.cos(grid.x)[:, np.newaxis, np.newaxis] * np.ones(grid.shape)
    assert np.max(np.abs(derivative - expected)) < 1.0e-11


def test_periodic_poisson_recovers_cosine() -> None:
    grid = PeriodicGrid(GridSpec((16, 8, 8), (2.0 * np.pi, 2.0 * np.pi, 2.0 * np.pi)))
    expected = np.cos(grid.x)[:, np.newaxis, np.newaxis] * np.ones(grid.shape)
    rhs = -expected
    solved = solve_periodic_poisson(rhs, grid)
    assert np.max(np.abs(solved - expected)) < 1.0e-11
