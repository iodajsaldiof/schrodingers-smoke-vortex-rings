"""验证归一化约束和常量态的速度。"""

from __future__ import annotations

import numpy as np

from isf_rings.config import GridSpec
from isf_rings.grid import PeriodicGrid
from isf_rings.wavefunction import density, normalize_spinor, velocity_from_spinor


def test_normalize_spinor_sets_unit_density() -> None:
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(2, 8, 8, 8)) + 1j * rng.normal(size=(2, 8, 8, 8))
    normalized = normalize_spinor(raw)
    assert np.max(np.abs(density(normalized) - 1.0)) < 1.0e-12


def test_constant_spinor_has_zero_velocity() -> None:
    grid = PeriodicGrid(GridSpec((8, 8, 8), (1.0, 1.0, 1.0)))
    psi = np.zeros((2, *grid.shape), dtype=complex)
    psi[0] = 1.0
    velocity = velocity_from_spinor(psi, grid, hbar=0.1)
    assert np.max(np.abs(velocity)) < 1.0e-12
