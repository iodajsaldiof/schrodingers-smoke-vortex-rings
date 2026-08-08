"""验证双涡环初值的形状、归一化和初始投影。"""

from __future__ import annotations

import numpy as np

from isf_rings.config import SimulationConfig
from isf_rings.diagnostics import extract_two_coaxial_rings_from_wavefunction
from isf_rings.grid import PeriodicGrid
from isf_rings.vortex_init import initialize_two_coaxial_rings
from isf_rings.wavefunction import density


def test_two_ring_initial_state_is_normalized() -> None:
    config = SimulationConfig.development()
    grid = PeriodicGrid(config.grid)
    psi, diagnostics = initialize_two_coaxial_rings(grid, config)

    assert psi.shape == (2, *grid.shape)
    assert np.max(np.abs(density(psi) - 1.0)) < 1.0e-12
    assert diagnostics.divergence_l2_after <= diagnostics.divergence_l2_before

    ring_1, ring_2 = extract_two_coaxial_rings_from_wavefunction(psi, grid)
    extracted_x = sorted((ring_1.x, ring_2.x))
    expected_x = sorted((4.7, 5.3))
    assert np.allclose(extracted_x, expected_x, atol=1.5 * grid.dx)
