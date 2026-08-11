"""验证论文相位盘双涡环初值的归一化、投影与离散涡量位置。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from isf_rings.config import SimulationConfig
from isf_rings.diagnostics import extract_two_coaxial_rings_from_wavefunction
from isf_rings.grid import PeriodicGrid
from isf_rings.vortex_init import initialize_two_coaxial_rings
from isf_rings.wavefunction import density


def test_paper_phase_disk_initial_state_is_normalized_with_constant_guard() -> None:
    config = SimulationConfig.development()
    grid = PeriodicGrid(config.grid)
    psi, diagnostics = initialize_two_coaxial_rings(grid, config)

    assert psi.shape == (2, *grid.shape)
    assert np.max(np.abs(density(psi) - 1.0)) < 1.0e-12
    assert diagnostics.divergence_l2_after <= diagnostics.divergence_l2_before

    # AddCircle 只改变 psi_1 的相位；归一化和压力投影都是纯相位操作，
    # 因而作为防奇点保护的 psi_2 模应仍是统一常数。
    assert np.std(np.abs(psi[1])) < 1.0e-12


def test_analytic_zero_set_remains_available_as_control_initialization() -> None:
    config = replace(SimulationConfig.development(), initialization_mode="analytic_zero_set")
    grid = PeriodicGrid(config.grid)
    psi, _ = initialize_two_coaxial_rings(grid, config)
    ring_1, ring_2 = extract_two_coaxial_rings_from_wavefunction(psi, grid, config.hbar)
    extracted_x = sorted((ring_1.x, ring_2.x))
    expected_x = sorted((4.7, 5.3))
    assert np.allclose(extracted_x, expected_x, atol=2.0 * grid.dx)
