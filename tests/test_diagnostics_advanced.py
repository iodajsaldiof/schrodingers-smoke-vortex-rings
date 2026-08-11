"""针对任务 2 新增诊断器的单元测试。"""

from __future__ import annotations

import numpy as np

from isf_rings.config import SimulationConfig
from isf_rings.diagnostics import (
    LeapfroggingCycleDetector,
    RingObservation,
    RingTracker,
    isf_energy,
    linked_ring_circulation,
)
from isf_rings.grid import PeriodicGrid
from isf_rings.vortex_init import initialize_two_coaxial_rings


def test_constant_spinor_has_zero_isf_energy() -> None:
    config = SimulationConfig.development()
    grid = PeriodicGrid(config.grid)
    psi = np.zeros((2, *grid.shape), dtype=complex)
    psi[0] = 1.0
    energy = isf_energy(psi, grid, config.hbar)
    assert energy.kinetic < 1.0e-14
    assert energy.landau_lifshitz < 1.0e-14
    assert energy.spinor_total < 1.0e-14


def test_initial_ring_has_quantized_linked_circulation() -> None:
    config = SimulationConfig.development()
    grid = PeriodicGrid(config.grid)
    psi, _ = initialize_two_coaxial_rings(grid, config)
    observation = RingObservation(
        x=0.5 * grid.lengths[0] - 0.5 * config.ring_separation,
        y=0.5 * grid.lengths[1],
        z=0.5 * grid.lengths[2],
        radius=config.ring_radius,
        enstrophy=1.0,
    )
    circulation = linked_ring_circulation(
        psi,
        grid,
        config.hbar,
        observation,
        loop_half_width=max(2.0 * config.core_radius, 3.0 * grid.dx),
    )
    expected = 2.0 * np.pi * config.hbar
    assert np.isclose(abs(circulation), expected, rtol=0.15)


def test_tracker_unwraps_periodic_axis() -> None:
    config = SimulationConfig.development()
    grid = PeriodicGrid(config.grid)
    tracker = RingTracker(grid)
    tracker.update(
        (
            RingObservation(1.0, 2.5, 2.5, 0.9, 1.0),
            RingObservation(9.8, 2.5, 2.5, 1.1, 1.0),
        )
    )
    tracker.update(
        (
            RingObservation(1.1, 2.5, 2.5, 0.9, 1.0),
            RingObservation(0.1, 2.5, 2.5, 1.1, 1.0),
        )
    )
    assert np.isclose(tracker.unwrapped_x[0], 1.1)
    assert np.isclose(tracker.unwrapped_x[1], 10.1)


def test_cycle_detector_requires_two_hysteretic_exchanges() -> None:
    detector = LeapfroggingCycleDetector(separation_hysteresis=0.1)
    initial = detector.update(0.0, 0.0, 0.6)
    first_exchange = detector.update(1.0, 0.6, 0.0)
    completed = detector.update(2.0, 0.0, 0.6)
    assert not initial.completed
    assert first_exchange.crossing_count == 1
    assert completed.completed
    assert completed.cycle_time == 2.0


def test_cycle_detector_rejects_two_crossings_without_radius_recurrence() -> None:
    detector = LeapfroggingCycleDetector(
        separation_hysteresis=0.1,
        recurrence_radius_tolerance=0.1,
    )
    detector.update(0.0, 0.0, 0.6, 1.0, 1.0)
    detector.update(1.0, 0.6, 0.0, 1.4, 0.7)
    rejected = detector.update(2.0, 0.0, 0.6, 1.4, 0.7)
    assert not rejected.completed
    assert any(event.name == "cycle_recurrence_rejected" for event in detector.events)
