"""运行 B 题任务 2 的双涡环交替穿越实验。

示例：
    python scripts/run_leapfrogging.py --preset assignment --until-cycle \
        --max-steps 2400 --output outputs/assignment_cycle

本脚本默认不与 Git 交互。计算产物位于 outputs/，便于人工检查后再决定
应当提交哪些源代码文件。
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

# 允许直接执行本脚本，而不要求使用者预先安装为 Python 包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isf_rings.config import GridSpec, SimulationConfig
from isf_rings.diagnostics import (
    LeapfroggingCycleDetector,
    RingObservation,
    RingTracker,
    CycleStatus,
    discrete_vorticity_magnitude,
    discrete_divergence_l2,
    isf_energy,
    linked_ring_circulation,
    vorticity_from_velocity,
    vorticity_magnitude,
)
from isf_rings.grid import PeriodicGrid
from isf_rings.isf_solver import ISFSolver
from isf_rings.visualization import (
    save_center_trajectories,
    save_constraint_diagnostics,
    save_energy_diagnostics,
    save_radius_trajectories,
    save_phase_vorticity_isosurface,
    save_vorticity_isosurface,
    save_vorticity_slice,
    save_wavefunction_core_isosurface,
)
from isf_rings.vortex_init import initialize_two_coaxial_rings
from isf_rings.wavefunction import max_density_error, velocity_from_spinor


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ISF 双涡环交替穿越实验")
    parser.add_argument(
        "--preset",
        choices=("development", "assignment", "paper-reference"),
        default="development",
        help="development 用于快速检查；assignment 使用题目给出的正式网格。",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="固定运行的总步数；不使用 --until-cycle 时默认为 development 的 48 步。",
    )
    parser.add_argument(
        "--initialization",
        choices=("paper_phase_disk", "analytic_zero_set"),
        help="覆盖预设的初值方案；正式复现实验默认使用论文作者的相位圆盘。",
    )
    parser.add_argument(
        "--until-cycle",
        action="store_true",
        help="直到检测到两次可靠换序（一个 leapfrogging 周期）后自动停止。",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2400,
        help="--until-cycle 的最大步数；Chern et al. 的示例量级约为 2000 步。",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        help="只用于收敛性实验的网格覆盖；物理长度与其他参数保持不变。",
    )
    parser.add_argument("--dt", type=float, help="只用于时间步收敛实验的时间步覆盖。")
    parser.add_argument(
        "--energy-interval",
        type=int,
        default=12,
        help="每隔多少步计算一次总能量、环量和独立散度诊断。",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=240,
        help="除换序事件外，每隔多少步保存一次关键时刻图像。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "leapfrogging",
        help="图片、CSV、JSON 输出目录；该目录被 .gitignore 排除。",
    )
    parser.add_argument(
        "--skip-isosurfaces",
        action="store_true",
        help="仅用于快速数值检查；正式结果不要启用此选项。",
    )
    parser.add_argument(
        "--core-level",
        type=float,
        default=0.25,
        help="波函数涡核等值面 |psi_2|^2 的固定 level。",
    )
    parser.add_argument(
        "--phase-vorticity-level",
        type=float,
        help="Algorithm 3 离散相位涡核等值面的固定 level；省略时取初始最大值的 10%%。",
    )
    parser.add_argument(
        "--vorticity-level",
        type=float,
        help="固定涡量等值面 level；省略时取初始最大涡量的 10%%。",
    )
    return parser.parse_args()


def save_rows(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    """把时间序列诊断量写成 CSV，保留 NaN 以显式标记未采样条目。"""

    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def save_json(data: dict[str, object], output_path: Path) -> None:
    """保存可机器读取的运行元数据或实验总结。"""

    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(_json_value(data), stream, indent=2, ensure_ascii=False, allow_nan=False)


def git_revision() -> str:
    """尽力记录代码版本；失败时不影响数值实验。"""

    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unavailable"
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def configure(arguments: argparse.Namespace) -> SimulationConfig:
    """从预设构造配置，并只在收敛脚本显式要求时覆盖网格或 dt。"""

    if arguments.preset == "development":
        config = SimulationConfig.development()
    elif arguments.preset == "assignment":
        config = SimulationConfig.assignment()
    else:
        config = SimulationConfig.paper_reference()
    if arguments.shape is not None:
        config = replace(
            config,
            grid=GridSpec(shape=tuple(arguments.shape), lengths=config.grid.lengths),
        )
    if arguments.dt is not None:
        config = replace(config, dt=arguments.dt)
    if arguments.initialization is not None:
        config = replace(config, initialization_mode=arguments.initialization)
    return config


def resolve_steps(arguments: argparse.Namespace) -> int:
    """统一固定步数和“运行至周期”的最大步数约定。"""

    if arguments.until_cycle:
        return arguments.max_steps
    if arguments.steps is not None:
        return arguments.steps
    return 48


def geometry_row(
    psi: np.ndarray,
    time: float,
    step: int,
    grid: PeriodicGrid,
    hbar: float,
    tracker: RingTracker,
    detector: LeapfroggingCycleDetector,
) -> tuple[dict[str, float | int], tuple[RingObservation, RingObservation] | None, CycleStatus | None, tuple[str, ...]]:
    """每步提取几何量；提取失败时保留 NaN 而不伪造一个穿越事件。"""

    events_before = len(detector.events)
    try:
        ring_1, ring_2 = tracker.update_from_wavefunction(psi, hbar)
        x_1_unwrapped, x_2_unwrapped = tracker.unwrapped_x
        cycle = detector.update(
            time,
            x_1_unwrapped,
            x_2_unwrapped,
            ring_1.radius,
            ring_2.radius,
        )
        new_events = tuple(event.name for event in detector.events[events_before:])
        return (
            {
                "step": step,
                "time": time,
                "ring_1_x": ring_1.x,
                "ring_2_x": ring_2.x,
                "ring_1_x_unwrapped": x_1_unwrapped,
                "ring_2_x_unwrapped": x_2_unwrapped,
                "ring_1_radius": ring_1.radius,
                "ring_2_radius": ring_2.radius,
                "ring_1_fit_spread": ring_1.fit_spread,
                "ring_2_fit_spread": ring_2.fit_spread,
                "ring_1_core_strength": ring_1.core_strength,
                "ring_2_core_strength": ring_2.core_strength,
                "relative_axial_separation": cycle.axial_separation,
                "order_exchange_count": cycle.crossing_count,
                "cycle_completed": int(cycle.completed),
            },
            (ring_1, ring_2),
            cycle,
            new_events,
        )
    except ValueError as error:
        print(f"warning: step={step} ring extraction unavailable: {error}")
        nan = float("nan")
        return (
            {
                "step": step,
                "time": time,
                "ring_1_x": nan,
                "ring_2_x": nan,
                "ring_1_x_unwrapped": nan,
                "ring_2_x_unwrapped": nan,
                "ring_1_radius": nan,
                "ring_2_radius": nan,
                "ring_1_fit_spread": nan,
                "ring_2_fit_spread": nan,
                "ring_1_core_strength": nan,
                "ring_2_core_strength": nan,
                "relative_axial_separation": nan,
                "order_exchange_count": len(detector.events),
                "cycle_completed": 0,
            },
            None,
            None,
            (),
        )


def add_physical_diagnostics(
    row: dict[str, float | int],
    psi: np.ndarray,
    grid: PeriodicGrid,
    config: SimulationConfig,
    rings: tuple[RingObservation, RingObservation] | None,
) -> None:
    """补充 Chern 能量、量子化环量和独立离散散度诊断。"""

    energy = isf_energy(psi, grid, config.hbar)
    row.update(
        {
            "kinetic_energy": energy.kinetic,
            "landau_lifshitz_energy": energy.landau_lifshitz,
            "isf_total_energy": energy.total,
            "spinor_gradient_energy": energy.spinor_total,
            "energy_decomposition_residual": energy.decomposition_residual,
            "edge_divergence_l2": discrete_divergence_l2(psi, grid, config.hbar),
        }
    )
    if rings is None:
        row["ring_1_circulation"] = float("nan")
        row["ring_2_circulation"] = float("nan")
        return
    loop_half_width = max(2.0 * config.core_radius, 3.0 * max(grid.dx, grid.dy))
    row["ring_1_circulation"] = linked_ring_circulation(
        psi, grid, config.hbar, rings[0], loop_half_width
    )
    row["ring_2_circulation"] = linked_ring_circulation(
        psi, grid, config.hbar, rings[1], loop_half_width
    )


def capture_snapshot(
    psi: np.ndarray,
    step: int,
    tag: str,
    output: Path,
    grid: PeriodicGrid,
    config: SimulationConfig,
    rings: tuple[RingObservation, RingObservation] | None,
    vorticity_level: float | None,
    phase_vorticity_level: float | None,
    core_level: float,
    skip_isosurfaces: bool,
) -> tuple[float | None, float | None]:
    """保存统一视角的连续涡量与相位/波函数涡核等值面。"""

    velocity = velocity_from_spinor(psi, grid, config.hbar)
    omega_magnitude = vorticity_magnitude(vorticity_from_velocity(velocity, grid))
    if vorticity_level is None:
        vorticity_level = 0.10 * float(np.max(omega_magnitude))
    phase_vorticity = discrete_vorticity_magnitude(psi, grid, config.hbar)
    if phase_vorticity_level is None:
        phase_vorticity_level = 0.10 * float(np.max(phase_vorticity))
    stem = f"snapshot_{step:05d}_{tag}"
    x_position = None if rings is None else 0.5 * (rings[0].x + rings[1].x)
    save_vorticity_slice(
        omega_magnitude,
        grid,
        output / f"{stem}_vorticity_slice.png",
        x_position=x_position,
        color_limit=vorticity_level,
    )
    if not skip_isosurfaces:
        try:
            save_vorticity_isosurface(
                omega_magnitude,
                grid,
                output / f"{stem}_vorticity_isosurface.png",
                level=vorticity_level,
            )
            if config.initialization_mode == "paper_phase_disk":
                save_phase_vorticity_isosurface(
                    phase_vorticity,
                    grid,
                    output / f"{stem}_phase_vorticity_isosurface.png",
                    level=phase_vorticity_level,
                )
            else:
                save_wavefunction_core_isosurface(
                    psi,
                    grid,
                    output / f"{stem}_core_isosurface.png",
                    level=core_level,
                )
        except ValueError as error:
            print(f"warning: step={step} skipped vorticity isosurface: {error}")
    return vorticity_level, phase_vorticity_level


def main() -> None:
    arguments = parse_arguments()
    steps = resolve_steps(arguments)
    if steps < 1 or arguments.energy_interval < 1 or arguments.snapshot_interval < 1:
        raise ValueError("步数、--energy-interval 和 --snapshot-interval 都必须为正数。")

    config = configure(arguments)
    grid = PeriodicGrid(config.grid)
    solver = ISFSolver(grid, config)
    psi, initial_projection = initialize_two_coaxial_rings(grid, config)
    arguments.output.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "purpose": "B题任务2：ISF 同轴双涡环交替穿越",
            "paper_basis": {
                "solver": "Chern et al. Algorithm 1--3",
                "energy": "Chern et al. Eq. (9), Fig. 12",
                "initialization": "paper_phase_disk follows the authors' AddCircle/example_leapfrog source; analytic_zero_set is a controlled comparison",
                "tracking": "task-specialized axisymmetric (x,r) tracker; phase-disk data uses Algorithm 3 discrete phase vorticity, while Weissmann and Pinkall gives the future general-filament route",
            },
            "config": asdict(config),
            "arguments": vars(arguments),
            "python": sys.version,
            "platform": platform.platform(),
            "git_revision": git_revision(),
        },
        arguments.output / "run_metadata.json",
    )

    print(f"preset={arguments.preset}, grid={grid.shape}, dt={config.dt:.8f}, steps={steps}")
    print(
        "initial projection divergence: "
        f"{initial_projection.divergence_l2_before:.3e} -> "
        f"{initial_projection.divergence_l2_after:.3e}"
    )

    tracker = RingTracker(
        grid,
        core_field=(
            "phase_vorticity"
            if config.initialization_mode == "paper_phase_disk"
            else "wavefunction"
        ),
    )
    detector = LeapfroggingCycleDetector(
        separation_hysteresis=max(2.0 * grid.dx, 0.12),
        recurrence_radius_tolerance=max(0.25, 3.0 * grid.dx),
        min_exchange_interval=1.0,
    )
    rows: list[dict[str, float | int]] = []
    saved_snapshot_steps: set[int] = set()
    time = 0.0
    vorticity_level: float | None = arguments.vorticity_level
    phase_vorticity_level: float | None = arguments.phase_vorticity_level

    row, rings, cycle, _ = geometry_row(
        psi, time, 0, grid, config.hbar, tracker, detector
    )
    row["density_error"] = max_density_error(psi)
    row["projection_divergence_before"] = initial_projection.divergence_l2_before
    row["projection_divergence_after"] = initial_projection.divergence_l2_after
    add_physical_diagnostics(row, psi, grid, config, rings)
    rows.append(row)
    vorticity_level, phase_vorticity_level = capture_snapshot(
        psi,
        0,
        "initial",
        arguments.output,
        grid,
        config,
        rings,
        vorticity_level,
        phase_vorticity_level,
        arguments.core_level,
        arguments.skip_isosurfaces,
    )
    saved_snapshot_steps.add(0)

    for step in range(1, steps + 1):
        psi, diagnostics = solver.step(psi, time)
        time = diagnostics.time
        row, rings, cycle, new_events = geometry_row(
            psi, time, step, grid, config.hbar, tracker, detector
        )
        row["density_error"] = diagnostics.density_error
        row["projection_divergence_before"] = diagnostics.projection.divergence_l2_before
        row["projection_divergence_after"] = diagnostics.projection.divergence_l2_after
        expensive = step % arguments.energy_interval == 0 or bool(new_events)
        if expensive:
            add_physical_diagnostics(row, psi, grid, config, rings)
        rows.append(row)

        snapshot_tag: str | None = None
        if new_events:
            snapshot_tag = new_events[-1]
        elif step % arguments.snapshot_interval == 0:
            snapshot_tag = "interval"
        if snapshot_tag is not None and step not in saved_snapshot_steps:
            vorticity_level, phase_vorticity_level = capture_snapshot(
                psi,
                step,
                snapshot_tag,
                arguments.output,
                grid,
                config,
                rings,
                vorticity_level,
                phase_vorticity_level,
                arguments.core_level,
                arguments.skip_isosurfaces,
            )
            saved_snapshot_steps.add(step)

        if expensive or snapshot_tag is not None:
            print(
                f"step={step:5d}, time={time:8.4f}, "
                f"density_error={diagnostics.density_error:.3e}, "
                f"div={diagnostics.projection.divergence_l2_after:.3e}, "
                f"exchanges={0 if cycle is None else cycle.crossing_count}"
            )
        if arguments.until_cycle and cycle is not None and cycle.completed:
            print(f"complete leapfrogging cycle detected at time={cycle.cycle_time:.6f}")
            break

    final_step = int(rows[-1]["step"])
    final_cycle = cycle
    if final_step not in saved_snapshot_steps:
        _, final_rings, final_cycle, _ = geometry_row(
            psi, time, final_step, grid, config.hbar, tracker, detector
        )
        vorticity_level, phase_vorticity_level = capture_snapshot(
            psi,
            final_step,
            "final",
            arguments.output,
            grid,
            config,
            final_rings,
            vorticity_level,
            phase_vorticity_level,
            arguments.core_level,
            arguments.skip_isosurfaces,
        )

    save_rows(rows, arguments.output / "diagnostics.csv")
    times = np.asarray([float(row["time"]) for row in rows])
    ring_1_x = np.asarray([float(row["ring_1_x_unwrapped"]) for row in rows])
    ring_2_x = np.asarray([float(row["ring_2_x_unwrapped"]) for row in rows])
    ring_1_radius = np.asarray([float(row["ring_1_radius"]) for row in rows])
    ring_2_radius = np.asarray([float(row["ring_2_radius"]) for row in rows])
    events = detector.events
    exchange_times = [event.time for event in events if event.name.startswith("order_exchange")]
    save_center_trajectories(
        times, ring_1_x, ring_2_x, arguments.output / "ring_centers.png", exchange_times
    )
    save_radius_trajectories(
        times, ring_1_radius, ring_2_radius, arguments.output / "ring_radii.png"
    )
    kinetic = np.asarray([float(row.get("kinetic_energy", np.nan)) for row in rows])
    landau_lifshitz = np.asarray(
        [float(row.get("landau_lifshitz_energy", np.nan)) for row in rows]
    )
    total = np.asarray([float(row.get("isf_total_energy", np.nan)) for row in rows])
    save_energy_diagnostics(times, kinetic, landau_lifshitz, total, arguments.output / "energy.png")
    density_error = np.asarray([float(row["density_error"]) for row in rows])
    edge_divergence = np.asarray(
        [float(row.get("edge_divergence_l2", np.nan)) for row in rows]
    )
    finite_divergence = np.isfinite(edge_divergence)
    if np.any(finite_divergence):
        edge_divergence = np.interp(times, times[finite_divergence], edge_divergence[finite_divergence])
    else:
        edge_divergence = np.asarray(
            [float(row["projection_divergence_after"]) for row in rows]
        )
    save_constraint_diagnostics(
        times, density_error, edge_divergence, arguments.output / "constraints.png"
    )
    cycle_status = final_cycle if final_cycle is not None else detector.update(
        time,
        ring_1_x[-1],
        ring_2_x[-1],
        ring_1_radius[-1],
        ring_2_radius[-1],
    )
    save_json(
        {
            "completed_steps": final_step,
            "final_time": time,
            "complete_cycle_detected": cycle_status.completed,
            "cycle_time": cycle_status.cycle_time,
            "order_exchange_count": cycle_status.crossing_count,
            "events": [asdict(event) for event in detector.events],
            "target_circulation": 2.0 * np.pi * config.hbar,
            "vorticity_isosurface_level": vorticity_level,
            "phase_vorticity_isosurface_level": phase_vorticity_level,
        },
        arguments.output / "cycle_summary.json",
    )
    print(f"output: {arguments.output}")


if __name__ == "__main__":
    main()
