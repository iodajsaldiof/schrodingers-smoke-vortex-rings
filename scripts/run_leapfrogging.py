"""手动运行双涡环交替穿越实验。

本文件不会在导入时自动启动模拟。请在仓库根目录显式执行：
python scripts/run_leapfrogging.py --preset development --steps 48
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# 允许直接执行本脚本，而不要求使用者预先安装为 Python 包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isf_rings.config import SimulationConfig
from isf_rings.diagnostics import (
    RingTracker,
    extract_two_coaxial_rings_from_wavefunction,
    kinetic_energy,
    vorticity_from_velocity,
    vorticity_magnitude,
)
from isf_rings.grid import PeriodicGrid
from isf_rings.isf_solver import ISFSolver
from isf_rings.visualization import (
    save_center_trajectories,
    save_vorticity_isosurface,
    save_vorticity_slice,
)
from isf_rings.vortex_init import initialize_two_coaxial_rings
from isf_rings.wavefunction import velocity_from_spinor


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ISF 双涡环交替穿越实验")
    parser.add_argument(
        "--preset",
        choices=("development", "assignment"),
        default="development",
        help="development 为小网格检查；assignment 为题目正式网格。",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=48,
        help="总时间步数。完整周期需要在验证后设置更大的值。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "leapfrogging",
        help="图片与 CSV 的输出目录；该目录不会被 Git 跟踪。",
    )
    parser.add_argument(
        "--isosurface",
        action="store_true",
        help="额外保存初末时刻的三维涡量等值面图。",
    )
    return parser.parse_args()


def save_rows(rows: list[dict[str, float]], output_path: Path) -> None:
    """把每个时间步的诊断量写成 CSV，供后续绘图或报告使用。"""

    if not rows:
        return
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def observe(
    psi: np.ndarray,
    time: float,
    grid: PeriodicGrid,
    config: SimulationConfig,
    tracker: RingTracker,
) -> tuple[dict[str, float], np.ndarray]:
    """由当前波函数计算速度、涡量和两个涡环的几何观测量。"""

    velocity = velocity_from_spinor(psi, grid, config.hbar)
    vorticity = vorticity_from_velocity(velocity, grid)
    try:
        ring_1, ring_2 = tracker.update(
            extract_two_coaxial_rings_from_wavefunction(psi, grid)
        )
        row = {
            "time": time,
            "ring_1_x": ring_1.x,
            "ring_1_radius": ring_1.radius,
            "ring_2_x": ring_2.x,
            "ring_2_radius": ring_2.radius,
            "kinetic_energy": kinetic_energy(velocity),
        }
    except ValueError:
        # 两环极近时，轻量诊断器可能无法分离它们。保留 NaN 而不是伪造数据。
        row = {
            "time": time,
            "ring_1_x": float("nan"),
            "ring_1_radius": float("nan"),
            "ring_2_x": float("nan"),
            "ring_2_radius": float("nan"),
            "kinetic_energy": kinetic_energy(velocity),
        }
    return row, vorticity_magnitude(vorticity)


def main() -> None:
    arguments = parse_arguments()
    if arguments.steps < 1:
        raise ValueError("--steps 至少为 1。")

    config = (
        SimulationConfig.development()
        if arguments.preset == "development"
        else SimulationConfig.assignment()
    )
    grid = PeriodicGrid(config.grid)
    solver = ISFSolver(grid, config)
    psi, initial_projection = initialize_two_coaxial_rings(grid, config)
    arguments.output.mkdir(parents=True, exist_ok=True)

    print(f"preset={arguments.preset}, grid={grid.shape}, steps={arguments.steps}")
    print(
        "initial projection divergence: "
        f"{initial_projection.divergence_l2_before:.3e} -> "
        f"{initial_projection.divergence_l2_after:.3e}"
    )

    tracker = RingTracker(grid)
    rows: list[dict[str, float]] = []
    time = 0.0
    row, omega_magnitude = observe(psi, time, grid, config, tracker)
    rows.append(row)
    save_vorticity_slice(omega_magnitude, grid, arguments.output / "vorticity_00000.png")
    if arguments.isosurface:
        save_vorticity_isosurface(omega_magnitude, grid, arguments.output / "isosurface_00000.png")

    for step in range(1, arguments.steps + 1):
        psi, diagnostics = solver.step(psi, time)
        time = diagnostics.time
        row, omega_magnitude = observe(psi, time, grid, config, tracker)
        row["density_error"] = diagnostics.density_error
        row["projection_divergence_before"] = diagnostics.projection.divergence_l2_before
        row["projection_divergence_after"] = diagnostics.projection.divergence_l2_after
        rows.append(row)

        if step % config.output_interval == 0 or step == arguments.steps:
            save_vorticity_slice(
                omega_magnitude, grid, arguments.output / f"vorticity_{step:05d}.png"
            )
            print(
                f"step={step:5d}, time={time:8.4f}, "
                f"density_error={diagnostics.density_error:.3e}, "
                f"div={diagnostics.projection.divergence_l2_after:.3e}"
            )

    if arguments.isosurface:
        save_vorticity_isosurface(
            omega_magnitude, grid, arguments.output / "isosurface_final.png"
        )

    save_rows(rows, arguments.output / "diagnostics.csv")
    time_values = np.array([row["time"] for row in rows])
    x1_values = np.array([row["ring_1_x"] for row in rows])
    x2_values = np.array([row["ring_2_x"] for row in rows])
    save_center_trajectories(
        time_values, x1_values, x2_values, arguments.output / "ring_centers.png"
    )
    print(f"output: {arguments.output}")


if __name__ == "__main__":
    main()
