"""执行 B 题任务 2 的网格与时间步收敛性实验。

短时示例：
    python scripts/run_convergence.py --kind both --t-end 4

完整周期验证较耗时，可显式使用：
    python scripts/run_convergence.py --kind both --until-cycle --max-steps 2400
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isf_rings.visualization import save_convergence_comparison


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ISF 双涡环网格与时间步收敛性实验")
    parser.add_argument("--kind", choices=("grid", "time", "both"), default="both")
    parser.add_argument(
        "--t-end", type=float, default=4.0, help="固定终止时间；短时收敛检查默认到 t=4。"
    )
    parser.add_argument(
        "--until-cycle", action="store_true", help="每个配置均运行到自动周期判据满足。"
    )
    parser.add_argument("--max-steps", type=int, default=2400)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "outputs" / "convergence"
    )
    return parser.parse_args()


def _last_finite(rows: list[dict[str, str]], field: str) -> float:
    for row in reversed(rows):
        raw = row.get(field, "")
        try:
            value = float(raw)
        except ValueError:
            continue
        if np.isfinite(value):
            return value
    return float("nan")


def _first_finite(rows: list[dict[str, str]], field: str) -> float:
    for row in rows:
        raw = row.get(field, "")
        try:
            value = float(raw)
        except ValueError:
            continue
        if np.isfinite(value):
            return value
    return float("nan")


def summarize_run(output: Path, label: str) -> dict[str, float | str]:
    """从主运行器的 CSV/JSON 提取适合横向比较的指标。"""

    with (output / "diagnostics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (output / "cycle_summary.json").open(encoding="utf-8") as stream:
        summary = json.load(stream)
    initial_energy = _first_finite(rows, "isf_total_energy")
    final_energy = _last_finite(rows, "isf_total_energy")
    energy_drift = (
        (final_energy - initial_energy) / initial_energy
        if np.isfinite(initial_energy) and initial_energy != 0.0 and np.isfinite(final_energy)
        else float("nan")
    )
    target_circulation = float(summary["target_circulation"])
    circulation = _last_finite(rows, "ring_1_circulation")
    circulation_error = (
        abs(abs(circulation) - target_circulation) / target_circulation
        if np.isfinite(circulation)
        else float("nan")
    )
    return {
        "label": label,
        "final_time": float(summary["final_time"]),
        "cycle_time": float(summary["cycle_time"]) if summary["cycle_time"] is not None else float("nan"),
        "order_exchange_count": int(summary["order_exchange_count"]),
        "energy_relative_change": energy_drift,
        "circulation_relative_error": circulation_error,
        "max_density_error": max(float(row["density_error"]) for row in rows),
        "max_projection_divergence": max(
            float(row["projection_divergence_after"]) for row in rows
        ),
    }


def run_case(
    label: str,
    shape: tuple[int, int, int],
    dt: float,
    arguments: argparse.Namespace,
) -> dict[str, float | str]:
    """调用主实验运行器，确保收敛性和正式实验共享同一数值代码。"""

    output = arguments.output / label
    if arguments.until_cycle:
        duration_arguments = ["--until-cycle", "--max-steps", str(arguments.max_steps)]
    else:
        steps = max(1, int(round(arguments.t_end / dt)))
        duration_arguments = ["--steps", str(steps)]
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_leapfrogging.py"),
        "--preset",
        "assignment",
        "--shape",
        *(str(item) for item in shape),
        "--dt",
        f"{dt:.17g}",
        "--energy-interval",
        "12",
        "--snapshot-interval",
        str(arguments.max_steps if arguments.until_cycle else max(1, int(round(arguments.t_end / dt)))),
        "--skip-isosurfaces",
        "--output",
        str(output),
        *duration_arguments,
    ]
    print("running:", " ".join(command))
    subprocess.run(command, check=True)
    return summarize_run(output, label)


def save_table(rows: list[dict[str, float | str]], output: Path) -> None:
    if not rows:
        return
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    arguments = parse_arguments()
    if arguments.t_end <= 0.0 or arguments.max_steps < 1:
        raise ValueError("--t-end 与 --max-steps 必须为正数。")
    arguments.output.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, float | str]] = []
    if arguments.kind in ("grid", "both"):
        # 三套网格保持题目物理尺度、hbar、涡环半径和涡核宽度不变。
        for label, shape in (
            ("grid_064x032x032", (64, 32, 32)),
            ("grid_128x064x064", (128, 64, 64)),
            ("grid_192x096x096", (192, 96, 96)),
        ):
            row = run_case(label, shape, 1.0 / 24.0, arguments)
            row["study"] = "grid"
            all_rows.append(row)
    if arguments.kind in ("time", "both"):
        # 固定正式网格，仅改变 dt；固定终止时间时比较的是同一物理时刻。
        for label, dt in (
            ("dt_1_12", 1.0 / 12.0),
            ("dt_1_24", 1.0 / 24.0),
            ("dt_1_48", 1.0 / 48.0),
        ):
            row = run_case(label, (128, 64, 64), dt, arguments)
            row["study"] = "time"
            all_rows.append(row)

    save_table(all_rows, arguments.output / "convergence.csv")
    for study in ("grid", "time"):
        rows = [row for row in all_rows if row["study"] == study]
        if not rows:
            continue
        save_convergence_comparison(
            [str(row["label"]) for row in rows],
            np.asarray([float(row["cycle_time"]) for row in rows]),
            np.asarray([float(row["energy_relative_change"]) for row in rows]),
            arguments.output / f"{study}_convergence.png",
        )
    print(f"output: {arguments.output}")


if __name__ == "__main__":
    main()
