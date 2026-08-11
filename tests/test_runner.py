"""主实验脚本的短程集成测试，覆盖 CSV、JSON 与自动快照流程。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_short_experiment_writes_reproducible_outputs(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output = tmp_path / "short_run"
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "run_leapfrogging.py"),
            "--preset",
            "development",
            "--steps",
            "3",
            "--energy-interval",
            "1",
            "--snapshot-interval",
            "3",
            "--skip-isosurfaces",
            "--output",
            str(output),
        ],
        check=True,
    )
    summary = json.loads((output / "cycle_summary.json").read_text(encoding="utf-8"))
    assert (output / "diagnostics.csv").is_file()
    assert (output / "run_metadata.json").is_file()
    assert summary["completed_steps"] == 3
