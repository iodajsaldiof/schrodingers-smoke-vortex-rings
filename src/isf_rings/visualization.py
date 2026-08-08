"""任务 2 所需的轨迹曲线、涡量切片和可选三维等值面输出。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .grid import PeriodicGrid


def _pyplot():
    """在真正需要画图时才加载 matplotlib，保持数值核心可独立运行。"""

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "保存图像需要 matplotlib。请先在仓库根目录执行 python -m pip install -e ."
        ) from error
    return plt


def save_center_trajectories(
    times: np.ndarray, x1: np.ndarray, x2: np.ndarray, output_path: Path
) -> None:
    """保存两个涡环环心的轴向位置-时间曲线。"""

    plt = _pyplot()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    axis.plot(times, x1, label="ring 1", linewidth=2.0)
    axis.plot(times, x2, label="ring 2", linewidth=2.0)
    axis.set_xlabel("time")
    axis.set_ylabel("axial center x")
    axis.set_title("Coaxial vortex-ring centers")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_vorticity_slice(
    omega_magnitude: np.ndarray, grid: PeriodicGrid, output_path: Path
) -> None:
    """保存 y-z 中心截面的涡量模，用于快速检查涡环是否存在。"""

    plt = _pyplot()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_index = grid.shape[0] // 2
    figure, axis = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    image = axis.imshow(
        omega_magnitude[x_index].T,
        origin="lower",
        extent=(0.0, grid.lengths[1], 0.0, grid.lengths[2]),
        aspect="equal",
        cmap="magma",
    )
    axis.set_xlabel("y")
    axis.set_ylabel("z")
    axis.set_title("Vorticity magnitude at middle x slice")
    figure.colorbar(image, ax=axis, label="|omega|")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_vorticity_isosurface(
    omega_magnitude: np.ndarray,
    grid: PeriodicGrid,
    output_path: Path,
    level: float | None = None,
) -> None:
    """保存涡量模等值面；只在需要三维展示时调用。"""

    plt = _pyplot()
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes

    output_path.parent.mkdir(parents=True, exist_ok=True)
    chosen_level = level if level is not None else float(np.percentile(omega_magnitude, 99.0))
    if not 0.0 < chosen_level < float(np.max(omega_magnitude)):
        raise ValueError("等值面 level 必须位于数据的有效范围内。")

    vertices, faces, _, _ = marching_cubes(
        omega_magnitude, level=chosen_level, spacing=(grid.dx, grid.dy, grid.dz)
    )
    figure = plt.figure(figsize=(7.0, 5.5), constrained_layout=True)
    axis = figure.add_subplot(projection="3d")
    surface = Poly3DCollection(vertices[faces], alpha=0.75, linewidth=0.0)
    surface.set_facecolor("tab:cyan")
    axis.add_collection3d(surface)
    axis.set(xlim=(0.0, grid.lengths[0]), ylim=(0.0, grid.lengths[1]), zlim=(0.0, grid.lengths[2]))
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.set_title("Vorticity isosurface")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
