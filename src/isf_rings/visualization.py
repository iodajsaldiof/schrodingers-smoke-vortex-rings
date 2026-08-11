"""任务 2 的轨迹、能量与可复核三维涡结构可视化。"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

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
    times: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    output_path: Path,
    event_times: Sequence[float] = (),
) -> None:
    """保存两个已编号涡环的连续轴向位置，并标出可靠换序时刻。"""

    plt = _pyplot()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
    axis.plot(times, x1, label="ring 1", linewidth=2.0, color="#0072b2")
    axis.plot(times, x2, label="ring 2", linewidth=2.0, color="#d55e00")
    for index, event_time in enumerate(event_times):
        axis.axvline(
            event_time,
            color="#4d4d4d",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label="order exchange" if index == 0 else None,
        )
    axis.set_xlabel("time")
    axis.set_ylabel("unwrapped axial center x")
    axis.set_title("Coaxial vortex-ring centers")
    axis.grid(alpha=0.28)
    axis.legend()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_radius_trajectories(
    times: np.ndarray, radius_1: np.ndarray, radius_2: np.ndarray, output_path: Path
) -> None:
    """保存两环半径演化，用于证明穿越时的收缩-膨胀交替。"""

    plt = _pyplot()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.4, 4.1), constrained_layout=True)
    axis.plot(times, radius_1, label="ring 1", linewidth=2.0, color="#0072b2")
    axis.plot(times, radius_2, label="ring 2", linewidth=2.0, color="#d55e00")
    axis.set_xlabel("time")
    axis.set_ylabel("fitted ring radius")
    axis.set_title("Alternating vortex-ring radii")
    axis.grid(alpha=0.28)
    axis.legend()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_energy_diagnostics(
    times: np.ndarray,
    kinetic: np.ndarray,
    landau_lifshitz: np.ndarray,
    total: np.ndarray,
    output_path: Path,
) -> None:
    """绘制 Chern et al. Fig. 12 对应的 ISF 能量分解。"""

    plt = _pyplot()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid = np.isfinite(kinetic) & np.isfinite(landau_lifshitz) & np.isfinite(total)
    if not np.any(valid):
        return
    figure, axis = plt.subplots(figsize=(7.4, 4.1), constrained_layout=True)
    axis.plot(times[valid], kinetic[valid], label="kinetic", color="#0072b2", linewidth=1.9)
    axis.plot(
        times[valid],
        landau_lifshitz[valid],
        label="Landau-Lifshitz",
        color="#009e73",
        linewidth=1.9,
    )
    axis.plot(times[valid], total[valid], label="ISF total", color="#cc79a7", linewidth=2.1)
    axis.set_xlabel("time")
    axis.set_ylabel("volume-averaged energy")
    axis.set_title("ISF energy diagnostics")
    axis.grid(alpha=0.28)
    axis.legend()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_constraint_diagnostics(
    times: np.ndarray,
    density_error: np.ndarray,
    divergence_l2: np.ndarray,
    output_path: Path,
) -> None:
    """以对数坐标保存密度约束与离散无散约束残差。"""

    plt = _pyplot()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.4, 4.1), constrained_layout=True)
    axis.semilogy(times, np.maximum(density_error, 1.0e-18), label="max |rho - 1|", color="#0072b2")
    axis.semilogy(
        times,
        np.maximum(divergence_l2, 1.0e-18),
        label="edge divergence L2",
        color="#d55e00",
    )
    axis.set_xlabel("time")
    axis.set_ylabel("constraint residual")
    axis.set_title("Normalization and incompressibility checks")
    axis.grid(alpha=0.28, which="both")
    axis.legend()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_vorticity_slice(
    omega_magnitude: np.ndarray,
    grid: PeriodicGrid,
    output_path: Path,
    x_position: float | None = None,
    color_limit: float | None = None,
) -> None:
    """保存指定轴向位置的 y-z 涡量截面；可使用统一色标比较时刻。"""

    plt = _pyplot()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if x_position is None:
        x_index = grid.shape[0] // 2
    else:
        x_index = int(np.floor(x_position / grid.dx + 0.5)) % grid.shape[0]
    figure, axis = plt.subplots(figsize=(5.4, 4.4), constrained_layout=True)
    image = axis.imshow(
        omega_magnitude[x_index].T,
        origin="lower",
        extent=(0.0, grid.lengths[1], 0.0, grid.lengths[2]),
        aspect="equal",
        cmap="magma",
        vmin=0.0,
        vmax=color_limit,
    )
    axis.set_xlabel("y")
    axis.set_ylabel("z")
    axis.set_title(f"Vorticity magnitude at x = {grid.x[x_index]:.3f}")
    figure.colorbar(image, ax=axis, label="|omega|")
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _save_scalar_isosurface(
    scalar: np.ndarray,
    grid: PeriodicGrid,
    output_path: Path,
    level: float,
    title: str,
    color: str,
) -> None:
    """以固定视角和固定等值保存标量场的三维等值面。"""

    plt = _pyplot()
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes

    output_path.parent.mkdir(parents=True, exist_ok=True)
    maximum = float(np.max(scalar))
    if not 0.0 < level < maximum:
        raise ValueError(
            f"等值面 level={level:.4g} 不在当前数据的有效范围 (0, {maximum:.4g}) 内。"
        )
    vertices, faces, _, _ = marching_cubes(
        scalar, level=level, spacing=(grid.dx, grid.dy, grid.dz)
    )
    figure = plt.figure(figsize=(7.4, 5.8), constrained_layout=True)
    axis = figure.add_subplot(projection="3d")
    surface = Poly3DCollection(vertices[faces], alpha=0.76, linewidth=0.0)
    surface.set_facecolor(color)
    axis.add_collection3d(surface)
    axis.set(
        xlim=(0.0, grid.lengths[0]),
        ylim=(0.0, grid.lengths[1]),
        zlim=(0.0, grid.lengths[2]),
    )
    axis.set_box_aspect(grid.lengths)
    axis.view_init(elev=23.0, azim=-56.0)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.set_title(title)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_vorticity_isosurface(
    omega_magnitude: np.ndarray,
    grid: PeriodicGrid,
    output_path: Path,
    level: float,
) -> None:
    """保存 ``|omega|`` 等值面；调用方应在整个实验中传入同一 level。"""

    _save_scalar_isosurface(
        omega_magnitude,
        grid,
        output_path,
        level,
        "Vorticity isosurface",
        "#0072b2",
    )


def save_phase_vorticity_isosurface(
    phase_vorticity_magnitude: np.ndarray,
    grid: PeriodicGrid,
    output_path: Path,
    level: float,
) -> None:
    """保存由 Algorithm 3 面元相位环量得到的离散涡核等值面。"""

    _save_scalar_isosurface(
        phase_vorticity_magnitude,
        grid,
        output_path,
        level,
        "Discrete phase-circulation vortex core",
        "#d55e00",
    )


def save_wavefunction_core_isosurface(
    psi: np.ndarray, grid: PeriodicGrid, output_path: Path, level: float = 0.5
) -> None:
    """保存 ``|psi_2|^2 = 0.5`` 的涡核等值面。

    对单位旋量这等价于 Chern et al. §4.5 中 ``s_x = 0`` 的涡管可视化，
    使用固定 level 使不同关键时刻可直接比较。
    """

    if psi.ndim != 4 or psi.shape[0] != 2:
        raise ValueError("psi 的形状必须为 (2, Nx, Ny, Nz)。")
    core_indicator = np.abs(psi[1]) ** 2
    _save_scalar_isosurface(
        core_indicator,
        grid,
        output_path,
        level,
        "Wavefunction vortex-core isosurface",
        "#d55e00",
    )


def save_convergence_comparison(
    labels: Sequence[str],
    cycle_times: np.ndarray,
    energy_drift: np.ndarray,
    output_path: Path,
) -> None:
    """用并列柱状图保存网格或时间步收敛实验的核心量。"""

    plt = _pyplot()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    locations = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), constrained_layout=True)
    axes[0].bar(locations, cycle_times, color="#0072b2")
    axes[0].set_xticks(locations, labels, rotation=20, ha="right")
    axes[0].set_ylabel("detected cycle time")
    axes[0].set_title("Leapfrogging period")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(locations, 100.0 * energy_drift, color="#d55e00")
    axes[1].set_xticks(locations, labels, rotation=20, ha="right")
    axes[1].set_ylabel("relative ISF energy change (%)")
    axes[1].set_title("Energy drift at common end time")
    axes[1].grid(axis="y", alpha=0.25)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
