"""周期网格及其谱空间波数。"""

from __future__ import annotations

import numpy as np

from .config import GridSpec


class PeriodicGrid:
    """为 FFT 计算准备均匀周期网格和三维波数网格。"""

    def __init__(self, spec: GridSpec) -> None:
        self.spec = spec
        self.shape = spec.shape
        self.lengths = spec.lengths
        self.dx, self.dy, self.dz = spec.spacing

        # 采用 [0, L) 的节点，终点不重复，以符合周期 FFT 的约定。
        self.x = np.arange(self.shape[0], dtype=float) * self.dx
        self.y = np.arange(self.shape[1], dtype=float) * self.dy
        self.z = np.arange(self.shape[2], dtype=float) * self.dz

        kx = 2.0 * np.pi * np.fft.fftfreq(self.shape[0], d=self.dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.shape[1], d=self.dy)
        kz = 2.0 * np.pi * np.fft.fftfreq(self.shape[2], d=self.dz)
        self.kx, self.ky, self.kz = np.meshgrid(kx, ky, kz, indexing="ij")
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2

    def periodic_displacement(self, axis: int, center: float) -> np.ndarray:
        """返回到 center 的最短周期位移，供涡环初值使用。"""

        coordinates = (self.x, self.y, self.z)[axis]
        length = self.lengths[axis]
        return (coordinates - center + 0.5 * length) % length - 0.5 * length

    def coordinate_mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """按需要生成物理坐标网格，避免在无需求时长期占用内存。"""

        return np.meshgrid(self.x, self.y, self.z, indexing="ij")
