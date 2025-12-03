# ======================== tsdf_o3d.py ========================
"""
Thread-safe Open3D 0.18 o3d.t.geometry.VoxelBlockGrid TSDF 封装
"""

from __future__ import annotations
import numpy as np, open3d as o3d, threading
from typing import Tuple

class TSDFVolume:
    def __init__(self,
                 voxel_size      : float = 0.01,
                 sdf_trunc_factor: float = 3.0,
                 depth_max       : float = 5.0,
                 block_resolution: int   = 6,
                 block_count     : int   = 3_000,
                 device          : str   = "CPU:0"):
        self._voxel = float(voxel_size)
        self._trunc = self._voxel * sdf_trunc_factor      # 截断距离 m
        self._dmax  = float(depth_max)                    # 最大融合深度 m
        self._dev   = o3d.core.Device(device)

        self._vbg = o3d.t.geometry.VoxelBlockGrid(
            ("tsdf", "weight", "color"),
            (o3d.core.float32,)*3,
            ((1,), (1,), (3,)),
            self._voxel, block_resolution, block_count, self._dev)

        self._mtx = threading.RLock()     # 读写互斥
        self.changed = False              # 有新写入 → True

    # --------------------------------------------------------
    def integrate(self,
                  color: np.ndarray,
                  depth: np.ndarray,
                  K: np.ndarray,
                  T: np.ndarray,
                  *, depth_scale: float = 1.0,
                  obs_weight  : float = 1) -> bool:
        """
        写入一次 TSDF。返回 True=写入成功，False=整帧无有效深度。
        """
        if np.count_nonzero(depth) == 0:
            return False                         # 整帧空深度

        if depth.dtype  != np.float32: depth  = depth.astype(np.float32)
        if color.dtype  != np.float32: color  = color.astype(np.float32)/255.

        d_img = o3d.t.geometry.Image(depth).to(self._dev)
        c_img = o3d.t.geometry.Image(color).to(self._dev)
        K_t   = o3d.core.Tensor(K, o3d.core.float64, self._dev)
        T_t   = o3d.core.Tensor(T, o3d.core.float64, self._dev)

        with self._mtx:
            try:
                blk = self._vbg.compute_unique_block_coordinates(
                    d_img, K_t, T_t, depth_scale, self._dmax)
            except RuntimeError:
                return False

            n = max(1, int(round(obs_weight)))
            for _ in range(n):
                self._vbg.integrate(
                    blk, d_img, c_img, K_t, T_t, depth_scale, self._dmax)

            self.changed = True
        return True

    # --------------------------------------------------------
    def get_mesh(self, weight_th: float = 0.1
                 ) -> Tuple[np.ndarray, np.ndarray,
                            np.ndarray, np.ndarray]:
        """返回 (V,F,N,C) 均为空数组时表示尚无网格"""
        with self._mtx:
            if self._vbg.hashmap().size() == 0:
                return (np.empty((0,3),np.float32),)*4
            try:
                mesh_t = self._vbg.extract_triangle_mesh(
                    weight_threshold=weight_th)
            except RuntimeError:
                return (np.empty((0,3),np.float32),)*4

        mesh = mesh_t.to(o3d.core.Device("CPU:0")).to_legacy()
        if len(mesh.vertices) == 0:
            return (np.empty((0,3),np.float32),)*4
        mesh.compute_vertex_normals()

        self.changed = False        # 已被消费
        return (np.asarray(mesh.vertices,       np.float32),
                np.asarray(mesh.triangles,      np.int32),
                np.asarray(mesh.vertex_normals, np.float32),
                (np.asarray(mesh.vertex_colors)*255).astype(np.uint8))
