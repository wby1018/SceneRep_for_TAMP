# scene_object.py
# -*- coding: utf-8 -*-
"""
SceneObject
-----------

A lightweight container that tracks everything we care about for a single
object instance observed in a 3-D scene:

* **detections** – every (label, score) tuple ever seen
* **label**      – the label whose *cumulative* score is highest so far
* **pose**       – the **immutable** 6-DoF pose (4×4 SE(3) matrix) assigned at
                   construction time (world ← object₀)
* **points**     – an accumulated point cloud, automatically down-sampled so
                   it never grows beyond ``max_points``
* **tsdf**       – a *TSDFVolume* instance (your own implementation) used to
                   fuse depth
* **T**          – current SE(3) transform *from the initial object frame* to
                   its latest pose (identity at start)

Author : you
Date   : 2025-07-11
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

# -----------------------------------------------------------------------------#
#  Optional imports (comment out if not available)
# -----------------------------------------------------------------------------#
try:
    import open3d as o3d
except ModuleNotFoundError:
    o3d = None  # fall back to simple random sampling

# Your own TSDF implementation must be in the import path
from .tsdf_o3d import TSDFVolume  # type: ignore


class SceneObject:
    """An object instance that lives (once) in a static scene."""

    # --------------------------------------------------------------------- #
    #  Construction
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        pose: np.ndarray,
        id,
        initial_label: str | None = None,
        initial_score: float = 0.2,
        voxel_size: float = 0.002,
        max_points: int = 6_000,
    ) -> None:
        """
        Parameters
        ----------
        pose : (4,4) np.ndarray
            World ← object₀ transform assigned **once** when the object is first
            discovered.
        initial_label : str | None
            Optional label to seed the score accumulator with score = 0.
        voxel_size : float
            Forwarded to ``TSDFVolume``.
        max_points : int
            Hard upper bound on the stored point-cloud size.
        """
        assert pose.shape == (4, 4), "`pose` must be 4×4 SE(3)"

        # assign internal unique id
        self.id: int = id
        self.pose_init: np.ndarray = pose.copy()
        self.pose_cur: np.ndarray = pose.copy()

        # ––– detections & running tally ––––––––––––––––––––––––––––––– #
        self.detections: List[Tuple[str, float]] = []
        self._score_sum: Dict[str, float] = defaultdict(float)
        self._count: int = 0 
        self._score: float = 0.0
        if initial_label is not None:
            self._score_sum[initial_label] = 0.0
        self.initial_score = initial_score

        # ––– derived label (arg-max) –––––––––––––––––––––––––––––––––– #
        self._label: str | None = initial_label

        # ––– accumulated point cloud –––––––––––––––––––––––––––––––––– #
        self._points = np.empty((0, 3), np.float32)
        self._colors = np.empty((0, 3), np.float32)
        self._max_points = int(max_points)

        # ––– for each observation, record points and colors and viewpoint, only one observation is stored for each viewpoint
        self.observation_sequence = []
        self.points_vp = np.empty((0, 3), np.float32)
        self.colors_vp = np.empty((0, 3), np.float32)

        # ––– TSDF volume –––––––––––––––––––––––––––––––––––––––––––––– #
        self.tsdf = TSDFVolume(voxel_size=voxel_size)

        # ––– relative transform (object₀ ← objectᵗ) –––––––––––––––––– #
        # self.T = np.eye(4, dtype=np.float32)

        # ––– Parameters for pose update ––––––––––––––––––––––––– #
        self.T_oe = None # object pose in ee frame
        self.moving = False # object is moving or not   
        self.fixed_pts = np.empty((0, 3), np.float32) # points for colored ICP
        self.fixed_cls = np.empty((0, 3), np.float32) # colors for colored ICP
        self.fixed_pose = None # pose for colored ICP
        self.last_update_frame = 0 # last frame when object is updated
        self.to_be_rebuild = False # whether to rebuild the object
        self.to_be_repaint = True

        self.pose_uncertain = False
        self.latest_observation_pts = np.empty((0, 3), np.float32)
        self.latest_observation_cls = np.empty((0, 3), np.float32)
        self.latest_observation_pose = None

        self.child_objs = {} # child objects: list of [id, relative_pose] where relative_pose is 4x4 transformation matrix
        self.parent_obj_id = None


    # --------------------------------------------------------------------- #
    #  Public properties
    # --------------------------------------------------------------------- #
    @property
    def label(self) -> str | None:
        """Most probable label so far (may be *None* if no detection yet)."""
        return self._label

    @property
    def points(self) -> np.ndarray:
        """(N, 3) accumulated, *down-sampled* point cloud."""
        return self._points
    
    @property
    def colors(self) -> np.ndarray:               # 新增
        return self._colors

    @property
    def detections_log(self) -> List[Tuple[str, float]]:
        """Shallow copy of the raw detection list."""
        return self.detections.copy()

    # --------------------------------------------------------------------- #
    #  Core API
    # --------------------------------------------------------------------- #
    # ––– 1. record a new detection –––––––––––––––––––––––––––––––––––– #
    def add_detection(self, label: str, score: float) -> None:
        """Append a new observation and update the label tally."""
        self.detections.append((label, float(score)))
        self._score_sum[label] += float(score)
        self._count += 1

        # update arg-max
        self._label = max(self._score_sum.items(), key=lambda kv: kv[1])[0]
        self._score = self._score_sum[self._label] / max(1, self._count)

    # ––– 2. add raw points (world frame) –––––––––––––––––––––––––––––– #
    def add_points(
        self,
        pts: np.ndarray,
        colors: np.ndarray | None = None,
    ) -> None:
        """
        Parameters
        ----------
        pts : (N,3) 或 (N,6) ndarray
            若为 (N,6)，后 3 列视作 RGB (0‑1 or 0‑255)。
        colors : (N,3) ndarray | None
            可显式传入颜色；若未给且 `pts` 为 (N,6) 自动拆分。
        """
        if pts.ndim != 2 or pts.shape[1] not in (3, 6):
            raise ValueError("`pts` must be (N,3) or (N,6)")

        pts = pts.astype(np.float32)

        if colors is None and pts.shape[1] == 6:
            pts, colors = pts[:, :3], pts[:, 3:]
        elif colors is None:
            raise ValueError("颜色缺失：请提供 `colors` 或传入 (N,6) 数组")
        else:
            colors = colors.astype(np.float32)
            if colors.shape != (pts.shape[0], 3):
                raise ValueError("`colors` 与 `pts` 行数不一致")

        self._points = (
            np.vstack((self._points, pts)) if self._points.size else pts
        )
        self._colors = (
            np.vstack((self._colors, colors)) if self._colors.size else colors
        )
        self._downsample_inplace()

    def add_points_vp(
        self,
        pts: np.ndarray,
        colors: np.ndarray | None = None,
    ) -> None:
        """
        Parameters
        ----------
        pts : (N,3) 或 (N,6) ndarray
            若为 (N,6)，后 3 列视作 RGB (0‑1 or 0‑255)。
        colors : (N,3) ndarray | None
            可显式传入颜色；若未给且 `pts` 为 (N,6) 自动拆分。
        """
        if pts.ndim != 2 or pts.shape[1] not in (3, 6):
            raise ValueError("`pts` must be (N,3) or (N,6)")

        pts = pts.astype(np.float32)

        if colors is None and pts.shape[1] == 6:
            pts, colors = pts[:, :3], pts[:, 3:]
        elif colors is None:
            raise ValueError("颜色缺失：请提供 `colors` 或传入 (N,6) 数组")
        else:
            colors = colors.astype(np.float32)
            if colors.shape != (pts.shape[0], 3):
                raise ValueError("`colors` 与 `pts` 行数不一致")

        self.points_vp = (
            np.vstack((self.points_vp, pts)) if self.points_vp.size else pts
        )
        self.colors_vp = (
            np.vstack((self.colors_vp, colors)) if self.colors_vp.size else colors
        )
        # self._downsample_inplace()


    # --------------------------------------------------------------------- #
    #  Helpers
    # --------------------------------------------------------------------- #
    def _downsample_inplace(self) -> None:
        """Voxel grid (Open3D) or random decimation to enforce max_points."""
        n = len(self._points)
        if n <= self._max_points:
            return

        # if o3d is not None:
        #     pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self._points))
        #     # voxel_size = 0.5 * np.linalg.norm(
        #     #     self._points.max(axis=0) - self._points.min(axis=0)
        #     # ) / (self._max_points ** (1.0 / 3.0) + 1e-6)
        #     voxel_size = 0.005         
        #     pcd = pcd.voxel_down_sample(max(1e-4, voxel_size))
        #     self._points = np.asarray(pcd.points, np.float32)

        # Fallback: uniform random sampling
        if len(self._points) > self._max_points:
            idx = random.sample(range(len(self._points)), self._max_points)
            self._points = self._points[idx]
            self._colors = self._colors[idx]

    # --------------------------------------------------------------------- #
    #  Representation
    # --------------------------------------------------------------------- #
    def __repr__(self) -> str:
        return (
            f"SceneObject(label={self._label}, "
            # f"n_obs={len(self.observation_sequence)}, "
            f"n_pts={len(self.points_vp)}, n_pts_total={len(self.points)})"
            # f"pos_uncertain={self.pose_uncertain}"
            # f"repainted={self.to_be_repaint}"
        )
