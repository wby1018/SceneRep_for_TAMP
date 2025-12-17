
from __future__ import annotations
import numpy as np
from typing import List, Any, Optional, Dict

import open3d as o3d

from utils.utils import _mask_to_world_pts_colors, find_object_by_id

# ------------------ 基础工具 ------------------ #
def _is_SE3(T: np.ndarray) -> bool:
    return (
        isinstance(T, np.ndarray)
        and T.shape == (4, 4)
        and np.allclose(T[3], [0, 0, 0, 1], atol=1e-6)
    )


def _invert(T: np.ndarray) -> np.ndarray:
    """SE(3) 逆: [R t; 0 1]^{-1} = [R^T -R^T t; 0 1]"""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=T.dtype)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    """SVD 纠正旋转矩阵数值漂移"""
    U, _, Vt = np.linalg.svd(R)
    R_new = U @ Vt
    if np.linalg.det(R_new) < 0:
        U[:, -1] *= -1
        R_new = U @ Vt
    return R_new

def _make_pcd(pts: np.ndarray, colors: np.ndarray | None = None):
    """(N,3)[,(N,3)] -> o3d PointCloud"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    if colors is not None:
        c = colors.copy()
        if c.max() > 1.5:          # 0-255 -> 0-1
            c /= 255.0
        pcd.colors = o3d.utility.Vector3dVector(c.astype(np.float64))
    return pcd


# ------------------ 核心更新函数 ------------------ #
def update_obj_pose_ee(
    objects: List[Any],
    obj_id_in_ee: int,
    T_cw: np.ndarray,               # camera → world
    T_ec: np.ndarray,               # ee → camera
) -> bool:
    # if objects is None or not (0 <= obj_id_in_ee < len(objects)):
    #     print(f"objects: {objects}")
    #     print("objects is None or obj_id_in_ee is not in objects")
    #     return False

    obj = find_object_by_id(obj_id_in_ee, objects)

    for name in ("pose_init", "pose_cur"):
        if not hasattr(obj, name):
            raise AttributeError(f"Object missing required attribute '{name}'")


    # 基本合法性
    matrices = [("T_cw", T_cw), ("T_ec", T_ec), ("obj.pose_init", obj.pose_init), ("obj.pose_cur", obj.pose_cur)]
    for n, M in matrices:
        if not _is_SE3(M):
            raise ValueError(f"{n} is not a valid 4x4 SE(3) matrix")

    # 计算 ee → world
    T_ew = T_cw @ T_ec

    # 当前世界下物体位姿 object_current → world
    T_ow_current = obj.pose_cur

    # 初始化 ee → object (抓取瞬间锁定)
    if obj.T_oe is None:
        # T_eo = (ee→world)^{-1} @ (object_current→world)
        # 使得: T_ew @ T_eo = T_ow_current
        obj.T_oe = _invert(T_ew) @ T_ow_current
        if not _is_SE3(obj.T_oe):
            raise ValueError("Initialized T_eo is invalid")

    obj.pose_cur = T_ew @ obj.T_oe
    _update_related_objects_using_relative_poses(
        obj, objects
    )
    return True


def gather_used_source_points(pcd_source: o3d.geometry.PointCloud,
                              pcd_target: o3d.geometry.PointCloud,
                              return_indices: bool = False,
                              k: int = 5):
    """
    从 pcd_source 中选出：作为 pcd_target 中每个点的 k-NN 而被“使用过”的所有 source 点（去重）。
    使用 Open3D 的 batched KNN；无显式 for 循环。
    """
    src_pts = np.asarray(pcd_source.points)
    tgt_pts = np.asarray(pcd_target.points)

    if src_pts.size == 0 or tgt_pts.size == 0:
        empty = o3d.geometry.PointCloud()
        if return_indices:
            return empty, np.empty((0,), dtype=np.int64)
        return empty

    # k 不能超过源点数
    k = int(max(1, min(k, len(src_pts))))

    dev = o3d.core.Device("CPU:0")
    src_t = o3d.core.Tensor(src_pts.astype(np.float32), device=dev)
    tgt_t = o3d.core.Tensor(tgt_pts.astype(np.float32), device=dev)

    nns = o3d.core.nns.NearestNeighborSearch(src_t)
    _ = nns.knn_index()
    knn_inds, _ = nns.knn_search(tgt_t, k)   # knn_inds: [M, k]

    used_idx = np.unique(knn_inds.numpy().reshape(-1))

    pcd_subset = o3d.geometry.PointCloud()
    pcd_subset.points = o3d.utility.Vector3dVector(src_pts[used_idx])

    if pcd_source.has_colors():
        src_cols = np.asarray(pcd_source.colors)
        if len(src_cols) == len(src_pts):
            pcd_subset.colors = o3d.utility.Vector3dVector(src_cols[used_idx])

    if pcd_source.has_normals():
        src_nrm = np.asarray(pcd_source.normals)
        if len(src_nrm) == len(src_pts):
            pcd_subset.normals = o3d.utility.Vector3dVector(src_nrm[used_idx])

    if return_indices:
        return pcd_subset, used_idx
    return pcd_subset

def icp_translation_only(pcd_source: o3d.geometry.PointCloud,
                         pcd_target: o3d.geometry.PointCloud,
                         *,
                         max_corr_dist: float = 0.05,
                         max_iter: int = 20,
                         method: str = "point_to_plane",  # "point_to_point" | "point_to_plane"
                         trim_ratio: float = 0.0          # 0~0.5，剔除最差对应比例
                         ):
    """
    仅估计平移向量 t 的 ICP。返回 T_delta(4x4)

    要求:
      - point_to_plane 需 pcd_target 有法向量；没有则先 estimate_normals()。
    """
    assert method in ("point_to_point", "point_to_plane")
    src_pts0 = np.asarray(pcd_source.points, dtype=np.float32)
    tgt_pts  = np.asarray(pcd_target.points, dtype=np.float32)

    if src_pts0.size == 0 or tgt_pts.size == 0:
        raise ValueError("empty point cloud")

    # KDTree on target
    kdt = o3d.geometry.KDTreeFlann(pcd_target)

    # translation only
    t = np.zeros(3, dtype=np.float64)

    # prepare normals if needed
    if method == "point_to_plane":
        if not pcd_target.has_normals():
            raise ValueError("point_to_plane 需要目标点云具有法向量（请先 estimate_normals）")
        tgt_nrm = np.asarray(pcd_target.normals, dtype=np.float32)

    src_pts = src_pts0.copy().astype(np.float64)

    for _ in range(max_iter):
        # 1) apply current translation
        src_cur = src_pts + t   # (N,3)

        # 2) radius-NN (keep nearest within radius)
        idx_src = []
        idx_tgt = []
        for i in range(src_cur.shape[0]):
            _, idx, dist2 = kdt.search_radius_vector_3d(
                o3d.utility.Vector3dVector([src_cur[i]])[0],
                max_corr_dist
            )
            if len(idx) == 0:
                continue
            j = idx[int(np.argmin(dist2))]
            idx_src.append(i)
            idx_tgt.append(j)

        if len(idx_src) < 4:
            break

        P = src_cur[np.asarray(idx_src)]
        Q = tgt_pts[np.asarray(idx_tgt)]

        # 3) optional trimming
        if trim_ratio > 0:
            resid = np.linalg.norm(P - Q, axis=1)
            k_keep = max(3, int((1.0 - trim_ratio) * resid.size))
            keep = np.argpartition(resid, k_keep-1)[:k_keep]
            P = P[keep]; Q = Q[keep]
            if method == "point_to_plane":
                N = tgt_nrm[np.asarray(idx_tgt)][keep]

        # 4) update translation only
        if method == "point_to_point":
            t_delta = (Q.mean(axis=0) - P.mean(axis=0)).astype(np.float64)
        else:
            N = tgt_nrm[np.asarray(idx_tgt)] if trim_ratio == 0 else N  # (M,3)
            dn = (Q - P)                                                # (M,3)
            A = (N[:, :, None] * N[:, None, :]).sum(axis=0)            # (3,3)
            proj = N * np.sum(N * dn, axis=1, keepdims=True)           # (M,3)
            b = proj.sum(axis=0)                                       # (3,)
            A += 1e-9 * np.eye(3)
            t_delta = np.linalg.solve(A, b).astype(np.float64)

        new_t = t + t_delta
        if np.linalg.norm(t_delta) < 1e-6:
            t = new_t
            break
        t = new_t

    # === 输出与 Open3D 一致风格：4x4 transformation （仅平移）===
    T_delta = np.eye(4, dtype=np.float64)
    T_delta[:3, 3] = t
    return T_delta

def icp_reappear(
    obj,
    T_cw: np.ndarray,
    # T_ec: np.ndarray,
    K: np.ndarray,
    tgt_mask,
    rgb: np.ndarray = None,
    depth: np.ndarray = None,
) -> bool:
    """
    只做colored ICP，不做抓取约束。source为obj.fixed_pts/cls，target为mask提取点云。
    ICP后将变换应用到obj.pose_cur。
    """
    # target点云
    # from utils import _mask_to_world_pts_colors
    tgt_pts, tgt_cls = _mask_to_world_pts_colors(
        tgt_mask, depth, rgb, K, T_cw, sample_step=2
    )
    if len(tgt_pts) < 30:
        print("[update_obj_pose_icp] target点云太少，跳过ICP")
        return True
    pcd_source = o3d.geometry.PointCloud()
    if len(obj.points_vp) > 0:
        pcd_source.points = o3d.utility.Vector3dVector(obj.points_vp.astype(np.float32))
        pcd_source.colors = o3d.utility.Vector3dVector(obj.colors_vp.astype(np.float32))
    else:
        pcd_source.points = o3d.utility.Vector3dVector(obj.latest_observation_pts.astype(np.float32))
        pcd_source.colors = o3d.utility.Vector3dVector(obj.latest_observation_cls.astype(np.float32))
    pcd_source.transform(_invert(obj.pose_init))
    # pcd_source.points = o3d.utility.Vector3dVector(obj.latest_observation_pts.astype(np.float32))
    # pcd_source.colors = o3d.utility.Vector3dVector(obj.latest_observation_cls.astype(np.float32))
    # pcd_source.transform(_invert(obj.latest_observation_pose))
    # pcd_source.transform(obj.pose_cur)
    pcd_source.voxel_down_sample(0.002)
    pcd_source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )
    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = o3d.utility.Vector3dVector(tgt_pts.astype(np.float32))
    pcd_target.colors = o3d.utility.Vector3dVector(tgt_cls.astype(np.float32))
    pcd_target.transform(_invert(obj.pose_cur))
    pcd_target.voxel_down_sample(0.002)
    pcd_target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )


    pcd_source = gather_used_source_points(pcd_source, pcd_target)
    # # ICP前可视化（仅显示时赋色）
    # vis_source = o3d.geometry.PointCloud(pcd_source)
    # vis_source.colors = o3d.utility.Vector3dVector(np.tile([0,1,0], (len(vis_source.points),1)))
    # vis_target = o3d.geometry.PointCloud(pcd_target)
    # vis_target.colors = o3d.utility.Vector3dVector(np.tile([1,0,0], (len(vis_target.points),1)))
    
    # # 添加世界坐标系
    # world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    
    # o3d.visualization.draw_geometries([vis_source, vis_target, world_frame], window_name="ICP前: 绿色source, 红色target, 灰色世界坐标系")
    try:
        # reg = o3d.pipelines.registration.registration_colored_icp(
        #     pcd_source,
        #     pcd_target,
        #     max_correspondence_distance=0.05,  # 先大后小
        #     init=np.eye(4, dtype=np.float32),
        #     estimation_method=o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        #     criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
        #         relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50
        #     ),
        # )

        # 鲁棒 ICP
        loss = o3d.pipelines.registration.TukeyLoss(k=0.05)
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)

        reg = o3d.pipelines.registration.registration_icp(
            pcd_source, pcd_target,
            max_correspondence_distance=0.05,
            init=np.eye(4),
            estimation_method=estimation,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50
            ),
        )

        # estimation = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()

        # reg = o3d.pipelines.registration.registration_generalized_icp(
        #     pcd_source, pcd_target,
        #     max_correspondence_distance=0.02,
        #     init=np.eye(4),
        #     estimation_method=estimation,
        #     criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        # )

        # T_delta = icp_translation_only(pcd_source, pcd_target)

        T_delta = reg.transformation  # source → target
        print("[update_obj_pose_icp] ICP T_delta=\n", T_delta)
        print("registrated reappeared object")
        # print(f"[ICP] fitness={reg.fitness:.8f}, rmse={reg.inlier_rmse:.8f}")

        translation_distance = np.linalg.norm(T_delta[:3, 3])
        
        # 旋转角度（从旋转矩阵提取）
        R = T_delta[:3, :3]
        # 使用旋转矩阵的迹计算角度：tr(R) = 1 + 2*cos(theta)
        cos_theta = (np.trace(R) - 1) / 2
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 数值稳定性
        rotation_angle = np.arccos(cos_theta)
        if translation_distance > 0.05 or rotation_angle > 0.2:
            print("icp pose change too large, rotation angle: ", {rotation_angle}, "distance:", {translation_distance})
            return False
        if reg.fitness < 0.9:
            
            print("Warning: ICP未收敛，fitness过低！")
            return False
        if reg.inlier_rmse < 0.01:
            print("ICP performed")
            obj.pose_cur = obj.pose_cur @ T_delta
            # obj.T_oe = _invert(T_cw @ T_ec) @ obj.pose_cur
            obj.pose_cur[:3, :3] = _orthonormalize(obj.pose_cur[:3, :3])
            # ICP后可视化
            # # 重新变换source点云
            # pcd_source_after = o3d.geometry.PointCloud()
            # pcd_source_after.points = o3d.utility.Vector3dVector(obj.points_vp.astype(np.float32))
            # pcd_source_after.colors = o3d.utility.Vector3dVector(obj.colors_vp.astype(np.float32))
            # pcd_source_after.transform(_invert(obj.pose_init))
            # # pcd_source_after.points = o3d.utility.Vector3dVector(obj.latest_observation_pts.astype(np.float32))
            # # pcd_source_after.colors = o3d.utility.Vector3dVector(obj.latest_observation_cls.astype(np.float32))
            # # pcd_source_after.transform(_invert(obj.latest_observation_pose))
            # pcd_source_after.transform(obj.pose_cur)
            # pcd_source_after.voxel_down_sample(0.01)
            # pcd_source_after.estimate_normals(
            #     o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
            # )
            # vis_source_after = o3d.geometry.PointCloud(pcd_source_after)
            # vis_source_after.colors = o3d.utility.Vector3dVector(np.tile([0,1,0], (len(vis_source_after.points),1)))
            # vis_target_after = o3d.geometry.PointCloud(pcd_target)
            # vis_target_after.transform(obj.pose_cur @ _invert(T_delta))
            # vis_target_after.colors = o3d.utility.Vector3dVector(np.tile([1,0,0], (len(vis_target_after.points),1)))
            
            # # # 添加世界坐标系
            # # world_frame_after = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            
            # o3d.visualization.draw_geometries([vis_source_after, vis_target_after], window_name="ICP后: 绿色source, 红色target, 灰色世界坐标系")
            return True
        else:
            print("ICP rmse too large, skip")
            return False
            # obj.to_be_rebuild = True
            # 平移距离

    except Exception as e:
        print(f"[update_obj_pose_icp] ICP refinement skipped: {e}")
    return True


# ========== 限制旋转（避免对称物体乱转） ==========
def clamp_rotation(T_delta, max_deg=10.0):
    R = T_delta[:3, :3]
    angle = np.arccos(np.clip((np.trace(R) - 1)/2, -1, 1))
    max_angle = np.deg2rad(max_deg)

    if angle > max_angle:
        print(f"[ICP] rotation {np.rad2deg(angle):.2f}° too large -> clamped")
        axis = np.array([
            R[2,1] - R[1,2],
            R[0,2] - R[2,0],
            R[1,0] - R[0,1],
        ])
        axis = axis / (np.linalg.norm(axis)+1e-8)
        R_clamped = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * max_angle)
        T_delta[:3,:3] = R_clamped
    return T_delta


def update_obj_pose_icp(
    objects: list,
    obj_id_in_ee: int,
    # relations: Dict[int, Dict[str, List[int]]],
    T_cw: np.ndarray,
    T_ec: np.ndarray,
    K: np.ndarray,
    masks: list = None,
    rgb: np.ndarray = None,
    depth: np.ndarray = None,
) -> bool:
    """
    只做colored ICP，不做抓取约束。source为obj.fixed_pts/cls，target为mask提取点云。
    ICP后将变换应用到obj.pose_cur。
    """
    if objects is None or not (0 <= obj_id_in_ee < len(objects)):
        return False
    obj = find_object_by_id(obj_id_in_ee, objects)
    if getattr(obj, "fixed_pts", None) is None or len(obj.fixed_pts) < 30:
        print("[update_obj_pose_icp] fixed_pts为空或太少，跳过ICP")
        return True
    if masks is None or rgb is None or depth is None:
        return True
    # 找到与obj label匹配的mask
    label = getattr(obj, "label", None)
    tgt_mask = None
    for m in masks:
        if m.get("id") == obj_id_in_ee:
            tgt_mask = m["mask"]
            break
    if tgt_mask is None:
        print(f"[update_obj_pose_icp] 未找到label={label}的mask，跳过ICP")
        return True
    # target点云
    # from utils import _mask_to_world_pts_colors
    tgt_pts, tgt_cls = _mask_to_world_pts_colors(
        tgt_mask, depth, rgb, K, T_cw, sample_step=2
    )
    if len(tgt_pts) < 30:
        print("[update_obj_pose_icp] target点云太少，跳过ICP")
        return True
    # import open3d as o3d
    pcd_source = o3d.geometry.PointCloud()
    pcd_source.points = o3d.utility.Vector3dVector(obj.fixed_pts.astype(np.float32))
    pcd_source.colors = o3d.utility.Vector3dVector(obj.fixed_cls.astype(np.float32))
    pcd_source.transform(_invert(obj.fixed_pose))
    # pcd_source.transform(obj.pose_cur)
    pcd_source.voxel_down_sample(0.001)
    pcd_source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )
    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = o3d.utility.Vector3dVector(tgt_pts.astype(np.float32))
    pcd_target.colors = o3d.utility.Vector3dVector(tgt_cls.astype(np.float32))
    pcd_target.transform(_invert(obj.pose_cur))
    pcd_target.voxel_down_sample(0.001)
    pcd_target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )
    # # ICP前可视化（仅显示时赋色）
    # vis_source = o3d.geometry.PointCloud(pcd_source)
    # vis_source.colors = o3d.utility.Vector3dVector(np.tile([0,1,0], (len(vis_source.points),1)))
    # vis_target = o3d.geometry.PointCloud(pcd_target)
    # vis_target.colors = o3d.utility.Vector3dVector(np.tile([1,0,0], (len(vis_target.points),1)))
    # o3d.visualization.draw_geometries([vis_source, vis_target], window_name="ICP前: 绿色source, 红色target")
    try:
        # reg = o3d.pipelines.registration.registration_colored_icp(
        #     pcd_source,
        #     pcd_target,
        #     max_correspondence_distance=0.05,  # 先大后小
        #     init=np.eye(4, dtype=np.float32),
        #     estimation_method=o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        #     criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
        #         relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50
        #     ),
        # )

        # 鲁棒 ICP
        loss = o3d.pipelines.registration.TukeyLoss(k=0.09)
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)

        reg = o3d.pipelines.registration.registration_icp(
            pcd_source, pcd_target,
            max_correspondence_distance=0.05,
            init=np.eye(4),
            estimation_method=estimation,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50
            ),
        )

        # estimation = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()

        # reg = o3d.pipelines.registration.registration_generalized_icp(
        #     pcd_source, pcd_target,
        #     max_correspondence_distance=0.02,
        #     init=np.eye(4),
        #     estimation_method=estimation,
        #     criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        # )

        T_delta = reg.transformation  # source → target
        T_delta = clamp_rotation(T_delta, max_deg=10.0)
        # print("[update_obj_pose_icp] ICP T_delta=\n", T_delta)
        print(f"[ICP] fitness={reg.fitness:.8f}, rmse={reg.inlier_rmse:.8f}")
        if reg.fitness < 0.1:
            print("Warning: ICP未收敛，fitness过低！")
        if reg.inlier_rmse < 0.01:
            print("ICP performed")
            obj.pose_cur = obj.pose_cur @ T_delta
            obj.T_oe = _invert(T_cw @ T_ec) @ obj.pose_cur
            obj.pose_cur[:3, :3] = _orthonormalize(obj.pose_cur[:3, :3])
            _update_related_objects_using_relative_poses(
                obj, objects
            )
            # # ICP后可视化
            # # 重新变换source点云
            # pcd_source_after = o3d.geometry.PointCloud()
            # pcd_source_after.points = o3d.utility.Vector3dVector(obj.fixed_pts.astype(np.float32))
            # pcd_source_after.colors = o3d.utility.Vector3dVector(obj.fixed_cls.astype(np.float32))
            # pcd_source_after.transform(_invert(obj.fixed_pose))
            # pcd_source_after.transform(obj.pose_cur)
            # pcd_source_after.voxel_down_sample(0.01)
            # pcd_source_after.estimate_normals(
            #     o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
            # )
            # vis_source_after = o3d.geometry.PointCloud(pcd_source_after)
            # vis_source_after.colors = o3d.utility.Vector3dVector(np.tile([0,1,0], (len(vis_source_after.points),1)))
            # vis_target_after = o3d.geometry.PointCloud(pcd_target)
            # vis_target_after.transform(obj.pose_cur @ _invert(T_delta))
            # vis_target_after.colors = o3d.utility.Vector3dVector(np.tile([1,0,0], (len(vis_target_after.points),1)))
            # o3d.visualization.draw_geometries([vis_source_after, vis_target_after], window_name="ICP后: 绿色source, 红色target")
        else:
            print("ICP rmse too large, skip")
    except Exception as e:
        print(f"[update_obj_pose_icp] ICP refinement skipped: {e}")
    return True


def _update_related_objects_using_relative_poses(
    obj_in_ee: Any,
    objects: List[Any],
    # relations: Dict[int, Dict[str, List[int]]],
) -> None:
    """
    使用记录下来的相对位姿关系更新所有相关物体的位姿
    
    参数:
        obj_in_ee: 被操作的物体
        objects: 物体列表
        relations: 物体关系图
    """
    # if not hasattr(obj_in_ee, 'relative_poses'):
    #     return
    
    # 使用记录下来的相对位姿关系计算每个相关物体的新位姿
    for related_id, T_relative in obj_in_ee.child_objs.items():
        if related_id >= len(objects):
            continue
            
        related_obj = find_object_by_id(related_id, objects)
        # 计算新位姿：T_new = T_obj_in_ee_current @ T_relative
        related_obj.pose_cur = obj_in_ee.pose_cur @ T_relative
    
    print(f"Updated {len(obj_in_ee.child_objs)} related objects using relative poses")


def reset_relative_poses_recorded(
    objects: List[Any],
    obj_id_in_ee: int,
) -> None:
    """
    重置物体的相对位姿记录状态，用于物体被释放时
    
    参数:
        objects: 物体列表
        obj_id_in_ee: 要重置的物体ID
    """
    if obj_id_in_ee is None:
        return
        
    obj = find_object_by_id(obj_id_in_ee, objects)
    if obj is not None and hasattr(obj, 'relative_poses_recorded'):
        obj.relative_poses_recorded = False
        if hasattr(obj, 'relative_poses'):
            obj.relative_poses.clear()
        print(f"Reset relative poses recorded for object {obj_id_in_ee}")


def update_child_objects_pose_icp(
    objects: List[Any],
    obj_id_in_ee: int,
    relations: Dict[int, Dict[str, List[int]]],
    T_cw: np.ndarray,
    T_ec: np.ndarray,
    K: np.ndarray,
    masks: List[Dict],
    rgb: np.ndarray,
    depth: np.ndarray,
) -> None:
    """
    在操作obj_in_ee过程中，使用ICP更新其子物体的位姿
    
    参数:
        objects: 物体列表
        obj_id_in_ee: 被操作的物体ID
        relations: 物体关系图
        T_cw, T_ec, K, masks, rgb, depth: 用于位姿更新的参数
    """
    if obj_id_in_ee is None:
        return
    
    # 找到被操作的物体
    obj_in_ee = find_object_by_id(obj_id_in_ee, objects)
    if obj_in_ee is None:
        return
    
    # # 获取所有相关的子物体ID
    # related_obj_ids = set()
    # if obj_in_ee.id in relations:
    #     current_relations = relations[obj_in_ee.id]
        
    #     # 包含关系：被当前物体包含的物体
    #     for contained_id in current_relations.get("contain", []):
    #         if contained_id < len(objects):
    #             related_obj_ids.add(contained_id)
        
    #     # 支撑关系：被当前物体支撑的物体
    #     for supported_id in current_relations.get("under", []):
    #         if supported_id < len(objects):
    #             related_obj_ids.add(supported_id)
    
    # 为每个子物体更新位姿
    for related_id, T in obj_in_ee.child_objs.items():
        _update_single_child_pose_icp(
            objects, related_id, obj_in_ee, T_cw, T_ec, K, masks, rgb, depth
        )

def _update_single_child_pose_icp(
    objects: List[Any],
    child_id: int,
    parent_obj: Any,
    T_cw: np.ndarray,
    T_ec: np.ndarray,
    K: np.ndarray,
    masks: List[Dict],
    rgb: np.ndarray,
    depth: np.ndarray,
) -> None:
    """
    更新单个子物体的位姿，使用ICP方法
    
    参数:
        objects: 物体列表
        child_id: 子物体ID
        parent_obj: 父物体
        T_cw, T_ec, K, masks, rgb, depth: 用于位姿更新的参数
    """
    if child_id >= len(objects):
        return
        
    child_obj = find_object_by_id(child_id, objects)
    
    # 找到与子物体匹配的mask
    tgt_mask = None
    for m in masks:
        if m.get("id") == child_id:
            tgt_mask = m["mask"]
            break
    
    if tgt_mask is None:
        return
    
    # # 如果还没有记录固定观测，则记录当前mask下的点云
    # if not hasattr(child_obj, 'fixed_pts_child') or child_obj.fixed_pts_child is None:
    #     child_obj.fixed_pts_child, child_obj.fixed_cls_child = _mask_to_world_pts_colors(
    #         tgt_mask, depth, rgb, K, T_cw, sample_step=2
    #     )
    #     child_obj.fixed_pose_child = child_obj.pose_cur.copy()
    #     print(f"Recorded fixed observation for child object {child_id}")
    #     return
    
    # 提取当前mask下的点云作为target
    tgt_pts, tgt_cls = _mask_to_world_pts_colors(
        tgt_mask, depth, rgb, K, T_cw, sample_step=2
    )
    
    if len(tgt_pts) < 30:
        return
    
    pcd_source = o3d.geometry.PointCloud()
    pcd_source.points = o3d.utility.Vector3dVector(child_obj.points_vp.astype(np.float32))
    pcd_source.colors = o3d.utility.Vector3dVector(child_obj.colors_vp.astype(np.float32))
    pcd_source.transform(_invert(child_obj.pose_init))
    # pcd_source.points = o3d.utility.Vector3dVector(obj.latest_observation_pts.astype(np.float32))
    # pcd_source.colors = o3d.utility.Vector3dVector(obj.latest_observation_cls.astype(np.float32))
    # pcd_source.transform(_invert(obj.latest_observation_pose))
    # pcd_source.transform(obj.pose_cur)
    pcd_source.voxel_down_sample(0.002)
    pcd_source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )
    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = o3d.utility.Vector3dVector(tgt_pts.astype(np.float32))
    pcd_target.colors = o3d.utility.Vector3dVector(tgt_cls.astype(np.float32))
    pcd_target.transform(_invert(child_obj.pose_cur))
    pcd_target.voxel_down_sample(0.002)
    pcd_target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )


    pcd_source = gather_used_source_points(pcd_source, pcd_target)
    
    try:
        # 执行ICP
        loss = o3d.pipelines.registration.TukeyLoss(k=0.05)
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
        
        reg = o3d.pipelines.registration.registration_icp(
            pcd_source, pcd_target,
            max_correspondence_distance=0.05,
            init=np.eye(4),
            estimation_method=estimation,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50
            ),
        )
        
        T_delta = reg.transformation  # source → target
        
        print(f"[Child ICP {child_id}] fitness={reg.fitness:.8f}, rmse={reg.inlier_rmse:.8f}")
        
        if reg.fitness < 0.1:
            print(f"Warning: Child {child_id} ICP未收敛，fitness过低！")
            return
            
        if reg.inlier_rmse < 0.01:
            # 更新子物体位姿
            child_obj.pose_cur = child_obj.pose_cur @ T_delta
            child_obj.pose_cur[:3, :3] = _orthonormalize(child_obj.pose_cur[:3, :3])
            
            # 更新子物体与父物体的相对位姿关系
            # if hasattr(parent_obj, 'relative_poses') and child_id in parent_obj.relative_poses:
                # 重新计算相对位姿：T_relative = T_parent^(-1) @ T_child
            T_relative = np.linalg.inv(parent_obj.pose_cur) @ child_obj.pose_cur
            parent_obj.child_objs[child_id] = T_relative.copy()
            print(f"Updated relative pose for child {child_id}")
            
            print(f"Child {child_id} ICP performed")
        else:
            print(f"Child {child_id} ICP rmse too large, skip")
            
    except Exception as e:
        print(f"[_update_single_child_pose_icp] Child {child_id} ICP refinement skipped: {e}")


def clear_child_fixed_observations(
    objects: List[Any],
    obj_id_in_ee: int,
    # relations: Dict[int, Dict[str, List[int]]],
) -> None:
    """
    清除子物体的固定观测，用于操作结束后
    
    参数:
        objects: 物体列表
        obj_id_in_ee: 被操作的物体ID
        relations: 物体关系图
    """
    if obj_id_in_ee is None:
        return
        
    obj_in_ee = find_object_by_id(obj_id_in_ee, objects)
    if obj_in_ee is None:
        return
    
    # 清除每个子物体的固定观测
    for related_id in obj_in_ee.child_objs.keys():
        if related_id >= len(objects):
            continue
            
        child_obj = find_object_by_id(related_id, objects)
        if hasattr(child_obj, 'fixed_pts_child'):
            child_obj.fixed_pts_child = None
        if hasattr(child_obj, 'fixed_cls_child'):
            child_obj.fixed_cls_child = None
        if hasattr(child_obj, 'fixed_pose_child'):
            child_obj.fixed_pose_child = None
    
    print(f"Cleared fixed observations for {len(obj_in_ee.child_objs.keys())} child objects")
