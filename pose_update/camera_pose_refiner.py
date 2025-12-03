from __future__ import annotations
from typing import List, Dict, Optional, Tuple

import numpy as np

try:
    import open3d as o3d  # type: ignore
    O3D_OK = True
except Exception:
    o3d = None
    O3D_OK = False

from utils.utils import _mask_to_world_pts_colors, find_object_by_id
from scene.scene_object import SceneObject


def _find_object_for_mask(objects: List[SceneObject], m: Dict) -> Optional[SceneObject]:
    mid = m.get("id")
    if mid is not None:
        try:
            mid_int = int(mid)
            for obj in objects:
                if getattr(obj, "id", None) == mid_int:
                    return obj
        except Exception:
            pass
    # # fallback by label
    # label = m.get("label")
    # if label is not None:
    #     for obj in objects:
    #         if obj.label == label:
    #             return obj
    return None


def refine_camera_pose(
    masks: List[Dict],
    objects: List[SceneObject],
    depth: np.ndarray,
    rgb: np.ndarray,
    K: np.ndarray,
    T_cw: np.ndarray,
    *,
    exclude_obj_id_in_ee: Optional[int] = None,
    min_objects: int = 2,
    sample_step: int = 2,
    voxel_size: float = 0.002,
    distance_thresh: float = 0.03,
    max_iter: int = 30,
    min_points_each: int = 20,
    min_points_total: int = 1000,
    min_fitness: float = 0.2,
    visualize: bool = False,
) -> Tuple[np.ndarray, bool]:
    """
    使用多个可见物体进行小范围ICP来微调当前帧的相机位姿 T_cw。
    返回 (T_cw_refined, refined_flag)
    """
    if not O3D_OK:
        print("o3d not ready")
        return T_cw, False

    # 选择可用的对象：排除抓取中的对象
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []  # list of (src_world_pts, tgt_world_pts)
    used_objs = set()
    valid_mask = 0
    
    # 获取当前在手中的物体及其子物体ID
    exclude_obj_ids = set()
    if exclude_obj_id_in_ee is not None:
        exclude_obj_ids.add(exclude_obj_id_in_ee)
        # 获取子物体ID
        obj_in_ee = find_object_by_id(exclude_obj_id_in_ee, objects)  # 修复参数顺序
        if obj_in_ee is not None and hasattr(obj_in_ee, 'child_objs'):
            # child_objs 是字典格式 {child_id: relative_pose}
            for child_id in obj_in_ee.child_objs.keys():
                exclude_obj_ids.add(child_id)
    
    for m in masks:
        obj = _find_object_for_mask(objects, m)
        if obj is None:
            continue
        # 检查是否是要排除的物体（包括子物体）
        if obj.id in exclude_obj_ids:
            continue
        # 提取源点（世界系）
        if obj.pose_uncertain:
            continue
        src_pts, src_cls = _mask_to_world_pts_colors(m["mask"], depth, rgb, K, T_cw, sample_step=sample_step)
        valid_mask += 1
        if src_pts.size == 0:
            continue
        # 目标点：对象累计点
        tgt_pts = getattr(obj, "points_vp", None)
        tgt_cls = getattr(obj, "colors_vp", None)
        
        if tgt_pts is not None and len(tgt_pts) > 0:            
            # 计算相对变换矩阵
            T_init_inv = np.linalg.inv(obj.pose_init)
            T_relative = obj.pose_cur @ T_init_inv
            
            # 应用变换
            points_homo = np.hstack([tgt_pts, np.ones((len(tgt_pts), 1))])
            tgt_pts = (T_relative @ points_homo.T).T[:, :3]
        if tgt_pts is None or len(tgt_pts) < min_points_each:
            continue
            
        # 将点云和颜色信息一起添加到pairs中
        pairs.append((src_pts.astype(np.float32), tgt_pts.astype(np.float32), 
                     src_cls.astype(np.float32), tgt_cls.astype(np.float32)))
        used_objs.add(obj.id)

    # 如果当前帧有足够多的可见物体，则初始化观测点云
    if valid_mask >= min_objects:
        # print(f"valid_mask: {valid_mask}")
        for m in masks:
            obj = _find_object_for_mask(objects, m)
            if obj is None:
                continue
            if exclude_obj_id_in_ee is not None and getattr(obj, "id", None) == exclude_obj_id_in_ee:
                continue
            # 提取源点（世界系）
            if not obj.observation_sequence:
                T_warp   = obj.pose_init @ np.linalg.inv(obj.pose_cur)
                # print(obj.pose_cur, obj.pose_init)
                T_cw_fix = T_warp @ T_cw
                # print(T_cw_fix, T_cw)
                pts, cols = _mask_to_world_pts_colors(
                    m["mask"], depth, rgb, K, T_cw_fix, sample_step=sample_step
                )
                obj.observation_sequence.append((T_cw_fix, pts, cols))
                obj.add_points_vp(pts, cols)
                obj.to_be_repaint = True

    
    if len(used_objs) < min_objects:
        print(f"less than min_objects: {min_objects}")
        return T_cw, False

    

    # 合并为一个大源/目标点云
    src_all = np.concatenate([p[0] for p in pairs], axis=0)
    src_cls = np.concatenate([p[2] for p in pairs], axis=0)
    tgt_all = np.concatenate([p[1] for p in pairs], axis=0)
    tgt_cls = np.concatenate([p[3] for p in pairs], axis=0)
    if len(src_all) < min_points_total or len(tgt_all) < min_points_total:
        print("too few points")
        return T_cw, False

    # 构建O3D点云并下采样
    pcd_src = o3d.geometry.PointCloud()
    pcd_tgt = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(src_all.astype(np.float64))
    pcd_src.colors = o3d.utility.Vector3dVector(src_cls.astype(np.float64))
    pcd_tgt.points = o3d.utility.Vector3dVector(tgt_all.astype(np.float64))
    pcd_tgt.colors = o3d.utility.Vector3dVector(tgt_cls.astype(np.float64))

    # 下采样点云
    if voxel_size and voxel_size > 0:
        pcd_src = pcd_src.voxel_down_sample(voxel_size)
        pcd_tgt = pcd_tgt.voxel_down_sample(voxel_size)

    # 估计法向量（彩色ICP必需）
    pcd_src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    pcd_tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    
    # 检查点云数量
    if np.asarray(pcd_src.points).shape[0] < 200 or np.asarray(pcd_tgt.points).shape[0] < 200:
        print("too few points")
        # print(np.asarray(pcd_src.points).shape[0])
        # print(np.asarray(pcd_tgt.points).shape[0])
        return T_cw, False

    if visualize:
        # 复制点云用于可视化，避免修改原始数据
        pcd_src_viz = o3d.geometry.PointCloud()
        pcd_tgt_viz = o3d.geometry.PointCloud()
        
        # 复制点云数据
        pcd_src_viz.points = pcd_src.points
        pcd_src_viz.colors = pcd_src.colors
        pcd_src_viz.normals = pcd_src.normals
        
        pcd_tgt_viz.points = pcd_tgt.points
        pcd_tgt_viz.colors = pcd_tgt.colors
        pcd_tgt_viz.normals = pcd_tgt.normals
        
        # 设置源点云为绿色，目标点云为红色
        pcd_src_viz.paint_uniform_color([0.0, 1.0, 0.0])  # 绿色
        pcd_tgt_viz.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
        print("camera pose refine visualization before")
        
        o3d.visualization.draw_geometries([pcd_src_viz, pcd_tgt_viz])

    # 彩色ICP
    pcd_src.transform(np.linalg.inv(T_cw))
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max_iter))
    result = o3d.pipelines.registration.registration_icp(
        pcd_src,
        pcd_tgt,
        max_correspondence_distance=float(distance_thresh),
        init=T_cw,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=criteria,
    )

    # result = o3d.pipelines.registration.registration_colored_icp(
    #     pcd_src,
    #     pcd_tgt,
    #     max_correspondence_distance=0.05,  # 先大后小
    #     init=T_cw,
    #     estimation_method=o3d.pipelines.registration.TransformationEstimationForColoredICP(),
    #     criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
    #         relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50
    #     ),
    # )

    # T_align = result.transformation.astype(np.float32)
    if result.fitness < float(min_fitness):
        print("camera pose adjust failed")
        return T_cw, False

    # 应用对齐: 新的 T_cw' = T_align @ T_cw
    # T_cw_refined = T_align @ T_cw
    T_cw_refined = result.transformation.astype(np.float32)
    # print(f"T_cw: {T_cw_refined @np.linalg.inv(T_cw)}")
    T_check = T_cw_refined @np.linalg.inv(T_cw)
    if abs(T_check[2,3]) > 0.05:
        print(f"T_check: {T_check}")
        return T_cw, False

    if visualize:
        # 复制点云用于可视化，避免修改原始数据
        pcd_src_viz = o3d.geometry.PointCloud()
        pcd_tgt_viz = o3d.geometry.PointCloud()
        
        # 复制点云数据
        pcd_src_viz.points = pcd_src.points
        pcd_src_viz.colors = pcd_src.colors
        pcd_src_viz.normals = pcd_src.normals
        
        pcd_tgt_viz.points = pcd_tgt.points
        pcd_tgt_viz.colors = pcd_tgt.colors
        pcd_tgt_viz.normals = pcd_tgt.normals
        
        # 应用变换到源点云
        pcd_src_viz.transform(T_cw_refined)
        
        # 设置源点云为绿色，目标点云为红色
        # pcd_src_viz.transform(T_cw_refined)
        pcd_src_viz.paint_uniform_color([0.0, 1.0, 0.0])  # 绿色
        pcd_tgt_viz.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
        print("camera pose refine visualization after")
        o3d.visualization.draw_geometries([pcd_src_viz, pcd_tgt_viz])
    # 获得可信camera pose，增加此视角下的观测点云，如果此视角下已经观察过则不再添加
    for m in masks:
        obj = _find_object_for_mask(objects, m)
        if obj is None:
            continue
        if exclude_obj_id_in_ee is not None and getattr(obj, "id", None) == exclude_obj_id_in_ee:
            continue
        # 提取源点（世界系）
        # 检查是否已经从这个视角观察过
        already_observed = False
        if hasattr(obj, 'observation_sequence') and obj.observation_sequence:
            for prev_T_cw, prev_pts, prev_cols in obj.observation_sequence:
                # 计算相机位姿的差异
                T_warp = obj.pose_init @ np.linalg.inv(obj.pose_cur)
                T_cw_fix = T_warp @ T_cw_refined
                T_diff = np.linalg.inv(prev_T_cw) @ T_cw_fix
                
                # 计算位置差异（米）
                pos_diff = np.linalg.norm(T_diff[:3, 3])
                
                # 计算旋转差异（弧度）
                rot_diff = np.arccos(np.clip((np.trace(T_diff[:3, :3]) - 1) / 2, -1, 1))
                
                # 如果位置差异小于阈值且旋转差异小于阈值，认为视角相似
                if pos_diff < 0.1 and rot_diff < 0.1:  # 5cm, 约5.7度
                    already_observed = True
                    break
        
        if not already_observed and not obj.pose_uncertain:
            T_warp = obj.pose_init @ np.linalg.inv(obj.pose_cur)
            T_cw_fix = T_warp @ T_cw_refined
            pts, cols = _mask_to_world_pts_colors(
                m["mask"], depth, rgb, K, T_cw_fix, sample_step=sample_step
            )
            obj.observation_sequence.append((T_cw_fix, pts, cols))
            obj.add_points_vp(pts, cols)
            obj.to_be_repaint = True
    return T_cw_refined.astype(np.float32), True 


def compute_pose_change(T_prev: np.ndarray, T_curr: np.ndarray) -> Tuple[float, float]:
    """
    计算两帧间位姿变化：
    - rotation_angle: 旋转角度（弧度）
    - translation_distance: 平移距离（米）
    """
    T_diff = np.linalg.inv(T_prev) @ T_curr
    
    # 平移距离
    translation_distance = np.linalg.norm(T_diff[:3, 3])
    
    # 旋转角度（从旋转矩阵提取）
    R = T_diff[:3, :3]
    # 使用旋转矩阵的迹计算角度：tr(R) = 1 + 2*cos(theta)
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 数值稳定性
    rotation_angle = np.arccos(cos_theta)
    
    return rotation_angle, translation_distance


def should_skip_object_fusion(
    T_prev: np.ndarray,
    T_curr: np.ndarray,
    *,
    max_rotation: float = 0.1,  # 弧度，约5.7度
    max_translation: float = 0.05,  # 米，5cm
) -> bool:
    """
    判断是否应该跳过物体TSDF融合：
    当位姿变化超过阈值时返回True（跳过融合）
    """
    if T_prev is None:
        return False
    
    rotation_angle, translation_distance = compute_pose_change(T_prev, T_curr)
    
    skip = (rotation_angle > max_rotation) or (translation_distance > max_translation)
    
    if skip:
        print(f"Pose change too large - rotation: {rotation_angle:.4f} rad, translation: {translation_distance:.4f} m")
    
    return skip 