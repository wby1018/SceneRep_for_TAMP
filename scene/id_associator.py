# -*- coding: utf-8 -*-
"""
Label-based (name) object association
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
author: you, 2025-07-14 (simplified 2025-07-19)

逻辑:
    对每个当前帧 mask:
        1) 提取其点云 (world)
        2) 按 label 在已存在 objects 中寻找第一个匹配对象
        3) 若找到 ⇒ 追加 detection + points + 融合
        4) 若找不到 ⇒ 用该点云质心初始化一个新对象，再融合

假设:
    SceneObject(pose=4x4, initial_label=str)
    obj.pose      : 初始世界位姿 (固定)
    obj.T         : 后续运动的相对变换 (由别处更新；此处不修改)
    obj.points    : 已累计世界点 (N,3)
    obj.add_detection(label, score)
    obj.add_points(pts_world, colors=None)
    obj.tsdf.integrate(color_img, depth_img, K, T_cw_inv)  # 与你原先接口一致

如需“同 label 多实例”可在 `_find_object_for_label` 内调整策略。
"""

from __future__ import annotations
from typing import List, Dict, Optional

import numpy as np
from .scene_object import SceneObject
from utils.utils import _mask_to_world_pts_colors

# -----------------------------------------------------------------------------#
#  Util – mask 像素 → 世界点 + 颜色
# -----------------------------------------------------------------------------#



# -----------------------------------------------------------------------------#
#  查找已有对象 (最简单: 第一个 initial_label == label)
# -----------------------------------------------------------------------------#
def _find_object_for_internal_id(objects: List[SceneObject], internal_id: int | str) -> Optional[int]:
    for i, obj in enumerate(objects):
        if obj.id == internal_id:
            return i
    return None


def _find_object_for_label(objects: List[SceneObject], label: str) -> Optional[int]:
    for i, obj in enumerate(objects):
        if obj.label == label:
            return i
    return None


def _compute_point_cloud_distance(points1: np.ndarray, points2: np.ndarray, max_points: int = 1000) -> float:
    """
    计算两个点云之间的距离（使用最近邻距离的平均值）
    如果点云太大，随机采样以提高效率
    """
    if len(points1) > max_points:
        indices = np.random.choice(len(points1), max_points, replace=False)
        points1 = points1[indices]
    if len(points2) > max_points:
        indices = np.random.choice(len(points2), max_points, replace=False)
        points2 = points2[indices]
    
    # 计算每个点到另一个点云的最小距离
    distances = []
    for pt in points1:
        min_dist = np.min(np.linalg.norm(points2 - pt, axis=1))
        distances.append(min_dist)
    
    return np.mean(distances)


# -----------------------------------------------------------------------------#
#  核心 – 基于 label 的关联
# -----------------------------------------------------------------------------#
def associate_by_id(
    masks: List[Dict],          # 每个 dict 至少包含: "mask"；可包含: "id", "label", "score"
    depth: np.ndarray,
    rgb:   np.ndarray,
    K:     np.ndarray,
    T_cw:  np.ndarray,
    objects: List[SceneObject],
    frame_id: int,
    sample_step: int = 1,
    voxel_size: float = 0.002,
    integrate: bool = True,
    max_distance_thresh: float = 0.1,  # 最大距离阈值（米）
    pose_ok: bool = True
) -> bool:  # 修改返回类型为 bool
    if not masks:
        return False  # 没有masks时返回False

    has_new_objects = False  # 标记是否有新物体加入

    for mid, m in enumerate(masks):
        mask = m["mask"]
        external_id = m.get("id")
        label = m.get("label", "unknown")
        score = float(m.get("score", 1.0))

        # 深度过滤：只保留在有效范围内的深度值
        depth_obj = depth.copy()
        depth_obj[mask == 0] = 0
        
        # 计算mask区域的深度均值，只保留在均值±0.1范围内的深度值
        valid_depths = depth_obj[mask == 1]
        # print(m["id"])
        # print(len(valid_depths))
        # 深度过滤：使用统计学特征动态确定有效深度范围
        if len(valid_depths) < 500:
            continue
        if len(valid_depths) > 0:
            depth_mean = np.mean(valid_depths)
            depth_std = np.std(valid_depths)
            
            # # 从配置文件读取参数（如果可用）
            # try:
            #     from data_demo_config import load_config
            #     config = load_config()
            #     k_factor = config.get('tsdf', {}).get('depth_filter', {}).get('statistics_filter', {}).get('k_factor', 1.5)
            #     min_abs_error = config.get('tsdf', {}).get('depth_filter', {}).get('statistics_filter', {}).get('min_abs_error', 0.05)
            #     min_depth = config.get('tsdf', {}).get('depth_filter', {}).get('statistics_filter', {}).get('min_depth', 0.1)
            #     max_depth = config.get('tsdf', {}).get('depth_filter', {}).get('statistics_filter', {}).get('max_depth', 10.0)
            #     use_percentile = config.get('tsdf', {}).get('depth_filter', {}).get('alternative_methods', {}).get('percentile_method', False)
            #     iqr_factor = config.get('tsdf', {}).get('depth_filter', {}).get('alternative_methods', {}).get('iqr_factor', 1.5)
            # except:
                # 如果无法读取配置，使用默认值
            k_factor = 1
            min_abs_error = 0.05
            min_depth = 0.1
            max_depth = 10.0
            use_percentile = True
            iqr_factor = 1.1
            
            if use_percentile and len(valid_depths) >= 4:
                # 方法2：基于百分位数的范围
                depth_25 = np.percentile(valid_depths, 5)
                depth_75 = np.percentile(valid_depths, 95)
                iqr = depth_75 - depth_25
                depth_min = depth_25 - iqr_factor * iqr
                depth_max = depth_75 + iqr_factor * iqr
                method_name = "百分位数方法"
            else:
                # 方法1：基于标准差的自适应范围（推荐）
                # 使用 mean ± k*std，其中k可以根据数据质量调整
                depth_min = depth_mean - k_factor * depth_std
                depth_max = depth_mean + k_factor * depth_std
                method_name = "标准差方法"
                
                # 方法3：基于绝对误差的保守范围（备选方案）
                # 如果标准差太小，使用最小绝对误差
                if k_factor * depth_std < min_abs_error:
                    depth_min = depth_mean - min_abs_error
                    depth_max = depth_mean + min_abs_error
                    method_name = "绝对误差方法"    
            # 确保深度范围在合理范围内
            depth_min = max(depth_min, depth_mean - 0.3)
            depth_max = min(depth_max, depth_mean + 0.3)
            
            # 应用深度过滤
            depth_filter_mask = (depth_obj >= depth_min) & (depth_obj <= depth_max)
            # depth_obj[~depth_filter_mask] = 0
            
            # 打印统计信息
            # filtered_pixels = np.sum(depth_filter_mask)
            # total_pixels = np.sum(mask == 1)
            # print(f"物体 {external_id} 深度过滤({method_name}): mean={depth_mean:.3f}m, std={depth_std:.3f}m, "
            #       f"范围=[{depth_min:.3f}, {depth_max:.3f}]m, "
            #       f"有效像素={filtered_pixels}/{total_pixels} ({filtered_pixels/total_pixels*100:.1f}%)")
        # mask = depth_filter_mask
    
        color_obj = rgb.copy();   color_obj[mask == 0] = 0

        pts_world, cols = _mask_to_world_pts_colors(
            mask, depth, rgb, K, T_cw, sample_step=sample_step
        )
        if pts_world.size == 0:
            print(f"Object {external_id} (label: {label}) no points, skipping...")
            continue

        oid: Optional[int] = None
        # 1) 若上游提供 id，则尝试与内部 obj.id 匹配
        if external_id is not None:
            try:
                oid = _find_object_for_internal_id(objects, int(external_id))
            except Exception:
                oid = None
        # 2) 否则回退到 label 匹配（保持兼容）
        # if oid is None and label is not None:
        #     oid = _find_object_for_label(objects, label)

        if oid is not None:
            # 检查距离：如果当前观测与已有对象距离过大，则重建
            obj = objects[oid]
            if hasattr(obj, '_points') and len(obj._points) > 0:
                # # 将对象累积点云变换到当前位姿：先逆向pose_init，再正向pose_cur
                # obj_points_current = obj._points.copy()
                # # 逆向变换：从初始位姿回到对象局部坐标系
                # obj_points_local = np.linalg.inv(obj.pose_init) @ np.concatenate([obj_points_current, np.ones((len(obj_points_current), 1))], axis=1).T
                # obj_points_local = obj_points_local[:3].T
                # # 正向变换：从对象局部坐标系到当前位姿
                # obj_points_current = obj.pose_cur @ np.concatenate([obj_points_local, np.ones((len(obj_points_local), 1))], axis=1).T
                # obj_points_current = obj_points_current[:3].T
                
                # distance = _compute_point_cloud_distance(pts_world, obj_points_current)
                # if distance > max_distance_thresh:
                #     print(f"Object {obj.id} (label: {label}) distance too large ({distance:.3f}m > {max_distance_thresh}m), rebuilding...")
                #     # 删除旧对象
                #     objects.pop(oid)
                #     oid = None
                #     # 继续执行重建逻辑
                if obj.to_be_rebuild and oid is not None:
                    print(f"Object {obj.id} (label: {label}) icp failed, rebuilding...")
                    # 删除旧对象
                    objects.pop(oid)
                    oid = None
                    # 继续执行重建逻辑

        if oid is not None:
            if objects[oid].pose_uncertain or not pose_ok:
                continue
            # 关联到已有对象
            objects[oid].add_detection(label, score)
            objects[oid].last_update_frame = frame_id
            objects[oid].to_be_repaint = True
            T_warp   = objects[oid].pose_init @ np.linalg.inv(objects[oid].pose_cur)
            T_cw_fix = T_warp @ T_cw
            pts, cols = _mask_to_world_pts_colors(
                mask, depth, rgb, K, T_cw_fix, sample_step=1
            )
            if pts.size:
                objects[oid].add_points(pts, colors=cols)
                objects[oid].latest_observation_pts = pts
                objects[oid].latest_observation_cls = cols
                objects[oid].latest_observation_pose = objects[oid].pose_init
                # print(f"update object latest_observation_pts for {objects[oid].label} {objects[oid].id}")
            if integrate:
                objects[oid].tsdf.integrate(color_obj, depth_obj, K,
                                            np.linalg.inv(T_cw_fix).astype(np.float64))
        else:
            # 创建新对象
            has_new_objects = True  # 标记有新物体加入
            Tw = np.eye(4, dtype=np.float32)
            Tw[:3, 3] = pts_world.mean(axis=0)
            new_obj = SceneObject(pose=Tw.copy(), id=external_id, initial_label=label, initial_score=score, voxel_size=voxel_size)
            new_obj.add_detection(label, score)
            new_obj.last_update_frame = frame_id
            T_warp   = new_obj.pose_init @ np.linalg.inv(new_obj.pose_cur)
            T_cw_fix = T_warp @ T_cw
            new_obj.add_points(pts_world, colors=cols)
            new_obj.latest_observation_pts = pts_world
            new_obj.latest_observation_cls = cols
            new_obj.latest_observation_pose = new_obj.pose_init
            if integrate:
                new_obj.tsdf.integrate(color_obj, depth_obj, K,
                                       np.linalg.inv(T_cw_fix).astype(np.float64))
            print(f"{new_obj.id} : {new_obj.pose_init}")
            objects.append(new_obj)

    return has_new_objects  # 返回是否有新物体加入
