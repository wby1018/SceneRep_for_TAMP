from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import numpy as np
from .scene_object import SceneObject
import open3d as o3d


def compute_bounding_box(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算点云的边界框
    返回: (min_corner, max_corner) 每个都是 (x, y, z)
    """
    if len(points) == 0:
        return np.array([0, 0, 0]), np.array([0, 0, 0])
    
    min_corner = np.min(points, axis=0)
    max_corner = np.max(points, axis=0)
    return min_corner, max_corner


def is_point_inside_box(point: np.ndarray, box_min: np.ndarray, box_max: np.ndarray, 
                       tolerance: float = 0.01) -> bool:
    """
    判断点是否在边界框内（考虑容差）
    """
    return np.all(point >= box_min - tolerance) and np.all(point <= box_max + tolerance)


def compute_spatial_relations(objects: List[SceneObject], 
                            tolerance: float = 0.02,
                            overlap_threshold: float = 0.3) -> Dict[int, Dict[str, List[int]]]:
    """
    计算物体间的空间几何关系
    
    参数:
        objects: 物体列表
        tolerance: 空间容差（米）
        overlap_threshold: 重叠阈值，用于判断"on"关系
    
    返回:
        关系图: {obj_id: {"in": [obj_ids], "on": [obj_ids], "under": [obj_ids], "contain": [obj_ids]}}
    """
    relations = {}
    
    # 为每个物体计算边界框
    object_boxes = {}
    for obj in objects:
        if hasattr(obj, '_points') and len(obj._points) > 0:
            
            # 将点云变换到当前位姿
            obj_points_current = obj._points.copy()
            # 逆向变换：从初始位姿回到对象局部坐标系
            obj_points_local = np.linalg.inv(obj.pose_init) @ np.concatenate([obj_points_current, np.ones((len(obj_points_current), 1))], axis=1).T
            obj_points_local = obj_points_local[:3].T
            # 正向变换：从对象局部坐标系到当前位姿
            obj_points_current = obj.pose_cur @ np.concatenate([obj_points_local, np.ones((len(obj_points_local), 1))], axis=1).T
            obj_points_current = obj_points_current[:3].T
            
            # 使用xyz三个方向的百分位数过滤点云
            if len(obj_points_current) > 0:
                percentile_low = 10.0  # 下百分位数
                percentile_high = 90.0  # 上百分位数
                expand_ratio = 1.2  # 扩展比例，用于延长过滤范围（在原始范围内外各扩展20%）
                
                # 计算xyz三个方向的百分位数阈值
                x_low_raw = np.percentile(obj_points_current[:, 0], percentile_low)
                x_high_raw = np.percentile(obj_points_current[:, 0], percentile_high)
                y_low_raw = np.percentile(obj_points_current[:, 1], percentile_low)
                y_high_raw = np.percentile(obj_points_current[:, 1], percentile_high)
                z_low_raw = np.percentile(obj_points_current[:, 2], percentile_low)
                z_high_raw = np.percentile(obj_points_current[:, 2], percentile_high)
                
                # 计算中点和距离，然后使用固定比例扩展
                # X方向
                x_mid = (x_low_raw + x_high_raw) / 2.0
                x_distance = (x_high_raw - x_low_raw) / 2.0
                x_low = x_mid - x_distance * expand_ratio
                x_high = x_mid + x_distance * expand_ratio
                
                # Y方向
                y_mid = (y_low_raw + y_high_raw) / 2.0
                y_distance = (y_high_raw - y_low_raw) / 2.0
                y_low = y_mid - y_distance * expand_ratio
                y_high = y_mid + y_distance * expand_ratio
                
                # Z方向
                z_mid = (z_low_raw + z_high_raw) / 2.0
                z_distance = (z_high_raw - z_low_raw) / 2.0
                z_low = z_mid - z_distance * expand_ratio
                z_high = z_mid + z_distance * expand_ratio
                
                # 创建过滤掩码：只保留在xyz三个方向都在扩展范围内的点
                filter_mask = (obj_points_current[:, 0] >= x_low) & (obj_points_current[:, 0] <= x_high) & \
                              (obj_points_current[:, 1] >= y_low) & (obj_points_current[:, 1] <= y_high) & \
                              (obj_points_current[:, 2] >= z_low) & (obj_points_current[:, 2] <= z_high)
                
                # 检查过滤后是否还有足够的点
                filtered_count = np.sum(filter_mask)
                original_count = len(obj_points_current)
                
                # 如果过滤后点太少（少于原始点的10%或少于100个点），则不应用过滤
                # 这样可以防止过滤太严格导致所有点都被过滤掉
                if filtered_count >= max(original_count * 0.1, min(100, original_count)):
                    # 应用过滤
                    obj_points_current = obj_points_current[filter_mask]
                # 否则不应用过滤，保持原始点云
            
            min_corner, max_corner = compute_bounding_box(obj_points_current)
            object_boxes[obj.id] = (min_corner, max_corner)
        else:
            object_boxes[obj.id] = (np.array([0, 0, 0]), np.array([0, 0, 0]))
    
    # 初始化关系字典
    for obj in objects:
        relations[obj.id] = {"in": [], "on": [], "under": [], "contain": []}
        obj.child_objs = {}
        obj.parent_obj_id = None
    
    # 计算物体间关系
    for i, obj1 in enumerate(objects):
        obj1_id = obj1.id
        
        if obj1_id not in object_boxes:
            continue
            
        box1_min, box1_max = object_boxes[obj1_id]
        
        for j, obj2 in enumerate(objects):
            if i == j:
                continue
                
            obj2_id = obj2.id
            if obj2_id not in object_boxes:
                continue
                
            box2_min, box2_max = object_boxes[obj2_id]
            
            # 判断关系类型
            is_inside, is_on = detect_spatial_relation(box1_min, box1_max, box2_min, box2_max, tolerance, overlap_threshold)
            # print(object_boxes)
            
            # 规则1：不允许同时出现ON和IN关系
            # if is_inside and is_on:
            #     # 优先选择IN关系（包含关系优先级更高）
            #     is_on = False
            
            # 设置关系
            if is_inside:
                # A in B
                relations[obj1_id]["in"].append(obj2_id)
                obj1.parent_obj_id = obj2_id
                # B contain A
                relations[obj2_id]["contain"].append(obj1_id)
                obj2.child_objs[obj1_id] = np.linalg.inv(obj2.pose_cur) @ obj1.pose_cur
            elif is_on:
                # A on B
                relations[obj1_id]["on"].append(obj2_id)
                obj1.parent_obj_id = obj2_id
                # gravity_simulation(obj1, obj2, object_boxes)
                # B under A
                relations[obj2_id]["under"].append(obj1_id)
                obj2.child_objs[obj1_id] = np.linalg.inv(obj2.pose_cur) @ obj1.pose_cur
    return relations

def detect_spatial_relation(box1_min: np.ndarray, box1_max: np.ndarray,
                           box2_min: np.ndarray, box2_max: np.ndarray,
                           tolerance: float, overlap_threshold: float) -> tuple[bool, bool]:
    """
    检测两个物体之间的空间关系
    
    新的判断逻辑：
    1. 首先xy面积重叠比例要在阈值以上（以面积更小的为分母）
    2. 如果面积小的z max高于面积大的z max，就是on关系
    3. 否则就是in关系（面积大的contain面积小的）
    4. 另外如果面积小的z min小于面积大的z min，就是面积大的on面积小的（很少见）
    
    返回: (is_in, is_on) 元组
    """
    # 计算x-y平面的重叠面积
    x_overlap = max(0, min(box1_max[0], box2_max[0]) - max(box1_min[0], box2_min[0]))
    y_overlap = max(0, min(box1_max[1], box2_max[1]) - max(box1_min[1], box2_min[1]))
    
    overlap_area = x_overlap * y_overlap
    box1_area = (box1_max[0] - box1_min[0]) * (box1_max[1] - box1_min[1])
    box2_area = (box2_max[0] - box2_min[0]) * (box2_max[1] - box2_min[1])
    
    # 重叠比例应该超过阈值（以面积更小的为分母）
    min_area = min(box1_area, box2_area)
    if min_area <= 0:
        return False, False
    
    overlap_ratio = overlap_area / min_area
    if overlap_ratio < overlap_threshold:
        return False, False
    
    # 判断关系类型
    is_on = False
    is_in = False
    
    # 情况1：面积小的z max高于面积大的z max（常见情况）
    if box1_area <= box2_area:
        # box1面积更小，检查box1的z max是否高于box2的z max
        if box1_max[2] > box2_max[2]:
            is_on = True
    else:
        # box2面积更小，检查box2的z max是否高于box1的z max
        if box2_min[2] < box1_min[2]:
            is_on = True
    
    # 情况2：面积小的z min小于面积大的z min（很少见的情况）
    if not is_on:  # 如果还没确定是on关系，继续检查
        if box1_area <= box2_area:
            # box1面积更小，检查box1的z min是否小于box2的z min
            if box1_max[2] < box2_max[2]:
                is_in = True
    
    return is_in, is_on


def print_relations(relations: Dict[int, Dict[str, List[int]]], objects: List[SceneObject]) -> None:
    """
    打印物体关系图
    """
    print("\n=== Object Spatial Relations ===")
    for obj_id, rels in relations.items():
        obj = next((obj for obj in objects if obj.id == obj_id), None)
        label = obj.label if obj else "unknown"
        
        in_objs = [f"obj_{id}" for id in rels["in"]]
        on_objs = [f"obj_{id}" for id in rels["on"]]
        under_objs = [f"obj_{id}" for id in rels["under"]]
        contain_objs = [f"obj_{id}" for id in rels["contain"]]
        
        print(f"Object {obj_id} ({label}):")
        if in_objs:
            print(f"  IN: {in_objs}")
        if on_objs:
            print(f"  ON: {on_objs}")
        if under_objs:
            print(f"  UNDER: {under_objs}")
        if contain_objs:
            print(f"  CONTAIN: {contain_objs}")
        if not any([in_objs, on_objs, under_objs, contain_objs]):
            print("  No spatial relations")
    print("===============================\n")


def visualize_bounding_boxes(objects: List[SceneObject], relations: Dict[int, Dict[str, List[int]]]) -> None:
    """
    使用Open3D可视化所有物体的边界框和空间关系
    
    参数:
        objects: 物体列表
        relations: 空间关系字典
    """
    geometries = []
    
    # 为每个物体创建边界框
    for obj in objects:
        if hasattr(obj, '_points') and len(obj._points) > 0:
            # 将点云变换到当前位姿
            obj_points_current = obj._points.copy()
            # 逆向变换：从初始位姿回到对象局部坐标系
            obj_points_local = np.linalg.inv(obj.pose_init) @ np.concatenate([obj_points_current, np.ones((len(obj_points_current), 1))], axis=1).T
            obj_points_local = obj_points_local[:3].T
            # 正向变换：从对象局部坐标系到当前位姿
            obj_points_current = obj.pose_cur @ np.concatenate([obj_points_local, np.ones((len(obj_points_local), 1))], axis=1).T
            obj_points_current = obj_points_current[:3].T
            
            # 计算边界框
            min_corner, max_corner = compute_bounding_box(obj_points_current)
            
            # 创建边界框的8个顶点
            bbox_vertices = np.array([
                [min_corner[0], min_corner[1], min_corner[2]],  # 0: 左下后
                [max_corner[0], min_corner[1], min_corner[2]],  # 1: 右下后
                [max_corner[0], max_corner[1], min_corner[2]],  # 2: 右下前
                [min_corner[0], max_corner[1], min_corner[2]],  # 3: 左下前
                [min_corner[0], min_corner[1], max_corner[2]],  # 4: 左上后
                [max_corner[0], min_corner[1], max_corner[2]],  # 5: 右上后
                [max_corner[0], max_corner[1], max_corner[2]],  # 6: 右上前
                [min_corner[0], max_corner[1], max_corner[2]],  # 7: 左上前
            ])
            
            # 定义立方体的12条边（线框）
            bbox_edges = np.array([
                # 底面4条边
                [0, 1], [1, 2], [2, 3], [3, 0],
                # 顶面4条边
                [4, 5], [5, 6], [6, 7], [7, 4],
                # 连接顶面和底面的4条边
                [0, 4], [1, 5], [2, 6], [3, 7]
            ])
            
            # 创建线框
            bbox_lineset = o3d.geometry.LineSet()
            bbox_lineset.points = o3d.utility.Vector3dVector(bbox_vertices)
            bbox_lineset.lines = o3d.utility.Vector2iVector(bbox_edges)
            
            # 根据物体ID设置不同颜色
            color = [0.8, 0.2, 0.2]  # 默认红色
            if obj.id == 0:
                color = [0.2, 0.8, 0.2]  # 绿色
            elif obj.id == 1:
                color = [0.2, 0.2, 0.8]  # 蓝色
            elif obj.id == 2:
                color = [0.8, 0.8, 0.2]  # 黄色
            elif obj.id == 3:
                color = [0.8, 0.2, 0.8]  # 紫色
            elif obj.id == 4:
                color = [0.2, 0.8, 0.8]  # 青色
            
            bbox_lineset.paint_uniform_color(color)
            
            # 可视化物体的当前位姿 (obj.pose_cur)
            pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
            pose_frame.transform(obj.pose_cur)
            
            # 添加物体标签（显示ID和标签）
            label_text = f"Obj{obj.id}:{obj.label}"
            print(f"物体 {obj.id} ({obj.label}) 位姿: 位置={obj.pose_cur[:3, 3]}, 边界框范围=[{min_corner}, {max_corner}]")
            
            geometries.append(bbox_lineset)
            geometries.append(pose_frame)
    
    # 添加世界坐标系
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    geometries.append(world_frame)
    
    # 可视化
    print(f"可视化 {len(objects)} 个物体的边界框和位姿...")
    o3d.visualization.draw_geometries(geometries, window_name="Object Bounding Boxes and Poses")


def get_relation_graph(objects: List[SceneObject], 
                      tolerance: float = 0.02,
                      overlap_threshold: float = 0.6,
                      verbose: bool = True,
                      visualize: bool = True) -> Dict[int, Dict[str, List[int]]]:
    """
    主函数：获取物体关系图
    
    参数:
        objects: 物体列表
        tolerance: 空间容差（米）
        overlap_threshold: 重叠阈值
        verbose: 是否打印关系图
        visualize: 是否可视化边界框
    
    返回:
        关系图字典，包含四种关系：in, on, under, contain
    """
    relations = compute_spatial_relations(objects, tolerance, overlap_threshold)
    
    if verbose:
        print_relations(relations, objects)
    
    # 可视化边界框
    # if visualize and len(objects) > 0:
    #     visualize_bounding_boxes(objects, relations)
    
    return relations 