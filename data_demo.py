import os
import sys
import argparse
import numpy as np
import cv2
import yaml
from tqdm import tqdm

from scene.tsdf_o3d import TSDFVolume
from scene.scene_object import SceneObject
from scene.id_associator import associate_by_id

from pose_update.object_pose_updater import update_obj_pose_ee, update_obj_pose_icp, update_child_objects_pose_icp, icp_reappear, clear_child_fixed_observations
from utils.mesh_filter_fast import filter_mesh_fast
from utils.utils import *
from utils.utils import _mask_to_world_pts_colors, find_object_by_id

import pyrender, trimesh
from utils.hand_mask_utils import generate_hand_mask, generate_end_effector_mask
from utils.inpaint_utils import inpaint_background
from pose_update.camera_pose_refiner import refine_camera_pose, should_skip_object_fusion
from scene.object_relation_graph import get_relation_graph
from pose_update.gravity_simulator import gravity_simulation
from detection import hungarian_detection
from detection.mask_extractor import extract_mask
import time
import json
from utils.eval_save_utils import ObjectPoseRecorder

def load_config(config_path="data_demo_config.yaml"):
    """加载YAML配置文件"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    except FileNotFoundError:
        print(f"错误: 找不到配置文件 {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"错误: 配置文件格式错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"错误: 加载配置文件失败: {e}")
        sys.exit(1)


def load_pose_txt(path):
    """读取 pose_txt，每行格式: idx tx ty tz qx qy qz qw"""
    poses = []
    with open(path, "r") as f:
        for line in f:
            arr = line.strip().split()
            if len(arr) != 8:
                continue
            _, tx, ty, tz, qx, qy, qz, qw = map(float, arr)
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T[:3, 3] = [tx, ty, tz]
            poses.append(T)
    return poses

def load_joints_pose_json(json_path):
    """
    读取关节位姿JSON文件
    返回字典，键为帧ID，值为关节位姿字典
    """
    with open(json_path, 'r') as f:
        joints_data = json.load(f)
    
    # 处理数据，将字符串姿态转换为便于使用的数据结构
    processed_data = {}
    for frame_id, joints in joints_data.items():
        processed_joints = {}
        for joint_name, pose in joints.items():
            # 提取位置和四元数
            position = np.array(pose[:3])  # 前3个值是位置 [x, y, z]
            quaternion = np.array(pose[3:])  # 后4个值是四元数 [qx, qy, qz, qw]
            
            # 创建旋转矩阵
            rotation = Rotation.from_quat(quaternion)
            rot_matrix = rotation.as_matrix()
            
            # 创建4x4变换矩阵
            transform = np.eye(4)
            transform[:3, :3] = rot_matrix
            transform[:3, 3] = position
            
            processed_joints[joint_name] = transform
        
        processed_data[frame_id] = processed_joints
    
    return processed_data

# def get_robot_state(idx, state_ranges):
#     """根据当前帧idx和状态区间列表返回当前状态和obj_id_in_ee"""
#     for start, end, state in state_ranges:
#         if start <= idx <= end:
#             return state
#     return "idle", None  # 默认

def get_robot_state(last_state, finger_d, last_finger_d):
    """
    根据夹爪手指间距离变化判断当前状态
    - idle: 初始状态或释放后状态
    - grasping: 手指距离减小（夹紧）
    - holding: 从grasping状态转换而来，距离基本不变
    - releasing: 手指距离增大（松开）
    """
    if last_finger_d is None:
        return "idle"
    
    d_diff = finger_d - last_finger_d
    threshold = 0.001  # 距离变化阈值（米）
    
    if d_diff < -threshold:  # 距离减小，正在夹紧
        return "grasping"
    elif d_diff > threshold:  # 距离增大，正在松开
        return "releasing"
    else:  # 距离基本不变（允许轻微抖动）
        if last_state == "grasping":
            return "holding"  # 从夹紧状态转为保持状态
        elif last_state == "releasing":
            return "idle"     # 从松开状态转为空闲状态
        else:
            return last_state  # 保持当前状态（idle或holding）

def get_obj_id_in_ee(T_ew, objects):
    """
    检测当前机械手中的物体
    在末端执行器附近定义8个点形成一个box，然后找到box内点云最多的物体
    
    Args:
        T_ew: 末端执行器到世界坐标系的变换矩阵 (4x4)
        objects: 场景中的物体列表
        
    Returns:
        int or None: 被抓取物体的ID，如果没有找到则返回None
    """
    if not objects or T_ew is None:
        return None
        
    # 定义抓取box的尺寸（米）
    box_size = np.array([0.1, 0.1, 0.08])  # 长宽高各15cm
    
    # 获取末端执行器在世界坐标系中的位置
    ee_center = T_ew[:3, 3]
    
    # 定义box的8个顶点（相对于末端执行器中心）
    # 使用末端执行器的坐标系方向
    ee_x = T_ew[:3, 0]  # x轴方向
    ee_y = T_ew[:3, 1]  # y轴方向  
    ee_z = T_ew[:3, 2]  # z轴方向
    
    # 计算box的8个顶点
    half_size = box_size / 2.0
    box_vertices = []
    for dx in [-0.1, 1.9]:
        for dy in [-1, 1]:
            for dz in [-1, 1]:
                vertex = ee_center + dx * half_size[0] * ee_x + dy * half_size[1] * ee_y + dz * half_size[2] * ee_z
                box_vertices.append(vertex)
    
    box_vertices = np.array(box_vertices)  # (8, 3)
    
    # 计算box的边界
    box_min = box_vertices.min(axis=0)
    box_max = box_vertices.max(axis=0)
    
    max_points_in_box = 0
    obj_id_in_ee = None
    
    # 遍历所有物体，统计在box内的点数
    for obj in objects:
        if obj is None or not hasattr(obj, 'tsdf'):
            continue
            
        try:
            # 从TSDF中获取物体的mesh
            V, F, N, C = obj.tsdf.get_mesh()
            if len(V) == 0:
                continue
                
            # 将物体点云变换到世界坐标系
            # V是物体局部坐标系中的点，需要变换到世界坐标系
            world_points = ((obj.pose_cur@ np.linalg.inv(obj.pose_init)) @ np.hstack([V, np.ones((len(V), 1))]).T).T[:, :3]
            
            # 检查哪些点在box内
            in_box_mask = np.all(world_points >= box_min, axis=1) & np.all(world_points <= box_max, axis=1)
            points_in_box = np.sum(in_box_mask)
            
            # 更新最大点数
            if points_in_box > max_points_in_box:
                max_points_in_box = points_in_box
                obj_id_in_ee = obj.id
                
        except Exception as e:
            rospy.logwarn(f"处理物体 {obj.id} 时出错: {e}")
            continue
    
    # 只有当box内有足够多的点时，才认为有物体被抓取
    min_points_threshold = 1  # 最少需要10个点在box内
    if max_points_in_box >= min_points_threshold:
        rospy.loginfo(f"检测到物体 {obj_id_in_ee} 在机械手中，box内点数: {max_points_in_box}")
        return obj_id_in_ee
    else:
        return None

def run_single_dataset(DATASET_PATH, config):
    # 掩码提取配置
    mask_config = config['mask']
    mask_method = mask_config['method']

    # 根据方法设置参数
    if mask_method == "color":
        color_map = {}
        for key, value in mask_config['color_map'].items():
            color_map[key] = tuple(value)
        mask_tol = mask_config['tolerance']
    elif mask_method == "json":
        detection_dir = os.path.join(DATASET_PATH, config['mask']['detection_dir'])
        hungarian_dir = os.path.join(DATASET_PATH, config['mask']['hungarian_dir'])
        score_threshold = mask_config['score_threshold']
    else:
        raise ValueError(f"不支持的mask提取方法: {mask_method}")
    # Mesh过滤配置
    mesh_filter_config = config['tsdf'].get('mesh_filter', {})
    mesh_filter_enabled = mesh_filter_config.get('enabled', False)
    trim_ratio = mesh_filter_config.get('trim_ratio', 0.02)

    
    # 1. 读取所有帧文件名
    rgb_dir = os.path.join(DATASET_PATH, "rgb")
    depth_dir = os.path.join(DATASET_PATH, "depth")
    pose_dir = os.path.join(DATASET_PATH, "pose_txt")
    hand_mask_dir = os.path.join(DATASET_PATH, "hand_mask")
    os.makedirs(hand_mask_dir, exist_ok=True)
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".png")])
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(".npy")])

    cam_poses = load_pose_txt(os.path.join(pose_dir, "camera_pose.txt"))
    ee_poses = load_pose_txt(os.path.join(pose_dir, "ee_pose.txt"))
    l_finger_poses = load_pose_txt(os.path.join(pose_dir, "l_gripper_pose.txt"))
    r_finger_poses = load_pose_txt(os.path.join(pose_dir, "r_gripper_pose.txt"))
    joints_pose_path = os.path.join(pose_dir, "joints_pose.json")
    joints_data = load_joints_pose_json(joints_pose_path)

    # 相机内参（从配置文件读取）
    camera_config = config['camera']
    fx, fy, cx, cy = camera_config['fx'], camera_config['fy'], camera_config['cx'], camera_config['cy']
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float32)

    # db_eps = 0.05
    # db_min = 30

    bg_tsdf = TSDFVolume(voxel_size=config['tsdf']['bg_voxel_size'])
    objects = []

    # Pyrender可视化
    scene = pyrender.Scene()
    viz_config = config['visualization']
    viewer = pyrender.Viewer(scene, run_in_thread=True, 
                           use_raymond_lighting=viz_config['use_raymond_lighting'], 
                           viewport_size=tuple(viz_config['viewport_size']))
    mesh_nodes = {}
    rlock = viewer.render_lock

    # 边界框可视化配置
    show_bounding_boxes = viz_config.get('show_bounding_boxes', True)
    bbox_color = viz_config.get('bbox_color', [0.0, 1.0, 0.0, 1.0])  # 默认绿色
    bbox_line_width = viz_config.get('bbox_line_width', 2.0)
    bbox_style = viz_config.get('bbox_style', 'wireframe')  # 边界框样式
    bbox_edge_thickness = viz_config.get('bbox_edge_thickness', 0.002)  # 边厚度
    
    # 位姿可视化配置
    show_object_poses = viz_config.get('show_object_poses', True)
    pose_axis_length = viz_config.get('pose_axis_length', 0.05)
    pose_axis_width = viz_config.get('pose_axis_width', 0.002)

    holding = False
    obj_id_in_ee = None
    use_icp = False
    last_state = None
    last_finger_d = None
    # last_obj_id_in_ee = None
    T_cw_prev = None  # 记录上一帧相机位姿
    relations = None
    obj_detected_in_grasping = False  # 标记是否已在grasping阶段检测过物体
    T_offset = None

    processed_frames = 0
    step = 1
    skip_fusion = False

    render_queue = {
        "bg": None,  # 背景网格
        "objects": {},  # 物体网格 {obj_id: (mesh, pose)}
        "bbox": {},     # 边界框 {obj_id: (mesh, pose)}
        "poses": {}     # 位姿可视化 {obj_id: [(mesh1, pose1), (mesh2, pose2), (mesh3, pose3)]}
    }

    pose_recorder = ObjectPoseRecorder(DATASET_PATH)

    for idx in tqdm(range(len(rgb_files)), desc="Processing dataset"):
        # if idx < 0 or idx % step == 1 or (idx >= 5 and idx <= 194):
        if idx < 0 or idx % step == 1:
            continue
        # 获取当前状态

        rgb_path = os.path.join(rgb_dir, rgb_files[idx])
        depth_path = os.path.join(depth_dir, depth_files[idx])
        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            print(f"警告: RGB或depth文件不存在，跳过帧 {idx}: {rgb_path} {depth_path}")
            continue
        frame_id = rgb_files[idx].split('_')[-1].split('.')[0]
        if idx < len(rgb_files) - step:
            next_frame_id = rgb_files[idx + step].split('_')[-1].split('.')[0]
        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        depth = np.load(depth_path)
        T_cw = cam_poses[int(frame_id)]
        T_ec = ee_poses[int(frame_id)]
        T_lfc = l_finger_poses[int(frame_id)]
        T_rfc = r_finger_poses[int(frame_id)]
        T_joints = joints_data[frame_id]


        finger_d = np.linalg.norm(T_lfc[:3, 3] - T_rfc[:3, 3])
        # state = get_robot_state(idx, state_ranges) 
        state = get_robot_state(last_state, finger_d, last_finger_d) 
        # print(f"current_state:{state}")
        last_finger_d = finger_d

        # 根据状态设置参数
        if state == "idle":
            holding = False
            use_icp = False
        elif state == "grasping":
            holding = False
            use_icp = True
        elif state == "holding":
            holding = True
            use_icp = False
        elif state == "releasing":
            holding = False
            use_icp = False
        else:
            holding = False
            use_icp = False
            
        # # 当状态从holding变为其他状态时，清除子物体的固定观测
        # if last_state == "holding" and state != "holding" and last_obj_id_in_ee is not None and relations is not None:
        #     clear_child_fixed_observations(objects, last_obj_id_in_ee, relations)

        # ---- 检查位姿变化，决定是否跳过物体融合 ----
        skip_fusion = False
        pose_thresh_cfg = config.get('tsdf', {}).get('pose_threshold', {})
        max_rotation = pose_thresh_cfg.get('max_rotation', 0.1)  # 弧度，约5.7度
        max_translation = pose_thresh_cfg.get('max_translation', 0.05)  # 米，5cm
        if T_cw_prev is not None:
            skip_fusion = should_skip_object_fusion(
                T_cw_prev, T_cw,
                max_rotation=max_rotation,
                max_translation=max_translation
            )
        T_cw_prev = T_cw.copy()

        hand_mask = generate_hand_mask(T_ec, K, depth.shape, T_lfc, T_rfc, T_joints, depth=depth)
        vis_mask = rgb.copy()
        overlay = np.zeros_like(vis_mask)
        overlay[hand_mask > 0] = [255, 0, 0]  # 红色掩码
        # vis_mask[hand_mask > 0] = [255, 0, 0]  # 红色掩码

        # 使用alpha混合实现半透明效果
        alpha = 0.5  # 50%的透明度
        vis_mask = cv2.addWeighted(vis_mask, 1.0, overlay, alpha, 0)
        hand_mask_path = os.path.join(hand_mask_dir, f"hand_mask_{int(frame_id):06d}.png")
        # cv2.imwrite(hand_mask_path, cv2.cvtColor(vis_mask, cv2.COLOR_RGB2BGR))
        if T_offset is not None:
            T_cw_offset = T_offset @ T_cw
        else:
            T_cw_offset = T_cw
        if holding and objects and obj_id_in_ee is not None:
            update_obj_pose_ee(objects, obj_id_in_ee, T_cw_offset, T_ec)
            update_child_objects_pose_icp(objects, obj_id_in_ee, relations, T_cw_offset, T_ec, K, masks, rgb, depth)
        
        start_time = time.time()
        ee_label = "none"
        for obj in objects:
            if obj.id == obj_id_in_ee:
                ee_label = obj.label
        hungarian_detection.process_single_frame(DATASET_PATH, detection_dir, hungarian_dir, int(frame_id), objects, state, skip_fusion,
                                                  config['mask'].get('shrink_kernel_size', 5), hand_mask, ee_label)
        end_time = time.time()
        print(f"处理帧 {frame_id} 的检测耗时: {end_time - start_time:.2f} 秒")

        # ---- masks
        if mask_method == "color":
            masks = extract_mask("color", image=rgb, color_map=color_map, tolerance=mask_tol)
        elif mask_method == "json":
            masks = extract_mask("json", fid=idx, detection_dir=hungarian_dir, 
                               score_threshold=score_threshold)
        # elif mask_method == "json_with_id":
        #     masks = extract_mask("json", fid=idx, detection_dir=detection_dir, 
        #                        score_threshold=score_threshold)
        else:
            masks = []

        vis_detection = True
        window_x = 908  # 距离屏幕左边的像素数
        window_y = 7  # 距离屏幕顶部的像素数
        scale_factor = 0.64  # 缩放因子
        new_width = None
        new_height = None
        
        if vis_detection:
            # 读取并显示检测结果图像
            detection_image_path = os.path.join(hungarian_dir, f"detection_{idx:06d}_final.png")
            if os.path.exists(detection_image_path):
                detection_img = cv2.imread(detection_image_path)
                if detection_img is not None:
                    # 调整图像大小以便显示
                    height, width = detection_img.shape[:2]
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    detection_img_resized = cv2.resize(detection_img, (new_width, new_height))
                    
                    # 创建手部掩码叠加图像（与检测结果图像相同尺寸）
                    vis_mask = rgb.copy()
                    vis_mask[hand_mask > 0] = [255, 0, 0]
                    vis_mask = cv2.cvtColor(vis_mask, cv2.COLOR_RGB2BGR)
                    vis_mask_resized = cv2.resize(vis_mask, (new_width, new_height))
                    
                    
                    # 将两张图像垂直拼接（检测结果在上，手部掩码在下）
                    combined_image = np.vstack([detection_img_resized, vis_mask_resized])
                    
                    # 显示拼接后的图像
                    cv2.imshow("Detection Result", combined_image)
                    
                    # 设置窗口位置
                    cv2.moveWindow("Detection Result", window_x, window_y)
        
        # 可视化深度图
        vis_depth = True
        if vis_depth and depth is not None:
            # 将深度图归一化到0-255范围
            depth_valid = depth[depth > 0]  # 只考虑有效深度值
            if len(depth_valid) > 0:
                depth_min = np.percentile(depth_valid, 2)  # 使用2%和98%百分位数去除极值
                depth_max = np.percentile(depth_valid, 98)
                depth_normalized = (depth - depth_min) / (depth_max - depth_min + 1e-8)
                depth_normalized = np.clip(depth_normalized, 0, 1)
                # 将无效深度值设为0
                depth_normalized[depth == 0] = 0
                # 转换为8位图像
                depth_vis = (depth_normalized * 255).astype(np.uint8)
                # 应用颜色映射（JET colormap）
                depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                # 将无效深度值设为黑色
                depth_colored[depth == 0] = [0, 0, 0]
                
                # 调整深度图大小以便显示（与检测结果图像相同的缩放因子）
                if new_width is not None and new_height is not None:
                    depth_colored_resized = cv2.resize(depth_colored, (new_width, new_height))
                else:
                    height, width = depth_colored.shape[:2]
                    depth_colored_resized = cv2.resize(depth_colored, (int(width * scale_factor), int(height * scale_factor)))
                
                # 显示深度图
                cv2.imshow("Depth Visualization", depth_colored_resized)
                
                # 设置深度图窗口位置（在检测结果窗口旁边）
                if new_width is not None:
                    depth_window_x = window_x + new_width + 10
                else:
                    depth_window_x = window_x + int(depth_colored.shape[1] * scale_factor) + 10
                depth_window_y = window_y
                cv2.moveWindow("Depth Visualization", depth_window_x, depth_window_y)
        
        # 等待按键，允许窗口更新
        if vis_detection or vis_depth:
            cv2.waitKey(1)


        end_effector_mask = generate_end_effector_mask(T_ec, K, rgb.shape[:2], T_lfc, T_rfc, depth=depth)

        T_lfw = T_cw @ T_lfc
        T_rfw = T_cw @ T_rfc
        # finger_d = np.linalg.norm(T_lfc[:3, 3] - T_rfc[:3, 3])
        # print(f"finger d: {finger_d}")
        # obj_id_in_ee = get_object_id_in_ee(idx, state_ranges, masks, end_effector_mask, relations, T_lfw, T_rfw, objects)
        
        # 只在grasping阶段开始的第一帧检测一次物体
        if state == "grasping" and last_state != "grasping" and not obj_detected_in_grasping:
            obj_id_in_ee = get_obj_id_in_ee(T_cw @ T_ec, objects)
            obj_detected_in_grasping = True
            print(f"在grasping阶段检测到物体ID: {obj_id_in_ee}")
        elif state == "idle":
            obj_id_in_ee = None
            obj_detected_in_grasping = False
        # 其他状态保持obj_id_in_ee不变

        print(f"当前状态: {state}, 物体ID在夹爪中: {obj_id_in_ee}")

        # ---- 可选：相机位姿微调（根据可用对象）----
        pose_ref_cfg = config.get('tsdf', {}).get('pose_refine', {})
        if pose_ref_cfg.get('enabled', False):
            pose_ok = False
        else:
            pose_ok = True
        if pose_ref_cfg.get('enabled', False) and objects and not skip_fusion:
            # print("camera pose refinement start")
            vis = False
            # if idx < 50: vis = False
            # else: vis = True
            # if idx >= 500:
            #     print("state: ", state)
            #     print("idx: ", idx)
            #     vis = True
            T_cw_refined, pose_ok = refine_camera_pose(
                masks, objects, depth, rgb, K, T_cw,
                exclude_obj_id_in_ee=obj_id_in_ee,
                min_objects=pose_ref_cfg.get('min_objects', 2),
                sample_step=pose_ref_cfg.get('sample_step', 2),
                voxel_size=pose_ref_cfg.get('voxel_size', 0.002),
                distance_thresh=pose_ref_cfg.get('distance_thresh', 0.03),
                max_iter=pose_ref_cfg.get('max_iter', 30),
                min_points_each=pose_ref_cfg.get('min_points_each', 20),
                min_points_total=pose_ref_cfg.get('min_points_total', 1000),
                min_fitness=pose_ref_cfg.get('min_fitness', 0.2),
                visualize=vis
            )
            # print("camera pose refinement done")
            print(f"camera pose refine statues: {pose_ok}")
            if pose_ok:
                T_offset = T_cw_refined @ np.linalg.inv(T_cw)
                T_cw = T_cw_refined
                # print(T_offset)

        
        # if masks:  # 位姿变化过大时跳过物体处理
        for mask_info in masks:
            mask = mask_info["mask"]
            # if state == "idle":
            # mask[hand_mask > 0] = 0
            
            # 缩小mask：使用形态学腐蚀操作
            if config['mask'].get('shrink_mask', True):
                # 将mask转换为uint8格式
                mask_uint8 = (mask * 255).astype(np.uint8)
                # 创建腐蚀核
                kernel_size = config['mask'].get('shrink_kernel_size', 5)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                # 执行腐蚀操作
                mask_eroded = cv2.erode(mask_uint8, kernel, iterations=1)
                # 转换回0-1格式
                mask = (mask_eroded > 0).astype(np.uint8)
            
            mask_info["mask"] = mask
        # 首次进入grasping时，初始化fixed_pts和fixed_cls
        if (((state == "grasping" and last_state != "grasping")
            or (state == "releasing" and last_state != "releasing")
            or (state == "holding" and last_state != "holding"))):
            # and obj_id_in_ee is not None and objects and obj_id_in_ee < len(objects)
            obj = find_object_by_id(obj_id_in_ee, objects)
            if obj is not None:
                label = obj.label
                tgt_mask = None
                for m in masks:
                    if m.get("id") == obj_id_in_ee:
                        tgt_mask = m["mask"]
                        break
                if tgt_mask is not None:
                    fixed_pts, fixed_cls = _mask_to_world_pts_colors(
                        tgt_mask, depth, rgb, K, T_cw, sample_step=2
                    )
                    obj.fixed_pts = fixed_pts
                    obj.fixed_cls = fixed_cls
                    obj.fixed_pose = obj.pose_cur
                    print(f"Set fixed_pts/fixed_cls for obj {obj_id_in_ee}, num_pts={len(fixed_pts)}")
                else:
                    print(f"未找到label={label}的mask，无法初始化fixed_pts")
            else:
                print(f"未找到id={obj_id_in_ee}的物体，无法初始化fixed_pts")
        # update the pose of the object in the gripper
        if holding and objects and obj_id_in_ee is not None:
            update_obj_pose_ee(objects, obj_id_in_ee, T_cw, T_ec)
            # print("1111111111111111111111111111111111111111111111111111111111111111111111111111")
            # 同时更新子物体的位姿
            update_child_objects_pose_icp(objects, obj_id_in_ee, relations, T_cw, T_ec, K, masks, rgb, depth)

        if use_icp and objects and obj_id_in_ee is not None:
            update_obj_pose_icp(objects, obj_id_in_ee, T_cw, T_ec, K, masks, rgb, depth)
            update_child_objects_pose_icp(objects, obj_id_in_ee, relations, T_cw, T_ec, K, masks, rgb, depth)
        # if state == "releasing" and last_state != "releasing":

        # update object pose if its pose is uncertain
        if state != "releasing" and pose_ok:
            for m in masks:
                obj = find_object_by_id(m.get("id"), objects)
                if obj is not None and obj.pose_uncertain:
                    success = icp_reappear(obj, T_cw, K, m["mask"], rgb, depth)
                    if success:
                        obj.to_be_repaint = True
                        obj.pose_uncertain = False

        # pose update done, integrate observation
        object_voxel_size = config['tsdf'].get('object_voxel_size', 0.01)
        max_distance_thresh = config['tsdf'].get('max_distance_thresh', 0.03)
        has_new_obj = False
        if (not skip_fusion and state != "grasping" and state != "releasing") or idx <= 3:
            # print(objects)
            has_new_obj = associate_by_id(masks, depth, rgb, K, T_cw, objects, processed_frames,
                            voxel_size=object_voxel_size, max_distance_thresh=max_distance_thresh, pose_ok=pose_ok)
            # print("1111111111")
            # print(objects)
        if has_new_obj or (state == "releasing" and last_state != "releasing"):
            relations = get_relation_graph(
                objects, 
                tolerance=0.02,        # 空间容差（米）
                overlap_threshold=0.3, # 重叠阈值
                verbose=True           # 是否打印关系图
            )
            # if idx > 1056: relations = {0: {}, 1: {}, 2: {}, 3: {}, 4: {"contain": [5]}, 5: {"in": [4]}}  # 添加根节点
            obj = find_object_by_id(obj_id_in_ee, objects)
            if obj is not None:
                obj.T_eo = None
                obj.pose_uncertain = True
                
                # 根据关系图，将所有与当前手中物体有关系的物体的pose_uncertain都设置为True
                if relations and obj.id is not None and obj.id in relations:
                    obj_relations = relations[obj.id]                    
                    # 检查所有关系类型
                    related_obj_ids = set()
                    related_obj_ids.update(obj_relations.get("in", []))
                    related_obj_ids.update(obj_relations.get("on", []))
                    related_obj_ids.update(obj_relations.get("under", []))
                    related_obj_ids.update(obj_relations.get("contain", []))
                    # 设置所有相关物体的pose_uncertain为True
                    for related_id in related_obj_ids:
                        if related_id is not None:  # 额外检查related_id是否有效
                            related_obj = find_object_by_id(related_id, objects)
                            if related_obj is not None:
                                related_obj.pose_uncertain = True
                                print(f"设置物体 {related_id} 的 pose_uncertain 为 True（与手中物体 {obj.id} 相关）")
                elif relations and obj.id is not None:
                    print(f"警告：物体 {obj.id} 不在关系图中")
                elif obj.id is None:
                    print(f"警告：手中物体的ID为None")
        if state == "releasing" and last_state != "releasing":
            gravity_simulation(obj_id_in_ee, objects, bg_tsdf)
        if state != "releasing" and last_state == "releasing":
            clear_child_fixed_observations(objects, obj_id_in_ee)

        pose_recorder.record_pose(objects, T_cw, frame_id, skip_fusion)
        if idx == len(rgb_files) - 1:
            pose_recorder.finalize(frame_id)
        # elif skip_fusion:
        #     print(f"Frame {idx}: Skipping object fusion due to large pose change")
        # 更新last_state和last_obj_id_in_ee
        last_state = state
        last_obj_id_in_ee = obj_id_in_ee
        # T_cw_prev = T_cw.copy()  # 保存当前帧位姿供下一帧比较

        # ---- carve bg + integrate
        depth_bg, color_bg = depth.copy(), rgb.copy()
        none_mask = depth_bg == 0
        for m in masks:
            # 膨胀mask以扩大区域
            mask = m["mask"] > 0
            mask = mask.astype(np.uint8)
            kernel_size = config['mask'].get('dilate_kernel_size', 7)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask = cv2.dilate(mask, kernel, iterations=1)
            mask = mask.astype(bool)
            depth_bg[mask] = 0
            color_bg[mask] = 0

        # # 可选：对背景进行插值修补
        inpaint_cfg = config.get('tsdf', {}).get('inpaint', {})
        if inpaint_cfg.get('enabled', False):
            depth_bg, color_bg = inpaint_background(
                depth_bg, color_bg, none_mask, hand_mask
            )
        # print(depth_bg[240, :])
        # print(none_mask[240, :])
        # print(color_bg[240, :])
        # 清除画面最下方区域，防止机械臂被当做背景融入
        img_height = depth_bg.shape[0]
        clear_height = int(img_height * config['mask'].get('clear_bottom_ratio', 0.1))
        clear_start = img_height - clear_height
        depth_bg[clear_start:, :] = 0
        color_bg[clear_start:, :] = 0


        depth_bg[hand_mask > 0] = 0
        color_bg[hand_mask > 0] = 0


        # print(f"T_cw:{T_cw}")
        # 3. 可视化
        

        # vis = depth_bg.copy()
        # vis[np.isnan(vis)] = 0
        # vis = np.clip(vis, 0, np.percentile(vis, 98))
        # vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-8)
        # vis = (vis * 255).astype(np.uint8)

        # color_vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        # # color_vis[hand_mask > 0] = [255, 255, 255]
        # cv2.imshow("Hand Mask Overlay", cv2.cvtColor(color_vis, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(1)

        # vis_mask = rgb.copy()
        # vis_mask[hand_mask > 0] = [255, 0, 0]
        # cv2.imshow("Hand Mask Overlay", cv2.cvtColor(vis_mask, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(1)

        bg_tsdf.integrate(color_bg, depth_bg, K, invert(T_cw))

        # ---- viewer
        # if getattr(bg_tsdf, "changed", True):
        V,F,N,C = bg_tsdf.get_mesh()
        if len(V):
            mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(V,F,vertex_normals=N,vertex_colors=C,process=False))
            # with rlock:
            #     if "bg" in mesh_nodes:
            #         try: scene.remove_node(mesh_nodes["bg"])
            #         except Exception: pass
            #     mesh_nodes["bg"] = scene.add(mesh, pose=np.eye(4))
            render_queue["bg"] = (mesh, np.eye(4))

        obj = find_object_by_id(obj_id_in_ee, objects)
        if obj is not None:
            obj.to_be_repaint = True
            # 根据关系图，将所有与当前手中物体有关系的物体的pose_uncertain都设置为True
            if relations and obj.id is not None and obj.id in relations:
                obj_relations = relations[obj.id]                    
                # 检查所有关系类型
                related_obj_ids = set()
                related_obj_ids.update(obj_relations.get("in", []))
                related_obj_ids.update(obj_relations.get("on", []))
                related_obj_ids.update(obj_relations.get("under", []))
                related_obj_ids.update(obj_relations.get("contain", []))
                # 设置所有相关物体的pose_uncertain为True
                for related_id in related_obj_ids:
                    if related_id is not None:  # 额外检查related_id是否有效
                        related_obj = find_object_by_id(related_id, objects)
                        if related_obj is not None:
                            related_obj.to_be_repaint = True

        for i,obj in enumerate(objects):
            # if not obj.to_be_repaint:
            #     continue
            obj.to_be_repaint = False
            tsdf=obj.tsdf
            V,F,N,C = tsdf.get_mesh()
            if not len(V):
                continue
                
            # 应用mesh过滤
            if mesh_filter_enabled:
                V, F, N, C = filter_mesh_fast(
                    V, F, N, C,
                    trim_ratio=trim_ratio
                )
                if not len(V):  # 过滤后为空，跳过
                    continue
                    
            mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(V,F,vertex_normals=N,vertex_colors=C,process=False))
            T = obj.pose_cur @ invert(obj.pose_init)
            # with rlock:
            #     key = f"obj{i}"
            #     if key in mesh_nodes:
            #         try: scene.remove_node(mesh_nodes[key])
            #         except Exception: pass
            #     mesh_nodes[key] = scene.add(mesh, pose=T)
            render_queue["objects"][i] = (mesh, T)
    


            # 添加边界框可视化
            if show_bounding_boxes:
                # bbox_key = f"bbox{i}"
                # if bbox_key in mesh_nodes:
                #     try: scene.remove_node(mesh_nodes[bbox_key])
                #     except Exception: pass
                
                # 计算边界框
                if len(V) > 0:
                    # 确保V是numpy数组
                    V_array = np.array(V) if isinstance(V, list) else V
                    
                    # 调试信息
                    # print(f"V type: {type(V)}, V_array type: {type(V_array)}, V_array shape: {V_array.shape}")
                    # print(f"T type: {type(T)}, T shape: {T.shape}")
                    
                    # 将点云转换到世界坐标系
                    # 创建齐次坐标
                    V_homogeneous = np.hstack([V_array, np.ones((len(V_array), 1))])
                    # print(f"V_homogeneous type: {type(V_homogeneous)}, shape: {V_homogeneous.shape}")
                    
                    # 转换到世界坐标系
                    V_world = (T @ V_homogeneous.T).T[:, :3]
                    
                    # 计算边界框
                    min_coords = V_world.min(axis=0)
                    max_coords = V_world.max(axis=0)
                    
                    # 创建边界框的8个顶点
                    bbox_vertices = np.array([
                        [min_coords[0], min_coords[1], min_coords[2]],  # 0: 左下后
                        [max_coords[0], min_coords[1], min_coords[2]],  # 1: 右下后
                        [max_coords[0], max_coords[1], min_coords[2]],  # 2: 右下前
                        [min_coords[0], max_coords[1], min_coords[2]],  # 3: 左下前
                        [min_coords[0], min_coords[1], max_coords[2]],  # 4: 左上后
                        [max_coords[0], min_coords[1], max_coords[2]],  # 5: 右上后
                        [max_coords[0], max_coords[1], max_coords[2]],  # 6: 右上前
                        [min_coords[0], max_coords[1], max_coords[2]],  # 7: 左上前
                    ])
                    
                    # 根据配置选择边界框样式
                    if bbox_style == "wireframe":
                        # 创建线框样式
                        # 定义立方体的12条边（线框）
                        bbox_edges = np.array([
                            # 底面4条边
                            [0, 1], [1, 2], [2, 3], [3, 0],
                            # 顶面4条边
                            [4, 5], [5, 6], [6, 7], [7, 4],
                            # 连接顶面和底面的4条边
                            [0, 4], [1, 5], [2, 6], [3, 7]
                        ])
                        
                        # 对于线框，我们需要创建细长的立方体来表示每条边
                        # 将每条边转换为细长的立方体
                        bbox_wireframe_vertices = []
                        bbox_wireframe_faces = []
                        
                        edge_thickness = bbox_edge_thickness  # 从配置文件读取边的厚度
                        
                        for edge in bbox_edges:
                            start_vertex = bbox_vertices[edge[0]]
                            end_vertex = bbox_vertices[edge[1]]
                            
                            # 计算边的方向向量
                            edge_vector = end_vertex - start_vertex
                            edge_length = np.linalg.norm(edge_vector)
                            
                            if edge_length > 0:
                                # 归一化方向向量
                                edge_direction = edge_vector / edge_length
                                
                                # 创建垂直于边的两个向量
                                if abs(edge_direction[2]) < 0.9:  # 如果不是垂直边
                                    perp1 = np.array([-edge_direction[1], edge_direction[0], 0])
                                else:
                                    perp1 = np.array([1, 0, 0])
                                
                                perp1 = perp1 / np.linalg.norm(perp1)
                                perp2 = np.cross(edge_direction, perp1)
                                
                                # 创建细长立方体的8个顶点
                                half_thickness = edge_thickness / 2
                                edge_vertices = []
                                
                                for t in [0, 1]:  # 起点和终点
                                    for p1 in [-1, 1]:  # 第一个垂直方向
                                        for p2 in [-1, 1]:  # 第二个垂直方向
                                            vertex = (start_vertex + t * edge_vector + 
                                                        p1 * half_thickness * perp1 + 
                                                        p2 * half_thickness * perp2)
                                            edge_vertices.append(vertex)
                                
                                # 添加顶点到总列表
                                start_idx = len(bbox_wireframe_vertices)
                                bbox_wireframe_vertices.extend(edge_vertices)
                                
                                # 为这个细长立方体添加面（12个三角形）
                                edge_faces = [
                                    # 底面
                                    [start_idx+0, start_idx+1, start_idx+2], [start_idx+0, start_idx+2, start_idx+3],
                                    # 顶面
                                    [start_idx+4, start_idx+6, start_idx+5], [start_idx+4, start_idx+7, start_idx+6],
                                    # 侧面
                                    [start_idx+0, start_idx+4, start_idx+5], [start_idx+0, start_idx+5, start_idx+1],
                                    [start_idx+1, start_idx+5, start_idx+6], [start_idx+1, start_idx+6, start_idx+2],
                                    [start_idx+2, start_idx+6, start_idx+7], [start_idx+2, start_idx+7, start_idx+3],
                                    [start_idx+3, start_idx+7, start_idx+4], [start_idx+3, start_idx+4, start_idx+0]
                                ]
                                
                                bbox_wireframe_faces.extend(edge_faces)
                        
                        # 创建线框网格
                        if len(bbox_wireframe_vertices) > 0:
                            bbox_mesh = pyrender.Mesh.from_trimesh(
                                trimesh.Trimesh(
                                    vertices=np.array(bbox_wireframe_vertices),
                                    faces=np.array(bbox_wireframe_faces),
                                    process=False
                                ),
                                material=pyrender.material.MetallicRoughnessMaterial(
                                    baseColorFactor=bbox_color,
                                    metallicFactor=0.0,
                                    roughnessFactor=0.5
                                )
                            )
                        else:
                            # 如果线框创建失败，回退到实心样式
                            bbox_style = "solid"
                            continue
                    else:
                        # 创建实心样式
                        # 定义立方体的6个面（每个面由2个三角形组成）
                        bbox_faces = np.array([
                            # 底面
                            [0, 1, 2], [0, 2, 3],
                            # 顶面
                            [4, 6, 5], [4, 7, 6],
                            # 前面
                            [3, 2, 6], [3, 6, 7],
                            # 后面
                            [0, 5, 1], [0, 4, 5],
                            # 左面
                            [0, 3, 7], [0, 7, 4],
                            # 右面
                            [1, 5, 6], [1, 6, 2]
                        ])
                        
                        # 创建实心网格
                        bbox_mesh = pyrender.Mesh.from_trimesh(
                            trimesh.Trimesh(
                                vertices=bbox_vertices,
                                faces=bbox_faces,
                                process=False
                            ),
                            material=pyrender.material.MetallicRoughnessMaterial(
                                baseColorFactor=bbox_color,
                                metallicFactor=0.0,
                                roughnessFactor=0.5
                            )
                        )
                    
                    # mesh_nodes[bbox_key] = scene.add(bbox_mesh, pose=np.eye(4))
                    render_queue["bbox"][i] = (bbox_mesh, np.eye(4))
                    
                    # 添加物体当前位姿的可视化（坐标系）
                    if show_object_poses:
                        
                        # 添加坐标轴（X轴：红色，Y轴：绿色，Z轴：蓝色）
                        axis_length = pose_axis_length  # 从配置文件读取坐标轴长度
                        
                        # 使用细长的立方体来表示坐标轴，避免trimesh的问题
                        # X轴（红色）
                        x_axis_vertices = np.array([
                            [0, -0.001, -0.001], [axis_length, -0.001, -0.001], [axis_length, 0.001, -0.001], [0, 0.001, -0.001],  # 底面
                            [0, -0.001, 0.001], [axis_length, -0.001, 0.001], [axis_length, 0.001, 0.001], [0, 0.001, 0.001]   # 顶面
                        ])
                        x_axis_faces = np.array([
                            # 底面
                            [0, 1, 2], [0, 2, 3],
                            # 顶面
                            [4, 6, 5], [4, 7, 6],
                            # 侧面
                            [0, 4, 5], [0, 5, 1],
                            [1, 5, 6], [1, 6, 2],
                            [2, 6, 7], [2, 7, 3],
                            [3, 7, 4], [3, 4, 0]
                        ])
                        
                        x_axis = pyrender.Mesh.from_trimesh(
                            trimesh.Trimesh(
                                vertices=x_axis_vertices,
                                faces=x_axis_faces,
                                process=False
                            ),
                            material=pyrender.material.MetallicRoughnessMaterial(
                                baseColorFactor=[1.0, 0.0, 0.0, 1.0],  # 红色
                                metallicFactor=0.0,
                                roughnessFactor=0.5
                            )
                        )
                        
                        # Y轴（绿色）
                        y_axis_vertices = np.array([
                            [-0.001, 0, -0.001], [-0.001, axis_length, -0.001], [0.001, axis_length, -0.001], [0.001, 0, -0.001],  # 底面
                            [-0.001, 0, 0.001], [-0.001, axis_length, 0.001], [0.001, axis_length, 0.001], [0.001, 0, 0.001]   # 顶面
                        ])
                        y_axis_faces = np.array([
                            # 底面
                            [0, 1, 2], [0, 2, 3],
                            # 顶面
                            [4, 6, 5], [4, 7, 6],
                            # 侧面
                            [0, 4, 5], [0, 5, 1],
                            [1, 5, 6], [1, 6, 2],
                            [2, 6, 7], [2, 7, 3],
                            [3, 7, 4], [3, 4, 0]
                        ])
                        
                        y_axis = pyrender.Mesh.from_trimesh(
                            trimesh.Trimesh(
                                vertices=y_axis_vertices,
                                faces=y_axis_faces,
                                process=False
                            ),
                            material=pyrender.material.MetallicRoughnessMaterial(
                                baseColorFactor=[0.0, 1.0, 0.0, 1.0],  # 绿色
                                metallicFactor=0.0,
                                roughnessFactor=0.5
                            )
                        )
                        
                        # Z轴（蓝色）
                        z_axis_vertices = np.array([
                            [-0.001, -0.001, 0], [-0.001, -0.001, axis_length], [0.001, -0.001, axis_length], [0.001, -0.001, 0],  # 底面
                            [-0.001, 0.001, 0], [-0.001, 0.001, axis_length], [0.001, 0.001, axis_length], [0.001, 0.001, 0]   # 顶面
                        ])
                        z_axis_faces = np.array([
                            # 底面
                            [0, 1, 2], [0, 2, 3],
                            # 顶面
                            [4, 6, 5], [4, 7, 6],
                            # 侧面
                            [0, 4, 5], [0, 5, 1],
                            [1, 5, 6], [1, 6, 2],
                            [2, 6, 7], [2, 7, 3],
                            [3, 7, 4], [3, 4, 0]
                        ])
                        
                        z_axis = pyrender.Mesh.from_trimesh(
                            trimesh.Trimesh(
                                vertices=z_axis_vertices,
                                faces=z_axis_faces,
                                process=False
                            ),
                            material=pyrender.material.MetallicRoughnessMaterial(
                                baseColorFactor=[0.0, 0.0, 1.0, 1.0],  # 蓝色
                                metallicFactor=0.0,
                                roughnessFactor=0.5
                            )
                        )
                        
                        # # 将坐标轴添加到场景中
                        # x_node = scene.add(x_axis, pose=obj.pose_cur)
                        # y_node = scene.add(y_axis, pose=obj.pose_cur)
                        # z_node = scene.add(z_axis, pose=obj.pose_cur)
                        
                        # # 保存所有位姿相关的节点
                        # mesh_nodes[pose_key] = [x_node, y_node, z_node]
                        render_queue["poses"][i] = ([x_axis, y_axis, z_axis], obj.pose_cur)
                        
                        # 打印位姿信息
                        pos = obj.pose_cur[:3, 3]
                        rot_matrix = obj.pose_cur[:3, :3]
                        # print(f"物体 {obj.id} ({obj.label}) 当前位姿:")
                        # print(f"  位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] 米")
                        # print(f"  旋转矩阵: \n{rot_matrix}")
        with rlock:
            # 清除现有节点
            for key in list(mesh_nodes.keys()):
                try:
                    if isinstance(mesh_nodes[key], list):
                        for node in mesh_nodes[key]:
                            scene.remove_node(node)
                    else:
                        scene.remove_node(mesh_nodes[key])
                except Exception:
                    pass
            mesh_nodes.clear()
            
            # 添加背景
            if render_queue["bg"] is not None:
                bg_mesh, bg_pose = render_queue["bg"]
                mesh_nodes["bg"] = scene.add(bg_mesh, pose=bg_pose)
            
            # 添加所有物体
            for i, (obj_mesh, obj_pose) in render_queue["objects"].items():
                key = f"obj{i}"
                mesh_nodes[key] = scene.add(obj_mesh, pose=obj_pose)
            
            # 添加所有边界框
            for i, (bbox_mesh, bbox_pose) in render_queue["bbox"].items():
                key = f"bbox{i}"
                mesh_nodes[key] = scene.add(bbox_mesh, pose=bbox_pose)
            
            # 添加所有位姿可视化
            for i, (axes_list, axis_pose) in render_queue["poses"].items():
                pose_nodes = []
                for axis_mesh in axes_list:
                    pose_nodes.append(scene.add(axis_mesh, pose=axis_pose))
                mesh_nodes[f"pose{i}"] = pose_nodes
        # print(objects)
    # ==== run_single_dataset 的最后 ====

    # 关闭可视化线程
    try:
        viewer.close()
    except:
        pass

# 强制退出当前子进程（避免线程阻塞）
    os._exit(0)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='数据演示程序')
    parser.add_argument('--config', '-c', type=str, default='data_demo_config.yaml',
                       help='配置文件路径 (默认: data_demo_config.yaml)')
    parser.add_argument('--dataset', type=str, help='覆盖 config 中的 dataset.path')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    # 如果外部传入 dataset，则覆盖 config['dataset']['path']
    if args.dataset is not None:
        DATASET_PATH = args.dataset
    else:
        DATASET_PATH = config['dataset']['path']

    run_single_dataset(DATASET_PATH, config)


