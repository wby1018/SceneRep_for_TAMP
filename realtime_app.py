#!/usr/bin/env python3
"""
Realtime TSDF fusion app (ROS1)
--------------------------------
- Merges the offline pipeline in `data_demo.py` with the ROS capture logic in `rosbag2dataset.py`.
- Consumes the *latest* RGB, depth, and poses after each frame finishes (no strict sync);
  i.e., once a frame is fully processed, we grab the most recent topics again for the next frame.
- Detection is intentionally left as a stub for you to plug in later.

Run (example):
  rosrun your_pkg realtime_tsdf_app.py _config:=/path/to/data_demo_config.yaml

Requires:
  - Your repository modules (scene/, utils/, detection/, pose_update/ ...)
  - ROS: rospy, tf2_ros, cv_bridge, sensor_msgs
  - Open3D >= 0.18, pyrender, trimesh, numpy, opencv-python

Notes:
  - Viewer uses pyrender in a separate thread; use a machine with OpenGL.
  - If you want exact message sync instead of "latest", swap in message_filters.
"""
import os
import sys
import time
import json
import threading
import numpy as np
import cv2
import yaml
import argparse

import rospy
from sensor_msgs.msg import Image, CompressedImage, JointState
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

import tf2_ros
import tf.transformations as tft
from scipy.spatial.transform import Rotation as R

# === Your repo imports (same as offline pipeline) ===
from scene.tsdf_o3d import TSDFVolume
from scene.scene_object import SceneObject
from detection.mask_extractor import extract_mask  # You can still use this if you want
from detection.det_client import detect_image
from detection import hungarian_detection
from scene.id_associator import associate_by_id
from pose_update.object_pose_updater import (
    update_obj_pose_ee, update_obj_pose_icp, update_child_objects_pose_icp,
    icp_reappear, clear_child_fixed_observations,
)
from utils.mesh_filter_fast import filter_mesh_fast
from utils.utils import *  # includes invert
from utils.utils import _mask_to_world_pts_colors, find_object_by_id
from utils.hand_mask_utils import generate_hand_mask, generate_end_effector_mask
from utils.inpaint_utils import inpaint_background
from pose_update.camera_pose_refiner import refine_camera_pose, should_skip_object_fusion
from scene.object_relation_graph import get_relation_graph
from pose_update.gravity_simulator import gravity_simulation

import pyrender, trimesh


# ------------------------------
# Config loading (ported from data_demo.py)
# ------------------------------
def load_config(config_path="data_demo_config.yaml"):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    # Minimal checks
    required = ['mask', 'camera', 'tsdf', 'visualization']
    for k in required:
        if k not in cfg:
            raise ValueError(f"配置缺少字段: {k}")
    return cfg


# ------------------------------
# ROS Helpers
# ------------------------------
class LatestCache:
    """Keep only the latest message and its stamp."""
    def __init__(self):
        self.msg = None
        self.stamp = None
        self.lock = threading.Lock()

    def set(self, msg, stamp):
        with self.lock:
            self.msg = msg
            self.stamp = stamp

    def get(self):
        with self.lock:
            return self.msg, self.stamp


def pose_to_matrix(position, orientation):
    Tt = tft.translation_matrix([position.x, position.y, position.z])
    Rq = tft.quaternion_matrix([orientation.x, orientation.y, orientation.z, orientation.w])
    return Tt @ Rq


def list_to_matrix(xyz_quat):
    x, y, z, qx, qy, qz, qw = xyz_quat
    Tt = tft.translation_matrix([x, y, z])
    Rq = tft.quaternion_matrix([qx, qy, qz, qw])
    return Tt @ Rq


# ------------------------------
# Detection stub (you will plug in your own)
# ------------------------------
def get_detections(rgb: np.ndarray, depth: np.ndarray, cfg: dict, frame_idx: int, hand_mask, objects, state, skip_fusion):
    """
    Return detection masks in the exact format your pipeline expects:
    [ {"id": int or None, "label": str or None, "mask": np.ndarray[H,W]{0/1}}, ... ]

    For now returns empty list. Replace with your detection (e.g., server, OWL-ViT, Hungarian, etc.).

    If you prefer to keep the original extract_mask behaviors (color/json), you can
    uncomment the sample below.
    """
    detect_image(rgb, frame_idx, cfg['workspace'])
    # Example of reusing offline extract_mask via config (optional):
    # mask_cfg = cfg['mask']
    # if mask_cfg['method'] == 'color':
    #     color_map = {k: tuple(v) for k, v in mask_cfg['color_map'].items()}
    #     tol = mask_cfg.get('tolerance', 20)
    #     return extract_mask("color", image=rgb, color_map=color_map, tolerance=tol)
    # elif mask_cfg['method'] == 'json':
    #     return extract_mask("json", fid=frame_idx, detection_dir=mask_cfg['hungarian_dir'],
    #                         score_threshold=mask_cfg.get('score_threshold', 0.5))
    # else:
    #     return []
    # start_time = time.time()
    detection_dir = os.path.join(cfg['workspace'], "detection_boxes")
    hungarian_dir = os.path.join(cfg['workspace'], "detection_h")
    # print(objects)
    hungarian_detection.process_single_frame(cfg['workspace'], detection_dir, hungarian_dir, int(frame_idx), objects, state, skip_fusion, cfg['mask'].get('shrink_kernel_size', 5), hand_mask)
    # end_time = time.time()
    # print(f"处理帧 {frame_idx} 的检测耗时: {end_time - start_time:.2f} 秒")
    masks = extract_mask("json", fid=frame_idx, detection_dir=hungarian_dir, score_threshold=0.0)

    return masks


# ------------------------------
# Main realtime app
# ------------------------------
class RealtimeTSDFApp:
    def __init__(self, config_path):
        rospy.init_node('realtime_tsdf_app')

        # === ROS params ===
        # self.config_path = rospy.get_param('~config', 'realtime_app_config.yaml')

        # Load config first to get ROS parameters
        self.cfg = load_config(config_path)
        
        # Get ROS parameters from config
        ros_cfg = self.cfg.get('ros', {})
        self.camera_frame = ros_cfg.get('camera_frame', 'head_camera_rgb_optical_frame')
        self.world_frame = ros_cfg.get('world_frame', 'map')
        self.rgb_topic = ros_cfg.get('rgb_topic', '/head_camera/rgb/image_raw/compressed')
        self.depth_topic = ros_cfg.get('depth_topic', '/head_camera/depth_registered/image_raw')
        self.ee_topic = ros_cfg.get('ee_topic', '/end_effector_pose')

        # Camera intrinsics
        cam_cfg = self.cfg['camera']
        self.K = np.array([[cam_cfg['fx'], 0, cam_cfg['cx']],
                           [0, cam_cfg['fy'], cam_cfg['cy']],
                           [0, 0, 1]], dtype=np.float32)

        # TSDF volumes / state containers
        self.bg_tsdf = TSDFVolume(voxel_size=self.cfg['tsdf']['bg_voxel_size'])
        self.objects = []
        self.relations = None
        self.holding = False
        self.use_icp = False
        self.obj_id_in_ee = None
        self.last_state = "idle"
        self.T_cw_prev = None
        self.T_offset = None
        self.last_finger_d = None
        self.processed_frames = 0

        # Mesh filter conf
        mesh_filter_config = self.cfg['tsdf'].get('mesh_filter', {})
        self.mesh_filter_enabled = mesh_filter_config.get('enabled', False)
        self.trim_ratio = mesh_filter_config.get('trim_ratio', 0.02)

        # Hand/EE mask display save dir (optional)
        # self.hand_mask_dir = rospy.get_param('~hand_mask_dir', '')
        # if self.hand_mask_dir:
        #     os.makedirs(self.hand_mask_dir, exist_ok=True)

        # Viewer
        viz_cfg = self.cfg['visualization']
        self.scene = pyrender.Scene()
        self.viewer = pyrender.Viewer(self.scene, run_in_thread=True,
                                      use_raymond_lighting=viz_cfg.get('use_raymond_lighting', True),
                                      viewport_size=tuple(viz_cfg.get('viewport_size', [1280, 720])))
        self.mesh_nodes = {}
        self.rlock = self.viewer.render_lock

        # BBox / pose axis visuals
        self.show_bboxes = viz_cfg.get('show_bounding_boxes', True)
        self.bbox_color = viz_cfg.get('bbox_color', [0.0, 1.0, 0.0, 1.0])
        self.bbox_edge_thickness = viz_cfg.get('bbox_edge_thickness', 0.002)
        self.bbox_style = viz_cfg.get('bbox_style', 'wireframe')
        self.show_obj_poses = viz_cfg.get('show_object_poses', True)
        self.pose_axis_length = viz_cfg.get('pose_axis_length', 0.05)

        # Pose change threshold
        pose_thresh = self.cfg.get('tsdf', {}).get('pose_threshold', {})
        self.max_rot = pose_thresh.get('max_rotation', 0.1)
        self.max_trans = pose_thresh.get('max_translation', 0.05)

        # Pose refine config
        self.pose_ref_cfg = self.cfg.get('tsdf', {}).get('pose_refine', {})

        # ROS infra
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Latest caches (we only grab latest after finishing each frame)
        self.rgb_cache   = LatestCache()
        self.depth_cache = LatestCache()
        self.ee_cache    = LatestCache()
        self.joint_transforms = {}

        # Joints we care about (same as rosbag2dataset)
        self.planning_joint_names = [
            'torso_lift_joint','shoulder_pan_joint','shoulder_lift_joint','upperarm_roll_joint',
            'elbow_flex_joint','forearm_roll_joint','wrist_flex_joint','wrist_roll_joint'
        ]

        # Subscribers
        self.sub_rgb   = rospy.Subscriber(self.rgb_topic, CompressedImage, self.cb_rgb, queue_size=1)
        self.sub_depth = rospy.Subscriber(self.depth_topic, Image,          self.cb_depth, queue_size=1)
        self.sub_ee    = rospy.Subscriber(self.ee_topic,    PoseStamped,    self.cb_ee, queue_size=1)


        # create dataset while running
        root_dir = self.cfg['workspace']
        
        # 创建输出目录
        os.makedirs(os.path.join(root_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(root_dir, "depth"), exist_ok=True)
        os.makedirs(os.path.join(root_dir, "pose_txt"), exist_ok=True)

        # 定义姿态文件路径
        self.cam_pose_path = os.path.join(root_dir, "pose_txt", "camera_pose.txt")
        self.ee_pose_path = os.path.join(root_dir, "pose_txt", "ee_pose.txt")
        self.l_gripper_pose_path = os.path.join(root_dir, "pose_txt", "l_gripper_pose.txt")
        self.r_gripper_pose_path = os.path.join(root_dir, "pose_txt", "r_gripper_pose.txt")
        self.base_pose_path = os.path.join(root_dir, "pose_txt", "base_pose.txt")
        self.joints_pose_path = os.path.join(root_dir, "pose_txt", "joints_pose.json")
        
        # 清空所有姿态文件 (Clear all pose files)
        for path in [self.cam_pose_path, self.ee_pose_path, 
                    self.l_gripper_pose_path, self.r_gripper_pose_path, 
                    self.base_pose_path]:
            with open(path, "w") as f:
                pass  # 清空文件
                
        # 清空关节位姿JSON文件 (Clear joint pose JSON file)
        with open(self.joints_pose_path, "w") as f:
            json.dump({}, f)
        
        # 数据保存计数器 (Data saving counter)
        self.save_idx = 0
        self.success_count = 0
        self.failure_count = 0

        rospy.loginfo("RealtimeTSDFApp initialized.")

    # --- Callbacks store only the latest ---
    def cb_rgb(self, msg: CompressedImage):
        self.rgb_cache.set(msg, msg.header.stamp.to_sec())

    def cb_depth(self, msg: Image):
        self.depth_cache.set(msg, msg.header.stamp.to_sec())

    def cb_ee(self, msg: PoseStamped):
        self.ee_cache.set(msg, msg.header.stamp.to_sec())
        self.update_joints()

    # --- TF helpers to compute needed poses in camera/world frames ---
    def get_T_camera_world(self):
        try:
            tr = self.tf_buffer.lookup_transform(self.world_frame, self.camera_frame, rospy.Time(0), rospy.Duration(1.0))
            return pose_to_matrix(tr.transform.translation, tr.transform.rotation)
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"TF get_T_camera_world failed: {e}")
            return None

    def get_T_base_world(self):
        try:
            tr = self.tf_buffer.lookup_transform(self.world_frame, 'base_link', rospy.Time(0), rospy.Duration(1.0))
            return pose_to_matrix(tr.transform.translation, tr.transform.rotation)
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"TF get_T_base_world failed: {e}")
            return None

    def update_joints(self):
        """更新TF变换"""
        try:
            # 更新关节变换
            self.joint_transforms = {}
            for joint_name in self.planning_joint_names:
                transform_name = joint_name.replace("_joint", "")
                try:
                    self.joint_transforms[transform_name] = self.tf_buffer.lookup_transform(
                        "head_camera_rgb_optical_frame",
                        joint_name.replace("joint", "link"),
                        rospy.Time(0),
                        rospy.Duration(0.1)
                    )
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    # 如果特定关节变换查找失败，就不更新该关节
                    pass
                    
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"更新变换失败: {e}")

    def lookup_matrix(self, target_frame, source_frame):
        try:
            tr = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(1.0))
            return pose_to_matrix(tr.transform.translation, tr.transform.rotation)
        except Exception:
            return None

    def collect_joint_poses_in_camera(self):
        T_joints = {}
        for jn in self.planning_joint_names:
            link = jn.replace('joint','link')
            T = self.lookup_matrix(self.camera_frame, link)
            if T is not None:
                T_joints[link] = T
        return T_joints

    def T_joints_to_camera(self):
        T_joints_camera = {}
        for joint_name in self.planning_joint_names:
            transform_name = joint_name.replace("_joint", "")
            if transform_name in self.joint_transforms:
                translation = self.joint_transforms[transform_name].transform.translation
                rotation = self.joint_transforms[transform_name].transform.rotation
                T_joints_camera[transform_name] = [
                    translation.x, translation.y, translation.z,
                    rotation.x, rotation.y, rotation.z, rotation.w
                ]
        processed_joints = {}
        for joint_name, pose in T_joints_camera.items():
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
        return processed_joints

    def get_robot_state(self, last_state, finger_d, last_finger_d):
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

    def get_obj_id_in_ee(self, T_ew, objects):
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
        box_size = np.array([0.19, 0.19, 0.18])  # 长宽高各15cm
        
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
                print(f"({obj.id}) V: {len(V)}")
                if len(V) == 0:
                    continue
                    
                # 将物体点云变换到世界坐标系
                # V是物体局部坐标系中的点，需要变换到世界坐标系
                world_points = ((obj.pose_cur@ np.linalg.inv(obj.pose_init)) @ np.hstack([V, np.ones((len(V), 1))]).T).T[:, :3]
                
                # 检查哪些点在box内
                in_box_mask = np.all(world_points >= box_min, axis=1) & np.all(world_points <= box_max, axis=1)
                points_in_box = np.sum(in_box_mask)
                print(f"({obj.id}) points: {points_in_box}")
                
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

    def pose_matrix_to_list(self, T):
        """将变换矩阵转换为列表形式 [idx, x, y, z, qx, qy, qz, qw] (Convert transformation matrix to list format)"""
        trans = T[:3, 3]
        quat = R.from_matrix(T[:3, :3]).as_quat()
        return [int(self.save_idx)] + list(trans) + list(quat)  # 1编号+3平移+4四元数

    def save_frame_data(self, rgb_np, depth_np, T_cw, T_ee_camera, T_lfc, T_rfc, T_bw, T_joints):
        """保存当前帧的所有数据 (Save all data for current frame)"""
        try:
            root_dir = self.cfg['workspace']
            
            # ==== 保存 RGB 图像 (Save RGB image) ====
            try:
                rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
                rgb_path = os.path.join(root_dir, "rgb", f"rgb_{self.save_idx:06d}.png")
                cv2.imwrite(rgb_path, rgb_bgr)
            except Exception as e:
                rospy.logerr(f"保存RGB图像失败 (Failed to save RGB image): {e}")

            # ==== 保存 Depth 图像 (Save depth image) ====
            try:
                depth_path = os.path.join(root_dir, "depth", f"depth_{self.save_idx:06d}.npy")
                np.save(depth_path, depth_np.astype(np.float32))
            except Exception as e:
                rospy.logerr(f"保存深度图像失败 (Failed to save depth image): {e}")

            # ==== 保存关节位姿到JSON文件 (Save joint poses to JSON file) ====
            try:
                joints_data = {}
                if os.path.exists(self.joints_pose_path):
                    try:
                        with open(self.joints_pose_path, "r") as f:
                            joints_data = json.load(f)
                    except json.JSONDecodeError:
                        joints_data = {}
                
                frame_key = f"{self.save_idx:06d}"
                # Convert joint transforms to list format
                T_joints_camera = {}
                for joint_name, T in T_joints.items():
                    trans = T[:3, 3]
                    quat = R.from_matrix(T[:3, :3]).as_quat()
                    T_joints_camera[joint_name] = [
                        trans[0], trans[1], trans[2],
                        quat[0], quat[1], quat[2], quat[3]
                    ]
                
                joints_data[frame_key] = T_joints_camera

                with open(self.joints_pose_path, "w") as f:
                    json.dump(joints_data, f, indent=2)
            except Exception as e:
                rospy.logerr(f"保存关节位姿失败 (Failed to save joint poses): {e}")

            # ==== 保存 Pose（txt） (Save poses to txt files) ====
            try:
                cam_line = self.pose_matrix_to_list(T_cw)
                ee_line = self.pose_matrix_to_list(T_ee_camera)
                l_gripper_line = self.pose_matrix_to_list(T_lfc)
                r_gripper_line = self.pose_matrix_to_list(T_rfc)
                
                # For base pose, we need to extract position and yaw from T_bw
                base_trans = T_bw[:3, 3]
                base_quat = R.from_matrix(T_bw[:3, :3]).as_quat()
                base_euler = R.from_quat(base_quat).as_euler('zyx', degrees=False)
                base_line = [int(self.save_idx), base_trans[0], base_trans[1], base_euler[0]]

                with open(self.cam_pose_path, "a") as f:
                    f.write(" ".join([f"{v}" for v in cam_line]) + "\n")
                with open(self.ee_pose_path, "a") as f:
                    f.write(" ".join([f"{v}" for v in ee_line]) + "\n")
                with open(self.l_gripper_pose_path, "a") as f:
                    f.write(" ".join([f"{v}" for v in l_gripper_line]) + "\n")
                with open(self.r_gripper_pose_path, "a") as f:
                    f.write(" ".join([f"{v}" for v in r_gripper_line]) + "\n")
                with open(self.base_pose_path, "a") as f:
                    f.write(" ".join([f"{v}" for v in base_line]) + "\n")
            except Exception as e:
                rospy.logerr(f"保存位姿失败 (Failed to save poses): {e}")

            # 更新保存编号并记录成功 (Update save index and record success)
            self.save_idx += 1
            self.success_count += 1
            
            if self.success_count % 10 == 0:  # 每10帧打印一次 (Print every 10 frames)
                rospy.loginfo(f"成功处理 {self.success_count} 帧，失败 {self.failure_count} 帧 (Successfully processed {self.success_count} frames, failed {self.failure_count} frames)")
                
        except Exception as e:
            self.failure_count += 1
            rospy.logerr(f"数据保存失败 (Data saving failed): {e}")
            import traceback
            traceback.print_exc()

    # --- Main processing loop ---
    def run(self):
        rate = rospy.Rate(3)  # Grab-latest cadence; actual processing time governs throughput
        last_used_rgb_stamp = -1.0

        while not rospy.is_shutdown():
            rgb_msg, rgb_stamp = self.rgb_cache.get()
            depth_msg, depth_stamp = self.depth_cache.get()
            ee_msg, ee_stamp = self.ee_cache.get()

            # Only proceed when we have at least one new RGB frame since last processing
            if rgb_msg is None or depth_msg is None or ee_msg is None or rgb_stamp == last_used_rgb_stamp:
                rate.sleep()
                continue

            # Snapshot inputs (latest at this moment)
            last_used_rgb_stamp = rgb_stamp

            # Convert images
            try:
                rgb_np = cv2.imdecode(np.frombuffer(rgb_msg.data, np.uint8), cv2.IMREAD_COLOR)
                rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2RGB)
                depth_np = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough').astype(np.float32)
                depth_np = np.nan_to_num(depth_np, nan=0.0)
            except Exception as e:
                rospy.logwarn(f"Image conversion failed: {e}")
                rate.sleep()
                continue

            # Compute transforms
            T_cw = self.get_T_camera_world()
            if T_cw is None:
                rate.sleep(); continue
            T_bw = self.get_T_base_world()
            if T_bw is None:
                rate.sleep(); continue

            # end-effector pose in camera
            T_ee_base = pose_to_matrix(ee_msg.pose.position, ee_msg.pose.orientation)
            T_ee_camera = invert(T_cw) @ T_bw @ T_ee_base


            # joints in camera
            T_joints = self.collect_joint_poses_in_camera()
            T_joints_camera = self.T_joints_to_camera()

            # --- pose-change gate to skip fusion when motion is too big ---
            skip_fusion = False
            if self.T_cw_prev is not None:
                skip_fusion = should_skip_object_fusion(self.T_cw_prev, T_cw, max_rotation=self.max_rot, max_translation=self.max_trans)
            self.T_cw_prev = T_cw.copy()

            # finger links in camera
            T_lfc = invert(T_cw) @ (T_bw @ self.lookup_matrix('base_link', 'l_gripper_finger_link')) if self.lookup_matrix('base_link','l_gripper_finger_link') is not None else np.eye(4)
            T_rfc = invert(T_cw) @ (T_bw @ self.lookup_matrix('base_link', 'r_gripper_finger_link')) if self.lookup_matrix('base_link','r_gripper_finger_link') is not None else np.eye(4)

            if self.T_offset is not None:
                T_cw_offset = self.T_offset @ T_cw
            else:
                T_cw_offset = T_cw
            self.save_frame_data(rgb_np, depth_np, T_cw_offset, T_ee_camera, T_lfc, T_rfc, T_bw, T_joints)

            # --- Hand / EE masks
            hand_mask = generate_hand_mask(T_ee_camera, self.K, depth_np.shape, T_lfc, T_rfc, T_joints_camera, depth=depth_np)
            # end_eff_mask = generate_end_effector_mask(T_ee_camera, self.K, rgb_np.shape[:2], T_lfc, T_rfc, depth=depth_np)

            # Optional save for debugging
            # if self.hand_mask_dir:
            #     overlay = rgb_np.copy()
            #     m = (hand_mask > 0)
            #     ov = overlay.copy(); ov[m] = [255, 0, 0]
            #     alpha = 0.5
            #     vis = cv2.addWeighted(overlay, 1.0, ov, alpha, 0)
            #     outp = os.path.join(self.hand_mask_dir, f"hand_mask_{self.processed_frames:06d}.png")
            #     cv2.imwrite(outp, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

            finger_d = np.linalg.norm(T_lfc[:3, 3] - T_rfc[:3, 3])
            state = self.get_robot_state(self.last_state, finger_d, self.last_finger_d)
            # print(f"Robot state: {state}")
            self.last_state = state
            self.last_finger_d = finger_d

            if state == "idle":
                self.obj_id_in_ee = None
                self.holding = False
                # self.use_icp = False
            elif state == "grasping":
                T_ee_world = T_cw_offset @ T_ee_camera
                self.obj_id_in_ee = self.get_obj_id_in_ee(T_ee_world, self.objects)
                # print("111111111111111111111111111111111111111111111111111111111111111111111111")
                self.holding = False
                # self.use_icp = True
            elif state == "holding":
                # self.obj_id_in_ee = self.get_obj_id_in_ee(T_ee_world, self.objects)
                self.holding = True
                # self.use_icp = False
            elif state == "releasing":
                self.holding = False
                # self.use_icp = False
            else:
                self.obj_id_in_ee = None
                self.holding = False
                # self.use_icp = False
            
            print(f"Robot state: {state}, obj_id_in_ee: {self.obj_id_in_ee}")

            if self.holding and self.objects and self.obj_id_in_ee is not None:
                update_obj_pose_ee(self.objects, self.obj_id_in_ee, T_cw_offset, T_ee_camera)
                

            # --- Detection ---
            masks = get_detections(rgb_np, depth_np, self.cfg, self.processed_frames, hand_mask, self.objects, state, skip_fusion)


            vis_detection = True
            hungarian_dir = self.cfg['workspace'] + "/detection_h"
            if vis_detection:
                # 读取并显示检测结果图像
                detection_image_path = os.path.join(hungarian_dir, f"detection_{self.processed_frames:06d}_final.png")
                if os.path.exists(detection_image_path):
                    detection_img = cv2.imread(detection_image_path)
                    if detection_img is not None:
                        # 调整图像大小以便显示
                        height, width = detection_img.shape[:2]
                        scale_factor = 0.64 # 缩放因子
                        new_width = int(width * scale_factor)
                        new_height = int(height * scale_factor)
                        detection_img_resized = cv2.resize(detection_img, (new_width, new_height))
                        
                        # 创建手部掩码叠加图像（与检测结果图像相同尺寸）
                        vis_mask = rgb_np.copy()
                        vis_mask[hand_mask > 0] = [255, 0, 0]
                        vis_mask = cv2.cvtColor(vis_mask, cv2.COLOR_RGB2BGR)
                        vis_mask_resized = cv2.resize(vis_mask, (new_width, new_height))
                        
                        
                        # 将两张图像垂直拼接（检测结果在上，手部掩码在下）
                        combined_image = np.vstack([detection_img_resized, vis_mask_resized])
                        
                        # 显示拼接后的图像
                        cv2.imshow("Detection Result", combined_image)
                        
                        # 设置窗口位置（x, y坐标，左上角为原点）
                        # 可以根据需要调整这些数值
                        window_x = 908  # 距离屏幕左边的像素数
                        window_y = 7  # 距离屏幕顶部的像素数
                        cv2.moveWindow("Detection Result", window_x, window_y)
                        
                        cv2.waitKey(1)  # 显示1ms，允许其他窗口更新
                        # print(f"显示检测结果图像: {detection_image_path}")
                    else:
                        print(f"无法读取检测结果图像: {detection_image_path}")
                else:
                    print(f"检测结果图像不存在: {detection_image_path}")
            
            # Optional shrink (same as offline)
            if self.cfg['mask'].get('shrink_mask', True):
                ksize = int(self.cfg['mask'].get('shrink_kernel_size', 5))
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
                for mi in masks:
                    mu8 = (mi['mask'].astype(np.uint8) * 255)
                    mi['mask'] = (cv2.erode(mu8, kernel, iterations=1) > 0).astype(np.uint8)

            # --- (Optional) camera pose refinement ---
            if self.pose_ref_cfg.get('enabled', False):
                pose_ok = False
            else:
                pose_ok = True
            if self.pose_ref_cfg.get('enabled', False) and self.objects and not skip_fusion:
                T_cw_refined, pose_ok = refine_camera_pose(
                    masks, self.objects, depth_np, rgb_np, self.K, T_cw,
                    exclude_obj_id_in_ee=self.obj_id_in_ee,
                    min_objects=self.pose_ref_cfg.get('min_objects', 2),
                    sample_step=self.pose_ref_cfg.get('sample_step', 2),
                    voxel_size=self.pose_ref_cfg.get('voxel_size', 0.002),
                    distance_thresh=self.pose_ref_cfg.get('distance_thresh', 0.03),
                    max_iter=self.pose_ref_cfg.get('max_iter', 30),
                    min_points_each=self.pose_ref_cfg.get('min_points_each', 20),
                    min_points_total=self.pose_ref_cfg.get('min_points_total', 1000),
                    min_fitness=self.pose_ref_cfg.get('min_fitness', 0.2),
                    visualize=False
                )
                print(f"camera pose refine statues: {pose_ok}")
                if pose_ok:
                    self.T_offset = T_cw_refined @ np.linalg.inv(T_cw)
                    T_cw = T_cw_refined
                # else:
            if self.T_offset is not None and not pose_ok:
                T_cw = self.T_offset @ T_cw

            # # --- object pose updates (simplified state: idle) ---
            # state = 'idle'  # you can add your own state machine
            # self.holding = False
            # self.use_icp = False

            if self.holding and self.objects and self.obj_id_in_ee is not None:
                update_obj_pose_ee(self.objects, self.obj_id_in_ee, T_cw, T_ee_camera)
            if state == "grasping":
                for m in masks:
                    obj = find_object_by_id(m.get("id"), self.objects)
                    if obj is not None and obj.id == self.obj_id_in_ee:
                        success = icp_reappear(obj, T_cw, self.K, m["mask"], rgb_np, depth_np)
                        if success:
                            obj.to_be_repaint = True
                            obj.pose_uncertain = False
                            T_ew = T_cw @ T_ee_camera
                            obj.T_oe = np.linalg.inv(T_ew) @ obj.pose_cur

            # initialize fixed points if entering grasping/holding/releasing (skipped here)
            # update object in gripper, child objects, etc. (skipped if no detection)
            if self.holding and self.objects and self.obj_id_in_ee is not None:
                # update_obj_pose_ee(self.objects, self.obj_id_in_ee, T_cw, T_ee_camera)
                update_child_objects_pose_icp(self.objects, self.obj_id_in_ee, self.relations, T_cw, T_ee_camera, self.K, masks, rgb_np, depth_np)

            # if self.use_icp and self.objects and self.obj_id_in_ee is not None:
            #     update_obj_pose_icp(self.objects, self.obj_id_in_ee, T_cw, T_ee_camera, self.K, masks, rgb_np, depth_np)
            #     update_child_objects_pose_icp(self.objects, self.obj_id_in_ee, self.relations, T_cw, T_ee_camera, self.K, masks, rgb_np, depth_np)
            if state == "releasing":
                obj = find_object_by_id(self.obj_id_in_ee, self.objects)
                if obj is not None:
                    obj.T_oe = None
                    
            # Re-appear update for uncertain objects
            if state != 'releasing' and pose_ok:
                for m in masks:
                    obj = find_object_by_id(m.get("id"), self.objects)
                    if obj is not None and obj.pose_uncertain:
                        success = icp_reappear(obj, T_cw, self.K, m['mask'], rgb_np, depth_np)
                        if success:
                            obj.to_be_repaint = True
                            obj.pose_uncertain = False


            # Associate / create object TSDFs (when not skipping fusion)
            has_new_obj = False
            if (not skip_fusion and state not in ("grasping","releasing")) or self.processed_frames <= 3:
                obj_voxel = self.cfg['tsdf'].get('object_voxel_size', 0.01)
                max_dist = self.cfg['tsdf'].get('max_distance_thresh', 0.03)
                has_new_obj = associate_by_id(masks, depth_np, rgb_np, self.K, T_cw, self.objects, self.processed_frames,
                                              voxel_size=obj_voxel, max_distance_thresh=max_dist, pose_ok=pose_ok)

            if has_new_obj:
                self.relations = get_relation_graph(self.objects, tolerance=0.02, overlap_threshold=0.3, verbose=False)

            # Background carve & integrate
            depth_bg, color_bg = depth_np.copy(), rgb_np.copy()
            none_mask = (depth_bg == 0)

            for mi in masks:
                m = (mi['mask'] > 0).astype(np.uint8)
                dil_k = int(self.cfg['mask'].get('dilate_kernel_size', 7))
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_k, dil_k))
                m = cv2.dilate(m, kernel, iterations=1).astype(bool)
                depth_bg[m] = 0
                color_bg[m] = 0

            # inpaint background (optional)
            inpaint_cfg = self.cfg.get('tsdf', {}).get('inpaint', {})
            if inpaint_cfg.get('enabled', False):
                depth_bg, color_bg = inpaint_background(depth_bg, color_bg, none_mask, hand_mask)

            # clear bottom strip
            img_h = depth_bg.shape[0]
            clear_h = int(img_h * self.cfg['mask'].get('clear_bottom_ratio', 0.1))
            depth_bg[img_h-clear_h:, :] = 0
            color_bg[img_h-clear_h:, :] = 0

            # remove hands from background
            depth_bg[hand_mask > 0] = 0
            color_bg[hand_mask > 0] = 0

            # integrate
            self.bg_tsdf.integrate(color_bg, depth_bg, self.K, invert(T_cw))

            # --- Render queue update ---
            render_queue = {"bg": None, "objects": {}, "bbox": {}, "poses": {}}

            V,F,N,C = self.bg_tsdf.get_mesh()
            if len(V):
                bg_mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(V,F,vertex_normals=N,vertex_colors=C,process=False))
                render_queue['bg'] = (bg_mesh, np.eye(4))

            # repaint objects whose mesh changed / flagged
            for i, obj in enumerate(self.objects):
                # if not getattr(obj, 'to_be_repaint', True):
                #     continue
                obj.to_be_repaint = False
                V,F,N,C = obj.tsdf.get_mesh()
                if not len(V):
                    continue
                if self.mesh_filter_enabled:
                    V,F,N,C = filter_mesh_fast(V,F,N,C, trim_ratio=self.trim_ratio)
                    if not len(V):
                        continue
                mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(V,F,vertex_normals=N,vertex_colors=C,process=False))
                T = obj.pose_cur @ invert(obj.pose_init)
                render_queue['objects'][i] = (mesh, T)

                # bbox / pose axes
                if self.show_bboxes:
                    bbox_mesh = self._make_bbox_mesh(V, T)
                    if bbox_mesh is not None:
                        render_queue['bbox'][i] = (bbox_mesh, np.eye(4))
                if self.show_obj_poses:
                    axes = self._make_pose_axes()
                    render_queue['poses'][i] = (axes, obj.pose_cur)

            # Draw to viewer atomically
            with self.rlock:
                # remove old
                for key in list(self.mesh_nodes.keys()):
                    try:
                        if isinstance(self.mesh_nodes[key], list):
                            for n in self.mesh_nodes[key]:
                                self.scene.remove_node(n)
                        else:
                            self.scene.remove_node(self.mesh_nodes[key])
                    except Exception:
                        pass
                self.mesh_nodes.clear()

                # add new
                if render_queue['bg'] is not None:
                    m,p = render_queue['bg']
                    self.mesh_nodes['bg'] = self.scene.add(m, pose=p)
                for i,(m,p) in render_queue['objects'].items():
                    self.mesh_nodes[f'obj{i}'] = self.scene.add(m, pose=p)
                for i,(m,p) in render_queue['bbox'].items():
                    self.mesh_nodes[f'bbox{i}'] = self.scene.add(m, pose=p)
                for i,(axes,pose) in render_queue['poses'].items():
                    nodes=[]
                    for ax in axes:
                        nodes.append(self.scene.add(ax, pose=pose))
                    self.mesh_nodes[f'pose{i}'] = nodes

            self.processed_frames += 1
            # rospy.loginfo_throttle(2.0, f"Processed frames: {self.processed_frames}")
            # Loop will now grab the newest topics again for the next frame

            rate.sleep()

    # --- helpers to create bbox & axes meshes ---
    def _make_bbox_mesh(self, V, T):
        V = np.asarray(V)
        V_h = np.hstack([V, np.ones((len(V),1))])
        Vw = (T @ V_h.T).T[:,:3]
        mn = Vw.min(axis=0); mx = Vw.max(axis=0)
        verts = np.array([
            [mn[0], mn[1], mn[2]],[mx[0], mn[1], mn[2]],[mx[0], mx[1], mn[2]],[mn[0], mx[1], mn[2]],
            [mn[0], mn[1], mx[2]],[mx[0], mn[1], mx[2]],[mx[0], mx[1], mx[2]],[mn[0], mx[1], mx[2]],
        ])
        if self.bbox_style == 'wireframe':
            edges = np.array([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]])
            tubes_v = []; tubes_f = []
            thick = self.bbox_edge_thickness
            for e0,e1 in edges:
                a = verts[e0]; b = verts[e1]; v = b-a; L = np.linalg.norm(v)
                if L <= 1e-8: continue
                d = v/L
                perp1 = np.array([-d[1], d[0], 0.0], dtype=np.float64) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
                perp1 /= (np.linalg.norm(perp1)+1e-12)
                perp2 = np.cross(d, perp1)
                h = thick/2
                edge_vs = []
                for t in [0,1]:
                    for p1 in [-1,1]:
                        for p2 in [-1,1]:
                            edge_vs.append(a + t*v + p1*h*perp1 + p2*h*perp2)
                si = len(tubes_v)
                tubes_v.extend(edge_vs)
                tubes_f = tubes_f + [
                    [si+0,si+1,si+2],[si+0,si+2,si+3],
                    [si+4,si+6,si+5],[si+4,si+7,si+6],
                    [si+0,si+4,si+5],[si+0,si+5,si+1],
                    [si+1,si+5,si+6],[si+1,si+6,si+2],
                    [si+2,si+6,si+7],[si+2,si+7,si+3],
                    [si+3,si+7,si+4],[si+3,si+4,si+0],
                ]
            if len(tubes_v) == 0:
                return None
            return pyrender.Mesh.from_trimesh(
                trimesh.Trimesh(vertices=np.array(tubes_v), faces=np.array(tubes_f), process=False),
                material=pyrender.material.MetallicRoughnessMaterial(
                    baseColorFactor=self.bbox_color, metallicFactor=0.0, roughnessFactor=0.5)
            )
        else:
            faces = np.array([
                [0,1,2],[0,2,3],[4,6,5],[4,7,6],
                [3,2,6],[3,6,7],[0,5,1],[0,4,5],
                [0,3,7],[0,7,4],[1,5,6],[1,6,2]
            ])
            return pyrender.Mesh.from_trimesh(
                trimesh.Trimesh(vertices=verts, faces=faces, process=False),
                material=pyrender.material.MetallicRoughnessMaterial(
                    baseColorFactor=self.bbox_color, metallicFactor=0.0, roughnessFactor=0.5)
            )

    def _make_pose_axes(self):
        L = self.pose_axis_length
        def box(ax_len, axis="x"):
            if axis == 'x':
                vs = np.array([[0,-0.001,-0.001],[ax_len,-0.001,-0.001],[ax_len,0.001,-0.001],[0,0.001,-0.001],
                               [0,-0.001,0.001],[ax_len,-0.001,0.001],[ax_len,0.001,0.001],[0,0.001,0.001]])
            elif axis == 'y':
                vs = np.array([[-0.001,0,-0.001],[-0.001,ax_len,-0.001],[0.001,ax_len,-0.001],[0.001,0,-0.001],
                               [-0.001,0,0.001],[-0.001,ax_len,0.001],[0.001,ax_len,0.001],[0.001,0,0.001]])
            else:
                vs = np.array([[-0.001,-0.001,0],[-0.001,-0.001,ax_len],[0.001,-0.001,ax_len],[0.001,-0.001,0],
                               [-0.001,0.001,0],[-0.001,0.001,ax_len],[0.001,0.001,ax_len],[0.001,0.001,0]])
            fs = np.array([[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,4,5],[0,5,1],[1,5,6],[1,6,2],[2,6,7],[2,7,3],[3,7,4],[3,4,0]])
            return vs, fs
        def mesh(vs, fs, color):
            return pyrender.Mesh.from_trimesh(
                trimesh.Trimesh(vertices=vs, faces=fs, process=False),
                material=pyrender.material.MetallicRoughnessMaterial(baseColorFactor=color, metallicFactor=0.0, roughnessFactor=0.5)
            )
        vx,fx = box(L,'x'); vy,fy = box(L,'y'); vz,fz = box(L,'z')
        x_axis = mesh(vx,fx,[1.0,0.0,0.0,1.0])
        y_axis = mesh(vy,fy,[0.0,1.0,0.0,1.0])
        z_axis = mesh(vz,fz,[0.0,0.0,1.0,1.0])
        return [x_axis, y_axis, z_axis]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="configs/realtime_app.yaml",
        help="Path to YAML config file"
    )
    args, unknown = parser.parse_known_args()  # 保留其他 ROS 参数

    app = RealtimeTSDFApp(args.config)
    try:
        app.run()
    except rospy.ROSInterruptException:
        pass
