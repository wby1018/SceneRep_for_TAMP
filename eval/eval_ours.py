#!/usr/bin/env python3
# filepath: /home/zhy/rosbag_to_dataset/eval_ours.py
import numpy as np
import os
from pathlib import Path
import pandas as pd
import json
import datetime
import cv2
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
import base64
import io
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d  # added for Open3D visualization
import time  # for small sleeps during visualization

# 路径配置
DATASET_DIR = "/media/zhy/bcd58cff-609f-4e23-89f6-9fc2e8b36fea/datasets/apple_1"  # 请替换为实际的数据集路径
EVAL_FILE = os.path.join(DATASET_DIR, "eval", "object_2.txt") # 请替换为实际的评估文件名
EVAL_CAM_FILE = os.path.join(DATASET_DIR, "eval", "camera.txt")
SEGMENT_FILE = os.path.join(DATASET_DIR, "eval", "segments.json")
BASE_FILE = os.path.join(DATASET_DIR, "pose_txt", "base_pose.txt")
TIMESTAMP_FILE = os.path.join(DATASET_DIR, "pose_txt", "timestamps.txt")
MOCAP_START_FILE = os.path.join(DATASET_DIR, "mocap_start.txt")
CSV_FILE = os.path.join(DATASET_DIR, "Take 2025-09-23 06.01.55 PM.csv")  # 请替换为实际的CSV文件名
DETECTION_DIR = os.path.join(DATASET_DIR, "detection_h")
OBJECT_NAME = "apple" # 请替换为实际的物体名称
OBJECT_ID = 2 # 请替换为实际的物体ID
MOCAP_ROBOT_FILE = os.path.join('eval', "transform_matrix.txt") 
T_HB_FILE = os.path.join('eval', "transform_hb_matrix.txt")


# ADD和ADD-S的阈值（单位：米）
ADD_THRESHOLD = 0.05
ADDS_THRESHOLD = 0.05

class PoseEvaluator:
    def __init__(self):
        # 读取动捕开始时间
        self.mocap_start_time = self.read_mocap_start_time()
        print(f"动捕开始时间: {self.mocap_start_time}")
        
        # 读取动捕CSV数据
        self.csv_data = self.read_csv_data()
        print(f"CSV数据加载完成，形状: {self.csv_data.shape}")

        # 读取相机位姿数据
        self.camera_poses = self.read_camera_poses()
        print(f"加载了 {len(self.camera_poses)} 个相机位姿")

        # 读取base位姿
        self.base_poses = self.read_base_poses()
        print(f"加载了 {len(self.base_poses)} 个base位姿")
        
        # 读取估计的姿态数据
        self.estimated_poses, self.evaluation_segments = self.read_estimated_poses()
        print(f"加载了 {len(self.estimated_poses)} 个估计姿态和 {len(self.evaluation_segments)} 个评估段")
        
        # 坐标系转换矩阵（第一帧初始化）
        self.mocap_robot = np.loadtxt(MOCAP_ROBOT_FILE)
        self.mocap_robot = np.linalg.inv(self.mocap_robot)
        self.obj_transformation = None
        self.camera_transformation = None
        self.first_flag = False

        # Open3D visualizer state (persistent across frames)
        self.o3d_vis = None  # Open3D Visualizer instance
        self.o3d_geoms = {}  # hold geometries for world, gt_cam, gt_obj, est_obj
        self.o3d_last = {}   # last transforms for incremental update
    
    def read_mocap_start_time(self):
        """读取动捕系统开始时间"""
        with open(MOCAP_START_FILE, 'r') as f:
            time_str = f.read().strip()
        dt = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
        # 转换为Unix时间戳（秒）
        return dt.timestamp()
    
    def read_camera_poses(self):
        """读取相机位姿数据"""
        camera_poses = {}
        with open(EVAL_CAM_FILE, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            parts = line.split()

            frame_idx = int(parts[0])

            position = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            quaternion = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])
            
            # 构建4x4变换矩阵
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
            transform[:3, 3] = position
            
            camera_poses[frame_idx] = transform
        return camera_poses
    
    def read_base_poses(self):
        """读取base位姿数据"""
        base_poses = {}
        with open(BASE_FILE, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            parts = line.split()

            frame_idx = int(parts[0])

            x = float(parts[1])
            y = float(parts[2])
            z = 0.0
            yaw = float(parts[3])
            position = np.array([x, y, z])
            quaternion = Rotation.from_euler('z', yaw).as_quat()  # 仅绕Z轴旋转
            
            # 构建4x4变换矩阵
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
            transform[:3, 3] = position
            
            base_poses[frame_idx] = transform
        return base_poses
        
    def read_csv_data(self):
        """读取CSV文件中的动捕数据"""
        # 跳过前两行，没有表头
        df = pd.read_csv(CSV_FILE, skiprows=2, header=None, low_memory=False)
        print("CSV文件前几行:")
        print(df.iloc[:10, :5])  # 打印前10行的前5列用于调试
        return df
    
    def read_estimated_poses(self):
        """
        1. 从 eval/segments.json 读取分段信息
        2. 从 eval/object_X.txt 读取位姿数据
        3. 从 pose_txt/timestamps.txt 读取时间戳
        """
        estimated_poses = {}
        segments = []
        
        # 读取分段信息
        if os.path.exists(SEGMENT_FILE):
            try:
                with open(SEGMENT_FILE, 'r') as f:
                    segments_data = json.load(f)
                
                # 从JSON解析分段
                for segment_info in segments_data:
                    start_frame = segment_info.get('start_frame', 0)
                    end_frame = segment_info.get('end_frame', 0)
                    
                    if start_frame <= end_frame:
                        segment = list(range(start_frame, end_frame + 1))
                        segments.append(segment)
                        print(f"添加评估段: 帧 {start_frame} 到 {end_frame} ({len(segment)} 帧)")
            except Exception as e:
                print(f"读取分段信息时出错: {e}")
                segments = []
        
        # 读取时间戳
        timestamps = {}
        try:
            with open(TIMESTAMP_FILE, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:  # 帧号 秒 纳秒
                        frame_idx = int(parts[0])
                        secs = int(parts[1])
                        nsecs = int(parts[2])
                        timestamps[frame_idx] = (secs, nsecs)
            print(f"从 {TIMESTAMP_FILE} 读取到 {len(timestamps)} 个时间戳")
        except Exception as e:
            print(f"读取时间戳文件时出错: {e}")
            return {}, segments
        
        # 读取姿态数据
        try:
            with open(EVAL_FILE, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                    
                # 解析姿态数据: 帧号 x y z qx qy qz qw
                parts = line.split()
                if len(parts) < 8:
                    print(f"警告: 行格式错误 '{line}'")
                    continue
                    
                try:
                    frame_idx = int(parts[0])
                    position = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                    quaternion = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])
                    
                    # 获取时间戳
                    timestamp = timestamps.get(frame_idx)

                    # 构建4x4变换矩阵
                    transform = np.eye(4)
                    transform[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
                    transform[:3, 3] = position
                    
                    estimated_poses[frame_idx] = {
                        'timestamp': timestamp,
                        'transform': transform
                    }
                except (ValueError, IndexError) as e:
                    print(f"解析行 '{line}' 时出错: {e}")
            
            print(f"从 {EVAL_FILE} 读取到 {len(estimated_poses)} 个姿态")
        except Exception as e:
            print(f"读取姿态文件时出错: {e}")
        
        # 过滤分段，仅保留有姿态数据的帧
        filtered_segments = []
        for segment in segments:
            filtered_segment = [idx for idx in segment if idx in estimated_poses]
            if filtered_segment:
                filtered_segments.append(filtered_segment)
        
        return estimated_poses, filtered_segments

        
    def find_nearest_mocap_idx(self, timestamp, last_nearest_idx=None):
        """根据时间戳找到最接近的动捕姿态"""
        secs, nsecs = timestamp
        target_time = secs + nsecs / 1e9 - self.mocap_start_time
        
        # 查找时间列（第二列，索引为1）
        min_diff = float('inf')
        nearest_row_idx = None
        start_idx = max(last_nearest_idx - 50, 5) if last_nearest_idx is not None else 5
        end_idx = min(start_idx + 200, len(self.csv_data)) if last_nearest_idx is not None else len(self.csv_data)
        
        for i in range(start_idx, end_idx):
            try:
                time_val = float(self.csv_data.iloc[i, 1])
                diff = abs(time_val - target_time)
                
                if diff < min_diff:
                    min_diff = diff
                    nearest_row_idx = i
            except (ValueError, TypeError):
                continue
        print(f"目标时间: {target_time:.6f}, 最近时间差: {min_diff:.6f} (行 {nearest_row_idx})")
                
        if nearest_row_idx is None:
            return None, min_diff
        return nearest_row_idx, min_diff

    def extract_mocap_pose(self, nearest_row_idx):
        # 提取苹果的7位姿态（位置和四元数）
        obj_pose_cols = self.find_obj_pose_columns()
        cam_pose_cols = self.find_cam_pose_columns()

        # try:
        obj_position = np.array([
            float(self.csv_data.iloc[nearest_row_idx, obj_pose_cols['x']]),
            float(self.csv_data.iloc[nearest_row_idx, obj_pose_cols['y']]),
            float(self.csv_data.iloc[nearest_row_idx, obj_pose_cols['z']])
        ])
        
        obj_quaternion = np.array([
            float(self.csv_data.iloc[nearest_row_idx, obj_pose_cols['qx']]),
            float(self.csv_data.iloc[nearest_row_idx, obj_pose_cols['qy']]),
            float(self.csv_data.iloc[nearest_row_idx, obj_pose_cols['qz']]),
            float(self.csv_data.iloc[nearest_row_idx, obj_pose_cols['qw']])
        ])
        mocap_obj_pose = np.eye(4)
        mocap_obj_pose[:3, :3] = Rotation.from_quat(obj_quaternion).as_matrix()
        mocap_obj_pose[:3, 3] = obj_position

        cam_position = np.array([
            float(self.csv_data.iloc[nearest_row_idx, cam_pose_cols['x']]),
            float(self.csv_data.iloc[nearest_row_idx, cam_pose_cols['y']]),
            float(self.csv_data.iloc[nearest_row_idx, cam_pose_cols['z']])
        ])
        cam_quaternion = np.array([
            float(self.csv_data.iloc[nearest_row_idx, cam_pose_cols['qx']]),
            float(self.csv_data.iloc[nearest_row_idx, cam_pose_cols['qy']]),
            float(self.csv_data.iloc[nearest_row_idx, cam_pose_cols['qz']]),
            float(self.csv_data.iloc[nearest_row_idx, cam_pose_cols['qw']])
        ])
        mocap_cam_pose = np.eye(4)
        mocap_cam_pose[:3, :3] = Rotation.from_quat(cam_quaternion).as_matrix()
        mocap_cam_pose[:3, 3] = cam_position
        
        return mocap_obj_pose, mocap_cam_pose
        
        # except (ValueError, TypeError, KeyError) as e:
        #     print(f"提取动捕姿态时出错: {e}")
        #     return None
    
    def find_cam_pose_columns(self):
        """找到CSV文件中相机位姿的列索引"""
        if hasattr(self, 'cam_cols'):
            return self.cam_cols
            
        self.cam_cols = {}
        
        for col in self.csv_data.columns:
            col_data = self.csv_data[col]
            if (col_data == "fetch").any() and (col_data == "Position").any() and (col_data == "X").any():
                self.cam_cols['x'] = col
            elif (col_data == "fetch").any() and (col_data == "Position").any() and (col_data == "Y").any():
                self.cam_cols['y'] = col
            elif (col_data == "fetch").any() and (col_data == "Position").any() and (col_data == "Z").any():
                self.cam_cols['z'] = col
            elif (col_data == "fetch").any() and (col_data == "Rotation").any() and (col_data == "X").any():
                self.cam_cols['qx'] = col
            elif (col_data == "fetch").any() and (col_data == "Rotation").any() and (col_data == "Y").any():
                self.cam_cols['qy'] = col
            elif (col_data == "fetch").any() and (col_data == "Rotation").any() and (col_data == "Z").any():
                self.cam_cols['qz'] = col
            elif (col_data == "fetch").any() and (col_data == "Rotation").any() and (col_data == "W").any():
                self.cam_cols['qw'] = col
        
        print(f"找到的相机位姿列: {self.cam_cols}")
        
        return self.cam_cols
            
    def find_obj_pose_columns(self):
        """找到CSV文件中苹果位姿的列索引"""
        if hasattr(self, 'obj_cols'):
            return self.obj_cols
            
        self.obj_cols = {}
        
        for col in self.csv_data.columns:
            col_data = self.csv_data[col]
            if (col_data == OBJECT_NAME).any() and (col_data == "Position").any() and (col_data == "X").any():
                self.obj_cols['x'] = col
            elif (col_data == OBJECT_NAME).any() and (col_data == "Position").any() and (col_data == "Y").any():
                self.obj_cols['y'] = col
            elif (col_data == OBJECT_NAME).any() and (col_data == "Position").any() and (col_data == "Z").any():
                self.obj_cols['z'] = col
            elif (col_data == OBJECT_NAME).any() and (col_data == "Rotation").any() and (col_data == "X").any():
                self.obj_cols['qx'] = col
            elif (col_data == OBJECT_NAME).any() and (col_data == "Rotation").any() and (col_data == "Y").any():
                self.obj_cols['qy'] = col
            elif (col_data == OBJECT_NAME).any() and (col_data == "Rotation").any() and (col_data == "Z").any():
                self.obj_cols['qz'] = col
            elif (col_data == OBJECT_NAME).any() and (col_data == "Rotation").any() and (col_data == "W").any():
                self.obj_cols['qw'] = col
        
        print(f"找到的苹果位姿列: {self.obj_cols}")
        
        return self.obj_cols
    
    def get_point_cloud(self, frame_idx, gt_cam_pose):
        """从深度图和掩码生成物体的点云"""
        # 构建文件路径
        rgb_path = os.path.join(DATASET_DIR, "rgb", f"rgb_{frame_idx:06d}.png")
        depth_path = os.path.join(DATASET_DIR, "depth", f"depth_{frame_idx:06d}.npy")
        json_path = os.path.join(DETECTION_DIR, f"detection_{frame_idx:06d}_final.json")
        
        # 检查文件是否存在
        if not all(os.path.exists(p) for p in [rgb_path, depth_path, json_path]):
            print(f"帧 {frame_idx}: 文件不完整")
            return None
        
        try:
            # 读取检测结果
            with open(json_path, 'r') as f:
                detection_data = json.load(f)
                detections = detection_data.get("detections", [])
            
            obj_detection = None
            for det in detections:
                if isinstance(det, dict) and det.get('object_id') == OBJECT_ID:
                    obj_detection = det
                    break
                    
            if not obj_detection:
                print(f"帧 {frame_idx}: 未检测到苹果")
                return None
            
            rgb = cv2.imread(rgb_path)
            depth = np.load(depth_path)
                
            # 解码mask
            mask_data = obj_detection.get('mask')
            mask_bytes = base64.b64decode(mask_data)
            mask_image = Image.open(io.BytesIO(mask_bytes))
            mask = np.array(mask_image)
            
            # 相机内参
            fx = 554.3827
            fy = 554.3827
            cx = 320
            cy = 240
            
            # 创建点云
            points = []
            # 使用掩码提取物体点
            for y in range(rgb.shape[0]):
                for x in range(rgb.shape[1]):
                    if mask[y, x] > 0:
                        if 0 <= y < depth.shape[0] and 0 <= x < depth.shape[1]:
                            d = depth[y, x]
                                
                            if d > 0.1 and d < 10.0:
                                X = (x - cx) * d / fx
                                Y = (y - cy) * d / fy
                                Z = d
                                points.append([X, Y, Z])
            points = np.array(points)
            # 将点云从相机坐标系转换到动捕坐标系
            points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
            points_mocap = (gt_cam_pose @ points_hom.T).T
            points = points_mocap[:, :3]
                
            return points
            
        except Exception as e:
            import traceback
            print(f"帧 {frame_idx} 处理时发生错误: {e}")
            traceback.print_exc()
            return None
    
    def compute_transformation_matrix(self, first_frame_idx):
        """计算动捕坐标系到估计坐标系的转换矩阵（使用第一帧）"""
        # 获取第一帧的估计姿态
        if first_frame_idx not in self.estimated_poses:
            print(f"错误: 找不到帧 {first_frame_idx} 的估计姿态")
            return False
        
        # 获取对应的动捕真实姿态
        nearest_idx, _ = self.find_nearest_mocap_idx(
            self.estimated_poses[first_frame_idx]['timestamp']
        )
        mocap_pose, mocap_head_pose = self.extract_mocap_pose(nearest_idx)

        # T_hb = np.linalg.inv(self.base_poses[first_frame_idx]) @ self.mocap_robot @ mocap_head_pose
        # np.savetxt(T_HB_FILE, T_hb)
        T_hb = np.loadtxt(T_HB_FILE)
        T_oc = np.linalg.inv(self.camera_poses[first_frame_idx]) @ self.estimated_poses[first_frame_idx]['transform']
        T_cw = (self.mocap_robot @ mocap_head_pose @ np.linalg.inv(T_hb)) @ np.linalg.inv(self.base_poses[first_frame_idx]) @ self.camera_poses[first_frame_idx]
        T_ow = T_cw @ T_oc

        # print(f"estimated pose:\n{self.estimated_poses[first_frame_idx]['transform']}")
        # print(f"camera pose:\n{self.camera_poses[first_frame_idx]}")
        # print(f"T_cw:\n{T_cw}")
        # print(f"T_ow:\n{T_ow}")


        # 计算从动捕坐标系到估计坐标系的变换矩阵
        # self.obj_transformation = np.linalg.inv(self.mocap_robot @ mocap_pose) @ self.estimated_poses[first_frame_idx]['transform']
        # self.camera_transformation = np.linalg.inv(self.mocap_robot @ mocap_head_pose) @ self.camera_poses[first_frame_idx]
        self.obj_transformation = np.linalg.inv(self.mocap_robot @ mocap_pose) @ T_ow
        self.camera_transformation = np.linalg.inv(self.mocap_robot @ mocap_head_pose) @ T_cw
        
        
        print(f"已计算转换矩阵:\n{self.obj_transformation, self.camera_transformation}")
        return True
        
    def calculate_add(self, points1, points2):
        """计算平均距离（ADD指标）"""
        if len(points1) != len(points2):
            raise ValueError("点云大小不匹配")
            
        # 计算对应点之间的欧氏距离
        dists = np.linalg.norm(points1 - points2, axis=1)
        return np.mean(dists)
        
    def calculate_adds(self, points1, points2):
        """计算最近点距离（ADD-S指标）"""
        # 为每个点找到最近的点
        dists = []
        for p1 in points1:
            # 计算到所有点的距离
            point_dists = np.linalg.norm(points2 - p1, axis=1)
            # 找到最小距离
            dists.append(np.min(point_dists))
            
        return np.mean(dists)

    # -------- Open3D visualization helpers --------
    def _init_open3d(self):
        """Initialize Open3D visualizer and geometries."""
        if self.o3d_vis is not None:
            return
        # create visualizer
        self.o3d_vis = o3d.visualization.Visualizer()
        self.o3d_vis.create_window(window_name="Poses (world, gt_cam, gt_obj, est_obj)", width=960, height=720, visible=True)
        # world coordinate frame (identity)
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.o3d_geoms['world'] = world_frame
        self.o3d_last['world'] = np.eye(4)
        self.o3d_vis.add_geometry(world_frame)
        # gt camera frame
        gt_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
        self.o3d_geoms['gt_cam'] = gt_cam
        self.o3d_last['gt_cam'] = np.eye(4)
        self.o3d_vis.add_geometry(gt_cam)
        # gt object frame
        gt_obj = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
        self.o3d_geoms['gt_obj'] = gt_obj
        self.o3d_last['gt_obj'] = np.eye(4)
        self.o3d_vis.add_geometry(gt_obj)
        # estimated object frame
        est_obj = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
        self.o3d_geoms['est_obj'] = est_obj
        self.o3d_last['est_obj'] = np.eye(4)
        self.o3d_vis.add_geometry(est_obj)
        # set initial view
        self.o3d_vis.poll_events()
        self.o3d_vis.update_renderer()

    def _update_open3d(self, world_T, gt_cam_T, gt_obj_T, est_obj_T):
        """Update frames with new transforms using incremental delta transform.
        All transforms are 4x4 homogeneous matrices.
        """
        # ensure visualizer is initialized
        self._init_open3d()
        # apply incremental transforms for each geometry to avoid recreation
        targets = {
            'world': world_T,
            'gt_cam': gt_cam_T,
            'gt_obj': gt_obj_T,
            'est_obj': est_obj_T
        }
        for name, T_new in targets.items():
            T_last = self.o3d_last.get(name, np.eye(4))
            # delta transform from last to new
            delta = np.linalg.inv(T_last) @ T_new
            self.o3d_geoms[name].transform(delta)
            self.o3d_last[name] = T_new.copy()
        self.o3d_vis.poll_events()
        self.o3d_vis.update_renderer()
        # small sleep to allow UI to refresh without blocking too much
        time.sleep(0.01)

    def _destroy_open3d(self):
        """Destroy visualizer when done."""
        if self.o3d_vis is not None:
            self.o3d_vis.destroy_window()
            self.o3d_vis = None
            self.o3d_geoms = {}
            self.o3d_last = {}
    
    def visualize_frame_open3d(self, world_T, gt_cam_T, est_cam_T, gt_obj_T, est_obj_T):
        """逐帧可视化：创建一次窗口并阻塞，关闭后返回。"""
        # 四个坐标系
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        gt_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
        est_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
        gt_obj = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        est_obj = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        # 颜色标记：为 obj 的 gt/est 添加彩色球体以区分
        sphere_gt = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sphere_gt.compute_vertex_normals()
        sphere_gt.paint_uniform_color([0.0, 0.8, 0.0])  # 绿色: gt obj
        sphere_est = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sphere_est.compute_vertex_normals()
        sphere_est.paint_uniform_color([1.0, 0.85, 0.0])  # 黄色: est obj
        # 应用绝对位姿
        world_frame.transform(world_T)
        gt_cam.transform(gt_cam_T)
        est_cam.transform(est_cam_T)
        gt_obj.transform(gt_obj_T)
        est_obj.transform(est_obj_T)
        sphere_gt.transform(gt_obj_T)
        sphere_est.transform(est_obj_T)
        # 阻塞式显示，用户关闭窗口后进入下一帧
        o3d.visualization.draw_geometries(
            [world_frame, gt_cam, est_cam, gt_obj, est_obj, sphere_gt, sphere_est],
            window_name="Frame poses (world, gt_cam, gt_obj, est_obj)",
            width=960,
            height=720
        )
         
    def visualize_poses(self, est_poses, mocap_poses, save_path=None):
        """
        在3D中可视化两个姿态序列，XYZ 同尺度显示
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # 提取轨迹
        est_traj = np.array([pose[:3, 3] for pose in est_poses])
        mocap_traj = np.array([pose[:3, 3] for pose in mocap_poses])

        # 绘制轨迹
        ax.plot(est_traj[:, 0], est_traj[:, 1], est_traj[:, 2], 
                'b-', label="Estimated Trajectory")
        ax.plot(mocap_traj[:, 0], mocap_traj[:, 1], mocap_traj[:, 2], 
                'r-', label="MoCap Trajectory")

        # 绘制关键帧坐标系
        def draw_axes(pose, ax, length=0.05):
            origin = pose[:3, 3]
            R = pose[:3, :3]
            x_axis = origin + R[:, 0] * length
            y_axis = origin + R[:, 1] * length
            z_axis = origin + R[:, 2] * length
            ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], 'r')
            ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], 'g')
            ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], 'b')

        # 每隔 N 帧画一个坐标系
        step = max(1, len(est_poses)//72)
        for i in range(0, len(est_poses), step):
            draw_axes(est_poses[i], ax, length=0.03)
            draw_axes(mocap_poses[i], ax, length=0.03)

        # 设置坐标轴标签
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title("Estimated vs MoCap Trajectories")
        ax.legend()
        ax.grid(True)

        # === 保持等比例缩放 ===
        all_points = np.vstack((est_traj, mocap_traj))
        max_range = (all_points.max(axis=0) - all_points.min(axis=0)).max() / 2.0
        mid_x = (all_points[:,0].max() + all_points[:,0].min()) / 2.0
        mid_y = (all_points[:,1].max() + all_points[:,1].min()) / 2.0
        mid_z = (all_points[:,2].max() + all_points[:,2].min()) / 2.0
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # 如果 matplotlib 版本支持，可以直接用
        try:
            ax.set_box_aspect([1,1,1])  # 保证 x,y,z 比例相同
        except:
            pass

        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
        
        
    def evaluate_segment(self, segment):
        """评估一个时间段内的姿态估计"""
        # 使用第一帧计算坐标系转换矩阵
        first_frame_idx = segment[0]
        nearest_idx = None
        if self.first_flag == False:
            self.compute_transformation_matrix(first_frame_idx)
            self.first_flag = True
            # 非实时模式：无需初始化持久化窗口
            
        results = {
            'add_values': [],
            'adds_values': [],
            'add_correct': 0,
            'adds_correct': 0,
            'total_frames': 0
        }

        est_poses = []
        mocap_poses = []
        
        for frame_idx in segment:        
            # 获取估计姿态
            est_pose = self.estimated_poses[frame_idx]['transform']
            
            # 获取对应的动捕真实姿态
            nearest_idx, _ = self.find_nearest_mocap_idx(
                self.estimated_poses[frame_idx]['timestamp'], nearest_idx
            )
            mocap_obj_pose, mocap_cam_pose = self.extract_mocap_pose(nearest_idx)
                
            # 将动捕姿态转换到估计坐标系
            gt_obj_pose = self.mocap_robot @ mocap_obj_pose @ self.obj_transformation
            gt_cam_pose = self.mocap_robot @ mocap_cam_pose @ self.camera_transformation
            # gt_obj_pose = self.mocap_robot @ mocap_obj_pose
            # gt_cam_pose = self.mocap_robot @ mocap_cam_pose
            # print(est_pose, "\n", self.mocap_robot @ mocap_obj_pose, "\n", gt_obj_pose)

            # Open3D 逐帧可视化：每帧弹窗一次，关闭后继续
            # try:
            #     if frame_idx > -1: self.visualize_frame_open3d(np.eye(4), gt_cam_pose, self.camera_poses[frame_idx], gt_obj_pose, est_pose)
            #     print(gt_obj_pose, "\n", est_pose)
            # except Exception as e:
            #     print(f"Open3D 可视化失败: {e}")

            # 获取点云
            point_cloud = self.get_point_cloud(frame_idx, gt_cam_pose)
            if point_cloud is None or len(point_cloud) < 10:  # 至少需要10个点
                print(f"跳过帧 {frame_idx}: 点云无效")
                continue
            est_poses.append(est_pose)
            mocap_poses.append(gt_obj_pose)
            
            # 将点云转换到两个姿态下
            points_est = self.transform_points(point_cloud, est_pose @ np.linalg.inv(gt_cam_pose))
            points_mocap = self.transform_points(point_cloud, gt_obj_pose @ np.linalg.inv(gt_cam_pose))
            # print(points_est)
            # print(points_mocap)
            
            # 计算ADD和ADD-S
            add_value = self.calculate_add(points_est, points_mocap)
            adds_value = self.calculate_adds(points_est, points_mocap)
            
            results['add_values'].append(add_value)
            results['adds_values'].append(adds_value)
            results['total_frames'] += 1
            
            if add_value < ADD_THRESHOLD:
                results['add_correct'] += 1
            if adds_value < ADDS_THRESHOLD:
                results['adds_correct'] += 1
            
            print(f"帧 {frame_idx}: ADD={add_value:.4f}, ADD-S={adds_value:.4f}")
            
        # 计算成功率
        if results['total_frames'] > 0:
            results['add_success_rate'] = results['add_correct'] / results['total_frames']
            results['adds_success_rate'] = results['adds_correct'] / results['total_frames']
            results['add_mean'] = np.mean(results['add_values'])
            results['adds_mean'] = np.mean(results['adds_values'])

        self.visualize_poses(est_poses, mocap_poses)
        
        return results
        
    def transform_points(self, points, transform):
        """将点云应用变换矩阵"""
        # 转换为齐次坐标
        points_homogeneous = np.hstack((points, np.ones((len(points), 1))))
        
        # 应用变换
        transformed_points = (np.linalg.inv(transform) @ points_homogeneous.T).T
        
        # 返回3D坐标
        return transformed_points[:, :3]
    
        
    def evaluate(self):
        """评估所有时间段"""
        all_results = []
        
        for i, segment in enumerate(self.evaluation_segments):
            print(f"\n评估段 {i+1}/{len(self.evaluation_segments)}")
            results = self.evaluate_segment(segment)
            if results:
                all_results.append(results)
                
                # 打印段结果
                print(f"段 {i+1} 结果:")
                print(f"帧数: {results['total_frames']}")
                print(f"ADD 平均值: {results['add_mean']:.4f}")
                print(f"ADD-S 平均值: {results['adds_mean']:.4f}")
                print(f"ADD 成功率: {results['add_success_rate']*100:.2f}%")
                print(f"ADD-S 成功率: {results['adds_success_rate']*100:.2f}%")
        
        # 计算总体结果
        if all_results:
            total_frames = sum(r['total_frames'] for r in all_results)
            add_correct = sum(r['add_correct'] for r in all_results)
            adds_correct = sum(r['adds_correct'] for r in all_results)
            
            all_add_values = []
            all_adds_values = []
            for r in all_results:
                all_add_values.extend(r['add_values'])
                all_adds_values.extend(r['adds_values'])
            
            print("\n总体结果:")
            print(f"总帧数: {total_frames}")
            print(f"ADD 平均值: {np.mean(all_add_values):.4f}")
            print(f"ADD-S 平均值: {np.mean(all_adds_values):.4f}")
            print(f"ADD 成功率: {add_correct/total_frames*100:.2f}%")
            print(f"ADD-S 成功率: {adds_correct/total_frames*100:.2f}%")
            
            # 保存结果到文件
            self.save_results({
                'total_frames': total_frames,
                'add_mean': float(np.mean(all_add_values)),
                'adds_mean': float(np.mean(all_adds_values)),
                'add_success_rate': float(add_correct/total_frames),
                'adds_success_rate': float(adds_correct/total_frames)
            })
            
    def save_results(self, results):
        """保存评估结果到文件"""
        output_file = os.path.join(DATASET_DIR, "eval", "evaluation_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"结果已保存到 {output_file}")


def main():
    evaluator = PoseEvaluator()
    evaluator.evaluate()

if __name__ == "__main__":
    main()
