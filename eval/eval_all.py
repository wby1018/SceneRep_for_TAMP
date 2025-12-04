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
from scipy.spatial import cKDTree

import re
import glob

def auto_find_take_csv(takes_dir, timestamps_file):
    """
    自动匹配 takes 文件夹中与 timestamps.txt 的 ros 时间最接近的 CSV 文件
    """
    # 1. 读取 ROS 第一帧时间
    with open(timestamps_file, "r") as f:
        line = f.readline().strip()
    parts = line.split()
    ros_secs = int(parts[1])
    ros_nsecs = int(parts[2])
    ros_time = ros_secs + ros_nsecs * 1e-9  # Unix 时间戳

    ros_datetime = datetime.datetime.fromtimestamp(ros_time)
    print(f"[auto_find_take_csv] 第一帧 ROS 时间: {ros_time} (Unix seconds)")
    print(f"[auto_find_take_csv] 第一帧 ROS 日期时间: {ros_datetime}")

    # 2. 遍历 takes/ 下的所有 CSV
    csv_files = glob.glob(os.path.join(takes_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"未找到 takes 下的 CSV 文件：{takes_dir}")

    best_csv = None
    best_time_diff = float("inf")

    # 3. 解析每个 CSV 文件名的时间
    for csv_path in csv_files:
        name = os.path.basename(csv_path)

        # 匹配: Take 2025-09-25 04.43.51 PM.csv
        m = re.match(r"Take (\d{4}-\d{2}-\d{2}) (\d{2}\.\d{2}\.\d{2}) (AM|PM)\.csv", name)
        if not m:
            print(f"[auto_find_take_csv] 跳过无法解析的文件名: {name}")
            continue

        date_str = m.group(1)                # 2025-09-25
        time_str = m.group(2)                # 04.43.51
        ampm = m.group(3)                    # PM

        # 转成标准时间
        time_str2 = time_str.replace(".", ":")  # "04:43:51"
        datetime_str = f"{date_str} {time_str2} {ampm}"

        # 注意这里用 %I（12 小时制）+ %p
        dt = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %I:%M:%S %p")
        dt_unix = dt.timestamp()

        diff = abs(dt_unix - ros_time)
        # print(f"[auto_find_take_csv] 检查 {name}: {dt} diff={diff}")

        # 找到最接近的
        if diff < best_time_diff:
            best_time_diff = diff
            best_csv = csv_path

    if best_csv is None:
        raise RuntimeError("未找到匹配的 Take CSV 文件！")

    print(f"[auto_find_take_csv] 选中了: {best_csv}")
    return best_csv


def compute_mocap_start_time_from_releases(releases_path, ros_first_timestamp):
    """
    从 releases.txt 中选出与 ROS 第一帧时间最接近的点击时间
    作为最精确的 mocap 开始时间。
    """
    if not os.path.exists(releases_path):
        raise FileNotFoundError(f"releases.txt 未找到: {releases_path}")

    ros_secs, ros_nsecs = ros_first_timestamp
    ros_time = ros_secs + ros_nsecs * 1e-9  # float unix seconds

    best_time = None
    best_diff = float("inf")

    with open(releases_path, "r") as f:
        for line in f:
            line = line.strip()

            # 解析前面的时间戳
            m = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)", line)
            if not m:
                continue

            dt = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")
            t = dt.timestamp()

            diff = abs(t - ros_time)

            # 打印调试
            # print(f"[releases] 检查 {m.group(1)} diff={diff}")

            if diff < best_diff:
                best_diff = diff
                best_time = t

    if best_time is None:
        raise RuntimeError("未能从 releases.txt 提取任何时间戳！")

    best_dt = datetime.datetime.fromtimestamp(best_time)
    print(f"[mocap] 选中的动捕开始时间: {best_dt}  diff={best_diff}")

    return best_time



class PoseEvaluator:
    def __init__(self, dataset_dir):
        # 设置所有路径和配置变量为实例变量
        self.dataset_dir = dataset_dir
        self.eval_file = os.path.join(self.dataset_dir, "eval", "object_2.txt")
        self.eval_cam_file = os.path.join(self.dataset_dir, "eval", "camera.txt")
        self.segment_file = os.path.join(self.dataset_dir, "eval", "segments.json")
        self.base_file = os.path.join(self.dataset_dir, "pose_txt", "base_pose.txt")
        self.timestamp_file = os.path.join(self.dataset_dir, "pose_txt", "timestamps.txt")
        self.takes_dir = "/media/wby/6d811df4-bde7-479b-ab4c-679222653ea0/takes_cleaned_2"
        self.releases_file = "/media/wby/6d811df4-bde7-479b-ab4c-679222653ea0/takes/releases.txt"
        self.detection_dir = os.path.join(self.dataset_dir, "detection_h")
        self.object_name = "apple"
        self.object_id = 2
        self.obj_points = None
        self.mocap_robot_file = os.path.join("transform_matrix.txt")
        
        # ADD和ADD-S的阈值（单位：米）
        self.add_threshold = 0.05
        self.adds_threshold = 0.05
        
        # 读取 timestamps.txt 第一行
        with open(self.timestamp_file, "r") as f:
            line = f.readline().strip().split()
        ros_first_timestamp = (int(line[1]), int(line[2]))
        
        # 自动查找匹配的CSV文件
        self.csv_file = auto_find_take_csv(self.takes_dir, self.timestamp_file)

        self.mocap_start_time = compute_mocap_start_time_from_releases(
            self.releases_file,
            ros_first_timestamp
        )

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
        self.mocap_robot = np.loadtxt(self.mocap_robot_file)
        self.mocap_robot = np.linalg.inv(self.mocap_robot)
        self.obj_transformation = None
        self.camera_transformation = None
        self.first_flag = False

        # Open3D visualizer state (persistent across frames)
        self.o3d_vis = None  # Open3D Visualizer instance
        self.o3d_geoms = {}  # hold geometries for world, gt_cam, gt_obj, est_obj
        self.o3d_last = {}   # last transforms for incremental update

        self.mocap_start_time_offset = self.find_mocap_start_time_offset()
        # self.mocap_start_time_offset = 0.13
    
    def _evaluate_offset(self, offset, segments_to_eval):
        """
        评估给定 offset 的总 add_sum（辅助函数，用于优化）
        """
        # 重置状态
        self.mocap_start_time_offset = offset
        self.first_flag = False
        self.obj_transformation = None
        self.camera_transformation = None
        
        # 评估所有 segments 并计算总 add_sum
        total_add_sum = 0.0
        try:
            for segment in segments_to_eval:
                _, add_sum = self.evaluate_segment(segment)
                total_add_sum += add_sum
            return total_add_sum
        except Exception as e:
            print(f"Offset {offset:.4f}s 评估时出错: {e}")
            return float('inf')
    
    def find_mocap_start_time_offset(self):
        """
        通过调整 mocap_start_time_offset 来最小化所有 segments 的总 add_sum
        使用三分搜索（Ternary Search）找到最优的 offset 值
        """
        print("\n开始优化 mocap_start_time_offset（使用三分搜索）...")
        
        # 保存初始状态
        original_first_flag = self.first_flag
        original_obj_transformation = self.obj_transformation
        original_camera_transformation = self.camera_transformation
        
        # 搜索范围
        left = 0.05
        right = 0.2
        tolerance = 0.01  # 精度：0.001秒
        
        # 如果评估段太多，只使用第一个段来加速
        segments_to_eval = self.evaluation_segments[:1] if len(self.evaluation_segments) > 1 else self.evaluation_segments
        print(f"使用 {len(segments_to_eval)} 个评估段进行优化")
        print(f"搜索范围: [{left:.3f}, {right:.3f}] 秒, 精度: {tolerance:.3f} 秒")
        
        # 三分搜索
        iteration = 0
        while right - left > tolerance:
            iteration += 1
            m1 = left + (right - left) / 3
            m2 = right - (right - left) / 3
            
            f1 = self._evaluate_offset(m1, segments_to_eval)
            f2 = self._evaluate_offset(m2, segments_to_eval)
            
            print(f"迭代 {iteration}: left={left:.4f}, m1={m1:.4f} (sum={f1:.4f}), m2={m2:.4f} (sum={f2:.4f}), right={right:.4f}")
            
            if f1 < f2:
                right = m2
            else:
                left = m1
        
        # 最终的最优值在 [left, right] 区间中点
        best_offset = (left + right) / 2
        best_add_sum = self._evaluate_offset(best_offset, segments_to_eval)
        
        print(f"\n最优 offset: {best_offset:.4f}s, 最小 ADD sum: {best_add_sum:.4f} (共 {iteration} 次迭代)")
        
        # 恢复状态并设置最优 offset
        self.mocap_start_time_offset = best_offset
        self.first_flag = original_first_flag
        self.obj_transformation = original_obj_transformation
        self.camera_transformation = original_camera_transformation
        
        return best_offset
        
    
    def read_camera_poses(self):
        """读取相机位姿数据"""
        camera_poses = {}
        with open(self.eval_cam_file, 'r') as f:
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
        with open(self.base_file, 'r') as f:
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
        df = pd.read_csv(self.csv_file, skiprows=2, header=None, low_memory=False)
        # df = pd.read_csv(
        #     CSV_FILE,
        #     skiprows=2,
        #     header=None,
        #     low_memory=False,
        #     encoding="latin1",          # 解决编码问题
        #     encoding_errors="ignore"    # 遇到坏字节自动跳过
        # )

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
        segment = list(range(0, 5000))
        segments.append(segment)
        
        # 读取分段信息
        if os.path.exists(self.segment_file):
            try:
                with open(self.segment_file, 'r') as f:
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
            with open(self.timestamp_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:  # 帧号 秒 纳秒
                        frame_idx = int(parts[0])
                        secs = int(parts[1])
                        nsecs = int(parts[2])
                        timestamps[frame_idx] = (secs, nsecs)
            print(f"从 {self.timestamp_file} 读取到 {len(timestamps)} 个时间戳")
        except Exception as e:
            print(f"读取时间戳文件时出错: {e}")
            return {}, segments
        
        # 读取姿态数据
        try:
            with open(self.eval_file, 'r') as f:
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
            
            print(f"从 {self.eval_file} 读取到 {len(estimated_poses)} 个姿态")
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
        target_time = secs + nsecs / 1e9 - (self.mocap_start_time + self.mocap_start_time_offset)
        
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
        # print(f"目标时间: {target_time:.6f}, 最近时间差: {min_diff:.6f} (行 {nearest_row_idx})")
                
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
            if (col_data == self.object_name).any() and (col_data == "Position").any() and (col_data == "X").any():
                self.obj_cols['x'] = col
            elif (col_data == self.object_name).any() and (col_data == "Position").any() and (col_data == "Y").any():
                self.obj_cols['y'] = col
            elif (col_data == self.object_name).any() and (col_data == "Position").any() and (col_data == "Z").any():
                self.obj_cols['z'] = col
            elif (col_data == self.object_name).any() and (col_data == "Rotation").any() and (col_data == "X").any():
                self.obj_cols['qx'] = col
            elif (col_data == self.object_name).any() and (col_data == "Rotation").any() and (col_data == "Y").any():
                self.obj_cols['qy'] = col
            elif (col_data == self.object_name).any() and (col_data == "Rotation").any() and (col_data == "Z").any():
                self.obj_cols['qz'] = col
            elif (col_data == self.object_name).any() and (col_data == "Rotation").any() and (col_data == "W").any():
                self.obj_cols['qw'] = col
        
        print(f"找到的苹果位姿列: {self.obj_cols}")
        
        return self.obj_cols
    
    def get_point_cloud(self, frame_idx, gt_cam_pose):
        """从深度图和掩码生成物体的点云"""
        # 构建文件路径
        rgb_path = os.path.join(self.dataset_dir, "rgb", f"rgb_{frame_idx:06d}.png")
        depth_path = os.path.join(self.dataset_dir, "depth", f"depth_{frame_idx:06d}.npy")
        json_path = os.path.join(self.detection_dir, f"detection_{frame_idx:06d}_final.json")
        
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
                if isinstance(det, dict) and det.get('object_id') == self.object_id:
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
            
            # # 创建点云
            # points = []
            # # 使用掩码提取物体点
            # for y in range(rgb.shape[0]):
            #     for x in range(rgb.shape[1]):
            #         if mask[y, x] > 0:
            #             if 0 <= y < depth.shape[0] and 0 <= x < depth.shape[1]:
            #                 d = depth[y, x]
                                
            #                 if d > 0.1 and d < 10.0:
            #                     X = (x - cx) * d / fx
            #                     Y = (y - cy) * d / fy
            #                     Z = d
            #                     points.append([X, Y, Z])
            # points = np.array(points)
            # # 将点云从相机坐标系转换到动捕坐标系
            # points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
            # points_mocap = (gt_cam_pose @ points_hom.T).T
            # points = points_mocap[:, :3]
# ----------------------------
# Vectorized point cloud creation
# ----------------------------

            # 获取所有 mask > 0 的像素坐标
            ys, xs = np.where(mask > 0)

            # 对应深度
            ds = depth[ys, xs]

            # 深度合法性过滤
            valid = (ds > 0.1) & (ds < 10.0)
            xs = xs[valid]
            ys = ys[valid]
            ds = ds[valid]

            # 反投影到相机坐标系
            X = (xs - cx) * ds / fx
            Y = (ys - cy) * ds / fy
            Z = ds

            # 组合成点云 Nx3
            points = np.stack([X, Y, Z], axis=1)

            # ---------- 坐标变换 ----------
            points_hom = np.hstack([points, np.ones((points.shape[0], 1))])
            points_mocap = (gt_cam_pose @ points_hom.T).T
            points = points_mocap[:, :3]
            # print(points)

                
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

        T_hb = np.linalg.inv(self.base_poses[first_frame_idx]) @ self.mocap_robot @ mocap_head_pose
        # np.savetxt(T_HB_FILE, T_hb)
        # T_hb = np.loadtxt(T_HB_FILE)
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
        self.obj_points = self.get_point_cloud(first_frame_idx, T_cw)
        self.obj_points = self.transform_points(self.obj_points, np.linalg.inv(T_ow))
        # print(self.obj_points)
        
        
        print(f"已计算转换矩阵:\n{self.obj_transformation, self.camera_transformation}")
        return True
        
    def calculate_add(self, points1, points2):
        """计算平均距离（ADD指标）"""
        if len(points1) != len(points2):
            raise ValueError("点云大小不匹配")
            
        # 计算对应点之间的欧氏距离
        dists = np.linalg.norm(points1 - points2, axis=1)
        return np.mean(dists)
        
    # def calculate_adds(self, points1, points2):
    #     """计算最近点距离（ADD-S指标）"""
    #     # 为每个点找到最近的点
    #     dists = []
    #     for p1 in points1:
    #         # 计算到所有点的距离
    #         point_dists = np.linalg.norm(points2 - p1, axis=1)
    #         # 找到最小距离
    #         dists.append(np.min(point_dists))
            
    #     return np.mean(dists)


    def calculate_adds(self, points1, points2):
        tree = cKDTree(points2)
        dists, _ = tree.query(points1, k=1)
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
         

    def visualize_poses(self, est_poses, mocap_poses, gt_cam_poses=None, est_cam_poses=None):
        """
        使用 Open3D 在 3D 中可视化两个姿态序列：
        - 蓝色折线：Estimated 物体轨迹
        - 红色折线：MoCap 物体轨迹
        - 青色折线：GT 相机轨迹
        - 紫色折线：Estimated 相机轨迹
        - 每隔若干帧画一对坐标系（est & mocap 物体，以及相机）
        """
        if len(est_poses) == 0 or len(mocap_poses) == 0:
            print("可视化失败：est_poses 或 mocap_poses 为空")
            return

        # 轨迹点
        est_traj = np.array([pose[:3, 3] for pose in est_poses])
        mocap_traj = np.array([pose[:3, 3] for pose in mocap_poses])

        geometries = []

        # 世界坐标系
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        # geometries.append(world_frame)

        # ===== 1) 估计物体轨迹（蓝色折线） =====
        est_lines = [[i, i + 1] for i in range(len(est_traj) - 1)]
        est_ls = o3d.geometry.LineSet()
        est_ls.points = o3d.utility.Vector3dVector(est_traj)
        est_ls.lines = o3d.utility.Vector2iVector(est_lines)
        est_color = np.array([[0.0, 0.0, 1.0] for _ in est_lines])  # 蓝色
        est_ls.colors = o3d.utility.Vector3dVector(est_color)
        geometries.append(est_ls)

        # ===== 2) MoCap 物体轨迹（红色折线） =====
        mocap_lines = [[i, i + 1] for i in range(len(mocap_traj) - 1)]
        mocap_ls = o3d.geometry.LineSet()
        mocap_ls.points = o3d.utility.Vector3dVector(mocap_traj)
        mocap_ls.lines = o3d.utility.Vector2iVector(mocap_lines)
        mocap_color = np.array([[1.0, 0.0, 0.0] for _ in mocap_lines])  # 红色
        mocap_ls.colors = o3d.utility.Vector3dVector(mocap_color)
        geometries.append(mocap_ls)

        # ===== 3) GT 相机轨迹（青色折线） =====
        if gt_cam_poses is not None and len(gt_cam_poses) > 0:
            gt_cam_traj = np.array([pose[:3, 3] for pose in gt_cam_poses])
            if len(gt_cam_traj) > 1:
                gt_cam_lines = [[i, i + 1] for i in range(len(gt_cam_traj) - 1)]
                gt_cam_ls = o3d.geometry.LineSet()
                gt_cam_ls.points = o3d.utility.Vector3dVector(gt_cam_traj)
                gt_cam_ls.lines = o3d.utility.Vector2iVector(gt_cam_lines)
                gt_cam_color = np.array([[0.0, 1.0, 1.0] for _ in gt_cam_lines])  # 青色
                gt_cam_ls.colors = o3d.utility.Vector3dVector(gt_cam_color)
                geometries.append(gt_cam_ls)

        # ===== 4) Estimated 相机轨迹（紫色折线） =====
        if est_cam_poses is not None and len(est_cam_poses) > 0:
            est_cam_traj = np.array([pose[:3, 3] for pose in est_cam_poses])
            if len(est_cam_traj) > 1:
                est_cam_lines = [[i, i + 1] for i in range(len(est_cam_traj) - 1)]
                est_cam_ls = o3d.geometry.LineSet()
                est_cam_ls.points = o3d.utility.Vector3dVector(est_cam_traj)
                est_cam_ls.lines = o3d.utility.Vector2iVector(est_cam_lines)
                est_cam_color = np.array([[1.0, 0.0, 1.0] for _ in est_cam_lines])  # 紫色
                est_cam_ls.colors = o3d.utility.Vector3dVector(est_cam_color)
                geometries.append(est_cam_ls)

        # ===== 5) 关键帧坐标系 =====
        step = max(1, len(est_poses) // 72)
        for idx in range(0, len(est_poses), step):
            # est 物体坐标系（绿色）
            frame_est = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            frame_est.paint_uniform_color([0.0, 0.8, 0.0])
            frame_est.transform(est_poses[idx])
            geometries.append(frame_est)

            # mocap 物体坐标系（黄色）
            frame_gt = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            frame_gt.paint_uniform_color([1.0, 0.85, 0.0])
            frame_gt.transform(mocap_poses[idx])
            geometries.append(frame_gt)
            
            # GT 相机坐标系（青色，稍小）
            if gt_cam_poses is not None and idx < len(gt_cam_poses):
                frame_gt_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
                frame_gt_cam.paint_uniform_color([0.0, 1.0, 1.0])
                frame_gt_cam.transform(gt_cam_poses[idx])
                geometries.append(frame_gt_cam)
            
            # Estimated 相机坐标系（紫色，稍小）
            if est_cam_poses is not None and idx < len(est_cam_poses):
                frame_est_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
                frame_est_cam.paint_uniform_color([1.0, 0.0, 1.0])
                frame_est_cam.transform(est_cam_poses[idx])
                geometries.append(frame_est_cam)

        # ===== 6) 显示 =====
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Estimated vs MoCap Trajectories (Objects & Cameras)",
            width=960,
            height=720,
        )

        
        
    def evaluate_segment(self, segment):
        """评估一个时间段内的姿态估计"""
        # 使用第一帧计算坐标系转换矩阵
        first_frame_idx = segment[0]
        nearest_idx = None
        if self.first_flag == False:
            self.compute_transformation_matrix(first_frame_idx)
            self.first_flag = False
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
        gt_cam_poses = []
        est_cam_poses = []
        add_sum = 0
        
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
            est_cam_pose = self.camera_poses[frame_idx]
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
            gt_cam_poses.append(gt_cam_pose)
            est_cam_poses.append(est_cam_pose)

            est_pose = np.linalg.inv(est_cam_pose) @ est_pose
            gt_obj_pose = np.linalg.inv(gt_cam_pose) @ gt_obj_pose
            
            # 将点云转换到两个姿态下
            # points_est = self.transform_points(point_cloud, est_pose @ np.linalg.inv(gt_cam_pose))
            # points_mocap = self.transform_points(point_cloud, gt_obj_pose @ np.linalg.inv(gt_cam_pose))
            points_est = self.transform_points(self.obj_points, est_pose)
            points_mocap = self.transform_points(self.obj_points, gt_obj_pose)
            # print(points_est)
            # print(points_mocap)
            
            # 计算ADD和ADD-S
            add_value = self.calculate_add(points_est, points_mocap)
            adds_value = self.calculate_adds(points_est, points_mocap)
            
            results['add_values'].append(add_value)
            results['adds_values'].append(adds_value)
            results['total_frames'] += 1
            
            if add_value < self.add_threshold:
                results['add_correct'] += 1
            if adds_value < self.adds_threshold:
                results['adds_correct'] += 1
            
            print(f"帧 {frame_idx}: ADD={add_value:.4f}, ADD-S={adds_value:.4f}")
            add_sum += add_value
        # 计算成功率
        if results['total_frames'] > 0:
            results['add_success_rate'] = results['add_correct'] / results['total_frames']
            results['adds_success_rate'] = results['adds_correct'] / results['total_frames']
            results['add_mean'] = np.mean(results['add_values'])
            results['adds_mean'] = np.mean(results['adds_values'])

        self.visualize_poses(est_poses, mocap_poses, gt_cam_poses, est_cam_poses)
        
        return results, add_sum

    def evaluate_segment_foudation_pose(self, segment):
        """评估一个时间段内的姿态估计"""
        # 使用第一帧计算坐标系转换矩阵
        first_frame_idx = segment[0]
        nearest_idx = None
        if self.first_flag == False:
            self.compute_transformation_matrix(first_frame_idx)
            self.first_flag = False
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
        gt_cam_poses = []
        est_cam_poses = []
        add_sum = 0
        
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
            est_cam_pose = self.camera_poses[frame_idx]
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
            gt_cam_poses.append(gt_cam_pose)
            est_cam_poses.append(est_cam_pose)

            est_pose = np.linalg.inv(est_cam_pose) @ est_pose
            gt_obj_pose = np.linalg.inv(gt_cam_pose) @ gt_obj_pose
            
            # 将点云转换到两个姿态下
            # points_est = self.transform_points(point_cloud, est_pose @ np.linalg.inv(gt_cam_pose))
            # points_mocap = self.transform_points(point_cloud, gt_obj_pose @ np.linalg.inv(gt_cam_pose))
            points_est = self.transform_points(self.obj_points, est_pose)
            points_mocap = self.transform_points(self.obj_points, gt_obj_pose)
            # print(points_est)
            # print(points_mocap)
            
            # 计算ADD和ADD-S
            add_value = self.calculate_add(points_est, points_mocap)
            adds_value = self.calculate_adds(points_est, points_mocap)
            
            results['add_values'].append(add_value)
            results['adds_values'].append(adds_value)
            results['total_frames'] += 1
            
            if add_value < self.add_threshold:
                results['add_correct'] += 1
            if adds_value < self.adds_threshold:
                results['adds_correct'] += 1
            
            print(f"帧 {frame_idx}: ADD={add_value:.4f}, ADD-S={adds_value:.4f}")
            add_sum += add_value
        # 计算成功率
        if results['total_frames'] > 0:
            results['add_success_rate'] = results['add_correct'] / results['total_frames']
            results['adds_success_rate'] = results['adds_correct'] / results['total_frames']
            results['add_mean'] = np.mean(results['add_values'])
            results['adds_mean'] = np.mean(results['adds_values'])

        self.visualize_poses(est_poses, mocap_poses, gt_cam_poses, est_cam_poses)
        
        return results, add_sum
        
    def transform_points(self, points, transform):
        """将点云应用变换矩阵"""
        # 转换为齐次坐标
        points_homogeneous = np.hstack((points, np.ones((len(points), 1))))
        
        # 应用变换
        transformed_points = (transform @ points_homogeneous.T).T
        
        # 返回3D坐标
        return transformed_points[:, :3]
    
        
    def evaluate(self, object_id, object_name):
        """评估所有时间段"""
        # 更新对象ID和名称
        self.object_id = object_id
        self.object_name = object_name
        
        # 更新评估文件路径以匹配新的对象ID
        self.eval_file = os.path.join(self.dataset_dir, "eval", f"object_{object_id}.txt")
        
        # 清除对象位姿列的缓存，因为对象名称可能已改变
        if hasattr(self, 'obj_cols'):
            delattr(self, 'obj_cols')
        
        # 重置转换矩阵标志，以便基于新对象重新计算
        self.first_flag = False
        self.obj_transformation = None
        self.camera_transformation = None
        
        # 重新读取估计的姿态数据，因为eval_file可能已改变
        self.estimated_poses, self.evaluation_segments = self.read_estimated_poses()
        print(f"评估对象: {object_name} (ID: {object_id})")
        print(f"使用评估文件: {self.eval_file}")
        
        all_results = []
        
        for i, segment in enumerate(self.evaluation_segments):
            print(f"\n评估段 {i+1}/{len(self.evaluation_segments)}")
            results, _ = self.evaluate_segment(segment)
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
        output_file = os.path.join(self.dataset_dir, "eval", "evaluation_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"结果已保存到 {output_file}")


def main():
    dataset_dir = "/media/wby/6d811df4-bde7-479b-ab4c-679222653ea0/dataset_done/apple_1"  # 请替换为实际的数据集路径
    evaluator = PoseEvaluator(dataset_dir)
    object_id = 2
    object_name = "apple"
    evaluator.evaluate(object_id, object_name)

if __name__ == "__main__":
    main()
