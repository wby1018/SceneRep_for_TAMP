import os
import numpy as np
from datetime import datetime
from scipy.spatial.transform import Rotation as R
import json

class ObjectPoseRecorder:
    def __init__(self, base_dir):
        """
        初始化对象位姿记录器
        
        Args:
            base_dir: 记录文件的基础目录
        """
        self.pose_dir = os.path.join(base_dir, "pose_txt")
        self.eval_dir = os.path.join(base_dir, "eval")
        os.makedirs(self.pose_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)
        for filename in os.listdir(self.eval_dir):
            file_path = os.path.join(self.eval_dir, filename)
            os.unlink(file_path)
        
        # 读取已有的timestamps.txt文件
        self.timestamp_file = os.path.join(self.pose_dir, "timestamps.txt")
        self.timestamps = {}  # 帧索引到时间戳的映射
        
        # 加载时间戳文件
        if os.path.exists(self.timestamp_file):
            with open(self.timestamp_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        frame_id = int(parts[0])
                        timestamp1 = parts[1]
                        timestamp2 = parts[2]
                        self.timestamps[frame_id] = (timestamp1, timestamp2)
        
        self.last_skip_fusion = True  # 初始化为True，这样第一帧会被认为是新的记录周期

        # 用于记录段落信息的变量
        self.segments_file = os.path.join(self.eval_dir, "segments.json")
        self.segments = []
        self.current_segment = None
        
        # 如果存在旧的segments文件，则删除
        if os.path.exists(self.segments_file):
            os.unlink(self.segments_file)
    
    def record_pose(self, objects, camera_pose, frame_id, skip_fusion):
        """
        Args:
            objects: 场景对象列表
            camera_pose: 相机位姿
            frame_id: 当前帧ID，用于查找对应时间戳
            skip_fusion: 当前帧是否跳过融合
        """
        # 检测段落变化
        if skip_fusion != self.last_skip_fusion:
            if skip_fusion:  # 从非跳过变为跳过，当前段结束
                if self.current_segment is not None:
                    self.current_segment['end_frame'] = int(frame_id) - 1
                    if self.current_segment['end_frame'] - self.current_segment['start_frame'] >= 2:
                        self.segments.append(self.current_segment)
                    self.current_segment = None
                    self._save_segments()
            else:  # 从跳过变为非跳过，开始新段
                self.current_segment = {
                    'start_frame': int(frame_id),
                    'end_frame': None,
                }
        
        self.last_skip_fusion = skip_fusion
        
        # 记录相机位姿
        cam_file = os.path.join(self.eval_dir, "camera.txt")
        with open(cam_file, "a") as f:
            tx, ty, tz = camera_pose[:3, 3]
            rot = R.from_matrix(camera_pose[:3, :3])
            qx, qy, qz, qw = rot.as_quat()
            f.write(f"{int(frame_id)} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
        
        # 记录所有对象的位姿
        for obj in objects:
            if obj is None or not hasattr(obj, 'pose_cur') or obj.pose_cur is None:
                continue
            obj_id = obj.id
            pose = obj.pose_cur
            
            # 为每个对象创建或更新位姿文件
            obj_file = os.path.join(self.eval_dir, f"object_{obj_id}.txt")
            
            with open(obj_file, "a") as f:
                # 记录时间戳和位姿: timestamp tx ty tz qx qy qz qw
                tx, ty, tz = pose[:3, 3]
                rot = R.from_matrix(pose[:3, :3])
                qx, qy, qz, qw = rot.as_quat()
                
                # 将时间戳和位姿信息一起写入文件
                f.write(f"{int(frame_id)} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
    
    def _save_segments(self):
        """保存段落信息到JSON文件"""
        with open(self.segments_file, 'w') as f:
            json.dump(self.segments, f, indent=2)
    
    def finalize(self, frame_id):
        """完成记录，保存最后一个段落信息"""
        if self.current_segment is not None:
            self.current_segment['end_frame'] = int(frame_id)
            self.segments.append(self.current_segment)
            self.current_segment = None
        self._save_segments()