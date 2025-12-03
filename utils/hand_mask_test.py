from hand_mask_utils import generate_hand_mask
import os
import cv2
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import json

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

DATASET_PATH = "/media/wby/2AB9-4188/data_new_long_demo"
camera_config = {'fx': 554.3827, 'fy': 554.3827, 'cx': 320.5, 'cy': 240.5}
fx, fy, cx, cy = camera_config['fx'], camera_config['fy'], camera_config['cx'], camera_config['cy']
K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float32)
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
for idx in tqdm(range(len(rgb_files))):
    if idx < 1000:
        continue
    rgb_path = os.path.join(rgb_dir, rgb_files[idx])
    depth_path = os.path.join(depth_dir, depth_files[idx])
    frame_id = rgb_files[idx].split('_')[-1].split('.')[0]
    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    depth = np.load(depth_path)
    T_cw = cam_poses[int(frame_id)]
    T_ec = ee_poses[int(frame_id)]
    T_lfc = l_finger_poses[int(frame_id)]
    T_rfc = r_finger_poses[int(frame_id)]
    T_joints = joints_data[frame_id]
    hand_mask = generate_hand_mask(T_ec, K, depth.shape, T_lfc, T_rfc, T_joints, depth=depth)
    vis_mask = rgb.copy()
    overlay = np.zeros_like(vis_mask)
    overlay[hand_mask > 0] = [255, 0, 0]  # 红色掩码
    # vis_mask[hand_mask > 0] = [255, 0, 0]  # 红色掩码

    # 使用alpha混合实现半透明效果
    alpha = 0.5  # 50%的透明度
    vis_mask = cv2.addWeighted(vis_mask, 1.0, overlay, alpha, 0)
    hand_mask_path = os.path.join(hand_mask_dir, f"hand_mask_{int(frame_id):06d}.png")
    cv2.imwrite(hand_mask_path, cv2.cvtColor(vis_mask, cv2.COLOR_RGB2BGR))
    cv2.imshow("Hand Mask Overlay", cv2.cvtColor(vis_mask, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)