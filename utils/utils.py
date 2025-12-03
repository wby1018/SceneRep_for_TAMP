from pathlib import Path

import base64
import cv2, trimesh, pyrender
import numpy as np, open3d as o3d
from scipy.spatial.transform import Rotation

from typing import Dict, List, Tuple, Optional, Union

import rospy
from sensor_msgs.msg import Image as ImageMsg
from geometry_msgs.msg import Pose, PoseStamped
from cv_bridge import CvBridge, CvBridgeError

# -------------------------- util ---------------------------------
def load_pose_txt_line(txt_path: str, idx: int) -> np.ndarray:
    with open(txt_path) as f:
        _idx, tx, ty, tz, qx, qy, qz, qw = map(float, f.readlines()[idx].split())
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    T[:3, 3]  = [tx, ty, tz]
    return T

def make_T_cb() -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = [0.16378, 0.02, 1.061]
    T[:3, :3] = np.array([[ 0,  0,  1],
                          [-1,  0,  0],
                          [ 0, -1,  0]], np.float32)
    return T

def base_to_cam(T_bw)-> np.ndarray:
    T_cw = np.eye(4, dtype=np.float32)
    T_cw[:3, 3] = T_bw[:3, 3] + T_bw[:3, :3] @ [0.16378, 0.02, 1.061]
    T_cw[:3, :3] = T_bw[:3, :3] @ np.array([[ 0,  0,  1],
                          [-1,  0,  0],
                          [ 0, -1,  0]], np.float32)
    return T_cw

def decode_mask(b64: str, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8),
                       cv2.IMREAD_GRAYSCALE)
    return (cv2.resize(img, (w, h)) > 0).astype(np.uint8)



# ros version helpers

bridge = CvBridge()

def pose_to_mat(p: Pose) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [p.position.x, p.position.y, p.position.z]
    q = [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
    T[:3, :3] = Rotation.from_quat(q).as_matrix()
    return T

def pose_any_to_mat(msg: Union[Pose, PoseStamped]) -> np.ndarray:
    return pose_to_mat(msg.pose) if hasattr(msg, "pose") else pose_to_mat(msg)

def img_msg_to_rgb(msg: ImageMsg) -> np.ndarray:
    return bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

def depth_msg_to_np(msg: ImageMsg) -> np.ndarray:
    try:
        if msg.encoding == "32FC1":
            d = bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1").astype(np.float32)
        else:  # assume 16UC1 in millimetres
            raw = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            d = raw.astype(np.float32) / 1000.0
    except CvBridgeError as e:
        rospy.logerr_throttle(5.0, f"CvBridge depth error: {e}")
        d = np.zeros((1, 1), np.float32)
    return np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

def invert(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]; t = T[:3, 3]
    Ti = np.eye(4, dtype=T.dtype)
    Ti[:3, :3] = R.T
    Ti[:3, 3]  = -R.T @ t
    return Ti


def _mask_to_world_pts_colors(
    mask: np.ndarray,
    depth: np.ndarray,
    rgb: np.ndarray,
    K: np.ndarray,
    T_cw: np.ndarray,
    sample_step: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回:
        pts_world : (N,3) float32
        cols      : (N,3) float32, 0‑1
    """
    assert mask.shape == depth.shape[:2] == rgb.shape[:2]

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    v_idx, u_idx = np.nonzero(mask)
    if sample_step > 1:
        v_idx, u_idx = v_idx[::sample_step], u_idx[::sample_step]

    z = depth[v_idx, u_idx].astype(np.float32)
    valid = z > 0
    if not np.any(valid):
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)

    u, v, z = u_idx[valid], v_idx[valid], z[valid]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts_cam = np.vstack((x, y, z, np.ones_like(z))).T        # (N,4)
    pts_world = (T_cw @ pts_cam.T).T[:, :3].astype(np.float32)

    cols = rgb[v, u].astype(np.float32) / 255.0              # 0‑1
    return pts_world, cols

def project_hand_to_mask(hand_pts_ee: np.ndarray, T_ec: np.ndarray, K: np.ndarray, img_shape, dilate_px=8):
    """
    hand_pts_ee: (N,3) 机械手模型点（ee系）
    T_ec: (4,4) ee→camera
    K: (3,3) 相机内参
    img_shape: (H,W)
    dilate_px: mask膨胀像素
    返回: mask, uint8, 0/1
    """
    import cv2
    # 1. 变换到相机系
    pts_ee_h = np.hstack([hand_pts_ee, np.ones((len(hand_pts_ee),1))])  # (N,4)
    pts_cam = (T_ec @ pts_ee_h.T).T[:, :3]  # (N,3)

    # 2. 投影到像素
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    x, y, z = pts_cam[:,0], pts_cam[:,1], pts_cam[:,2]
    u = (x * fx / z + cx).astype(np.int32)
    v = (y * fy / z + cy).astype(np.int32)

    # 3. 生成mask
    H, W = img_shape
    mask = np.zeros((H, W), np.uint8)
    valid = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    mask[v[valid], u[valid]] = 1
    mask = cv2.dilate(mask, np.ones((dilate_px, dilate_px), np.uint8))
    return mask

def box_surface_points(xmin, xmax, xstep, ymin, ymax, ystep, zmin, zmax, zstep):
    """
    生成box六个面的点云，只采样表面。
    返回: (N,3) np.ndarray
    """
    x = np.arange(xmin, xmax + xstep, xstep)
    y = np.arange(ymin, ymax + ystep, ystep)
    z = np.arange(zmin, zmax + zstep, zstep)

    # zmin面
    xx, yy = np.meshgrid(x, y, indexing='ij')
    zz = np.full_like(xx, zmin)
    pts_zmin = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    # zmax面
    zz = np.full_like(xx, zmax)
    pts_zmax = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    # xmin面
    yy, zz = np.meshgrid(y, z, indexing='ij')
    xx = np.full_like(yy, xmin)
    pts_xmin = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    # xmax面
    xx = np.full_like(yy, xmax)
    pts_xmax = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    # ymin面
    xx, zz = np.meshgrid(x, z, indexing='ij')
    yy = np.full_like(xx, ymin)
    pts_ymin = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    # ymax面
    yy = np.full_like(xx, ymax)
    pts_ymax = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    # 合并
    pts = np.concatenate([pts_zmin, pts_zmax, pts_xmin, pts_xmax, pts_ymin, pts_ymax], axis=0)
    return pts

def find_object_by_id(obj_id_in_ee, objects):
    """
    根据obj_id_in_ee查找对应的物体
    
    Parameters:
    -----------
    obj_id_in_ee : int or None
        要查找的物体ID
    objects : list
        物体列表
        
    Returns:
    --------
    SceneObject or None
        找到的物体对象，如果没找到则返回None
    """
    if obj_id_in_ee is None:
        return None
    for obj in objects:
        if obj.id == obj_id_in_ee:
            return obj
    return None