import numpy as np
import cv2
# from collections import deque
# from scipy.ndimage import sobel
# from scipy.ndimage import sobel, binary_propagation

# 主gripper box参数
default_hand_config = {
    'xmin': -0.12,
    'xmax': 0.00,
    'xstep': 0.005,
    'ymin': -0.08,
    'ymax': 0.08,
    'ystep': 0.005,
    'zmin': -0.03,
    'zmax': 0.03,
    'zstep': 0.005,
    'dilate_px': 9
}
# l finger box参数
default_lfinger_config = {
    'xmin': -0.03,
    'xmax': 0.05,
    'xstep': 0.002,
    'ymin': -0.025,
    'ymax': 0.00,
    'ystep': 0.002,
    'zmin': -0.015,
    'zmax': 0.005,
    'zstep': 0.002,
    'dilate_px': 7
}
# r finger box参数
default_rfinger_config = {
    'xmin': -0.03,
    'xmax': 0.05,
    'xstep': 0.002,
    'ymin': -0.025,
    'ymax': 0.0,
    'ystep': 0.002,
    'zmin': -0.015,
    'zmax': 0.005,
    'zstep': 0.002,
    'dilate_px': 7
}

# planning_joint_names = [
#     # "torso_lift", 
#     # "shoulder_pan",
#     "shoulder_lift",
#     "upperarm_roll",
#     "elbow_flex",
#     "forearm_roll",
#     "wrist_flex",
#     "wrist_roll",
# ]
planning_joint_names = [
    # "torso_lift", 
    # "shoulder_pan",
    "shoulder_lift_link",
    "upperarm_roll_link",
    "elbow_flex_link",
    "forearm_roll_link",
    "wrist_flex_link",
    "wrist_roll_link",
]

default_joint_config = {
    'xmin': -0.05,
    'xmax': 0.15,
    'xstep': 0.005,
    'ymin': -0.09,
    'ymax': 0.09,
    'ystep': 0.005,
    'zmin': -0.06,
    'zmax': 0.06,
    'zstep': 0.005,
    'dilate_px': 25
}


# ---------- 小工具：核缓存，避免重复创建 ----------
_KERNEL_CACHE = {}
def _get_ellipse_kernel(k):
    if k <= 1:
        return None
    key = int(k)
    ker = _KERNEL_CACHE.get(key)
    if ker is None:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (key, key))
        _KERNEL_CACHE[key] = ker
    return ker

# ---------- 向量化生成盒子表面点 ----------
def box_surface_points(xmin, xmax, xstep, ymin, ymax, ystep, zmin, zmax, zstep):
    """
    向量化生成六个面的采样点 (N,3)。无 Python 循环。
    """
    xs = np.arange(xmin, xmax + 1e-12, xstep, dtype=np.float32)
    ys = np.arange(ymin, ymax + 1e-12, ystep, dtype=np.float32)
    zs = np.arange(zmin, zmax + 1e-12, zstep, dtype=np.float32)

    # x = xmin/xmax 两个面
    yy, zz = np.meshgrid(ys, zs, indexing='ij')
    face_xmin = np.column_stack([np.full(yy.size, xmin, dtype=np.float32),
                                 yy.ravel(), zz.ravel()])
    face_xmax = np.column_stack([np.full(yy.size, xmax, dtype=np.float32),
                                 yy.ravel(), zz.ravel()])

    # y = ymin/ymax 两个面
    xx, zz = np.meshgrid(xs, zs, indexing='ij')
    face_ymin = np.column_stack([xx.ravel(),
                                 np.full(xx.size, ymin, dtype=np.float32),
                                 zz.ravel()])
    face_ymax = np.column_stack([xx.ravel(),
                                 np.full(xx.size, ymax, dtype=np.float32),
                                 zz.ravel()])

    # z = zmin/zmax 两个面
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    face_zmin = np.column_stack([xx.ravel(), yy.ravel(),
                                 np.full(xx.size, zmin, dtype=np.float32)])
    face_zmax = np.column_stack([xx.ravel(), yy.ravel(),
                                 np.full(xx.size, zmax, dtype=np.float32)])

    pts = np.vstack([face_xmin, face_xmax, face_ymin, face_ymax, face_zmin, face_zmax])
    return pts  # (N,3) float32

# ---------- 向量化：点云 -> 掩码 ----------
def _project_pts_to_mask(pts_ee, T_ec, K, img_shape, dilate_px=5):
    """
    将 (N,3) 末端坐标系点云通过 4x4 T_ec、3x3 K 投影到像素平面，并写入掩码。
    全向量化，无逐点 for。
    """
    H, W = img_shape
    if pts_ee.size == 0:
        return np.zeros((H, W), np.uint8)

    # 相机系坐标：R(3x3), t(3,), Xc = R @ X + t
    R = T_ec[:3, :3].astype(np.float32)
    t = T_ec[:3, 3].astype(np.float32)
    Xc = pts_ee @ R.T + t  # (N,3)

    # 仅保留在相机前方的点
    z = Xc[:, 2]
    valid = z > 1e-6
    if not np.any(valid):
        return np.zeros((H, W), np.uint8)
    Xc = Xc[valid]

    # 投影 (以像素中心为准，使用四舍五入)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = (fx * (Xc[:, 0] / Xc[:, 2]) + cx)
    v = (fy * (Xc[:, 1] / Xc[:, 2]) + cy)

    # 量化到整数像素，并裁剪到图像
    u = np.rint(u).astype(np.int32)
    v = np.rint(v).astype(np.int32)
    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(in_img):
        return np.zeros((H, W), np.uint8)

    u = u[in_img]
    v = v[in_img]

    # 一次性写入掩码：ravel 索引
    mask = np.zeros((H, W), np.uint8)
    idx = v * W + u
    mask.ravel()[idx] = 1

    # 形态学膨胀（可选）
    if dilate_px and dilate_px > 1:
        ker = _get_ellipse_kernel(dilate_px)
        if ker is not None:
            mask = cv2.dilate(mask, ker, iterations=1)
    return mask

# ---------- 面向调用：投影一组盒子点 ----------
def project_hand_to_mask(hand_pts_ee, T_ec, K, img_shape, dilate_px=5):
    """
    兼容你原来的函数名，但内部已向量化。
    """
    return _project_pts_to_mask(hand_pts_ee, T_ec, K, img_shape, dilate_px)

# ---------- 生成末端执行器掩码（向量化投影 + 仅对“关节”循环） ----------
def generate_end_effector_mask(
    T_ec, K, img_shape, T_lfc=None, T_rfc=None, depth=None
):
    """
    生成末端执行器（夹爪）掩码，包含可选的左右手指。完全去掉逐点循环。
    对“不同刚体”的遍历（左指/右指/若干关节）仍保留轻量级 for（数量很小，对性能影响很小）。
    """
    H, W = img_shape
    hand_mask = np.zeros((H, W), np.uint8)

    if T_lfc is not None:
        lcfg = default_lfinger_config
        l_pts = box_surface_points(
            lcfg['xmin'], lcfg['xmax'], lcfg['xstep'],
            lcfg['ymin'], lcfg['ymax'], lcfg['ystep'],
            lcfg['zmin'], lcfg['zmax'], lcfg['zstep']
        )
        hand_mask |= _project_pts_to_mask(l_pts, T_lfc, K, img_shape, lcfg['dilate_px'])

    if T_rfc is not None:
        rcfg = default_rfinger_config
        r_pts = box_surface_points(
            rcfg['xmin'], rcfg['xmax'], rcfg['xstep'],
            rcfg['ymin'], rcfg['ymax'], rcfg['ystep'],
            rcfg['zmin'], rcfg['zmax'], rcfg['zstep']
        )
        hand_mask |= _project_pts_to_mask(r_pts, T_rfc, K, img_shape, rcfg['dilate_px'])

    return hand_mask.astype(np.uint8)

# ---------- 若需要 joints：用轻量 for（关节数量少，已无逐点循环） ----------
def generate_hand_mask(T_ec, K, img_shape, T_lfc=None, T_rfc=None, T_joints=None, depth=None):
    """
    主 gripper 目前关闭；左右指已向量化；关节集合（若给出）仅在“刚体数量”上循环。
    """
    H, W = img_shape
    hand_mask = np.zeros((H, W), np.uint8)

    # 左右手指
    if T_lfc is not None:
        lcfg = default_lfinger_config
        l_pts = box_surface_points(
            lcfg['xmin'], lcfg['xmax'], lcfg['xstep'],
            lcfg['ymin'], lcfg['ymax'], lcfg['ystep'],
            lcfg['zmin'], lcfg['zmax'], lcfg['zstep']
        )
        hand_mask |= _project_pts_to_mask(l_pts, T_lfc, K, img_shape, lcfg['dilate_px'])

    if T_rfc is not None:
        rcfg = default_rfinger_config
        r_pts = box_surface_points(
            rcfg['xmin'], rcfg['xmax'], rcfg['xstep'],
            rcfg['ymin'], rcfg['ymax'], rcfg['ystep'],
            rcfg['zmin'], rcfg['zmax'], rcfg['zstep']
        )
        hand_mask |= _project_pts_to_mask(r_pts, T_rfc, K, img_shape, rcfg['dilate_px'])

    # 关节（只在“不同 T_joint”上循环；内部投影是向量化的）
    if T_joints is not None:
        for joint_name, T_joint in T_joints.items():
            if joint_name in planning_joint_names:
                jcfg = default_joint_config
                j_pts = box_surface_points(
                    jcfg['xmin'], jcfg['xmax'], jcfg['xstep'],
                    jcfg['ymin'], jcfg['ymax'], jcfg['ystep'],
                    jcfg['zmin'], jcfg['zmax'], jcfg['zstep']
                )
                # 这里你原先硬编码了 dilate_px=9，如需一致可改为 jcfg['dilate_px']
                hand_mask |= _project_pts_to_mask(j_pts, T_joint, K, img_shape, dilate_px=9)

    return hand_mask.astype(np.uint8)