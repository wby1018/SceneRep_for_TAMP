from scipy.optimize import linear_sum_assignment
import numpy as np
import open3d as o3d
import json
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from scipy.spatial.transform import Rotation as R
import os
import time
from warnings import warn
import shutil
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity

OBJECTS = ["apple", "basket", "cup", "can", "tomato", "orange juice plastic bottle", "wooden box", "milk carton", "bowl", "wooden cube block"]
# OBJECT_THRESHOLDS = {
#     'milkbox': {'new': 0.5, 'pcd': 0.7},
#     'cola': {'new': 0.4, 'pcd': 0.5},
#     # 'cup': {'new': 0.5, 'pcd': 0.55},
#     'cup': {'new': 0.45, 'pcd': 0.45},
#     'apple': {'new': 0.5, 'pcd': 0.7},
#     'pear': {'new': 0.2, 'pcd': 0.2},
#     'flowerpot': {'new': 0.4, 'pcd': 0.5},
#     'tomato': {'new': 0.2, 'pcd': 0.1},
#     'bowl': {'new': 0.2, 'pcd': 0.5}
# }
NEW_SCORE_THRESHOLD = 0.2 # 认为是新物体的owl分数
EE_NEW_SCORE_THRESHOLD = 0.1 # 认为是新物体的owl分数（当手持物体时）
NEW_HAND_THRESHOLD = 0.95 # 新物体与手部掩码重合度阈值
NEW_PCD_DIST = 0.05 # 新物体与现有物体点云距离阈值
NEW_PCD_THRESHOLD = 0.1 # 新物体与现有物体点云重叠度阈值
NEW_MIN_MASK_AREA = 100 # 新物体掩码面积阈值

HUNGARIAN_THRESHOLD = 0.6 # 与上一帧物体的匈牙利算法匹配阈值
MATCH_HUNGARIAN_THRESHOLD = 0.6 # 与消失物体的匈牙利算法匹配阈值
DEPTH_RATIO_THRESHOLD = 0.95 # 新物体掩码中有效深度像素的比例阈值
STEP = 1
object_num = 0
disappeared_objects = {}
FIRST_FRAME_THRESH = 0.2

fx = 541.7874545077664
fy = 538.1475661088576
cx = 321.6009893833978
cy = 232.013175647453
K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float32)

def load_data_with_fallback(json_path, rgb_path, depth_path, camera_pose_path, frame_id, kernel_size, objects, max_retries=20):
    original_frame_id = frame_id
    retry_count = 0
    pre_flag = True
    
    while retry_count < max_retries:
        json_file = json_path.format(f"{frame_id:06d}")
        with open(json_file, 'r') as f:
            data = json.load(f)

        if data and "detections" in data and len(data["detections"]) > 0:
            bboxes, classes, scores, pcds, masks = load_data(
                json_path, rgb_path, depth_path, camera_pose_path, frame_id, kernel_size, objects, pre_flag
            )              
            return bboxes, classes, scores, pcds, masks, frame_id
        else:
            retry_count += 1
            frame_id -= STEP
            if frame_id < 0:
                print(f"错误: 无法找到有检测内容的帧")
                return np.array([]), np.array([]), np.array([]), [], np.array([]), original_frame_id

    return np.array([]), np.array([]), np.array([]), [], np.array([]), original_frame_id

def load_data(json_path, rgb_path, depth_path, camera_pose_path, frame_id, kernel_size, objects, pre_flag=False):
    json_file = json_path.format(f"{frame_id:06d}")
    rgb_file = rgb_path.format(f"{frame_id:06d}")
    depth_file = depth_path.format(f"{frame_id:06d}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    rgb_img = cv2.imread(rgb_file)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    depth_img = np.load(depth_file)
    
    # 直接搜索指定frame_id的相机位姿（从文件底部开始搜索）
    camera_pose = None
    with open(camera_pose_path, 'r') as f:
        lines = f.readlines()
        # 从最后一行开始向前搜索
        for line in reversed(lines):
            parts = line.strip().split()
            if len(parts) >= 8:
                frame = parts[0]
                if frame == str(frame_id):
                    position = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                    quaternion = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])
                    rotation_matrix = R.from_quat(quaternion).as_matrix()
                    transform = np.eye(4)
                    transform[:3, :3] = rotation_matrix
                    transform[:3, 3] = position
                    camera_pose = transform
                    break
    
    if camera_pose is None:
        raise ValueError(f"Camera pose not found for frame {frame_id}")
    
    depth_scale = 1
    bboxes = []
    classes = []
    scores = []
    pcds = []
    masks = []
    class_to_idx = {name: i for i, name in enumerate(OBJECTS)}
    detections = data.get("detections", [])

    def decode_mask(mask_base64):
        mask_bytes = base64.b64decode(mask_base64)
        mask_pil = Image.open(BytesIO(mask_bytes))
        return np.array(mask_pil) > 0
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        mask_futures = [executor.submit(decode_mask, det["mask"]) for det in detections]
        masks = [future.result() for future in mask_futures]
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        # masks_eroded = [cv2.erode(mask.astype(np.uint8), kernel, iterations=1) > 0 for mask in masks]
        masks_eroded = masks

    for i, det in enumerate(detections):
        box = det.get("box", [0, 0, 0, 0])

        if "detection" in det:
            detection_list = det.get("detection", [])

            best_label = ""
            best_score = -1
            for d in detection_list:
                label = d.get("label", "")
                score = d.get("score", 0)
                if score > best_score:
                    best_score = score
                    best_label = label
            class_idx = class_to_idx.get(best_label, 0) if best_label in class_to_idx else 0
            classes.append(class_idx)
            score_array = np.zeros(len(class_to_idx))
            for d in detection_list:
                label = d.get("label", "")
                score = d.get("score", 0)
                if label in class_to_idx:
                    score_array[class_to_idx[label]] = score
            
            scores.append(score_array)
        else:
            label = det.get("label", "")
            score = det.get("score", 0.0)
            obj_id = det.get("object_id", 0)
            class_idx = class_to_idx.get(label, 0) if label in class_to_idx else 0
            classes.append(class_idx)
            score_array = np.zeros(len(class_to_idx))
            score_array[class_idx] = score
            scores.append(score_array)
        
        bboxes.append(box)
        
        # mask_bytes = base64.b64decode(det["mask"])
        # mask_pil = Image.open(BytesIO(mask_bytes))
        # mask = np.array(mask_pil) > 0
        # # mask[hand_mask > 0] = 0
        # masks.append(mask)

        points = []
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        # mask_uint8 = (mask * 255).astype(np.uint8)
        # mask_eroded = cv2.erode(mask_uint8, kernel, iterations=1)
        if pre_flag == False:
            y_indices, x_indices = np.where(masks_eroded[i])
            if len(y_indices) > 0:
                if len(y_indices) > 2000:
                    sample_rate = len(y_indices) // 2000
                    y_indices = y_indices[::sample_rate]
                    x_indices = x_indices[::sample_rate]
                valid = (y_indices < depth_img.shape[0]) & (x_indices < depth_img.shape[1])
                y_indices = y_indices[valid]
                x_indices = x_indices[valid]

                valid_depth = depth_img[y_indices, x_indices] > 0
                if not np.any(valid_depth):  # 如果没有有效深度值，跳过
                    points = np.array([])
                    pcds.append(points)
                    continue

                # 只使用有效深度值的像素
                valid_y = y_indices[valid_depth]
                valid_x = x_indices[valid_depth]
                valid_depths = depth_img[valid_y, valid_x] / depth_scale

                # 计算3D坐标
                X = (valid_x - cx) * valid_depths / fx
                Y = (valid_y - cy) * valid_depths / fy
                Z = valid_depths

                points_cam = np.vstack([X, Y, Z, np.ones_like(X)]).T
                
                points_world = []
                
                points = (camera_pose @ points_cam.T).T[:, :3]
            else:
                points = np.array([])
        else:
            for obj in objects:
                if obj.id == obj_id:
                    points = obj.points
                    T_init_inv = np.linalg.inv(obj.pose_init)
                    T_relative = obj.pose_cur @ T_init_inv
                    
                    # 应用变换
                    points_homo = np.hstack([points, np.ones((len(points), 1))])
                    points = (T_relative @ points_homo.T).T[:, :3]
                    break
        pcds.append(points)
    
    return np.array(bboxes), np.array(classes), np.array(scores), pcds, np.array(masks)

def load_data_realtime(det_json, rgb_img, depth_img, camera_pose, kernel_size):
    """
    实时处理函数：直接从输入参数加载数据，而不是从文件读取
    """
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    
    depth_scale = 1
    bboxes = []
    classes = []
    scores = []
    pcds = []
    masks = []
    class_to_idx = {name: i for i, name in enumerate(OBJECTS)}
    detections = det_json.get("detections", [])

    def decode_mask(mask_base64):
        mask_bytes = base64.b64decode(mask_base64)
        mask_pil = Image.open(BytesIO(mask_bytes))
        return np.array(mask_pil) > 0
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        mask_futures = [executor.submit(decode_mask, det["mask"]) for det in detections]
        masks = [future.result() for future in mask_futures]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        masks_eroded = [cv2.erode(mask.astype(np.uint8), kernel, iterations=1) > 0 for mask in masks]

    for i, det in enumerate(detections):
        box = det.get("box", [0, 0, 0, 0])

        if "detection" in det:
            detection_list = det.get("detection", [])

            best_label = ""
            best_score = -1
            for d in detection_list:
                label = d.get("label", "")
                score = d.get("score", 0)
                if score > best_score:
                    best_score = score
                    best_label = label
            class_idx = class_to_idx.get(best_label, 0) if best_label in class_to_idx else 0
            classes.append(class_idx)
            score_array = np.zeros(len(class_to_idx))
            for d in detection_list:
                label = d.get("label", "")
                score = d.get("score", 0)
                if label in class_to_idx:
                    score_array[class_to_idx[label]] = score
            
            scores.append(score_array)
        else:
            label = det.get("label", "")
            score = det.get("score", 0.0)
            class_idx = class_to_idx.get(label, 0) if label in class_to_idx else 0
            classes.append(class_idx)
            score_array = np.zeros(len(class_to_idx))
            score_array[class_idx] = score
            scores.append(score_array)
        
        bboxes.append(box)

        points = []
        y_indices, x_indices = np.where(masks_eroded[i])
        if len(y_indices) > 0:
            if len(y_indices) > 2000:
                sample_rate = len(y_indices) // 2000
                y_indices = y_indices[::sample_rate]
                x_indices = x_indices[::sample_rate]
            valid = (y_indices < depth_img.shape[0]) & (x_indices < depth_img.shape[1])
            y_indices = y_indices[valid]
            x_indices = x_indices[valid]

            valid_depth = depth_img[y_indices, x_indices] > 0
            if not np.any(valid_depth):  # 如果没有有效深度值，跳过
                points = np.array([])
                pcds.append(points)
                continue

            # 只使用有效深度值的像素
            valid_y = y_indices[valid_depth]
            valid_x = x_indices[valid_depth]
            valid_depths = depth_img[valid_y, valid_x] / depth_scale

            # 计算3D坐标
            X = (valid_x - cx) * valid_depths / fx
            Y = (valid_y - cy) * valid_depths / fy
            Z = valid_depths

            points_cam = np.vstack([X, Y, Z, np.ones_like(X)]).T
            
            points = (camera_pose @ points_cam.T).T[:, :3]
        else:
            points = np.array([])
        pcds.append(points)
    
    return np.array(bboxes), np.array(classes), np.array(scores), pcds, np.array(masks)

def compute_point_cloud_overlap(pcd1, pcd2, distance_threshold=NEW_PCD_DIST, max_points=2000):

    if len(pcd1) == 0 or len(pcd2) == 0:
        return 0.0
    
    # 对大型点云进行下采样
    if len(pcd1) > max_points:
        step = len(pcd1) // max_points
        pcd1 = pcd1[::step]
        
    if len(pcd2) > max_points:
        step = len(pcd2) // max_points
        pcd2 = pcd2[::step]
    
    from scipy.spatial import cKDTree
    tree = cKDTree(pcd1)

    distances, _ = tree.query(pcd2, k=1)

    overlap_count = np.sum(distances < distance_threshold)
    overlap_ratio = overlap_count / len(pcd2) if len(pcd2) > 0 else 0
    
    return overlap_ratio

def compute_point_distance(points1, points2):
    if len(points1) == 0 or len(points2) == 0:
        return np.zeros((len(points1), len(points2)))

    points1 = np.asarray(points1, dtype=np.float32)
    points2 = np.asarray(points2, dtype=np.float32)
    if len(points1) * len(points2) > 10000:
        from scipy.spatial.distance import cdist
        return cdist(points1, points2, 'euclidean')

    diff = points1[:, np.newaxis, :] - points2[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff * diff, axis=-1))
    return dist_matrix

def pcd_distance(pcds, pcds_new):
    """
    使用ADDS (Average Distance of Distances) 计算点云之间的距离
    ADDS计算每个点云中每个点到另一个点云的最小距离的平均值
    """
    assert pcds is not None and pcds_new is not None, \
        "Point clouds are not provided for the object detection outputs. Please set 3D data."
    
    def compute_adds_distance(points1, points2, max_points=50):
        """
        计算两个点云之间的ADDS距离
        Args:
            points1: 第一个点云 (N, 3)
            points2: 第二个点云 (M, 3)
            max_points: 最大采样点数，用于提高计算效率
        Returns:
            ADDS距离值
        """
        if len(points1) == 0 or len(points2) == 0:
            return 1e6
        
        # 如果点云太大，随机采样以提高效率
        if len(points1) > max_points:
            indices = np.random.choice(len(points1), max_points, replace=False)
            points1 = points1[indices]
        if len(points2) > max_points:
            indices = np.random.choice(len(points2), max_points, replace=False)
            points2 = points2[indices]
        
        # 计算points1中每个点到points2的最小距离
        distances_1_to_2 = []
        for pt in points1:
            min_dist = np.min(np.linalg.norm(points2 - pt, axis=1))
            distances_1_to_2.append(min_dist)
        
        # 计算points2中每个点到points1的最小距离
        distances_2_to_1 = []
        for pt in points2:
            min_dist = np.min(np.linalg.norm(points1 - pt, axis=1))
            distances_2_to_1.append(min_dist)
        
        # ADDS是双向距离的平均值
        adds_distance = (8*np.mean(distances_1_to_2) + 2*np.mean(distances_2_to_1)) / 10.0
        return adds_distance*5
    
    # 计算所有点云对之间的ADDS距离矩阵
    n_pcds = len(pcds)
    n_pcds_new = len(pcds_new)
    point_dists = np.zeros((n_pcds, n_pcds_new))
    
    for i, pcd in enumerate(pcds):
        if pcd is not None and len(pcd) > 0:
            for j, pcd_new in enumerate(pcds_new):
                if pcd_new is not None and len(pcd_new) > 0:
                    point_dists[i, j] = compute_adds_distance(pcd, pcd_new)
                else:
                    point_dists[i, j] = 1e6
        else:
            point_dists[i, :] = 1e6
    
    # 仍然计算中心点用于其他用途
    centroids = np.array([pc.mean(axis=0) if pc is not None and len(pc) > 0 else np.zeros(3) for pc in pcds])
    centroids_new = np.array([pc.mean(axis=0) if pc is not None and len(pc) > 0 else np.zeros(3) for pc in pcds_new])
    
    return point_dists, centroids, centroids_new

# def pcd_distance(pcds, pcds_new):
#         assert pcds is not None and pcds_new is not None, \
#             "Point clouds are not provided for the object detection outputs. Please set 3D data."
#         centroids = np.array([pc.mean(axis=0) if pc.size > 0 else np.zeros(3) for pc in pcds])
#         centroids_new = np.array([pc.mean(axis=0) if pc.size > 0 else np.zeros(3) for pc in pcds_new])

#         point_dists = compute_point_distance(centroids, centroids_new)
#         return point_dists, centroids, centroids_new

def pcd_size_distance(pcds, pcds_new):
    """
    计算3D边界框的IoU距离矩阵
    使用点云数据计算3D边界框，然后计算IoU
    """
    def compute_3d_bbox(points):
        """
        从点云计算3D边界框
        Args:
            points: 点云数据 (N, 3)
        Returns:
            bbox: [min_x, min_y, min_z, max_x, max_y, max_z]
        """
        if points is None or len(points) == 0:
            return np.array([0, 0, 0, 0, 0, 0])
        
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        return np.concatenate([min_coords, max_coords])
    
    def compute_3d_iou(bbox1, bbox2):
        """
        计算两个3D边界框的IoU
        Args:
            bbox1: [min_x, min_y, min_z, max_x, max_y, max_z]
            bbox2: [min_x, min_y, min_z, max_x, max_y, max_z]
        Returns:
            IoU值 (0-1之间)
        """
        # 计算交集边界框
        inter_min = np.maximum(bbox1[:3], bbox2[:3])
        inter_max = np.minimum(bbox1[3:], bbox2[3:])
        
        # 检查是否有交集
        if np.any(inter_min >= inter_max):
            return 0.0
        
        # 计算交集体积
        inter_volume = np.prod(inter_max - inter_min)
        
        # 计算各自体积
        volume1 = np.prod(bbox1[3:] - bbox1[:3])
        volume2 = np.prod(bbox2[3:] - bbox2[:3])
        
        # 计算并集体积
        union_volume = volume1 + volume2 - inter_volume
        
        if union_volume <= 0:
            return 0.0
        
        return inter_volume / union_volume
    
    # 计算所有点云的3D边界框
    bboxes = [compute_3d_bbox(pc) if pc is not None and len(pc) > 0 else np.array([0, 0, 0, 0, 0, 0]) for pc in pcds]
    bboxes_new = [compute_3d_bbox(pc) if pc is not None and len(pc) > 0 else np.array([0, 0, 0, 0, 0, 0]) for pc in pcds_new]
    
    # 计算IoU距离矩阵 (1 - IoU，因为IoU越大距离越小)
    n_bboxes = len(bboxes)
    n_bboxes_new = len(bboxes_new)
    iou_matrix = np.zeros((n_bboxes, n_bboxes_new))
    
    for i, bbox in enumerate(bboxes):
        for j, bbox_new in enumerate(bboxes_new):
            iou = compute_3d_iou(bbox, bbox_new)
            # 将IoU转换为距离：1 - IoU，IoU越大距离越小
            iou_matrix[i, j] = 1.0 - iou
    
    return iou_matrix, bboxes, bboxes_new

# def pcd_size_distance(pcds, pcds_new):
#     sizes = [np.ptp(pc, axis=0) if pc.size > 0 else np.zeros(3) for pc in pcds]
#     sizes_new = [np.ptp(pc, axis=0) if pc.size > 0 else np.zeros(3) for pc in pcds_new]

#     sizes_array = np.array(sizes)
#     sizes_new_array = np.array(sizes_new)

#     diff = sizes_array[:, np.newaxis, :] - sizes_new_array[np.newaxis, :, :]
#     dist_matrix = np.sqrt(np.sum(diff * diff, axis=2))
#     # print(f"size_pcd: {sizes}")
#     # print(f"size_pcd_new: {sizes_new}")
#     return dist_matrix, sizes, sizes_new

def get_iou_masks(masks1, masks2):
    if len(masks1) == 0 or len(masks2) == 0:
        return np.zeros((len(masks1), len(masks2)))
    
    N = len(masks1)
    M = len(masks2)

    iou_matrix = np.zeros((N, M))

    for i in range(N):
        mask1 = masks1[i].astype(bool)
        area1 = np.sum(mask1)
        
        if area1 == 0:
            continue
            
        for j in range(M):
            mask2 = masks2[j].astype(bool)
            area2 = np.sum(mask2)
            
            if area2 == 0:
                continue

            intersection = np.sum(mask1 & mask2)

            union = area1 + area2 - intersection

            iou_matrix[i, j] = intersection / max(union, 1e-10)
    
    return iou_matrix

def visualize_pcd_with_boxes(pcds, pcds_new):
    vis_objects = []

    # 可视化 pcds
    for pc in pcds:
        if pc.size == 0:
            continue
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pc)
        pcd_o3d.paint_uniform_color([0, 0, 1])  # 蓝色
        vis_objects.append(pcd_o3d)

        # 计算包围盒
        bbox = pcd_o3d.get_axis_aligned_bounding_box()
        bbox.color = (1, 0, 0)  # 红色边框
        vis_objects.append(bbox)

    # 可视化 pcds_new
    for pc in pcds_new:
        if pc.size == 0:
            continue
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pc)
        pcd_o3d.paint_uniform_color([0, 1, 0])  # 绿色
        vis_objects.append(pcd_o3d)

        bbox = pcd_o3d.get_axis_aligned_bounding_box()
        bbox.color = (1, 1, 0)  # 黄色边框
        vis_objects.append(bbox)

    # 使用 Open3D 可视化
    o3d.visualization.draw_geometries(vis_objects)

def add_object_id(json_path, frame_id, hand_mask=None):
    global object_num
    json_file = json_path.format(f"{frame_id:06d}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)

    detections = data.get("detections", [])
    
    # 对第一帧进行筛选
    if frame_id == 0:
        filtered_detections = []
        for det in detections:
            # 处理检测结果格式
            if "detection" in det:
                detection_list = det.get("detection", [])
                best_score = detection_list[0].get("score", 0) if detection_list else 0
                best_label = detection_list[0].get("label", "") if detection_list else ""
                det["label"] = best_label
                det["score"] = best_score
                del det["detection"]
            
            # 应用第一帧筛选条件
            score = det.get("score", 0)
            if score >= FIRST_FRAME_THRESH:
                # 检查掩码面积（如果有掩码的话）
                if "mask" in det:
                    try:
                        mask_bytes = base64.b64decode(det["mask"])
                        mask_pil = Image.open(BytesIO(mask_bytes))
                        mask_np = np.array(mask_pil) > 0
                        mask_area = np.sum(mask_np)
                        
                        if mask_area < NEW_MIN_MASK_AREA:
                            # print(f"第一帧筛选：跳过检测，掩码面积 {mask_area} 小于阈值 {NEW_MIN_MASK_AREA}")
                            continue
                        
                        # 检查当前掩码是否与手部掩码重叠
                        if hand_mask is not None:
                            intersection = np.sum(mask_np & hand_mask)
                            overlap_ratio = intersection / mask_area if mask_area > 0 else 0
                            if overlap_ratio >= NEW_HAND_THRESHOLD:
                                # print(f"第一帧筛选：跳过检测，当前掩码与手部掩码重叠度 {overlap_ratio:.2f} 超过阈值 {NEW_HAND_THRESHOLD}")
                                continue
                        
                        filtered_detections.append(det)
                    except Exception as e:
                        print(f"第一帧筛选：无法处理掩码，跳过检测: {e}")
                        continue
                # else:
                #     # 没有掩码信息，直接添加
                #     filtered_detections.append(det)
            # else:
                # print(f"第一帧筛选：跳过检测，分数 {score} 小于阈值 {FIRST_FRAME_THRESH}")
        
        detections = filtered_detections
    
    # 为检测结果分配object_id
    for i, det in enumerate(detections):
        det["object_id"] = i
        object_num += 1

    # 更新数据
    data["detections"] = detections
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)

def update_detection(json_path, json_new_path, prev_frame_id, frame_id, match_dict, masks_new, pcds_new, dataset_name, objects, skip_fusion, hand_mask, state, ee_label):

    curr_json_file = json_new_path.format(f"{frame_id:06d}")
    prev_json_file = json_path.format(f"{prev_frame_id:06d}")
    output_json_file = json_path.format(f"{frame_id:06d}")
    depth_file = f"{dataset_name}/depth/depth_{frame_id:06d}.npy"
    depth_img = np.load(depth_file)
    
    with open(curr_json_file, 'r') as f:
        curr_data = json.load(f)
    with open(prev_json_file, 'r') as f:
        prev_data = json.load(f)
    
    # 获取上一帧的object_id映射
    prev_id_map = {}
    for i, det in enumerate(prev_data.get("detections", [])):
        if "object_id" in det:
            prev_id_map[i] = det["object_id"]
    
    curr_detections = curr_data.get("detections", [])

    new_detections = []
    ignored_detections = []
    new_pcds = []

    processed_indices = set()
    processed_masks = []
    ignored_boxes = set()

    for prev_idx, curr_idx in match_dict.items():
        if curr_idx < len(curr_detections) and prev_idx in prev_id_map:
            det = curr_detections[curr_idx]
            processed_indices.add(curr_idx)

            if "detection" in det:
                detection_list = det.get("detection", [])
                best_score = detection_list[0].get("score", 0) if detection_list else 0
                best_label = detection_list[0].get("label", "") if detection_list else ""

                det["label"] = best_label
                det["score"] = best_score
                del det["detection"]

            det["object_id"] = prev_id_map[prev_idx]
            mask_pil = Image.fromarray(masks_new[curr_idx].astype(np.uint8) * 255)
            buffer = BytesIO()
            mask_pil.save(buffer, format="PNG")
            buffer.seek(0)
            mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            det["mask"] = mask_base64

            new_detections.append(det)
            processed_masks.append(masks_new[curr_idx])
            if det["object_id"] in disappeared_objects:
                del disappeared_objects[det["object_id"]]

    for i, det in enumerate(curr_detections):
        if i not in processed_indices:
            if "detection" in det:
                detection_list = det.get("detection", [])
                best_score = detection_list[0].get("score", 0) if detection_list else 0
                best_label = detection_list[0].get("label", "") if detection_list else ""
                det["label"] = best_label
                det["score"] = best_score
                box = det.get("box", [0, 0, 0, 0])
                del det["detection"]

            # new_threshold = OBJECT_THRESHOLDS[det.get("label", "")].get('new', 0)
            if best_score > NEW_SCORE_THRESHOLD or (state == "holding" and det.get("label", "") == ee_label and best_score > EE_NEW_SCORE_THRESHOLD):
                mask = masks_new[i]
                mask_area = np.sum(mask)
                info = None

                # 1. 检查掩码面积是否小于阈值
                if mask_area < NEW_MIN_MASK_AREA:
                    info = f"跳过帧 {frame_id}，掩码面积 {mask_area} 小于阈值 {NEW_MIN_MASK_AREA}\n"
                    continue
                # 2. 查找所有现有对象的点云并比较重合度是否超过阈值
                ratio = {}
                ratio_flag = False
                if new_pcds is not None:
                    for new_pcd in new_pcds:
                        # 计算点云重叠度
                        overlap_ratio = compute_point_cloud_overlap(new_pcd, pcds_new[i])
                        ratio[-1] = overlap_ratio
                        
                        if overlap_ratio > NEW_PCD_THRESHOLD:
                            info = f"跳过帧 {frame_id}，与新检测点云的重叠度为 {overlap_ratio:.2f}，超过阈值 {NEW_PCD_THRESHOLD}"
                            skip_fusion = True
                            ratio_flag = True
                            break
                if objects is not None:
                    for obj in objects:
                        # if obj.id in prev_id_map.values():
                        # 获取物体点云并变换到当前世界坐标系
                        points_h = np.hstack([obj._points, np.ones((obj._points.shape[0], 1))])
                        T = obj.pose_cur @ np.linalg.inv(obj.pose_init)
                        obj_points_world = (T @ points_h.T).T[:, :3]
                        
                        # 计算点云重叠度
                        overlap_ratio = compute_point_cloud_overlap(obj_points_world, pcds_new[i])
                        ratio[obj.id] = overlap_ratio
                        
                        if overlap_ratio > NEW_PCD_THRESHOLD:
                            info = f"跳过帧 {frame_id}，与物体ID {obj.id} 的点云重叠度为 {overlap_ratio:.2f}，超过阈值 {NEW_PCD_THRESHOLD}"
                            skip_fusion = True
                            ratio_flag = True
                            break
                # 3. 检查当前掩码是否与手部掩码重叠
                if hand_mask is not None:
                    intersection = np.sum(mask & hand_mask)
                    overlap_ratio = intersection / mask_area if mask_area > 0 else 0
                    if overlap_ratio >= NEW_HAND_THRESHOLD and ratio_flag == False: 
                        info = f"跳过帧 {frame_id}，当前掩码与手部掩码重叠度 {overlap_ratio:.2f} 超过阈值 {NEW_HAND_THRESHOLD}"
                        continue
                
                # if det["label"] == "milkbox":
                #     skip_fusion = True

                # 计算掩码中有效深度像素的比例
                y_indices, x_indices = np.where(mask)
                total_points = len(y_indices)

                valid_indices = (y_indices < depth_img.shape[0]) & (x_indices < depth_img.shape[1])
                y_valid = y_indices[valid_indices]
                x_valid = x_indices[valid_indices]

                valid_depth_points = np.sum(depth_img[y_valid, x_valid] > 0)
                valid_depth_ratio = valid_depth_points / total_points
                
                if valid_depth_ratio > DEPTH_RATIO_THRESHOLD:
                    mask_pil = Image.fromarray(masks_new[i].astype(np.uint8) * 255)
                    buffer = BytesIO()
                    mask_pil.save(buffer, format="PNG")
                    buffer.seek(0)
                    mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    det["mask"] = mask_base64
                    updated_det = new_object(det, object_num, masks_new, i, frame_id, pcds_new, objects, skip_fusion)
                    if updated_det is not None: 
                        new_detections.append(updated_det)
                        new_pcds.append(pcds_new[i])
                    else: 
                        if ratio_flag != True: 
                            ignored_boxes.add(tuple(box))
                            ignored_detections.append(det)
                        if info is not None:
                            with open("hungarian_match.txt", "a") as log_file:
                                log_file.write(f"{info}; 指标: {mask_area}, {ratio}\n")

    new_data = {"detections": new_detections}

    with open(output_json_file, 'w') as f:
        json.dump(new_data, f, indent=2)

    visualize_detections(output_json_file, frame_id, dataset_name, ignored_boxes)

def new_object(det, new_id, masks_new, curr_idx, frame_id, pcds_new, objects, skip_fusion):
    global object_num, disappeared_objects
    
    # 如果没有消失的物体记录，则直接分配新ID
    if not disappeared_objects:
        if not skip_fusion:
            det["object_id"] = new_id
            object_num += 1
            return det
        else:
            return None
    
    # 准备当前检测对象的数据
    curr_mask = masks_new[curr_idx:curr_idx+1]
    curr_box = np.array([det.get("box", [0, 0, 0, 0])])

    curr_class_idx = OBJECTS.index(det.get("label", "")) if det.get("label", "") in OBJECTS else 0
    curr_classes = np.array([curr_class_idx])
    
    score_array = np.zeros(len(OBJECTS))
    score_array[curr_class_idx] = det.get("score", 0)
    curr_scores = np.array([score_array])
    curr_pcds = pcds_new[curr_idx:curr_idx+1]
    
    # 从disappeared_objects中提取所有消失物体的数据
    disappeared_ids = list(disappeared_objects.keys())
    disappeared_boxes = []
    disappeared_classes = []
    disappeared_scores = []
    disappeared_pcds = []
    disappeared_masks = []
    dis_frame_ids = []

    for idx, obj_id in enumerate(disappeared_ids):
        dis_frame_id = disappeared_objects[obj_id]
        # print(disappeared_objects)
        # print(objects)
        if objects is not None:
            for obj in objects:
                if obj.id == obj_id:
                    score = obj._score
                    label = obj._label
                    points_h = np.hstack([obj._points, np.ones((obj._points.shape[0], 1))])  # (N,4)
                    T = obj.pose_cur @ np.linalg.inv(obj.pose_init)  # 4x4
                    points_trans = (T @ points_h.T).T[:, :3]
                    pcd = points_trans
                    # with open("new_object.txt", "a") as log_file:
                    #     log_file.write(f"消失物体: ID {obj_id}, 帧 {frame_id}, 标签 {label}, 分数 {score}, 点云{pcd}\n")
                    break
        dis_frame_ids.append(dis_frame_id)

        disappeared_boxes.append([0, 0, 0, 0])
        disappeared_masks.append(np.zeros((480, 640), dtype=bool))

        class_idx = -1
        class_idx = OBJECTS.index(label)
        disappeared_classes.append(class_idx)

        score_array = np.zeros(len(OBJECTS))
        score_array[class_idx] = score
        disappeared_scores.append(score_array)

        disappeared_pcds.append(pcd)

    disappeared_boxes = np.array(disappeared_boxes)
    disappeared_masks = np.array(disappeared_masks)
    disappeared_classes = np.array(disappeared_classes)
    disappeared_scores = np.array(disappeared_scores)
    
    # 进行匹配
    match_dict, cent_dists, pcd_box_dists, total_dist, centroids, centroids_new, sizes, sizes_new = hungarian_match_raw(
        frame_id,
        curr_box, curr_classes, curr_scores, curr_pcds, curr_mask,
        disappeared_boxes, disappeared_classes, disappeared_scores, disappeared_pcds, disappeared_masks,
        cost_weights={"iou": 0.0, "score": 0.2, "pcds": 0.59, "pcd_box": 0.01}, threshold= MATCH_HUNGARIAN_THRESHOLD
    )

    if match_dict:
        disappeared_idx = list(match_dict.values())[0]
        matched_obj_id = disappeared_ids[disappeared_idx]

        frame_id_disappeared = dis_frame_ids[disappeared_idx]

        # print(f"帧 {frame_id} :")
        print(f"检测到物体重现: ID {matched_obj_id}, 消失于帧 {frame_id_disappeared}")
        with open("new_object.txt", "a") as log_file:
            log_file.write(f"帧 {frame_id}:\n")
            log_file.write(f"{disappeared_objects}\n")
            log_file.write(f"匹配结果: {match_dict}, 总距离: {total_dist}, box距离: {pcd_box_dists} 中心距离: {cent_dists}\n")
            log_file.write(f"centroids: {centroids}\n")
            log_file.write(f"centroids_new: {centroids_new}\n")
            log_file.write(f"sizes: {sizes}\n")
            log_file.write(f"sizes_new: {sizes_new}\n")
            log_file.write("-------------------------------------------\n")
        # print(disappeared_objects)
        # print(f"匹配结果: {match_dict}, 总距离: {total_dist}, 中心距离: {cent_dists}")
        # print(f"centroids: {centroids}, centroids_new: {centroids_new}")
        # 从disappeared_objects中移除
        if matched_obj_id in disappeared_objects:
            del disappeared_objects[matched_obj_id]

        # 记录物体重现信息到文件
        with open("new_object.txt", "a") as log_file:
            log_file.write(f"检测到物体重现: ID {matched_obj_id}, 消失于帧 {frame_id_disappeared}\n")
            log_file.write(f"new disappeared_objects: {disappeared_objects}\n")
            log_file.write("-------------------------------------------\n")
        

        det["object_id"] = matched_obj_id
        return det
    
    if not skip_fusion:
        # 没有匹配，分配新ID
        det["object_id"] = new_id
        # 记录新物体信息到文件
        with open("new_object.txt", "a") as log_file:
            log_file.write(f"帧 {frame_id}:\n")
            log_file.write(f"{disappeared_objects}\n")
            log_file.write(f"匹配结果: {match_dict}, 总距离: {total_dist}, box距离: {pcd_box_dists} 中心距离: {cent_dists}\n")
            log_file.write(f"centroids: {centroids}\n")
            log_file.write(f"centroids_new: {centroids_new}\n")
            log_file.write(f"sizes: {sizes}\n")
            log_file.write(f"sizes_new: {sizes_new}\n")
            log_file.write("-------------------------------------------\n")
        print(f"帧 {frame_id} :")
        print(f"创建新物体: ID {new_id}, 标签 {det.get('label', '')}, 分数 {det.get('score', 0)}")
        # print(disappeared_objects)
        # print(f"匹配结果: {match_dict}, 总距离: {total_dist}, 中心距离: {cent_dists}")
        # print(f"centroids: {centroids}, centroids_new: {centroids_new}")
        with open("new_object.txt", "a") as log_file:
            log_file.write(f"创建新物体: ID {new_id}, 标签 {det.get('label', '')}, 分数 {det.get('score', 0)}\n")
            log_file.write("-------------------------------------------\n")
        object_num += 1
        return det
    else:
        return None

def visualize_detections(json_file, frame_id, dataset_name, ignored_boxes):
    with open(json_file, 'r') as f:
        data = json.load(f)

    rgb_path = f"{dataset_name}/rgb/rgb_{frame_id:06d}.png"
    image = cv2.imread(rgb_path)

    colors = [
        (0, 0, 255),    # 红色
        (0, 255, 0),    # 绿色
        (255, 0, 0),    # 蓝色
        (0, 255, 255),  # 黄色
        (255, 0, 255),  # 紫色
        (255, 255, 0),  # 青色
        (128, 0, 0),    # 深蓝色
        (0, 128, 0),    # 深绿色
        (0, 0, 128),    # 深红色
    ]

    ignored_color = (255, 255, 255)  # 白色
    for box in ignored_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), ignored_color, 2)
        # 添加"Ignored"文本标签
        cv2.putText(image, "Ignored", (int(x1), int(y1) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, ignored_color, 2)

    for det in data.get("detections", []):
        box = det.get("box", [0, 0, 0, 0])
        label = det.get("label", "")
        score = det.get("score", 0)
        object_id = det.get("object_id", -1)
        
        x1, y1, x2, y2 = box

        color = colors[object_id % len(colors)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f"{label} ({object_id}): {score:.2f}"
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if "mask" in det:
            try:
                mask_bytes = base64.b64decode(det["mask"])
                mask_pil = Image.open(BytesIO(mask_bytes))
                mask_np = np.array(mask_pil) > 0

                mask_color = np.zeros_like(image)
                mask_color[mask_np] = color
                
                alpha = 0.3
                mask_area = mask_np.astype(bool)
                image[mask_area] = cv2.addWeighted(image[mask_area], 1-alpha, mask_color[mask_area], alpha, 0)

            except Exception as e:
                print(f"无法处理mask: {e}")

    output_image_path = json_file.replace('.json', '.png')
    cv2.imwrite(output_image_path, image)

    # window_name = "Object Detection Results"
    # cv2.imshow(window_name, image)
    # cv2.waitKey(1)

def hungarian_match_raw(
    frame_id, bboxes, classes, scores, pcds, masks,
    bboxes_new, classes_new, scores_new, pcds_new, masks_new,
    cost_weights={"iou": 0.4, "score": 0.2, "pcds": 0.2, "pcd_box": 0.2}, threshold=HUNGARIAN_THRESHOLD
):
    N, M = len(scores), len(scores)
    if N == 0 or M == 0:
        return {}, list(range(N)), list(range(M))

    ind_match_dict = {}

    _with_iou_dist = cost_weights.get("iou", 0.0) > 0.0
    _with_score_dist = cost_weights.get("score", 0.0) > 0.0
    _with_pcd_dist = cost_weights.get("pcds", 0.0) > 0.0
    _with_pcd_box_dist = cost_weights.get("pcd_box", 0.0) > 0.0

    def expand_scores(scores, classes):
        _scores = np.zeros((scores.shape[0], len(OBJECTS)))
        _scores[np.arange(scores.shape[0]), classes] = scores.squeeze()
        return _scores

    cent_dists = 1.
    if _with_pcd_dist:
        assert pcds is not None and pcds_new is not None, \
            "Point clouds are not provided for the object detection outputs. Please set 3D data."
        cent_dists, centroids, centroids_new = pcd_distance(pcds, pcds_new)
    
    pcd_box_dist = 1.
    if _with_pcd_box_dist:
        pcd_box_dist, sizes, sizes_new = pcd_size_distance(pcds, pcds_new)

    iou_dists = 1.
    if _with_iou_dist:
        iou_dists = 1 - get_iou_masks(masks, masks_new)
    
    score_dists = 1.
    if _with_score_dist:
        if scores.shape[1] == 1:
            scores = expand_scores(scores, classes)
        if scores_new.shape[1] == 1:
            scores_new = expand_scores(scores_new, classes_new)
        score_dists = 1 - np.sqrt(np.dot(scores, scores_new.T))
        # cosine_sim = 1 - cosine_similarity(scores, scores_new)
        # score_dists = (score_sim + cosine_sim) / 2.0
        # print(scores, scores_new, score_sim, cosine_sim)

    total_dists = cent_dists * cost_weights.get("pcds", 0.0) + \
        iou_dists * cost_weights.get("iou", 0.0) + \
        score_dists * cost_weights.get("score", 0.0) + \
        pcd_box_dist * cost_weights.get("pcd_box", 0.0)

    # print(f"total_dists: {total_dists}, score_dists: {score_dists}, cent_dists: {cent_dists}, pcd_box_dist: {pcd_box_dist}")
    # print("total_dists shape:", total_dists.shape)
    # print("nan?", np.isnan(total_dists).any(), "inf?", np.isinf(total_dists).any())
    # print(total_dists)

    
    row_ind, col_ind = linear_sum_assignment(total_dists)

    # for r, c in zip(row_ind, col_ind):
    #     if total_dists[r, c] > threshold:
    #         with open("hungarian_match.txt", "a") as log_file:
    #             if _with_iou_dist:
    #                 log_file.write(f"帧 {frame_id} 匈牙利匹配: {r} -> {c}, 距离: {total_dists[r, c]}, 点云距离: {cent_dists[r, c]}, iou距离: {iou_dists[r, c]}, score距离: {score_dists[r, c]}\n")
    #             else:
    #                 log_file.write(f"帧 {frame_id} 匈牙利匹配: {r} -> {c}, 距离: {total_dists[r, c]}, 点云距离: {cent_dists[r, c]}, score距离: {score_dists[r, c]}\n")
    #         print(frame_id, r, c, total_dists[r, c])
    ind_match_dict = {r: c for r, c in zip(row_ind, col_ind) if total_dists[r, c] < threshold}
    return ind_match_dict, cent_dists, pcd_box_dist, total_dists, centroids, centroids_new, sizes, sizes_new

def process_single_frame(dataset_name, input_folder, save_folder, frame_id, objects, state, skip_fusion, kernel_size, hand_mask, ee_label):
    # print(objects)
    json_path = f"{save_folder}/detection_{{}}_final.json"
    json_new_path = f"{input_folder}/detection_{{}}.json"
    rgb_path = f"{dataset_name}/rgb/rgb_{{}}.png"
    depth_path = f"{dataset_name}/depth/depth_{{}}.npy"
    camera_pose_path = f"{dataset_name}/pose_txt/camera_pose.txt"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    # 处理第一帧的情况
    if frame_id == 0:
        first_frame_json_dst = json_path.format("000000")
        first_frame_png_dst = first_frame_json_dst.replace('.json', '.png')
        first_frame_json_src = f"{input_folder}/detection_000000.json"
        first_frame_png_src = f"{input_folder}/detection_000000.png"
        if os.path.exists(first_frame_json_src) and os.path.exists(first_frame_png_src):
            shutil.copy2(first_frame_json_src, first_frame_json_dst)
            shutil.copy2(first_frame_png_src, first_frame_png_dst)
            add_object_id(json_path, frame_id, hand_mask)
            return True

    input_folder_path = f"{dataset_name}/rgb"
    files = [f for f in os.listdir(input_folder_path) if f.endswith('.png')]
    files.sort()
    current_idx = files.index(f"rgb_{frame_id:06d}.png")
    if current_idx - STEP >= 0:
        prev_frame_id = int(files[current_idx - STEP].split('_')[-1].split('.')[0])

    # 加载上一帧数据
    start_time = time.time()
    # if state == "holding":
    #     bboxes, classes, scores, pcds, masks, actual_frame_id = load_data_with_fallback(
    #         json_path, rgb_path, depth_path, camera_pose_path, prev_frame_id, kernel_size, objects
    #     )
    #     prev_frame_id = actual_frame_id
    # if state != "holding" or (state == "holding" and bboxes.size == 0):
    pre_flag = True
    bboxes, classes, scores, pcds, masks = load_data(
        json_path, rgb_path, depth_path, camera_pose_path, prev_frame_id, kernel_size, objects, pre_flag
    )
    end_time = time.time()
    # print(f"加载上一帧 {prev_frame_id} 数据耗时: {end_time - start_time:.2f} 秒")
    
    # 加载当前帧数据
    start_time = time.time()
    pre_flag = False
    bboxes_new, classes_new, scores_new, pcds_new, masks_new = load_data(
        json_new_path, rgb_path, depth_path, camera_pose_path, frame_id, kernel_size, objects, pre_flag
    )
    end_time = time.time()
    # print(f"加载帧 {frame_id} 数据耗时: {end_time - start_time:.2f} 秒")
    # print(f"pcd_size:{len(pcds[0])}, {len(pcds_new[0])}, {len(pcds_new[1])}, {len(pcds_new[2])}, {len(pcds_new[3])}")
    # print(f"pcd_size:{pcds[0]}, {pcds_new[0]}")
    
    # 保存点云数据到文件
    # def save_point_clouds(pcds_list, pcds_new_list, prev_frame_id, frame_id):
    #     """
    #     保存点云数据到work_space/pcds目录
    #     Args:
    #         pcds_list: 上一帧的点云列表
    #         pcds_new_list: 当前帧的点云列表
    #         prev_frame_id: 上一帧ID
    #         frame_id: 当前帧ID
    #     """
    #     # 创建保存目录
    #     save_dir = "work_space/pcds"
    #     os.makedirs(save_dir, exist_ok=True)
        
    #     # 保存上一帧点云（蓝色）
    #     prev_save_dir = os.path.join(save_dir, f"frame_{prev_frame_id}")
    #     os.makedirs(prev_save_dir, exist_ok=True)
        
    #     for i, pcd in enumerate(pcds_list):
    #         if pcd is not None and len(pcd.points) > 0:
    #             # 设置颜色为蓝色
    #             pcd.paint_uniform_color([0, 0, 1])
    #             # 保存为PLY格式
    #             pcd_path = os.path.join(prev_save_dir, f"pcd_{i}.ply")
    #             o3d.io.write_point_cloud(pcd_path, pcd)
    #             print(f"保存上一帧点云 {i}: {len(pcd.points)} 个点 -> {pcd_path}")
        
    #     # 保存当前帧点云（红色）
    #     new_save_dir = os.path.join(save_dir, f"frame_{frame_id}")
    #     os.makedirs(new_save_dir, exist_ok=True)
        
    #     for i, pcd in enumerate(pcds_new_list):
    #         if pcd is not None and len(pcd.points) > 0:
    #             # 设置颜色为红色
    #             pcd.paint_uniform_color([1, 0, 0])
    #             # 保存为PLY格式
    #             pcd_path = os.path.join(new_save_dir, f"pcd_{i}.ply")
    #             o3d.io.write_point_cloud(pcd_path, pcd)
    #             print(f"保存当前帧点云 {i}: {len(pcd.points)} 个点 -> {pcd_path}")
        
    #     # 创建合并的点云文件（用于对比）
    #     combined_pcd = o3d.geometry.PointCloud()
        
    #     # 添加上一帧所有点云
    #     for pcd in pcds_list:
    #         if pcd is not None and len(pcd.points) > 0:
    #             pcd_copy = o3d.geometry.PointCloud(pcd)
    #             pcd_copy.paint_uniform_color([0, 0, 1])  # 蓝色
    #             combined_pcd += pcd_copy
        
    #     # 添加当前帧所有点云
    #     for pcd in pcds_new_list:
    #         if pcd is not None and len(pcd.points) > 0:
    #             pcd_copy = o3d.geometry.PointCloud(pcd)
    #             pcd_copy.paint_uniform_color([1, 0, 0])  # 红色
    #             combined_pcd += pcd_copy
        
    #     # 保存合并的点云
    #     if len(combined_pcd.points) > 0:
    #         combined_path = os.path.join(save_dir, f"combined_frame_{prev_frame_id}_vs_{frame_id}.ply")
    #         o3d.io.write_point_cloud(combined_path, combined_pcd)
    #         print(f"保存合并点云: {len(combined_pcd.points)} 个点 -> {combined_path}")
        
    #     print(f"点云保存完成 - 上一帧: {len(pcds_list)} 个点云, 当前帧: {len(pcds_new_list)} 个点云")
    #     print(f"保存目录: {save_dir}")
    #     print("蓝色点云: 上一帧, 红色点云: 当前帧")
    
    # # 调用保存函数
    # try:
    #     save_point_clouds(pcds, pcds_new, prev_frame_id, frame_id)
    # except Exception as e:
    #     print(f"保存点云时出错: {e}")
    
    # 进行匈牙利匹配
    start_time = time.time()
    if bboxes_new.size == 0 or bboxes.size == 0:
        match_dict = {}
    else:
        match_dict, _, _, total_dists, _, _, _, _ = hungarian_match_raw(
            frame_id, bboxes, classes, scores, pcds, masks,
            bboxes_new, classes_new, scores_new, pcds_new, masks_new,
            cost_weights={"iou": 0.0, "score": 0.4, "pcds": 0.59, "pcd_box": 0.01}
        )
    # print(f"total_dists: {total_dists}")
    end_time = time.time()
    # print(f"匈牙利匹配耗时: {end_time - start_time:.2f} 秒")
    
    # 更新检测结果
    start_time = time.time()
    update_detection(json_path, json_new_path, prev_frame_id, frame_id, 
                        match_dict, masks_new, pcds_new, dataset_name, objects, skip_fusion, hand_mask, state, ee_label)
    end_time = time.time()
    # print(f"更新检测结果耗时: {end_time - start_time:.2f} 秒")
    
    # 处理未匹配的物体（消失的物体）
    unmatched_old = [i for i in range(len(bboxes)) if i not in match_dict.keys()]
    if unmatched_old:
        with open(json_path.format(f"{prev_frame_id:06d}"), 'r') as f:
            current_data = json.load(f)
            
        for box_index in unmatched_old:
            if box_index < len(current_data.get("detections", [])):
                det = current_data["detections"][box_index]
                object_id = det.get("object_id", -1)
                score = det.get("score", "unknown")
                label = det.get("label", "unknown")
                
                # if object_id in object_point_clouds:
                #     saved_frame_id, saved_pcd, saved_score, saved_label = object_point_clouds[object_id]
                #     disappeared_objects[object_id] = (frame_id, box_index, saved_score, saved_label, saved_pcd)
                # else:
                #     disappeared_objects[object_id] = (frame_id, box_index, score, label, pcds[box_index])
                disappeared_objects[object_id] = (prev_frame_id)

    return True

# def process_single_frame_img(rgb_img, depth_img, det_json, camera_pose, prev_det_json, prev_camera_pose, frame_id, objects, state, skip_fusion, kernel_size, hand_mask, save_folder=None):
#     """
#     实时处理单帧图像
#     参数:
#         rgb_img: RGB图像 (numpy array)
#         depth_img: 深度图像 (numpy array) 
#         det_json: 当前帧检测结果 (dict)
#         camera_pose: 当前帧相机位姿 (4x4 numpy array)
#         prev_det_json: 上一帧检测结果 (dict)
#         prev_camera_pose: 上一帧相机位姿 (4x4 numpy array)
#         frame_id: 当前帧ID
#         objects: 物体列表
#         state: 状态
#         skip_fusion: 是否跳过融合
#         kernel_size: 核大小
#         hand_mask: 手部掩码
#         save_folder: 保存文件夹（可选，用于保存结果）
#     """
#     if save_folder and not os.path.exists(save_folder):
#         os.makedirs(save_folder)
        
#     # 处理第一帧的情况
#     if frame_id == 0:
#         if save_folder:
#             first_frame_json_dst = f"{save_folder}/detection_000000_final.json"
            
#             # 保存第一帧数据
#             with open(first_frame_json_dst, 'w') as f:
#                 json.dump(det_json, f, indent=2)
            
#             # 可视化并保存图像
#             visualize_detections_realtime(det_json, rgb_img.copy(), frame_id, save_folder)
            
#             # 添加object_id
#             add_object_id_realtime(det_json, frame_id)
#             return det_json
#         else:
#             # 如果不保存，直接添加object_id并返回
#             add_object_id_realtime(det_json, frame_id)
#             return det_json

#     # 加载上一帧数据
#     start_time = time.time()
#     if prev_det_json is None:
#         bboxes, classes, scores, pcds, masks = np.array([]), np.array([]), np.array([]), [], np.array([])
#     else:
#         bboxes, classes, scores, pcds, masks = load_data_realtime(
#             prev_det_json, rgb_img, depth_img, prev_camera_pose, kernel_size
#         )
#     end_time = time.time()
#     print(f"加载上一帧数据耗时: {end_time - start_time:.2f} 秒")
    
#     # 加载当前帧数据
#     start_time = time.time()
#     bboxes_new, classes_new, scores_new, pcds_new, masks_new = load_data_realtime(
#         det_json, rgb_img, depth_img, camera_pose, kernel_size
#     )
#     end_time = time.time()
#     print(f"加载帧 {frame_id} 数据耗时: {end_time - start_time:.2f} 秒")
    
#     # 进行匈牙利匹配
#     start_time = time.time()
#     if bboxes_new.size == 0 or bboxes.size == 0:
#         match_dict = {}
#     else:
#         match_dict, _, _, _, _, _, _, _ = hungarian_match_raw(
#             frame_id, bboxes, classes, scores, pcds, masks,
#             bboxes_new, classes_new, scores_new, pcds_new, masks_new,
#             cost_weights={"iou": 0.3, "score": 0.3, "pcds": 0.2, "pcd_box": 0.2}
#         )
#     end_time = time.time()
#     print(f"匈牙利匹配耗时: {end_time - start_time:.2f} 秒")
    
#     # 更新检测结果
#     start_time = time.time()
#     updated_det_json = update_detection_realtime(
#         det_json, prev_det_json, match_dict, masks_new, pcds_new, 
#         frame_id, objects, skip_fusion, hand_mask, depth_img
#     )
#     end_time = time.time()
#     print(f"更新检测结果耗时: {end_time - start_time:.2f} 秒")
    
#     # 处理未匹配的物体（消失的物体）
#     unmatched_old = [i for i in range(len(bboxes)) if i not in match_dict.keys()]
#     if unmatched_old and prev_det_json:
#         for box_index in unmatched_old:
#             if box_index < len(prev_det_json.get("detections", [])):
#                 det = prev_det_json["detections"][box_index]
#                 object_id = det.get("object_id", -1)
#                 disappeared_objects[object_id] = frame_id

#     # 保存结果（如果指定了保存文件夹）
#     if save_folder:
#         output_json_file = f"{save_folder}/detection_{frame_id:06d}_final.json"
#         with open(output_json_file, 'w') as f:
#             json.dump(updated_det_json, f, indent=2)
        
#         visualize_detections_realtime(updated_det_json, rgb_img.copy(), frame_id, save_folder)

#     return updated_det_json

# def add_object_id_realtime(det_json, frame_id):
#     """为实时检测结果添加object_id"""
#     global object_num
#     detections = det_json.get("detections", [])
#     for i, det in enumerate(detections):
#         det["object_id"] = i
#         object_num += 1

# def update_detection_realtime(curr_det_json, prev_det_json, match_dict, masks_new, pcds_new, frame_id, objects, skip_fusion, hand_mask, depth_img):
#     """实时更新检测结果"""
#     if prev_det_json is None:
#         prev_det_json = {"detections": []}
    
#     # 获取上一帧的object_id映射
#     prev_id_map = {}
#     for i, det in enumerate(prev_det_json.get("detections", [])):
#         if "object_id" in det:
#             prev_id_map[i] = det["object_id"]
    
#     curr_detections = curr_det_json.get("detections", [])
#     new_detections = []
#     processed_indices = set()
#     processed_masks = []
#     ignored_boxes = set()

#     for prev_idx, curr_idx in match_dict.items():
#         if curr_idx < len(curr_detections) and prev_idx in prev_id_map:
#             det = curr_detections[curr_idx]
#             processed_indices.add(curr_idx)

#             if "detection" in det:
#                 detection_list = det.get("detection", [])
#                 best_score = detection_list[0].get("score", 0) if detection_list else 0
#                 best_label = detection_list[0].get("label", "") if detection_list else ""

#                 det["label"] = best_label
#                 det["score"] = best_score
#                 del det["detection"]

#             det["object_id"] = prev_id_map[prev_idx]
#             mask_pil = Image.fromarray(masks_new[curr_idx].astype(np.uint8) * 255)
#             buffer = BytesIO()
#             mask_pil.save(buffer, format="PNG")
#             buffer.seek(0)
#             mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
#             det["mask"] = mask_base64

#             new_detections.append(det)
#             processed_masks.append(masks_new[curr_idx])
#             if det["object_id"] in disappeared_objects:
#                 del disappeared_objects[det["object_id"]]

#     for i, det in enumerate(curr_detections):
#         if i not in processed_indices:
#             if "detection" in det:
#                 detection_list = det.get("detection", [])
#                 best_score = detection_list[0].get("score", 0) if detection_list else 0
#                 best_label = detection_list[0].get("label", "") if detection_list else ""
#                 det["label"] = best_label
#                 det["score"] = best_score
#                 box = det.get("box", [0, 0, 0, 0])
#                 del det["detection"]

#             if best_score > NEW_SCORE_THRESHOLD:
#                 mask = masks_new[i]
#                 mask_area = np.sum(mask)
#                 info = None

#                 # 1. 检查掩码面积是否小于阈值
#                 if mask_area < NEW_MIN_MASK_AREA:
#                     info = f"跳过帧 {frame_id}，掩码面积 {mask_area} 小于阈值 {NEW_MIN_MASK_AREA}\n"
#                     continue
                    
#                 # 3. 查找所有现有对象的点云并比较重合度是否超过阈值
#                 ratio = {}
#                 ratio_flag = False
#                 if objects is not None:
#                     for obj in objects:
#                         points_h = np.hstack([obj._points, np.ones((obj._points.shape[0], 1))])
#                         T = obj.pose_cur @ np.linalg.inv(obj.pose_init)
#                         obj_points_world = (T @ points_h.T).T[:, :3]
                        
#                         overlap_ratio = compute_point_cloud_overlap(obj_points_world, pcds_new[i])
#                         ratio[obj.id] = overlap_ratio
                        
#                         if overlap_ratio > NEW_PCD_THRESHOLD:
#                             info = f"跳过帧 {frame_id}，与物体ID {obj.id} 的点云重叠度为 {overlap_ratio:.2f}，超过阈值 {NEW_PCD_THRESHOLD}\n"
#                             skip_fusion = True
#                             ratio_flag = True
#                             break
                            
#                 # 2. 检查当前掩码是否与手部掩码重叠
#                 if hand_mask is not None:
#                     intersection = np.sum(mask & hand_mask)
#                     if intersection / mask_area == NEW_HAND_THRESHOLD and ratio_flag == False: 
#                         info = f"跳过帧 {frame_id}，当前掩码与手部掩码重叠{intersection / mask_area}\n"
#                         continue
                
#                 if det["label"] == "milkbox":
#                     skip_fusion = True

#                 # 计算掩码中有效深度像素的比例
#                 y_indices, x_indices = np.where(mask)
#                 total_points = len(y_indices)

#                 valid_indices = (y_indices < depth_img.shape[0]) & (x_indices < depth_img.shape[1])
#                 y_valid = y_indices[valid_indices]
#                 x_valid = x_indices[valid_indices]

#                 valid_depth_points = np.sum(depth_img[y_valid, x_valid] > 0)
#                 valid_depth_ratio = valid_depth_points / total_points if total_points > 0 else 0
                
#                 if valid_depth_ratio > DEPTH_RATIO_THRESHOLD:
#                     mask_pil = Image.fromarray(masks_new[i].astype(np.uint8) * 255)
#                     buffer = BytesIO()
#                     mask_pil.save(buffer, format="PNG")
#                     buffer.seek(0)
#                     mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
#                     det["mask"] = mask_base64
#                     updated_det = new_object(det, object_num, masks_new, i, frame_id, pcds_new, objects, skip_fusion)
#                     if updated_det is not None: 
#                         new_detections.append(updated_det)
#                     else: 
#                         ignored_boxes.add(tuple(box))
#                         if info is not None:
#                             with open("hungarian_match.txt", "a") as log_file:
#                                 log_file.write(f"{info}; 指标: {mask_area}, {ratio}")

#     return {"detections": new_detections}

# def visualize_detections_realtime(det_json, rgb_img, frame_id, save_folder):
#     """实时可视化检测结果"""
#     colors = [
#         (0, 0, 255),    # 红色
#         (0, 255, 0),    # 绿色
#         (255, 0, 0),    # 蓝色
#         (0, 255, 255),  # 黄色
#         (255, 0, 255),  # 紫色
#         (255, 255, 0),  # 青色
#         (128, 0, 0),    # 深蓝色
#         (0, 128, 0),    # 深绿色
#         (0, 0, 128),    # 深红色
#     ]

#     for det in det_json.get("detections", []):
#         box = det.get("box", [0, 0, 0, 0])
#         label = det.get("label", "")
#         score = det.get("score", 0)
#         object_id = det.get("object_id", -1)
        
#         x1, y1, x2, y2 = box

#         color = colors[object_id % len(colors)]
#         cv2.rectangle(rgb_img, (x1, y1), (x2, y2), color, 2)
#         text = f"{label} ({object_id}): {score:.2f}"
#         cv2.putText(rgb_img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#         if "mask" in det:
#             try:
#                 mask_bytes = base64.b64decode(det["mask"])
#                 mask_pil = Image.open(BytesIO(mask_bytes))
#                 mask_np = np.array(mask_pil) > 0

#                 mask_color = np.zeros_like(rgb_img)
#                 mask_color[mask_np] = color
                
#                 alpha = 0.3
#                 mask_area = mask_np.astype(bool)
#                 rgb_img[mask_area] = cv2.addWeighted(rgb_img[mask_area], 1-alpha, mask_color[mask_area], alpha, 0)

#             except Exception as e:
#                 print(f"无法处理mask: {e}")

#     output_image_path = f"{save_folder}/detection_{frame_id:06d}_final.png"
#     cv2.imwrite(output_image_path, rgb_img)

#     window_name = "Object Detection Results"
#     cv2.imshow(window_name, rgb_img)
#     cv2.waitKey(1)
