#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
amcl_reader_icp.py
-------------------
订阅:
  /map  (nav_msgs/OccupancyGrid)
  /base_scan (sensor_msgs/LaserScan)

TF:
  map <- laser_link(或scan的frame_id)
  map <- base_link

思路:
1) OccupancyGrid → 点云(target, 世界坐标, z=0) 作为 Open3D ICP 目标。
2) scan 原始极坐标(x=前, y=左) → 局部平面点云(z=0) 作为 source。
3) ICP 初值 = T_map_base (AMCL/TF) × T_base_scan（静态外参），即 map<-scan。
4) ICP 输出 map<-scan，再右乘 scan<-base 得 map<-base = 优化后的机器人位姿。

用法:
  rosrun your_pkg amcl_reader_icp.py \
      _scan_topic:=/base_scan _map_topic:=/map \
      _map_frame:=map _base_frame:=base_link _hz:=15 \
      _icp_max_iter:=30 _icp_max_corr:=0.25 _voxel_map:=0.05
"""

import numpy as np
import rospy
import tf2_ros
import tf.transformations as tft
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from collections import deque
import threading
import subprocess

import open3d as o3d

# -------------------- 工具函数 -------------------- #
def se2_from_xyyaw(x: float, y: float, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.array([[c, -s, x],
                  [s,  c, y],
                  [0,  0, 1]], dtype=np.float64)
    return T

def se2_to_4x4(T2: np.ndarray) -> np.ndarray:
    T4 = np.eye(4, dtype=np.float64)
    T4[:2, :2] = T2[:2, :2]
    T4[0, 3]   = T2[0, 2]
    T4[1, 3]   = T2[1, 2]
    return T4

def mat4_to_se2(T4: np.ndarray) -> np.ndarray:
    T2 = np.eye(3, dtype=np.float64)
    T2[:2, :2] = T4[:2, :2]
    T2[0, 2]   = T4[0, 3]
    T2[1, 2]   = T4[1, 3]
    return T2

def se2_to_xyyaw(T: np.ndarray):
    x, y = T[0, 2], T[1, 2]
    yaw = np.arctan2(T[1, 0], T[0, 0])
    return x, y, yaw

def to_homo(pts: np.ndarray) -> np.ndarray:
    return np.c_[pts, np.ones((pts.shape[0], 1), pts.dtype)]

def apply_T_2d(T: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    return (T @ to_homo(pts_xy).T).T[:, :2]

def np_to_pcd(xyz: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    return pcd


class AMCLReaderICP:
    def __init__(self):
        # ---------------- ROS param ----------------
        self.map_topic   = rospy.get_param("~map_topic",   "/map")
        self.scan_topic  = rospy.get_param("~scan_topic",  "/base_scan")
        self.map_frame   = rospy.get_param("~map_frame",   "map")
        self.base_frame  = rospy.get_param("~base_frame",  "base_link")
        self.hz          = rospy.get_param("~hz", 30)

        # ICP 参数
        self.icp_max_iter   = rospy.get_param("~icp_max_iter", 30)  # 增加最大迭代次数
        self.icp_max_corr   = rospy.get_param("~icp_max_corr", 0.2)  # 增大最近邻距离阈值
        self.voxel_map      = rospy.get_param("~voxel_map", 0.05)     # 地图体素下采样
        self.voxel_scan     = rospy.get_param("~voxel_scan", 0.02)    # scan 下采样 (2D 网格法)
        self.icp_min_pts    = rospy.get_param("~icp_min_pts", 50)

        # TF buffer
        self.tf_buf = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        self.tf_lst = tf2_ros.TransformListener(self.tf_buf)

        # 发布优化后位姿
        self.pub_opt_pose = rospy.Publisher("amcl_icp_pose", PoseStamped, queue_size=1)

        # 数据缓存
        self.lock = threading.Lock()
        self.map_img     = None
        self.map_ext     = None
        self.map_pts_xy  = None
        self.map_pcd     = None

        self.robot_pose  = None     # (x,y,yaw) 来自 TF
        self.traj        = deque(maxlen=50000)

        self.opt_pose    = None     # (x,y,yaw) ICP 优化
        self.opt_traj    = deque(maxlen=50000)

        self.scan_xy_amcl = np.empty((0, 2), np.float32)
        self.scan_xy_icp  = np.empty((0, 2), np.float32)

        # 激光外参缓存
        self.scan_frame_cached = None
        # 直接写死base<-scan外参（用户指定）
        self.T_bs_4x4 = np.array([
            [1.0, 0.0, 0.0, 0.235],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # 订阅
        rospy.Subscriber(self.map_topic,  OccupancyGrid, self._cb_map,  queue_size=1)
        rospy.Subscriber(self.scan_topic, LaserScan,     self._cb_scan, queue_size=1)

        # 定时 TF 取位姿
        rospy.Timer(rospy.Duration(1.0 / self.hz), self._timer_robot_pose)

    # ---------------- Callbacks ----------------
    def _cb_map(self, msg: OccupancyGrid):
        w, h = msg.info.width, msg.info.height
        data = np.array(msg.data, dtype=np.int16).reshape(h, w)
        img = np.empty_like(data, dtype=np.float32)
        img[data < 0]  = 0.5
        img[data == 0] = 1.0
        img[data > 50] = 0.0

        res = msg.info.resolution
        ox  = msg.info.origin.position.x
        oy  = msg.info.origin.position.y
        extent = [ox, ox + w * res, oy, oy + h * res]

        occ = data > 50
        ys, xs = np.where(occ)
        if xs.size:
            mx = ox + (xs + 0.5) * res
            my = oy + (ys + 0.5) * res
            pts_xy = np.stack([mx, my], axis=1).astype(np.float32)
        else:
            pts_xy = None

        if pts_xy is not None and pts_xy.shape[0] > 0:
            xyz = np.c_[pts_xy, np.zeros(len(pts_xy), dtype=np.float32)]
            pcd = np_to_pcd(xyz)
            if self.voxel_map > 0:
                pcd = pcd.voxel_down_sample(self.voxel_map)
        else:
            pcd = None

        with self.lock:
            self.map_img    = img
            self.map_ext    = extent
            self.map_pts_xy = pts_xy
            self.map_pcd    = pcd

    def _cb_scan(self, msg: LaserScan):
        scan_frame = msg.header.frame_id

        # ---- 1. 构造 source 点云 (scan 局部坐标: 原点+X轴前) ----
        ang = msg.angle_min + np.arange(len(msg.ranges)) * msg.angle_increment
        ranges = np.array(msg.ranges, dtype=np.float32)
        # 翻转ranges数组
        ranges = np.flip(ranges)
        valid = np.isfinite(ranges)
        ranges, ang = ranges[valid], ang[valid]
        if ranges.size == 0:
            return

        xs = ranges * np.cos(ang)
        ys = ranges * np.sin(ang)
        src_xy = np.stack([xs, ys], axis=1).astype(np.float32)

        # 2D voxel-like 下采样（简单整数网格法）
        if self.voxel_scan > 0 and src_xy.shape[0] > 0:
            grid = (src_xy / self.voxel_scan).round().astype(np.int32)
            _, idx = np.unique(grid, axis=0, return_index=True)
            src_xy = src_xy[idx]

        xyz_src = np.c_[src_xy, np.zeros(len(src_xy), dtype=np.float32)]
        pcd_src = np_to_pcd(xyz_src)

        # ---- 2. 取 AMCL pose 和 map_pcd ----
        with self.lock:
            pose_tf = self.robot_pose
            map_pcd = self.map_pcd

        if pose_tf is None or map_pcd is None or len(map_pcd.points) == 0 or len(pcd_src.points) < self.icp_min_pts:
            return

        x0, y0, yaw0 = pose_tf
        T_mb_2d  = se2_from_xyyaw(x0, y0, yaw0)  # map <- base
        T_mb_4x4 = se2_to_4x4(T_mb_2d)

        if self.T_bs_4x4 is None:
            return

        # map <- scan 初值
        init_4x4 = T_mb_4x4 @ self.T_bs_4x4
        rospy.logdebug(f"init_4x4: {init_4x4}")

        # ---- 3. ICP ----
        reg = o3d.pipelines.registration.registration_icp(
            pcd_src, map_pcd,
            self.icp_max_corr,
            init_4x4,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=self.icp_max_iter
            )
        )
        rospy.logdebug(f"ICP fitness: {reg.fitness}, rmse: {reg.inlier_rmse}")
        T_ms_4x4 = reg.transformation                      # map <- scan
        T_sb_4x4 = np.linalg.inv(self.T_bs_4x4)            # scan <- base
        T_mb_opt_4x4 = T_ms_4x4 @ T_sb_4x4                 # map <- base (优化)
        rospy.logdebug(f"T_mb_opt_4x4: {T_mb_opt_4x4}")

        # ---- 4. 计算优化后的位姿 ----
        # scan @ AMCL
        T_ms_amcl_4x4 = T_mb_4x4 @ self.T_bs_4x4           # map <- scan (AMCL)
        T_ms_amcl_2d  = mat4_to_se2(T_ms_amcl_4x4)
        scan_xy_amcl  = apply_T_2d(T_ms_amcl_2d, src_xy)

        # scan @ ICP
        T_ms_icp_4x4  = T_mb_opt_4x4 @ self.T_bs_4x4       # map <- scan (ICP)
        T_ms_icp_2d   = mat4_to_se2(T_ms_icp_4x4)
        scan_xy_icp   = apply_T_2d(T_ms_icp_2d, src_xy)

        # ---- 5. 保存 / 发布 ----
        T_mb_opt_2d = mat4_to_se2(T_mb_opt_4x4)
        x2, y2, yaw2 = se2_to_xyyaw(T_mb_opt_2d)

        with self.lock:
            self.opt_pose       = (x2, y2, yaw2)
            self.opt_traj.append((x2, y2))
            self.scan_xy_amcl   = scan_xy_amcl.astype(np.float32)
            self.scan_xy_icp    = scan_xy_icp.astype(np.float32)

        self._publish_opt_pose(x2, y2, yaw2, msg.header.stamp)

    def _timer_robot_pose(self, _):
        try:
            tfmsg = self.tf_buf.lookup_transform(self.map_frame, self.base_frame,
                                                 rospy.Time(0), rospy.Duration(0.05))
            T = self._tf_to_mat(tfmsg)
            x, y = T[0, 3], T[1, 3]
            yaw = np.arctan2(T[1, 0], T[0, 0])
            with self.lock:
                self.robot_pose = (x, y, yaw)
                self.traj.append((x, y))
        except (LookupException, ConnectivityException, ExtrapolationException):
            pass

    # ------------- Utils -------------
    @staticmethod
    def _tf_to_mat(tfmsg):
        t = tfmsg.transform.translation
        q = tfmsg.transform.rotation
        M = tft.quaternion_matrix([q.x, q.y, q.z, q.w]).astype(np.float64)
        M[:3, 3] = [t.x, t.y, t.z]
        return M

    def _publish_opt_pose(self, x, y, yaw, stamp):
        msg = PoseStamped()
        msg.header.stamp = stamp if stamp and stamp.to_sec() != 0.0 else rospy.Time.now()
        msg.header.frame_id = self.map_frame
        msg.pose.position.x = x
        msg.pose.position.y = y
        q = tft.quaternion_from_euler(0, 0, yaw)
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]
        self.pub_opt_pose.publish(msg)

def start_map_server():
    map_file = "/home/zhy/rls/rls-digital-twin/rls_fetch_ws/src/apps/low_level_planning/maps/digital_twin_map.yaml"
    try:
        # 启动 map_server 作为子进程
        process = subprocess.Popen(
            ["rosrun", "map_server", "map_server", map_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        rospy.loginfo(f"map_server 已启动，加载地图文件: {map_file}")
        return process
    except Exception as e:
        rospy.logerr(f"启动 map_server 时出错: {e}")
        return None
    
def main():
    rospy.init_node("amcl_reader_icp", anonymous=False)
    # 启动 map_server
    map_server_process = start_map_server()
    if map_server_process is None:
        rospy.logerr("无法启动 map_server，程序退出")
        return

    # 启动 AMCLReaderICP 节点
    node = AMCLReaderICP()
    try:
        rospy.spin()
    finally:
        # 确保在程序退出时终止 map_server
        if map_server_process:
            map_server_process.terminate()
            rospy.loginfo("map_server 已终止")


if __name__ == "__main__":
    main()