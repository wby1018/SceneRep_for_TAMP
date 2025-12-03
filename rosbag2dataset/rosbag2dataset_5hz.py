#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Pose
from sensor_msgs.msg import JointState, CompressedImage, Image, LaserScan
from message_filters import Subscriber, Cache
from tf2_msgs.msg import TFMessage
import tf
import tf.transformations as tft
import numpy as np
import cv2
import tf2_ros
import os
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
import sensor_msgs.msg
import json
import struct
import tempfile
import shutil
import subprocess
import sys
import threading


DATASET_PATH = "/media/zhy/bcd58cff-609f-4e23-89f6-9fc2e8b36fea/datasets"
SAVE_FREQUENCY = 5  # 保存频率 (Hz)

class PoseProcessor:
    def __init__(self, rosbag_name):
        rospy.init_node("pose_processor")

        self.lock = threading.Lock()

        # 初始化缓存用于存储最新消息
        self.latest_js = None
        self.latest_ee = None
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_bs = None
        self.latest_pu = None
        self.latest_tf_base_map = None
        self.latest_tf_l_camera = None
        self.latest_tf_r_camera = None
        self.joint_transforms = {}
        self.latest_timestamp = None  # 存储最近一次处理的时间戳

        # 初始化消息订阅
        self.js_sub = rospy.Subscriber("/joint_states", JointState, self.js_callback)
        self.ee_sub = rospy.Subscriber("/end_effector_pose", PoseStamped, self.ee_callback)
        self.rgb_sub = rospy.Subscriber("/head_camera/rgb/image_raw/compressed", CompressedImage, self.rgb_callback)
        self.depth_sub = rospy.Subscriber("/head_camera/depth_registered/image_raw", Image, self.depth_callback)
        self.bs_sub = rospy.Subscriber("/base_scan", LaserScan, self.bs_callback)
        self.pu_sub = rospy.Subscriber("/amcl_icp_pose", PoseStamped, self.pu_callback)

        # 初始化发布者
        self.ee_camera_pub = rospy.Publisher("/ee_camera", PoseStamped, queue_size=10)
        self.camera_world_pub = rospy.Publisher("/camera_world", PoseStamped, queue_size=10)
        self.rgb_pub = rospy.Publisher("/head_camera/rgb/compressed", CompressedImage, queue_size=10)
        self.depth_pub = rospy.Publisher("/head_camera/depth_registered", Image, queue_size=10)
        
        # 初始化TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10))  # 增加缓存时间到10秒
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 初始化计数器和工具
        self.save_idx = 0
        self.bridge = CvBridge()
        self.root_dir = f"{DATASET_PATH}/{os.path.splitext(rosbag_name)[0]}"
        subdirs = ["rgb", "depth", "pose_txt"]
        os.makedirs(self.root_dir, exist_ok=True)
        
        for subdir in subdirs:
            dir_path = os.path.join(self.root_dir, subdir)

            if os.path.exists(dir_path):
                for file_name in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file_name)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
            os.makedirs(dir_path, exist_ok=True)
        
        # # 创建输出目录
        # os.makedirs(self.root_dir, exist_ok=True)
        # os.makedirs(os.path.join(self.root_dir, "rgb"), exist_ok=True)
        # os.makedirs(os.path.join(self.root_dir, "depth"), exist_ok=True)
        # os.makedirs(os.path.join(self.root_dir, "pose_txt"), exist_ok=True)

        # 定义姿态文件路径
        self.cam_pose_path = os.path.join(self.root_dir, "pose_txt", "camera_pose.txt")
        self.cam_pose_update_path = os.path.join(self.root_dir, "pose_txt", "camera_pose_update.txt")
        self.ee_pose_path = os.path.join(self.root_dir, "pose_txt", "ee_pose.txt")
        self.l_gripper_pose_path = os.path.join(self.root_dir, "pose_txt", "l_gripper_pose.txt")
        self.r_gripper_pose_path = os.path.join(self.root_dir, "pose_txt", "r_gripper_pose.txt")
        self.base_pose_path = os.path.join(self.root_dir, "pose_txt", "base_pose.txt")
        self.base_update_pose_path = os.path.join(self.root_dir, "pose_txt", "base_update_pose.txt")
        self.joints_pose_path = os.path.join(self.root_dir, "pose_txt", "joints_pose.json")
        self.timestamp_path = os.path.join(self.root_dir, "pose_txt", "timestamps.txt")
        
        # # 清空所有姿态文件
        # for path in [self.cam_pose_path, self.ee_pose_path, 
        #             self.l_gripper_pose_path, self.r_gripper_pose_path, 
        #             self.base_pose_path, self.base_update_pose_path, self.timestamp_path]:  # 添加时间戳文件
        #     with open(path, "w") as f:
        #         pass  # 清空文件
                
        # # 清空关节位姿JSON文件
        # with open(self.joints_pose_path, "w") as f:
        #     json.dump({}, f)

        # 定义关节名称
        self.planning_joint_names = [
            "torso_lift_joint", 
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "upperarm_roll_joint",
            "elbow_flex_joint",
            "forearm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
        ]
        
        # 记录处理成功和失败的次数
        self.success_count = 0
        self.failure_count = 0
        
        # 设置定时器，以5Hz频率处理数据
        self.process_timer = rospy.Timer(rospy.Duration(1.0/SAVE_FREQUENCY), self.process_data_timer)

        self.last_message_time = rospy.Time.now()
        self.message_timeout = rospy.Duration(0.5) 
        self.save_flag = False
        
        rospy.loginfo(f"PoseProcessor 初始化完成，将以 {SAVE_FREQUENCY}Hz 频率保存数据...")

    def pose_to_matrix(self, position, orientation):
        """将位置和方向转换为变换矩阵"""
        trans = tft.translation_matrix([position.x, position.y, position.z])
        rot = tft.quaternion_matrix([orientation.x, orientation.y, orientation.z, orientation.w])
        return np.dot(trans, rot)

    def matrix_to_pose(self, matrix):
        """将变换矩阵转换为位姿消息"""
        trans = tft.translation_from_matrix(matrix)
        quat = tft.quaternion_from_matrix(matrix)

        pose = PoseStamped()
        pose.pose.position.x = trans[0]
        pose.pose.position.y = trans[1]
        pose.pose.position.z = trans[2]
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]
        return pose
    
    def get_camera_pose(self, camera_frame='head_camera_rgb_optical_frame', world_frame='map'):
        """获取相机在世界坐标系中的位姿"""
        try:
            # 使用最新的变换
            transform_stamped = self.tf_buffer.lookup_transform(
                world_frame,     # 目标坐标系
                camera_frame,    # 源坐标系
                rospy.Time(0),   # 获取最新的变换
                rospy.Duration(1.0)  # 等待最多1秒
            )
            
            # 提取平移和旋转
            translation = transform_stamped.transform.translation
            rotation = transform_stamped.transform.rotation
            
            # 四元数转旋转矩阵
            quaternion = [rotation.x, rotation.y, rotation.z, rotation.w]
            rotation_matrix = tft.quaternion_matrix(quaternion)
            
            # 设置矩阵的平移部分
            rotation_matrix[0, 3] = translation.x
            rotation_matrix[1, 3] = translation.y
            rotation_matrix[2, 3] = translation.z
            
            return rotation_matrix
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"获取相机位姿失败: {e}")
            return None
    
    def pose_matrix_to_list(self, T):
        """将变换矩阵转换为列表形式 [idx, x, y, z, qx, qy, qz, qw]"""
        trans = T[:3, 3]
        quat = R.from_matrix(T[:3, :3]).as_quat()
        return [int(self.save_idx)] + list(trans) + list(quat)  # 1编号+3平移+4四元数

    def js_callback(self, msg):
        """关节状态回调"""
        # 检查是否包含所有需要的关节
        joint_dict = dict(zip(msg.name, msg.position))
        if all(joint_name in joint_dict for joint_name in self.planning_joint_names):
            self.latest_js = msg
            self.update_tf_transforms()  # 在关节状态更新时尝试更新TF
            self.last_message_time = rospy.Time.now()

    def ee_callback(self, msg):
        """末端执行器位姿回调"""
        self.latest_ee = msg
        self.latest_timestamp = msg.header.stamp if hasattr(msg, 'header') else rospy.Time.now()
        self.last_message_time = rospy.Time.now()

    def rgb_callback(self, msg):
        """RGB图像回调"""
        self.latest_rgb = msg
        self.last_message_time = rospy.Time.now()

    def depth_callback(self, msg):
        """深度图像回调"""
        self.latest_depth = msg
        self.last_message_time = rospy.Time.now()

    def bs_callback(self, msg):
        """激光扫描回调"""
        self.latest_bs = msg
        self.last_message_time = rospy.Time.now()

    def pu_callback(self, msg):
        """位姿更新回调"""
        self.latest_pu = msg
        self.last_message_time = rospy.Time.now()

    def update_tf_transforms(self):
        """更新TF变换"""
        try:
            # 使用最新的变换
            self.latest_tf_base_map = self.tf_buffer.lookup_transform(
                "map", "base_link", rospy.Time(0), rospy.Duration(0.1))
                
            self.latest_tf_l_camera = self.tf_buffer.lookup_transform(
                "base_link", "l_gripper_finger_link", rospy.Time(0), rospy.Duration(0.1))
                
            self.latest_tf_r_camera = self.tf_buffer.lookup_transform(
                "base_link", "r_gripper_finger_link", rospy.Time(0), rospy.Duration(0.1))
            
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
        except Exception as e:
            rospy.logwarn(f"更新变换失败: {e}")
                    
    
    def process_data_timer(self, event):
        """定时器回调函数，以固定频率处理数据"""
        # 检查是否超过消息超时时间
        time_since_last_msg = rospy.Time.now() - self.last_message_time
        if time_since_last_msg > self.message_timeout and self.save_flag == True:
            self.process_timer.shutdown()
            return
        
        if not self.lock.acquire(blocking=False):
            rospy.logwarn("上一次处理尚未完成，跳过当前触发")
            return
    
        # 首先更新TF变换，确保使用最新数据
        self.update_tf_transforms()
        
        try:
            # 检查所有必要的数据是否都可用
            if not (self.latest_js and self.latest_ee and self.latest_rgb and 
                    self.latest_depth and self.latest_bs and self.latest_pu and
                    self.latest_tf_base_map and self.latest_tf_l_camera and 
                    self.latest_tf_r_camera and len(self.joint_transforms) == len(self.planning_joint_names)):
                # 如果缺少任何必要数据，记录并返回
                rospy.logdebug("数据不完整，等待下一次尝试...")
                return
                
            try:
                # 使用最新消息的时间戳或当前时间
                if self.latest_timestamp is None:
                    t = rospy.Time.now()
                else:
                    t = self.latest_timestamp
                
                # 获取相机在基座坐标系中的位姿
                T_camera_base = self.get_camera_pose(world_frame="base_link")
                if T_camera_base is None:
                    rospy.logwarn("无法获取相机位姿，跳过当前帧")
                    return
                    
                # 计算变换矩阵
                T_base_world = self.pose_to_matrix(
                    self.latest_tf_base_map.transform.translation, 
                    self.latest_tf_base_map.transform.rotation
                )
                T_base_world_update = self.pose_to_matrix(
                    self.latest_pu.pose.position, 
                    self.latest_pu.pose.orientation
                )
                T_l_base = self.pose_to_matrix(
                    self.latest_tf_l_camera.transform.translation, 
                    self.latest_tf_l_camera.transform.rotation
                )
                T_r_base = self.pose_to_matrix(
                    self.latest_tf_r_camera.transform.translation, 
                    self.latest_tf_r_camera.transform.rotation
                )
                T_ee_base = self.pose_to_matrix(
                    self.latest_ee.pose.position, 
                    self.latest_ee.pose.orientation
                )

                # 计算相机位姿相对于世界和各部件相对于相机的变换
                T_camera_world = T_base_world @ T_camera_base
                T_camera_world_update = T_base_world_update @ T_camera_base
                T_ee_camera = np.linalg.inv(T_camera_base) @ T_ee_base
                T_l_camera = np.linalg.inv(T_camera_base) @ T_l_base
                T_r_camera = np.linalg.inv(T_camera_base) @ T_r_base
                
                # 计算关节在相机坐标系中的位置
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

                # 发布位姿消息
                ee_camera = self.matrix_to_pose(T_ee_camera)
                ee_camera.header.stamp = t
                ee_camera.header.frame_id = 'head_camera_rgb_optical_frame'
                self.ee_camera_pub.publish(ee_camera)

                camera_world = self.matrix_to_pose(T_camera_world)
                camera_world.header.stamp = t
                camera_world.header.frame_id = 'map'
                self.camera_world_pub.publish(camera_world)

                # 发布RGB和深度图像
                self.rgb_pub.publish(self.latest_rgb)
                self.depth_pub.publish(self.latest_depth)

                # ==== 保存 RGB 图像 ====
                try:
                    np_arr = np.frombuffer(self.latest_rgb.data, np.uint8)
                    rgb_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    rgb_path = os.path.join(self.root_dir, "rgb", f"rgb_{self.save_idx:06d}.png")
                    cv2.imwrite(rgb_path, rgb_img)
                except Exception as e:
                    rospy.logerr(f"保存RGB图像失败: {e}")

                # ==== 保存 Depth 图像 ====
                try:
                    depth_img = self.bridge.imgmsg_to_cv2(self.latest_depth, desired_encoding="passthrough")
                    depth_clean = np.nan_to_num(depth_img, nan=0)
                    
                    # 转换为米单位 (假设输入为毫米)
                    depth_meters = depth_clean.astype(np.float32)
                    
                    depth_path = os.path.join(self.root_dir, "depth", f"depth_{self.save_idx:06d}.npy")
                    np.save(depth_path, depth_meters)
                        
                except Exception as e:
                    rospy.logerr(f"保存深度图像失败: {e}")

                # ==== 保存关节位姿到JSON文件 ====
                try:
                    joints_data = {}
                    if os.path.exists(self.joints_pose_path):
                        try:
                            with open(self.joints_pose_path, "r") as f:
                                joints_data = json.load(f)
                        except json.JSONDecodeError:
                            joints_data = {}
                    
                    frame_key = f"{self.save_idx:06d}"
                    joints_data[frame_key] = T_joints_camera

                    with open(self.joints_pose_path, "w") as f:
                        json.dump(joints_data, f, indent=2)
                except Exception as e:
                    rospy.logerr(f"保存关节位姿失败: {e}")

                # ==== 保存 Pose（txt） ====
                try:
                    cam_line = self.pose_matrix_to_list(T_camera_world)
                    cam_line_update = self.pose_matrix_to_list(T_camera_world_update)
                    ee_line = self.pose_matrix_to_list(T_ee_camera)
                    l_gripper_line = self.pose_matrix_to_list(T_l_camera)
                    r_gripper_line = self.pose_matrix_to_list(T_r_camera)
                    base_line = [int(self.save_idx), 
                                self.latest_tf_base_map.transform.translation.x,
                                self.latest_tf_base_map.transform.translation.y,
                                R.from_quat([
                                    self.latest_tf_base_map.transform.rotation.x,
                                    self.latest_tf_base_map.transform.rotation.y,
                                    self.latest_tf_base_map.transform.rotation.z,
                                    self.latest_tf_base_map.transform.rotation.w
                                ]).as_euler('zyx', degrees=False)[0]
                                ]

                    with open(self.cam_pose_path, "a") as f:
                        f.write(" ".join([f"{v}" for v in cam_line]) + "\n")
                    with open(self.cam_pose_update_path, "a") as f:
                        f.write(" ".join([f"{v}" for v in cam_line_update]) + "\n")
                    with open(self.ee_pose_path, "a") as f:
                        f.write(" ".join([f"{v}" for v in ee_line]) + "\n")
                    with open(self.l_gripper_pose_path, "a") as f:
                        f.write(" ".join([f"{v}" for v in l_gripper_line]) + "\n")
                    with open(self.r_gripper_pose_path, "a") as f:
                        f.write(" ".join([f"{v}" for v in r_gripper_line]) + "\n")
                    with open(self.base_pose_path, "a") as f:
                        f.write(" ".join([f"{v}" for v in base_line]) + "\n")
                        
                    # ==== 保存时间戳 ====
                    with open(self.timestamp_path, "a") as f:
                        f.write(f"{self.save_idx} {t.secs} {t.nsecs}\n")
                        
                except Exception as e:
                    rospy.logerr(f"保存位姿或时间戳失败: {e}")

                # 更新保存编号并记录成功
                self.save_idx += 1
                self.success_count += 1
                self.save_flag = True
                
                if self.success_count % 10 == 0:  # 每10帧打印一次
                    rospy.loginfo(f"成功处理 {self.success_count} 帧，失败 {self.failure_count} 帧")
                    
            except Exception as e:
                self.failure_count += 1
                rospy.logerr(f"数据处理失败: {e}")
                import traceback
                traceback.print_exc()
        finally:
            self.lock.release()

def main():
    # 获取 rosbag 文件路径
    rosbag_name = sys.argv[1]
    rosbag_path = f"/media/zhy/bcd58cff-609f-4e23-89f6-9fc2e8b36fea/rosbags/{rosbag_name}"

    # 启动 rosbag play
    rosbag_process = subprocess.Popen(
        ["rosbag", "play", rosbag_path, "--clock"],
        stdout=None,
        stderr=None
    )

    # 启动 PoseProcessor
    try:
        processor = PoseProcessor(rosbag_name)
        rospy.loginfo("PoseProcessor 初始化完成，开始处理数据...")
        
        # 使用 rospy.is_shutdown() 检查 ROS 是否正在运行
        rate = rospy.Rate(10)  # 10Hz
        while not rospy.is_shutdown():
            if rosbag_process.poll() is not None:  # 检查 rosbag 是否已退出
                rospy.loginfo("rosbag 播放已完成，准备退出程序...")
                break
            rate.sleep()
    finally:
        # 终止 rosbag 播放
        rosbag_process.terminate()
        rospy.loginfo("已终止 rosbag 播放")

if __name__ == "__main__":
    main()
