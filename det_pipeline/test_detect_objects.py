#!/usr/bin/env python3
"""
测试 detect_objects_on_image 函数
"""
import numpy as np
import cv2
from detect_objects import detect_objects_on_image

def test_detect_objects():
    """测试目标检测函数"""
    # 创建一个测试图像 (RGB格式)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 在图像上画一些简单的形状来模拟物体
    cv2.rectangle(test_image, (100, 100), (200, 200), (255, 0, 0), -1)  # 红色矩形
    cv2.circle(test_image, (400, 300), 50, (0, 255, 0), -1)  # 绿色圆形
    
    # 测试检测函数
    frame_idx = 0
    datapath = "./test_output"
    
    print("开始测试目标检测...")
    try:
        results = detect_objects_on_image(test_image, frame_idx, datapath)
        print("检测完成！")
        print(f"检测结果: {results}")
        print(f"检测到的目标数量: {len(results['detections'])}")
        
        # 检查输出文件是否生成
        import os
        json_file = os.path.join(datapath, f"detection_{frame_idx:06d}.json")
        png_file = os.path.join(datapath, f"detection_{frame_idx:06d}.png")
        
        if os.path.exists(json_file):
            print(f"✓ JSON文件已生成: {json_file}")
        else:
            print(f"✗ JSON文件未生成: {json_file}")
            
        if os.path.exists(png_file):
            print(f"✓ PNG文件已生成: {png_file}")
        else:
            print(f"✗ PNG文件未生成: {png_file}")
            
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_detect_objects()
