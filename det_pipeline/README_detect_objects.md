# 目标检测函数使用说明

## 概述

`detect_objects_on_image` 函数是基于 OWL-ViT 模型的目标检测函数，可以将原始的 `main` 函数改造成一个可重用的函数。

## 函数签名

```python
def detect_objects_on_image(rgb_image, frame_idx, datapath, score_threshold=None):
    """
    在RGB图像上进行目标检测
    
    Args:
        rgb_image: RGB图像数组 (H, W, 3)，uint8格式
        frame_idx: 帧索引，用于生成文件名
        datapath: 数据路径，用于保存检测结果和渲染图像
        score_threshold: 分数阈值，如果为None则使用默认值
    
    Returns:
        dict: 检测结果，格式为 {"detections": [...]}
    """
```

## 参数说明

- **rgb_image**: 输入的RGB图像，必须是numpy数组，形状为 (H, W, 3)，数据类型为uint8
- **frame_idx**: 帧索引，用于生成输出文件名（如 detection_000000.png）
- **datapath**: 输出路径，函数会在此路径下保存检测结果和渲染图像
- **score_threshold**: 可选的分数阈值，如果为None则使用默认值（第一帧使用0.2，其他帧使用0.02）

## 返回值

函数返回一个字典，包含检测结果：

```python
{
    "detections": [
        {
            "detection": [
                {"label": "milkbox", "score": 0.85},
                {"label": "cola", "score": 0.12}
            ],
            "box": [x1, y1, x2, y2]  # 边界框坐标
        },
        # ... 更多检测结果
    ]
}
```

## 输出文件

函数会在指定的 `datapath` 目录下生成两个文件：

1. **detection_{frame_idx:06d}.json**: 包含检测结果的JSON文件
2. **detection_{frame_idx:06d}.png**: 包含检测框和标签的渲染图像

## 使用示例

### 基本使用

```python
import numpy as np
import cv2
from PIL import Image
from detect_objects import detect_objects_on_image

# 读取图像
img = Image.open("your_image.png").convert("RGB")
rgb_image = np.array(img)

# 进行检测
frame_idx = 0
datapath = "./detection_results"
results = detect_objects_on_image(rgb_image, frame_idx, datapath)

# 处理结果
print(f"检测到 {len(results['detections'])} 个目标")
for detection in results['detections']:
    print(f"边界框: {detection['box']}")
    for det in detection['detection']:
        print(f"  {det['label']}: {det['score']:.3f}")
```

### 批量处理

```python
import os
from PIL import Image
import numpy as np
from detect_objects import detect_objects_on_image

# 处理多个图像
input_folder = "path/to/images"
output_folder = "path/to/results"
os.makedirs(output_folder, exist_ok=True)

for i, filename in enumerate(os.listdir(input_folder)):
    if filename.endswith('.png'):
        # 读取图像
        img = Image.open(os.path.join(input_folder, filename)).convert("RGB")
        rgb_image = np.array(img)
        
        # 进行检测
        results = detect_objects_on_image(rgb_image, i, output_folder)
        print(f"处理 {filename}: 检测到 {len(results['detections'])} 个目标")
```

## 检测的目标类别

默认检测以下目标类别：
- milkbox
- cola
- cup
- apple
- pot
- flowerpot

## 性能优化

- 模型只会在第一次调用时初始化，后续调用会复用已加载的模型
- 支持GPU加速（如果可用）
- 自动清理GPU缓存

## 注意事项

1. 输入图像必须是RGB格式，形状为 (H, W, 3)
2. 函数会自动创建输出目录
3. 第一帧和其他帧使用不同的分数阈值
4. 返回的检测结果格式与原始 `owl_object_scores.py` 保持一致

## 依赖项

确保安装了以下依赖：
- jax
- numpy
- opencv-python
- PIL (Pillow)
- scipy
- scikit-image
- scenic (OWL-ViT)
- torch

## 错误处理

函数会抛出以下异常：
- `ValueError`: 当输入图像格式不正确时
- 其他异常：模型加载或检测过程中的错误
