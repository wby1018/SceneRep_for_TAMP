import sys
import os
import jax
import numpy as np
from scenic.projects.owl_vit import configs
from scipy.special import expit as sigmoid
import skimage
from skimage import io as skimage_io
from scenic.projects.owl_vit import models
import time
import cv2
import torch
import json

# 全局变量
OBJECTS = ['milkbox', 'cola', 'cup', 'apple', 'pot', 'flowerpot']
FIRST_THRESHOLD = 0.2
SCORE_THRESHOLD = 0.02

# 全局变量用于缓存模型
_global_config = None
_global_variables = None
_global_jitted = None
_global_tokenized_queries = None

def initialize_owl_vit():
    """初始化OWL-ViT模型（只初始化一次）"""
    global _global_config, _global_variables, _global_jitted, _global_tokenized_queries
    
    if _global_config is None:
        print("正在初始化OWL-ViT模型...")
        _global_config = configs.owl_v2_clip_b16.get_config(init_mode='canonical_checkpoint')
        module = models.TextZeroShotDetectionModule(
            body_configs=_global_config.model.body,
            objectness_head_configs=_global_config.model.objectness_head,
            normalize=_global_config.model.normalize,
            box_bias=_global_config.model.box_bias)
        _global_variables = module.load_variables(_global_config.init_from.checkpoint_path)
        _global_jitted = jax.jit(module.apply, static_argnames=('train',))
        
        text_queries = OBJECTS
        _global_tokenized_queries = np.array([
            module.tokenize(q, _global_config.dataset_configs.max_query_length)
            for q in text_queries
        ])
        
        _global_tokenized_queries = np.pad(
            _global_tokenized_queries,
            pad_width=((0, 100 - len(text_queries)), (0, 0)),
            constant_values=0)
        print("OWL-ViT模型初始化完成")

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
    # 确保模型已初始化
    initialize_owl_vit()
    
    # 使用全局变量
    config = _global_config
    variables = _global_variables
    jitted = _global_jitted
    tokenized_queries = _global_tokenized_queries
    
    # 设置分数阈值
    if score_threshold is None:
        score_threshold = SCORE_THRESHOLD if frame_idx != 0 else FIRST_THRESHOLD
    
    start_time = time.time()
    
    # 确保输入图像是RGB格式
    if len(rgb_image.shape) == 3 and rgb_image.shape[2] == 3:
        image_rgb = rgb_image.copy()
    else:
        raise ValueError("输入图像必须是RGB格式 (H, W, 3)")
    
    # 转换为float32并归一化
    image = image_rgb.astype(np.float32) / 255.0
    
    # 图像预处理
    h, w, _ = image.shape
    size = max(h, w)
    image_padded = np.pad(
        image, ((0, size - h), (0, size - w), (0, 0)), constant_values=0.5)

    input_image = skimage.transform.resize(
        image_padded,
        (config.dataset_configs.input_size, config.dataset_configs.input_size),
        anti_aliasing=True)
    
    text_queries = OBJECTS

    # 进行检测
    predictions = jitted(
        variables,
        input_image[None, ...],
        tokenized_queries[None, ...],
        train=False)

    predictions = jax.tree_util.tree_map(lambda x: np.array(x[0]), predictions)
    
    end_time = time.time()
    print(f"处理帧 {frame_idx:06d} 耗时 {end_time - start_time:.2f} 秒")

    # 创建输出目录
    os.makedirs(datapath, exist_ok=True)
    
    # 生成输出文件路径
    image_save_path = os.path.join(datapath, f"detection_{frame_idx:06d}.png")
    json_save_path = os.path.join(datapath, f"detection_{frame_idx:06d}.json")
    
    boxes = predictions['pred_boxes']
    logits = predictions['pred_logits'][..., :len(text_queries)]

    # 处理检测结果
    if frame_idx != 0:
        # 非第一帧的处理逻辑
        all_detections = []
        result_img = image_rgb.copy()
        
        for box_idx, box in enumerate(boxes):
            detection = []
            box_logits = logits[box_idx]
            box_scores = sigmoid(box_logits)
            if np.max(box_scores) < score_threshold:
                continue
            
            # 计算边界框坐标
            cx = int(box[0] * w)  # 使用原始图像宽度
            cy = int(box[1] * h)  # 使用原始图像高度
            w_box = int(box[2] * w)
            h_box = int(box[3] * h)
            x1 = int(cx - w_box/2)
            y1 = int(cy - h_box/2)
            x2 = int(cx + w_box/2)
            y2 = int(cy + h_box/2)

            for label_idx, (label_name, score) in enumerate(zip(text_queries, box_scores)):
                if score == max(box_scores):
                    color = (255, 0, 0)  # 红色

                    cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
                    text = f'{label_name}: {score:.2f}'

                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(result_img, 
                                    (x1, y1 - 25), 
                                    (x1 + text_size[0], y1), 
                                    (255, 255, 255), 
                                    -1)
                    cv2.putText(result_img, text, (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                detection.append({
                    "label": label_name,
                    "score": float(score),
                })
            
            detection = sorted(detection, key=lambda x: x["score"], reverse=True)
            detections = {
                "detection": detection,
                "box": [x1, y1, x2, y2]
            }
            all_detections.append(detections)

        json_results = {"detections": all_detections}
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        
        print(f"保存了 {len(all_detections)} 个检测框到 {json_save_path}")
    else:
        # 第一帧的处理逻辑
        scores = sigmoid(np.max(logits, axis=-1))
        labels = np.argmax(predictions['pred_logits'], axis=-1)

        json_results = {"detections": []}
        result_img = image_rgb.copy()

        for score, box, label in zip(scores, boxes, labels):
            label_name = text_queries[label]
            if label_name != 'tomato' and score < score_threshold:
                continue
            if label_name == 'tomato' and score < 0.05:
                continue
            
            # 计算边界框坐标
            cx = int(box[0] * w)  # 使用原始图像宽度
            cy = int(box[1] * h)  # 使用原始图像高度
            w_box = int(box[2] * w)
            h_box = int(box[3] * h)
            x1 = int(cx - w_box/2)
            y1 = int(cy - h_box/2)
            x2 = int(cx + w_box/2)
            y2 = int(cy + h_box/2)
            
            cv2.circle(result_img, (cx, cy), 5, (255, 0, 0), -1)
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            text = f'{text_queries[label]}: {score:1.2f}'
            cv2.rectangle(result_img, (x1, y2), (x1 + len(text)*10, y2 + 25), (255, 255, 255), -1)
            cv2.putText(result_img, text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            json_results["detections"].append({
                "label": text_queries[label],
                "score": float(score),
                "box": [x1, y1, x2, y2]
            })
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    
    # 保存结果
    cv2.imwrite(image_save_path, result_img)
    with open(json_save_path, "w") as f:
        json.dump(json_results, f, indent=2)
    
    return json_results

def main():
    """原始的main函数，用于批量处理"""
    # os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/home/wby/miniforge3/envs/owl/lib/python3.11/site-packages/PyQt5/Qt5/plugins/platforms'
    DATASET_NAME = "/media/wby/2AB9-4188/data_move"
    input_folder = f"{DATASET_NAME}/rgb"
    output_folder = f"{DATASET_NAME}/detection_boxes_{SCORE_THRESHOLD}"
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    files.sort()

    ### OWL_VIT initialization
    # sys.path.append('/home/zhy/Scene_Representation/rosbag_to_dataset/owl/big_vision/')
    config = configs.owl_v2_clip_b16.get_config(init_mode='canonical_checkpoint')
    module = models.TextZeroShotDetectionModule(
        body_configs=config.model.body,
        objectness_head_configs=config.model.objectness_head,
        normalize=config.model.normalize,
        box_bias=config.model.box_bias)
    variables = module.load_variables(config.init_from.checkpoint_path)
    jitted = jax.jit(module.apply, static_argnames=('train',))
    text_queries = OBJECTS
    tokenized_queries = np.array([
        module.tokenize(q, config.dataset_configs.max_query_length)
        for q in text_queries
    ])

    tokenized_queries = np.pad(
        tokenized_queries,
        pad_width=((0, 100 - len(text_queries)), (0, 0)),
        constant_values=0)

    ### OWL_VIT
    for i in range(len(files)):
        print(files[i])
        owl_vit(input_folder, output_folder, files[i], config, variables, jitted, tokenized_queries)
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
