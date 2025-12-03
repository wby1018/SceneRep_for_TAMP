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
import sys

DATASET_PATH = "/media/zhy/bcd58cff-609f-4e23-89f6-9fc2e8b36fea/datasets"
OBJECTS = ['apple', 'cup', 'bottle', 'cola']
FIRST_THRESHOLD = 0.2
SCORE_THRESHOLD = 0.02

def owl_vit(input_folder, output_folder, filename, config, variables, jitted, tokenized_queries):
    index = filename.split('_')[-1].split('.')[0]
    file_path = os.path.join(input_folder, filename)
    start_time = time.time()
    image_orignal = cv2.imread(file_path)
    image_orignal = cv2.cvtColor(image_orignal, cv2.COLOR_BGR2RGB)
    image_uint8 = skimage_io.imread(file_path)
    image = image_uint8.astype(np.float32) / 255.0

    h, w, _ = image.shape
    size = max(h, w)
    image_padded = np.pad(
        image, ((0, size - h), (0, size - w), (0, 0)), constant_values=0.5)

    input_image = skimage.transform.resize(
        image_padded,
        (config.dataset_configs.input_size, config.dataset_configs.input_size),
        anti_aliasing=True)
    
    text_queries = OBJECTS

    predictions = jitted(
        variables,
        input_image[None, ...],
        tokenized_queries[None, ...],
        train=False)

    predictions = jax.tree_util.tree_map(lambda x: np.array(x[0]), predictions )
    end_time = time.time()
    print(f"Processed {filename} in {end_time - start_time:.2f} seconds")

    image_save_path = os.path.join(output_folder, f"detection_{index}.png")
    json_save_path = os.path.join(output_folder, f"detection_{index}.json")
    boxes = predictions['pred_boxes']
    logits = predictions['pred_logits'][..., :len(text_queries)]

    if index != "000000":
        score_threshold = SCORE_THRESHOLD
        all_detections = []
        result_img = image_orignal.copy()
        
        for box_idx, box in enumerate(boxes):
            detection = []
            box_logits = logits[box_idx]
            box_scores = sigmoid(box_logits)
            if np.max(box_scores) < score_threshold:
                continue
            
            cx = int(box[0] * 640)
            cy = int(box[1] * 640)
            w = int(box[2] * 640)
            h = int(box[3] * 640)
            x1 = int(cx - w/2)
            y1 = int(cy - h/2)
            x2 = int(cx + w/2)
            y2 = int(cy + h/2)

            for label_idx, (label_name, score) in enumerate(zip(text_queries, box_scores)):
                # if score > score_threshold:
                # 为不同标签使用不同颜色
                # if 'wooden' in label_name:
                #     color = (0, 255, 0)  # 绿色(RGB)
                # elif 'flower' in label_name:
                #     color = (255, 0, 0)  # 红色(RGB)
                # else:
                #     color = (0, 0, 255)  # 蓝色(RGB)
                if score == max(box_scores):
                    color = (255, 0, 0)

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
                "box": [x1, y1, x2, y2]}
            all_detections.append(detections)

        json_results = {"detections": all_detections}
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        
        print(f"保存了 {len(all_detections)} 个检测框到 {json_save_path}")
    else:
        score_threshold = FIRST_THRESHOLD

        scores = sigmoid(np.max(logits, axis=-1))
        labels = np.argmax(predictions['pred_logits'], axis=-1)

        json_results = {"detections": []}
        result_img = image_orignal.copy()

        for score, box, label in zip(scores, boxes, labels):
            label_name = text_queries[label]
            if label_name in ['apple', 'bottle'] and score < score_threshold:
                continue
            if label_name == 'cola' and score < 0.3:
                continue
            if label_name == 'cup' and score < 0.1:
                continue
            # if score < score_threshold:
            #     continue
            cx = int(box[0] * 640)
            cy = int(box[1] * 640)
            w = int(box[2] * 640)
            h = int(box[3] * 640)
            x1 = int(cx - w/2)
            y1 = int(cy - h/2)
            x2 = int(cx + w/2)
            y2 = int(cy + h/2)
            
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
    cv2.imwrite(image_save_path, result_img)
    with open(json_save_path, "w") as f:
        json.dump(json_results, f, indent=2)

def main():
    dataset_name = sys.argv[1]
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/home/zhy/anaconda3/envs/owl/lib/python3.11/site-packages/PyQt5/Qt5/plugins/platforms'
    input_folder = f"{DATASET_PATH}/{dataset_name}/rgb"
    output_folder = f"{DATASET_PATH}/{dataset_name}/detection_boxes_{SCORE_THRESHOLD}"
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.png')]
    files.sort()

    ### OWL_VIT initialization
    sys.path.append('/home/zhy/scenerep/rosbag2dataset/owl/big_vision')
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


