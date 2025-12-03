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

DATASET_NAME = "/media/wby/2AB9-4188/data_move"
OBJECTS = ['milkbox', 'cola', 'cup', 'apple', 'pot', 'flowerpot']
FIRST_THRESHOLD = 0.2
SCORE_THRESHOLD = 0.02

# Global variables for OWL-ViT model (initialized once)
_owl_config = None
_owl_variables = None
_owl_jitted = None
_tokenized_queries = None

def initialize_owl_vit():
    """Initialize OWL-ViT model (call this once at the beginning)"""
    global _owl_config, _owl_variables, _owl_jitted, _tokenized_queries
    
    if _owl_config is None:
        print("Initializing OWL-ViT model...")
        _owl_config = configs.owl_v2_clip_b16.get_config(init_mode='canonical_checkpoint')
        module = models.TextZeroShotDetectionModule(
            body_configs=_owl_config.model.body,
            objectness_head_configs=_owl_config.model.objectness_head,
            normalize=_owl_config.model.normalize,
            box_bias=_owl_config.model.box_bias)
        _owl_variables = module.load_variables(_owl_config.init_from.checkpoint_path)
        _owl_jitted = jax.jit(module.apply, static_argnames=('train',))
        text_queries = OBJECTS
        _tokenized_queries = np.array([
            module.tokenize(q, _owl_config.dataset_configs.max_query_length)
            for q in text_queries
        ])
        _tokenized_queries = np.pad(
            _tokenized_queries,
            pad_width=((0, 100 - len(text_queries)), (0, 0)),
            constant_values=0)
        print("OWL-ViT model initialized successfully!")

def detect_objects_in_image(rgb_image, frame_idx, datapath, objects=None, score_threshold=None):
    """
    Detect objects in RGB image using OWL-ViT
    
    Args:
        rgb_image: RGB image as numpy array (H, W, 3) with values 0-255
        frame_idx: Frame index for saving files
        datapath: Directory path to save detection results and rendered images
        objects: List of object names to detect (default: OBJECTS)
        score_threshold: Detection threshold (default: SCORE_THRESHOLD)
    
    Returns:
        dict: Detection results in the same format as owl_object_scores
    """
    # Initialize model if not already done
    initialize_owl_vit()
    
    # Use default values if not provided
    if objects is None:
        objects = OBJECTS
    if score_threshold is None:
        score_threshold = SCORE_THRESHOLD
    
    # Create output directory
    os.makedirs(datapath, exist_ok=True)
    
    # Convert RGB image to the format expected by OWL-ViT
    image_uint8 = rgb_image.astype(np.uint8)
    image = image_uint8.astype(np.float32) / 255.0
    
    h, w, _ = image.shape
    size = max(h, w)
    image_padded = np.pad(
        image, ((0, size - h), (0, size - w), (0, 0)), constant_values=0.5)

    input_image = skimage.transform.resize(
        image_padded,
        (_owl_config.dataset_configs.input_size, _owl_config.dataset_configs.input_size),
        anti_aliasing=True)
    
    # Run inference
    start_time = time.time()
    predictions = _owl_jitted(
        _owl_variables,
        input_image[None, ...],
        _tokenized_queries[None, ...],
        train=False)

    predictions = jax.tree_util.tree_map(lambda x: np.array(x[0]), predictions)
    end_time = time.time()
    print(f"Processed frame {frame_idx} in {end_time - start_time:.2f} seconds")

    # Save paths
    image_save_path = os.path.join(datapath, f"detection_{frame_idx:06d}.png")
    json_save_path = os.path.join(datapath, f"detection_{frame_idx:06d}.json")
    
    boxes = predictions['pred_boxes']
    logits = predictions['pred_logits'][..., :len(objects)]

    # Process detections
    all_detections = []
    result_img = rgb_image.copy()
    
    for box_idx, box in enumerate(boxes):
        detection = []
        box_logits = logits[box_idx]
        box_scores = sigmoid(box_logits)
        if np.max(box_scores) < score_threshold:
            continue
        
        # Convert normalized coordinates to pixel coordinates
        cx = int(box[0] * w)
        cy = int(box[1] * h)
        box_w = int(box[2] * w)
        box_h = int(box[3] * h)
        x1 = int(cx - box_w/2)
        y1 = int(cy - box_h/2)
        x2 = int(cx + box_w/2)
        y2 = int(cy + box_h/2)

        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))

        for label_idx, (label_name, score) in enumerate(zip(objects, box_scores)):
            if score == max(box_scores):
                color = (255, 0, 0)  # Red color for bounding box
                
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

    # Prepare results
    json_results = {"detections": all_detections}
    
    # Save rendered image (convert RGB to BGR for OpenCV)
    result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_save_path, result_img_bgr)
    
    # Save JSON results
    with open(json_save_path, "w") as f:
        json.dump(json_results, f, indent=2)
    
    print(f"保存了 {len(all_detections)} 个检测框到 {json_save_path}")
    
    return json_results

def main():
    """Original main function for batch processing (kept for compatibility)"""
    # os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/home/wby/miniforge3/envs/owl/lib/python3.11/site-packages/PyQt5/Qt5/plugins/platforms'
    input_folder = f"{DATASET_NAME}/rgb"
    output_folder = f"{DATASET_NAME}/detection_boxes_{SCORE_THRESHOLD}"
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    files.sort()

    # Initialize OWL-ViT
    initialize_owl_vit()

    # Process all files
    for i in range(len(files)):
        print(files[i])
        # Read image
        file_path = os.path.join(input_folder, files[i])
        image = cv2.imread(file_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract frame index from filename
        frame_idx = int(files[i].split('_')[-1].split('.')[0])
        
        # Detect objects
        detect_objects_in_image(image_rgb, frame_idx, output_folder)
    
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
