import sys
import os
import jax
from matplotlib import pyplot as plt
import numpy as np
from scenic.projects.owl_vit import configs
from scipy.special import expit as sigmoid
import skimage
from skimage import io as skimage_io
from skimage import transform as skimage_transform
import time

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
from PIL import Image
import cv2
import open3d as o3d

import torch, gc
import matplotlib.cm as cm
import json
import io
import base64
import sys

DATASET_PATH = "/media/zhy/bcd58cff-609f-4e23-89f6-9fc2e8b36fea/datasets"
SCORE_THRESHOLD = 0.02

def sam_masks(input_folder, output_folder, filename, di_name, dj_name, predictor):
    torch.cuda.empty_cache()
    gc.collect()
    file_path = os.path.join(input_folder, filename)
    start_time = time.time()
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    dj_path = os.path.join(output_folder, dj_name)
    with open(dj_path, "r") as f:
        detection_data = json.load(f)
    boxes = []
    for det in detection_data["detections"]:
        boxes.append(det["box"])
    
    if boxes:
        input_boxes = torch.tensor(boxes, device=predictor.device)
        predictor.set_image(image)
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])

        masks, scores, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        dj_path = os.path.join(output_folder, dj_name)
        di_path = os.path.join(output_folder, di_name)
        colors = [
            [0, 255, 0],
            [255, 0, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
        ]
        
        masks_np = masks.cpu().numpy()
        detection_image = cv2.imread(di_path)
        result = detection_image.copy()
        for i, mask in enumerate(masks_np):
            color = colors[i % len(colors)]
            result = show_mask(result, mask, color)
        cv2.imwrite(di_path, result)

        for i, det in enumerate(detection_data["detections"]):
            if i < len(masks_np):
                mask_np = (masks_np[i][0] * 255).astype(np.uint8)
                pil_mask = Image.fromarray(mask_np)
                buffered = io.BytesIO()
                pil_mask.save(buffered, format="PNG")
                mask_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                det["mask"] = mask_base64
        with open(dj_path, "w") as f:
            json.dump(detection_data, f, indent=2)

        end_time = time.time()
        execution_time = end_time - start_time

        print(f"Processed {filename} in {execution_time:.2f} seconds")

        del masks, scores, input_boxes, transformed_boxes
        torch.cuda.empty_cache()
        gc.collect()
        return True

def show_mask(image, mask, color_):
    color = np.array(color_) 
    mask_bool = mask[0] > 0 
    alpha = 0.5 
    image_copy = image.copy()
    colored_mask = np.zeros_like(image_copy)
    colored_mask[mask_bool] = color
    cv2.addWeighted(image_copy, 1.0, colored_mask, alpha, 0, image_copy)
    return image_copy

def main():
    dataset_name = sys.argv[1]
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/home/zhy/anaconda3/envs/owl/lib/python3.11/site-packages/PyQt5/Qt5/plugins/platforms'
    input_folder = f"{DATASET_PATH}/{dataset_name}/rgb"
    output_folder = f"{DATASET_PATH}/{dataset_name}/detection_boxes_{SCORE_THRESHOLD}"
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    files.sort()
    detect_images = [f for f in os.listdir(output_folder) if f.endswith('.png')]
    detect_images.sort()
    detect_jsons = [f for f in os.listdir(output_folder) if f.endswith('.json')]
    detect_jsons.sort()

    ### SAM initialization
    sam = sam_model_registry["vit_b"](checkpoint="rosbag2dataset/sam/sam_vit_b_01ec64.pth")
    sam = sam.to('cuda')
    predictor = SamPredictor(sam)
    print("Loaded SAM model")

    ### SAM
    for i in range(len(files)):
        # if i < 961:
        #     continue
        sam_masks(input_folder, output_folder, files[i], detect_images[i], detect_jsons[i], predictor)
    
    

if __name__ == "__main__":
    main()


