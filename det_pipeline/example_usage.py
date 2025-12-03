#!/usr/bin/env python3
"""
Example usage of the new detect_objects_in_image function
"""
import cv2
import numpy as np
from owl_object_scores import detect_objects_in_image

def main():
    # Example: Load an RGB image
    image_path = "/path/to/your/image.png"  # Replace with actual image path
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Example: Detect objects
    frame_idx = 0
    datapath = "./detection_results"
    
    # Call the detection function
    detection_results = detect_objects_in_image(
        rgb_image=image_rgb,
        frame_idx=frame_idx,
        datapath=datapath,
        objects=['milkbox', 'cola', 'cup'],  # Optional: specify objects to detect
        score_threshold=0.02  # Optional: specify detection threshold
    )
    
    # Print results
    print(f"Detected {len(detection_results['detections'])} objects:")
    for i, detection in enumerate(detection_results['detections']):
        print(f"Object {i+1}: {detection['detection'][0]['label']} (score: {detection['detection'][0]['score']:.3f})")
        print(f"  Bounding box: {detection['box']}")

if __name__ == "__main__":
    main()
