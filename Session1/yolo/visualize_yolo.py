import os
import torch
import pickle
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob
import re
from ultralytics import YOLO
import numpy as np

# Paths
base_dir = "/ghome/c5mcv06/mcv/datasets/C5/KITTI-MOTS"
data_dir = "/ghome/c5mcv06/abril_working_dir/yolo/all_frames"  # Directory with test images
annotation_dir = os.path.join(base_dir, "instances_txt")
project = 'pred_outputs'
name = 'pred_with_visualization'
output_dir = f"{project}/{name}"

# Create output directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)

print("Loading YOLO model...")
model = YOLO("yolo11x.pt")

# YOLO uses COCO classes by default, we need only classes 0 (person) and 2 (car)
# Map them to KITTI-MOTS: 0->2 (pedestrian), 2->1 (car)
yolo_to_kitti = {0: 2, 2: 1}  # YOLO class -> KITTI class
class_names = {1: "car", 2: "pedestrian"}
valid_class_ids = [1, 2]  # car, pedestrian

# Get test sequences (adjust as needed)
test_sequences = ["0016", "0017", "0018", "0019", "0020"]

# Function to convert RLE to mask using pycocotools
def rle_to_mask(rle, height, width):
    try:
        from pycocotools import mask as mask_util
        if rle.startswith("WSV:"):
            rle = rle[4:]
        rle_dict = {"size": [height, width], "counts": rle.encode("utf-8")}
        mask = mask_util.decode(rle_dict)
        return mask
    except Exception as e:
        print(f"Error decoding RLE: {e}")
        return np.zeros((height, width), dtype=np.uint8)

# Function to convert mask to bbox with better error handling
def mask_to_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return [0, 0, 1, 1]  # Return 1x1 box instead of 0x0 to avoid errors
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return [int(x_min), int(y_min), int(x_max), int(y_max)]

# Visualization function
def visualize_predictions(
    image,
    predictions,
    ground_truth,
    class_names,
    score_threshold=0.5,
    output_dir=None,
    image_id=None
):
    """
    Visualize model predictions alongside ground truth
    
    Parameters:
    - image: The image to visualize
    - predictions: List of dictionaries with format:
        [{'image_id': str, 'category_id': int, 'bbox': [x, y, w, h], 'score': float, ...}]
    - ground_truth: List of annotations in COCO format
    - class_names: Dictionary mapping class IDs to class names
    - score_threshold: Minimum confidence score for showing predictions
    - output_dir: Directory to save visualization (optional)
    - image_id: ID to use for saving the visualization (optional)
    """
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    # --- Plot Ground Truth (green) ---
    for ann in ground_truth:
        x, y, w, h = ann["bbox"]
        gt_class_id = ann["category_id"]  # [1,2]
        if gt_class_id in class_names:
            class_name = class_names[gt_class_id]
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=1,
                edgecolor='#00FF21',
                facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                x, y - 5,
                f"GT: {class_name}",
                color='#00FF21',
                fontsize=5,
            )

    # --- Plot Predictions (magenta) ---
    for pred in predictions:
        # Skip predictions below threshold
        if pred["score"] < score_threshold:
            continue
            
        # Extract prediction data
        x, y, w, h = pred["bbox"]
        label_id = pred["category_id"]
        conf = pred["score"]

        if label_id in class_names:
            class_name = class_names[label_id]
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=1,
                edgecolor='#FF00DC',
                facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                x, y + h + 10,
                f"Pred: {class_name} ({conf:.2f})",
                color='#FF00DC',
                fontsize=5,
            )

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Save or show
    if output_dir and image_id:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{image_id}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

# Parse the KITTI-MOTS annotations
def parse_annotations(annotation_dir, test_sequences, valid_class_ids):
    print("Parsing KITTI-MOTS annotations...")
    
    ann_files = [f for f in os.listdir(annotation_dir) if f.endswith('.txt')]
    ann_files = [f for f in ann_files if any(seq in f for seq in test_sequences)]
    
    annotations_by_frame = {}
    
    for ann_file in tqdm(ann_files, desc="Parsing annotation files"):
        sequence_id = ann_file.split('.')[0]
        with open(os.path.join(annotation_dir, ann_file), 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 6:
                continue

            frame_id = int(parts[0])
            obj_id = int(parts[1])
            class_id = int(parts[2])  # KITTI IDs: 1=car, 2=pedestrian
            height = int(parts[3])
            width = int(parts[4])
            rle = parts[5]

            # Use only valid classes
            if class_id not in valid_class_ids:
                continue

            img_path = os.path.join(
                base_dir, 
                "training", 
                "image_02", 
                sequence_id, 
                f"{frame_id:06d}.png"
            )

            # Create unique image_id
            image_id = f"{sequence_id}_{frame_id:06d}"
            
            # Skip if image doesn't exist on disk
            if not os.path.exists(img_path):
                continue

            if image_id not in annotations_by_frame:
                annotations_by_frame[image_id] = {
                    "image_id": image_id,
                    "img_path": img_path,
                    "width": width,
                    "height": height,
                    "annotations": []
                }

            mask = rle_to_mask(rle, height, width)
            bbox = mask_to_bbox(mask)

            x_min, y_min, x_max, y_max = bbox
            width_box = x_max - x_min
            height_box = y_max - y_min

            annotations_by_frame[image_id]["annotations"].append({
                "id": obj_id,
                "category_id": class_id,  # Keep as [1,2] for ground truth
                "bbox": [x_min, y_min, width_box, height_box],
                "area": width_box * height_box,
                "iscrowd": 0
            })
    
    return list(annotations_by_frame.values())

# Parse ground truth annotations
print("Parsing ground truth annotations...")
annotations_data = parse_annotations(annotation_dir, test_sequences, valid_class_ids)
print(f"Found {len(annotations_data)} images with annotations")

# Get list of image paths for prediction
image_paths = [img_info["img_path"] for img_info in annotations_data]
print(f"Running predictions on {len(image_paths)} images")

# Run predictions in batches
print("Starting inference...")
batch_size = 16
results = []

for i in range(0, len(image_paths), batch_size):
    batch_files = image_paths[i:i+batch_size]
    batch_preds = model.predict(
        source=batch_files,
        conf=0.25,
        iou=0.65,
        imgsz=640,
        half=True,
        device='cuda',
        batch=batch_size,
        max_det=300,
        visualize=False,
        augment=False,
        agnostic_nms=False,
        classes=[0, 2],  # Person and Car in COCO
        embed=False,
        save=False,
        verbose=False
    )
    
    # Process each prediction
    for j, pred in enumerate(batch_preds):
        image_path = batch_files[j]
        
        # Extract sequence_id and frame_id from image path
        match = re.search(r"([0-9]+)/([0-9]+)\.png$", image_path)
        if match:
            sequence_id, frame_id = match.groups()
            image_id = f"{sequence_id}_{int(frame_id):06d}"
        else:
            # Use basename as fallback
            basename = os.path.splitext(os.path.basename(image_path))[0]
            image_id = basename
        
        # Find corresponding ground truth info
        gt_info = next((item for item in annotations_data if item["image_id"] == image_id), None)
        if gt_info is None:
            print(f"Warning: No ground truth found for image {image_id}")
            continue
        
        width, height = gt_info["width"], gt_info["height"]
        
        # Process YOLO predictions to new format
        coco_format_predictions = []
        
        if len(pred.boxes) > 0:
            boxes = pred.boxes.xyxy.cpu().numpy()
            cls = pred.boxes.cls.cpu().numpy()
            confs = pred.boxes.conf.cpu().numpy()
            
            for box, cls_id, conf in zip(boxes, cls, confs):
                # Convert YOLO class to KITTI class
                cls_id = int(cls_id)
                if cls_id not in yolo_to_kitti:
                    continue
                    
                kitti_cls = yolo_to_kitti[cls_id]
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                
                # COCO format prediction [x, y, width, height]
                coco_format_predictions.append({
                    "image_id": image_id,
                    "category_id": kitti_cls,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(conf),
                    "segmentation": [[x1, y1, x1, y2, x2, y2, x2, y1]],
                    "area": float(w * h),
                    "id": len(coco_format_predictions) + 1,
                    "iscrowd": 0
                })
        
        # Store results
        results.append({
            "image_id": image_id,
            "predictions": coco_format_predictions,
            "ground_truth": gt_info["annotations"],
            "original_size": (height, width),
            "image_path": image_path
        })
        
        # Visualize predictions (for every image)
        try:
            # Load the image
            img = np.array(Image.open(image_path))
            
            # Create visualization directory
            vis_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            
            # Generate and save visualization
            visualize_predictions(
                image=img,
                predictions=coco_format_predictions,
                ground_truth=gt_info["annotations"],
                class_names=class_names,
                score_threshold=0.3,  # Lower threshold to show more predictions
                output_dir=vis_dir,
                image_id=image_id
            )
        except Exception as e:
            print(f"Error visualizing image {image_id}: {e}")
    
    # Print progress
    print(f"Processed {min(i+batch_size, len(image_paths))}/{len(image_paths)} images")

# Save all results to pickle
with open(os.path.join(output_dir, "predictions.pkl"), "wb") as f:
    pickle.dump(results, f)

# Save results to a text file in KITTI-like format
result_file = os.path.join(output_dir, "inference_results.txt")
with open(result_file, 'w') as f:
    for result in results:
        image_id = result["image_id"]
        # Extract sequence_id and frame_id from image_id (format: "sequence_framenum")
        try:
            sequence_id, frame_num = image_id.split("_")
            frame_id = int(frame_num)
            
            for pred in result["predictions"]:
                bbox = pred["bbox"]
                x, y, w, h = bbox
                x2, y2 = x + w, y + h  # Convert to x2, y2 format for the output file
                line = (
                    f"{frame_id} {pred['category_id']} {pred['score']:.6f} "
                    f"{x:.1f} {y:.1f} {x2:.1f} {y2:.1f}\n"
                )
                f.write(line)
        except ValueError:
            print(f"Warning: Could not parse image_id: {image_id}")

print(f"Inference complete. Results saved to {result_file}")
print(f"Visualizations saved to {os.path.join(output_dir, 'visualizations')}")
