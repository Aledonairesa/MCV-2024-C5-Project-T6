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
from pycocotools import mask as mask_util

# Paths
base_dir = "/ghome/c5mcv06/mcv/datasets/C5/KITTI-MOTS"
data_dir = "/ghome/c5mcv06/abril_working_dir/yolo/all_frames"  # Directory with test images
annotation_dir = os.path.join(base_dir, "instances_txt")
project = 'pred_outputs'
name = 'pred_with_seg_vis_fine_tune_no_aug'
output_dir = f"{project}/{name}"

# Create output directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)

print("Loading YOLO segmentation model...")
# Load the segmentation model instead of the detection model
model = YOLO("/ghome/c5mcv06/biel_working_dir/runs/train_outputs/train_finetune_noaug/weights/best.pt")

# YOLO uses COCO classes by default, we need only classes 0 (person) and 2 (car)
# Map them to KITTI-MOTS: 0->2 (pedestrian), 2->1 (car)
yolo_to_kitti = {0: 1, 1: 2} # YOLO class -> KITTI class
class_names = {1: "car", 2: "pedestrian"}
valid_class_ids = [1, 2]  # car, pedestrian

# Get test sequences (adjust as needed)
test_sequences = ["0016", "0017", "0018", "0019", "0020"]

# Function to convert RLE to mask using pycocotools
def rle_to_mask(rle, height, width):
    try:
        if rle.startswith("WSV:"):
            rle = rle[4:]
        rle_dict = {"size": [height, width], "counts": rle.encode("utf-8")}
        mask = mask_util.decode(rle_dict)
        return mask
    except Exception as e:
        print(f"Error decoding RLE: {e}")
        return np.zeros((height, width), dtype=np.uint8)

# Function to convert mask to RLE encoding
def mask_to_rle(mask):
    try:
        mask = np.asfortranarray(mask.astype(np.uint8))
        rle = mask_util.encode(mask)
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle["counts"]
    except Exception as e:
        print(f"Error encoding mask to RLE: {e}")
        # Return a simple RLE for an empty mask
        return "0"

# Function to convert mask to bbox with better error handling
def mask_to_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return [0, 0, 1, 1]  # Return 1x1 box instead of 0x0 to avoid errors
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return [int(x_min), int(y_min), int(x_max), int(y_max)]

# Updated visualization function to show segmentation masks
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
    Visualize model predictions (segmentation masks) alongside ground truth (bounding boxes)
    
    Parameters:
    - image: The image to visualize
    - predictions: List of dictionaries with format:
        [{'image_id': str, 'category_id': int, 'bbox': [x, y, w, h], 'score': float, 'segmentation': mask, ...}]
    - ground_truth: List of annotations in COCO format
    - class_names: Dictionary mapping class IDs to class names
    - score_threshold: Minimum confidence score for showing predictions
    - output_dir: Directory to save visualization (optional)
    - image_id: ID to use for saving the visualization (optional)
    """
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    
    # Create a mask overlay for all predictions
    height, width = image.shape[:2]
    mask_overlay = np.zeros((height, width, 4), dtype=np.uint8)
    
    # --- Plot Ground Truth bounding boxes (green) ---
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

    # --- Plot Predictions (segmentation masks with colored overlay) ---
    for pred in predictions:
        # Skip predictions below threshold
        if pred["score"] < score_threshold:
            continue
            
        # Extract prediction data
        label_id = pred["category_id"]
        conf = pred["score"]
        
        # Get the segmentation mask
        if "segmentation_mask" in pred and pred["segmentation_mask"] is not None:
            # Binary mask from the model
            mask = pred["segmentation_mask"]
            
            # Choose color based on class
            if label_id == 1:  # car
                color = np.array([255, 0, 0, 100])  # red with alpha
            else:  # pedestrian
                color = np.array([0, 0, 255, 100])  # blue with alpha
                
            # Add to the overlay
            mask_3d = np.stack([mask, mask, mask, mask], axis=-1)
            mask_overlay = np.where(mask_3d > 0, 
                                    np.maximum(mask_overlay, color), 
                                    mask_overlay)
            
            # Get the bounding box from the mask for label positioning
            x, y, w, h = pred["bbox"]
            
            if label_id in class_names:
                class_name = class_names[label_id]
                ax.text(
                    x, y + h + 10,
                    f"Pred: {class_name} ({conf:.2f})",
                    color='#FF00DC',
                    fontsize=5,
                )
        else:
            # If no mask is available, show the bounding box
            x, y, w, h = pred["bbox"]
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
    
    # Add the mask overlay to the image
    ax.imshow(mask_overlay, alpha=0.5)
    
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
                "segmentation": rle,  # Keep the RLE string
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
        #classes=[0, 2],  # Person and Car in COCO
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
            # Get the bounding boxes
            boxes = pred.boxes.xyxy.cpu().numpy()
            cls = pred.boxes.cls.cpu().numpy()
            confs = pred.boxes.conf.cpu().numpy()
            
            # Get the segmentation masks if available
            masks = pred.masks.data.cpu().numpy() if hasattr(pred, 'masks') and pred.masks is not None else None
            
            for idx, (box, cls_id, conf) in enumerate(zip(boxes, cls, confs)):
                # Convert YOLO class to KITTI class
                cls_id = int(cls_id)
                if cls_id not in yolo_to_kitti:
                    continue
                    
                kitti_cls = yolo_to_kitti[cls_id]
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                
                # Process segmentation mask if available
                mask_data = None
                rle_string = None
                
                try:
                    if masks is not None and idx < len(masks):
                        # Get the mask for this prediction
                        mask = masks[idx]
                        
                        # Resize the mask to the original image dimensions
                        if mask.shape[0] != height or mask.shape[1] != width:
                            mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
                            mask_img = mask_img.resize((width, height), Image.NEAREST)
                            mask = np.array(mask_img) > 0
                        
                        # Convert mask to RLE format
                        rle_string = mask_to_rle(mask)
                        
                        # Save the binary mask for visualization
                        mask_data = mask.astype(bool)
                except Exception as e:
                    print(f"Error processing mask for prediction {idx} in image {image_id}: {e}")
                    # Create a simple mask from the bounding box as fallback
                    mask = np.zeros((height, width), dtype=np.uint8)
                    x1_int, y1_int, x2_int, y2_int = int(x1), int(y1), int(x2), int(y2)
                    mask[y1_int:y2_int, x1_int:x2_int] = 1
                    rle_string = mask_to_rle(mask)
                    mask_data = mask.astype(bool)
                
                # COCO format prediction with segmentation mask
                coco_format_predictions.append({
                    "image_id": image_id,
                    "category_id": kitti_cls,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(conf),
                    "segmentation": rle_string,
                    "segmentation_mask": mask_data,  # For visualization
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
try:
    with open(os.path.join(output_dir, "predictions.pkl"), "wb") as f:
        pickle.dump(results, f)
    print(f"Successfully saved predictions to {os.path.join(output_dir, 'predictions.pkl')}")
except Exception as e:
    print(f"Error saving predictions to pickle: {e}")

# Save results to a text file in KITTI-MOTS format
result_file = os.path.join(output_dir, "inference_results.txt")
try:
    with open(result_file, 'w') as f:
        for result in results:
            image_id = result["image_id"]
            # Extract sequence_id and frame_id from image_id (format: "sequence_framenum")
            try:
                sequence_id, frame_num = image_id.split("_")
                frame_id = int(frame_num)
                
                for pred in result["predictions"]:
                    category_id = pred["category_id"]
                    score = pred["score"]
                    
                    # For KITTI-MOTS format, we need: frame_id class_id id rle
                    # Use a dummy object ID (use prediction index for uniqueness)
                    obj_id = pred["id"]
                    
                    # Get the mask or bounding box info
                    if "segmentation" in pred and pred["segmentation"] is not None:
                        # We already have the RLE string from the model
                        rle_string = pred["segmentation"]
                        height, width = result["original_size"]
                        
                        # Write in KITTI-MOTS format: frame_id instance_id class_id img_height img_width rle
                        line = f"{frame_id} {obj_id} {category_id} {height} {width} {rle_string}\n"
                        f.write(line)
                    else:
                        # If no segmentation available, use bounding box to create a simple mask
                        try:
                            x, y, w, h = [int(v) for v in pred["bbox"]]
                            height, width = result["original_size"]
                            
                            # Create a simple mask from the bounding box
                            mask = np.zeros((height, width), dtype=np.uint8)
                            mask[y:y+h, x:x+w] = 1
                            
                            # Convert to RLE
                            rle_string = mask_to_rle(mask)
                            
                            # Write in KITTI-MOTS format
                            line = f"{frame_id} {obj_id} {category_id} {height} {width} {rle_string}\n"
                            f.write(line)
                        except Exception as e:
                            print(f"Error creating mask from bbox for prediction in image {image_id}: {e}")
            except ValueError as e:
                print(f"Warning: Could not parse image_id: {image_id}, error: {e}")
    print(f"Inference complete. Results saved to {result_file}")
except Exception as e:
    print(f"Error saving results to text file: {e}")

print(f"Visualizations saved to {os.path.join(output_dir, 'visualizations')}")