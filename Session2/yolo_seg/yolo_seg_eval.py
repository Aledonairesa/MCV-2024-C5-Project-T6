import os
import torch
import csv
from tqdm import tqdm
import pickle
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_util
import numpy as np
from PIL import Image
import glob
import re

# Paths
base_dir = "/ghome/c5mcv06/mcv/datasets/C5/KITTI-MOTS"
data_dir = "/ghome/c5mcv06/abril_working_dir/yolo/all_frames"  # Directory with test images
annotation_dir = os.path.join(base_dir, "instances_txt")
project = 'pred_outputs'
name = 'pred_with_seg_eval_fine_tune_no_aug'
output_dir = f"{project}/{name}"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)

print("Loading YOLO segmentation model...")
model = YOLO("/ghome/c5mcv06/biel_working_dir/runs/train_outputs/train_finetune_noaug/weights/best.pt")  # Changed to segmentation model

# YOLO uses COCO classes by default, we need only classes 0 (person) and 2 (car)
# Map them to KITTI-MOTS: 0->2 (pedestrian), 2->1 (car)
yolo_to_kitti = {0: 1, 1: 2}  # YOLO class -> KITTI class
class_names = {1: "car", 2: "pedestrian"}
valid_class_ids = [1, 2]  # car, pedestrian

# Get test sequences (adjust as needed)
test_sequences = ["0016", "0017", "0018", "0019", "0020"]

# For COCO evaluation
all_coco_predictions = []
all_coco_seg_predictions = []  # For segmentation evaluation
all_gt_annotations = []
all_gt_segmentations = []  # For segmentation evaluation
images_info = []
ann_id = 1

# Function to convert RLE to mask using COCO mask_util
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

# Function to convert mask to RLE
def mask_to_rle(mask):
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

# Function to convert polygon to mask
def polygon_to_mask(segmentation, height, width):
    from pycocotools import mask as mask_util
    mask = np.zeros((height, width), dtype=np.uint8)
    try:
        # Handle different segmentation formats
        if isinstance(segmentation, list) and len(segmentation) > 0:
            if isinstance(segmentation[0], list):  # Multiple polygons
                polygons = segmentation
            else:  # Single polygon
                polygons = [segmentation]
                
            import cv2
            for polygon in polygons:
                # Convert polygon to numpy array
                pts = np.array(polygon).reshape((-1, 2))
                # Convert to int32
                pts = pts.astype(np.int32)
                # Draw filled polygon
                cv2.fillPoly(mask, [pts], 1)
        return mask
    except Exception as e:
        print(f"Error converting polygon to mask: {e}")
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
                "iscrowd": 0,
                "segmentation": mask_to_rle(mask),  # Convert mask to RLE for evaluation
                "mask": mask  # Keep the mask for visualization
            })
    
    return list(annotations_by_frame.values())

# Parse ground truth annotations
print("Parsing ground truth annotations...")
annotations_data = parse_annotations(annotation_dir, test_sequences, valid_class_ids)
print(f"Found {len(annotations_data)} images with annotations")

# Create COCO ground truth structure
for idx, img_info in enumerate(annotations_data):
    image_id = img_info["image_id"]
    
    # Add image info
    images_info.append({
        "id": image_id,
        "file_name": os.path.basename(img_info["img_path"]),
        "width": img_info["width"],
        "height": img_info["height"]
    })
    
    # Add annotations
    for ann in img_info["annotations"]:
        all_gt_annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": ann["category_id"],
            "bbox": ann["bbox"],
            "area": ann["area"],
            "iscrowd": ann["iscrowd"]
        })
        
        # Add ground truth segmentation
        all_gt_segmentations.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": ann["category_id"],
            "segmentation": ann["segmentation"],
            "area": ann["area"],
            "iscrowd": ann["iscrowd"],
            "bbox": ann["bbox"]
        })
        
        ann_id += 1

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
        
        # Process YOLO predictions to KITTI and COCO formats
        kitti_predictions = []
        kitti_seg_predictions = []
        
        if len(pred.boxes) > 0:
            # For bounding box predictions
            boxes = pred.boxes.xyxy.cpu().numpy()
            cls = pred.boxes.cls.cpu().numpy()
            confs = pred.boxes.conf.cpu().numpy()
            
            # For segmentation predictions (check if masks are available)
            has_masks = hasattr(pred, 'masks') and pred.masks is not None
            
            for idx, (box, cls_id, conf) in enumerate(zip(boxes, cls, confs)):
                # Convert YOLO class to KITTI class
                cls_id = int(cls_id)
                if cls_id not in yolo_to_kitti:
                    continue
                    
                kitti_cls = yolo_to_kitti[cls_id]
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                
                # KITTI format prediction for bounding box
                kitti_predictions.append({
                    "image_id": image_id,
                    "category_id": kitti_cls,
                    "bbox": [x1, y1, x2, y2],  # [x1, y1, x2, y2]
                    "score": float(conf)
                })
                
                # COCO format prediction [x, y, width, height]
                all_coco_predictions.append({
                    "image_id": image_id,
                    "category_id": kitti_cls,
                    "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                    "score": round(float(conf), 3)
                })
                
                # Process segmentation if available
                if has_masks:
                    try:
                        # Get mask for this detection
                        mask = pred.masks[idx].data.cpu().numpy()[0]
                        
                        # Resize mask to original image size if needed
                        if mask.shape != (height, width):
                            from PIL import Image
                            mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
                            mask_img = mask_img.resize((width, height), Image.NEAREST)
                            mask = np.array(mask_img) > 0
                            
                        # Convert mask to RLE
                        rle = mask_to_rle(mask)
                        
                        # Create segmentation prediction
                        seg_prediction = {
                            "image_id": image_id,
                            "category_id": kitti_cls,
                            "segmentation": rle,
                            "score": round(float(conf), 3),
                            "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                            "area": round(float(w * h), 2)
                        }
                        
                        all_coco_seg_predictions.append(seg_prediction)
                        kitti_seg_predictions.append(seg_prediction)
                    except Exception as e:
                        print(f"Error processing segmentation: {e}")
        
        # Store results
        results.append({
            "image_id": image_id,
            "predictions": kitti_predictions,
            "segmentation_predictions": kitti_seg_predictions,
            "ground_truth": gt_info["annotations"],
            "original_size": (height, width),
            "image_path": image_path
        })
        
        # Optional: visualize predictions
        if i % 100 == 0 and j == 0:
            # Save the prediction visualization
            pred_img = pred.plot()
            save_path = os.path.join(output_dir, "visualizations", f"vis_{image_id}.jpg")
            if pred_img is not None:
                # Convert to PIL Image and save
                Image.fromarray(pred_img).save(save_path)
    
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
                x1, y1, x2, y2 = bbox
                line = (
                    f"{frame_id} {pred['category_id']} {pred['score']:.6f} "
                    f"{x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n"
                )
                f.write(line)
        except ValueError:
            print(f"Warning: Could not parse image_id: {image_id}")

# Save segmentation results in KITTI-MOTS format
seg_result_file = os.path.join(output_dir, "segmentation_results.txt")
with open(seg_result_file, 'w') as f:
    for result in results:
        image_id = result["image_id"]
        # Extract sequence_id and frame_id from image_id (format: "sequence_framenum")
        try:
            sequence_id, frame_num = image_id.split("_")
            frame_id = int(frame_num)
            
            for i, pred in enumerate(result.get("segmentation_predictions", [])):
                mask_rle = pred["segmentation"]["counts"]
                class_id = pred["category_id"]
                score = pred["score"]
                
                # Generate a unique track ID (use prediction index as part of it)
                track_id = int(f"{class_id}0{i+1}")
                
                # Format: frame_id track_id class_id img_height img_width rle_mask
                line = f"{frame_id} {track_id} {class_id} {result['original_size'][0]} {result['original_size'][1]} {mask_rle}\n"
                f.write(line)
        except ValueError:
            print(f"Warning: Could not parse image_id for segmentation: {image_id}")

print(f"Inference complete. Results saved to {result_file}")
print(f"Segmentation results saved to {seg_result_file}")
print(f"Visualizations (sampled) saved to {os.path.join(output_dir, 'visualizations')}")

# --- COCO Evaluation for Bounding Boxes ---
print("Performing COCO evaluation for bounding boxes...")
coco_gt_dict = {
    "images": images_info,
    "annotations": all_gt_annotations,
    "categories": [
        {"id": 1, "name": "car"},
        {"id": 2, "name": "pedestrian"}
    ]
}

coco_gt = COCO()
coco_gt.dataset = coco_gt_dict
coco_gt.createIndex()

# Check if we have valid predictions
if not all_coco_predictions:
    print("Warning: No bbox predictions found. Cannot perform bbox evaluation.")
else:
    try:
        coco_dt = coco_gt.loadRes(all_coco_predictions)
        
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        metric_names = [
            "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
            "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large"
        ]
        bbox_metrics = {name: float(coco_eval.stats[i]) for i, name in enumerate(metric_names)}
        
        # Save metrics to CSV
        csv_path = os.path.join(output_dir, "coco_bbox_metrics.csv")
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Metric", "Value"])
            for name, value in bbox_metrics.items():
                writer.writerow([name, value])
        
        print("\nOfficial COCO Bounding Box Evaluation Metrics:")
        for name, value in bbox_metrics.items():
            print(f"{name}: {value}")
        
        # Per-class metrics for bounding boxes
        print("\nPer-class AP50 (Bounding Boxes):")
        for cat_id in [1, 2]:  # car, pedestrian
            # Set evaluation parameters for this class
            coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
            coco_eval.params.catIds = [cat_id]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            print(f"Class {class_names[cat_id]}: AP50 = {coco_eval.stats[1]}")
            
    except Exception as e:
        print(f"Error during COCO bbox evaluation: {e}")
        # Save the problematic data for debugging
        with open(os.path.join(output_dir, "coco_gt_bbox.pkl"), "wb") as f:
            pickle.dump(coco_gt_dict, f)
        with open(os.path.join(output_dir, "coco_preds_bbox.pkl"), "wb") as f:
            pickle.dump(all_coco_predictions, f)

# --- COCO Evaluation for Segmentation ---
print("Performing COCO evaluation for segmentation...")
coco_gt_seg_dict = {
    "images": images_info,
    "annotations": all_gt_segmentations,
    "categories": [
        {"id": 1, "name": "car"},
        {"id": 2, "name": "pedestrian"}
    ]
}

if not all_coco_seg_predictions:
    print("Warning: No segmentation predictions found. Cannot perform segmentation evaluation.")
else:
    try:
        # Create new COCO objects for segmentation evaluation
        coco_gt_seg = COCO()
        coco_gt_seg.dataset = coco_gt_seg_dict
        coco_gt_seg.createIndex()
        
        coco_dt_seg = coco_gt_seg.loadRes(all_coco_seg_predictions)
        
        # Evaluate segmentation
        coco_seg_eval = COCOeval(coco_gt_seg, coco_dt_seg, iouType='segm')
        coco_seg_eval.evaluate()
        coco_seg_eval.accumulate()
        coco_seg_eval.summarize()
        
        seg_metrics = {name: float(coco_seg_eval.stats[i]) for i, name in enumerate(metric_names)}
        
        # Save segmentation metrics to CSV
        csv_path = os.path.join(output_dir, "coco_segm_metrics.csv")
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Metric", "Value"])
            for name, value in seg_metrics.items():
                writer.writerow([name, value])
        
        print("\nOfficial COCO Segmentation Evaluation Metrics:")
        for name, value in seg_metrics.items():
            print(f"{name}: {value}")
        
        # Per-class metrics for segmentation
        print("\nPer-class AP50 (Segmentation):")
        for cat_id in [1, 2]:  # car, pedestrian
            # Set evaluation parameters for this class
            coco_seg_eval = COCOeval(coco_gt_seg, coco_dt_seg, iouType='segm')
            coco_seg_eval.params.catIds = [cat_id]
            coco_seg_eval.evaluate()
            coco_seg_eval.accumulate()
            coco_seg_eval.summarize()
            print(f"Class {class_names[cat_id]}: AP50 = {coco_seg_eval.stats[1]}")
            
    except Exception as e:
        print(f"Error during COCO segmentation evaluation: {e}")
        # Save the problematic data for debugging
        with open(os.path.join(output_dir, "coco_gt_segm.pkl"), "wb") as f:
            pickle.dump(coco_gt_seg_dict, f)
        with open(os.path.join(output_dir, "coco_preds_segm.pkl"), "wb") as f:
            pickle.dump(all_coco_seg_predictions, f)

# Create a combined metrics report
print("\nGenerating combined metrics report...")
combined_metrics = {
    "detection": bbox_metrics if 'bbox_metrics' in locals() else {},
    "segmentation": seg_metrics if 'seg_metrics' in locals() else {}
}

# Save combined metrics to JSON
import json
with open(os.path.join(output_dir, "combined_metrics.json"), "w") as f:
    json.dump(combined_metrics, f, indent=4)

print("Evaluation complete!")