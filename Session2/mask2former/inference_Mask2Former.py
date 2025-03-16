import os
import glob
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import csv

# New imports for evaluation
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

# Mapping from model output to KITTI-MOTS labels:
# Model: 0 (person) -> KITTI-MOTS: 2 (pedestrian)
# Model: 2 (car)    -> KITTI-MOTS: 1 (car)
MODEL_TO_KITTI = {0: 2, 2: 1}
# For visualization, label name mapping
KITTI_LABEL_NAMES = {1: "car", 2: "pedestrian"}

def process_kitti_mots(root_dir, sequence_id, model, processor, device, output_dir=None, image_id_start=0):
    """
    Process a sequence from the KITTI-MOTS dataset using Mask2Former.
    
    Args:
        root_dir (str): Path to the KITTI-MOTS dataset root directory.
        sequence_id (str): ID of the sequence to process (e.g., "0000").
        model: Pre-loaded Mask2Former model.
        processor: Pre-loaded image processor.
        device: Device to run inference on.
        output_dir (str, optional): Directory to save visualization results.
        image_id_start (int): Starting image id (to be consistent across sequences).
        
    Returns:
        Returns (predictions, next_image_id), where predictions is a list of 
        dicts in COCO detection format and next_image_id is the updated counter.
    """
    # Setup paths
    image_dir = os.path.join(root_dir, "training", "image_02", sequence_id)
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get list of images in the sequence
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    
    predictions = []
    current_image_id = image_id_start

    # Process each image
    for i, image_path in enumerate(image_paths):
        # Load image
        image = Image.open(image_path)
        
        # Process image and run inference
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process outputs
        instance_results = processor.post_process_instance_segmentation(
            outputs,
            target_sizes=[image.size[::-1]],
            threshold=0.5
        )[0]
        
        # Filter segments: keep only predictions for person (model label 0) and car (model label 2)
        segments_info = instance_results.get("segments_info", [])
        segments_info = [s for s in segments_info if s["label_id"] in (0, 2)]
        
        instance_segmentation = instance_results.get("segmentation", None)
        
        if instance_segmentation is not None and len(segments_info) > 0:
            instance_segmentation = instance_segmentation.cpu().numpy()
            
            masks = []
            boxes = []
            scores = []
            labels = []
            
            for segment in segments_info:
                # Create binary mask for this instance
                mask = (instance_segmentation == segment["id"]).astype(np.uint8)
                masks.append(mask)
                
                # Get bounding box coordinates from mask
                y_indices, x_indices = np.where(mask)
                if len(y_indices) > 0 and len(x_indices) > 0:
                    x1, y1 = np.min(x_indices), np.min(y_indices)
                    x2, y2 = np.max(x_indices), np.max(y_indices)
                    boxes.append([x1, y1, x2, y2])
                else:
                    boxes.append([0, 0, 0, 0])
                
                scores.append(segment.get("score", 1.0))
                # Map model label to KITTI-MOTS label
                labels.append(MODEL_TO_KITTI.get(segment.get("label_id", 0), segment.get("label_id", 0)))
            
            # Visualize results
            if i % 10000 == 0:
                visualize_results(image, masks, boxes, scores, labels)
                
                # Save visualization if output directory is provided
                if output_dir:
                    frame_id = os.path.basename(image_path).split('.')[0]
                    save_path = os.path.join(output_dir, f"{frame_id}_segmented.png")
                    plt.savefig(save_path)
                    plt.close()
            
            # Collect predictions for evaluation (do not filter by score threshold here)
            for mask, score, label in zip(masks, scores, labels):
                rle = maskUtils.encode(np.asfortranarray(mask))
                predictions.append({
                    "image_id": current_image_id,
                    "category_id": label,  # already mapped to KITTI-MOTS convention
                    "segmentation": rle,
                    "score": float(score)
                })
        
        print(f"Processed image {i+1}/{len(image_paths)}: {image_path}")
        current_image_id += 1

    return predictions, current_image_id

def visualize_results(image, masks, boxes, scores, labels):
    """
    Visualize the segmentation results.
    
    Args:
        image: Original image.
        masks: Predicted segmentation masks.
        boxes: Predicted bounding boxes.
        scores: Prediction scores.
        labels: Predicted class labels (in KITTI-MOTS convention).
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    colors = plt.cm.tab10(np.linspace(0, 1, 20))
    
    for mask, box, score, label in zip(masks, boxes, scores, labels):
        # Apply a score threshold for visualization
        if score < 0.5:
            continue
        
        color = colors[label % len(colors)]
        
        # Create a colored mask overlay
        mask_image = np.zeros((mask.shape[0], mask.shape[1], 4))
        mask_image[:, :, 3] = mask * 0.5
        mask_image[:, :, :3] = color[:3]
        
        plt.imshow(mask_image)
        
        # Draw bounding box
        x1, y1, x2, y2 = box
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color=color[:3], linewidth=2)
        
        # Get class name from KITTI mapping
        class_name = KITTI_LABEL_NAMES.get(label, f"Class {label}")
        plt.text(x1, y1, f"{class_name}: {score:.2f}", 
                 color=color[:3], bbox=dict(facecolor='white', alpha=0.8))
    
    plt.axis('off')
    plt.tight_layout()

def process_sequences(root_dir, model, processor, device, sequences_to_process=None, output_root_dir=None):
    """
    Process either all sequences in the KITTI-MOTS dataset or only a user-defined subset.
    
    Args:
        root_dir (str): Path to the KITTI-MOTS dataset root directory.
        model: Pre-loaded Mask2Former model.
        processor: Pre-loaded image processor.
        device: Device to run inference on.
        sequences_to_process (list[str], optional): List of specific sequences (e.g., ["0016", "0017"]).
        output_root_dir (str, optional): Root directory for saving results.
        
    Returns:
        If collect_predictions is True, returns the list of predictions.
    """
    predictions_all = []
    image_id_counter = 0
    if sequences_to_process is None or len(sequences_to_process) == 0:
        seq_dirs = [
            d for d in os.listdir(os.path.join(root_dir, "training", "image_02")) 
            if os.path.isdir(os.path.join(root_dir, "training", "image_02", d))
        ]
    else:
        seq_dirs = sequences_to_process
    
    for seq_id in seq_dirs:
        print(f"Processing sequence {seq_id}")
        seq_output_dir = None
        if output_root_dir:
            seq_output_dir = os.path.join(output_root_dir, seq_id)
        
        preds, image_id_counter = process_kitti_mots(root_dir, seq_id, model, processor, device,
                                                     seq_output_dir, image_id_start=image_id_counter)
        predictions_all.extend(preds)
    
    return predictions_all

def build_ground_truth_coco(root_dir, sequences):
    """
    Build a COCO-style ground-truth dictionary from KITTI-MOTS annotations.
    
    Args:
        root_dir (str): Path to the KITTI-MOTS dataset root directory.
        sequences (list[str]): List of sequence IDs to include.
    
    Returns:
        dict: A dictionary in COCO format containing "images", "annotations", and "categories".
    """
    # KITTI-MOTS ground truth uses 1 for car and 2 for pedestrian.
    coco_gt = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "car"},
            {"id": 2, "name": "pedestrian"}
        ]
    }
    image_id = 0
    ann_id = 0
    for seq in sequences:
        # Read the instances_txt file.
        txt_path = os.path.join(root_dir, "instances_txt", f"{seq}.txt")
        gt_mapping = {}
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                frame = int(parts[0])
                obj_id = int(parts[1])
                # Compute category id from object id.
                cat_id = obj_id // 1000
                # Keep only KITTI-MOTS labels: 1 for car, 2 for pedestrian.
                if cat_id not in (1, 2):
                    continue  # This filters out ignore regions (e.g. label 10) and any other classes.
                if frame not in gt_mapping:
                    gt_mapping[frame] = {}
                gt_mapping[frame][obj_id] = cat_id
        
        # Process ground-truth segmentation masks
        seq_inst_dir = os.path.join(root_dir, "instances", seq)
        img_files = sorted(glob.glob(os.path.join(seq_inst_dir, "*.png")))
        for img_file in img_files:
            base = os.path.basename(img_file)
            # Assume image file name like "000000.png"
            frame_num = int(os.path.splitext(base)[0])
            gt_mask = np.array(Image.open(img_file))
            height, width = gt_mask.shape
            coco_gt["images"].append({
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": os.path.join(seq, base)
            })
            # For each unique instance in the mask (skip background=0)
            for inst_id in np.unique(gt_mask):
                if inst_id == 0:
                    continue
                # Only add annotation if the text file mapping contains it.
                if frame_num in gt_mapping and inst_id in gt_mapping[frame_num]:
                    mask = (gt_mask == inst_id).astype(np.uint8)
                    rle = maskUtils.encode(np.asfortranarray(mask))
                    area = float(maskUtils.area(rle))
                    bbox = maskUtils.toBbox(rle).tolist()  # [x, y, w, h]
                    coco_gt["annotations"].append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": gt_mapping[frame_num][inst_id],
                        "segmentation": rle,
                        "area": area,
                        "bbox": bbox,
                        "iscrowd": 0
                    })
                    ann_id += 1
            image_id += 1
    return coco_gt

def evaluate_coco(coco_gt_dict, predictions):
    """
    Evaluate predictions using COCO evaluation metrics.
    
    Args:
        coco_gt_dict (dict): Ground-truth annotations in COCO format.
        predictions (list): List of prediction dicts in COCO format.
    """
    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()
    
    coco_dt = coco_gt.loadRes(predictions)
    
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    import csv
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate_coco(coco_gt_dict, predictions, output_csv='coco_metrics.csv'):
    """
    Evaluate predictions using COCO evaluation metrics and save results to a CSV file.

    Args:
        coco_gt_dict (dict): Ground-truth annotations in COCO format.
        predictions (list): List of prediction dicts in COCO format.
        output_csv (str): File path for the output CSV file.
    """
    # Initialize COCO ground truth and detection results
    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()

    coco_dt = coco_gt.loadRes(predictions)

    # Run COCO evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Define metric names and corresponding values
    metric_names = [
        'AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
        'AR_1', 'AR_10', 'AR_100', 'AR_small', 'AR_medium', 'AR_large'
    ]
    metric_values = coco_eval.stats.tolist()

    # Write metrics to CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Value'])
        writer.writerows(zip(metric_names, metric_values))

    print(f"COCO metrics saved to {output_csv}")


if __name__ == "__main__":
    # Path to KITTI-MOTS dataset
    kitti_mots_root = "/ghome/c5mcv06/mcv/datasets/C5/KITTI-MOTS"
    
    # Sequences to process
    test_sequences = ["0016", "0017", "0018", "0019", "0020"] # Test sequences
    
    # Output directory for visualizations
    output_dir = "./inference_output/"
    
    # Load model and image processor
    model_name = "facebook/mask2former-swin-base-coco-instance"
    processor = Mask2FormerImageProcessor.from_pretrained(model_name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Process KITTI-MOTS sequences and collect predictions in COCO format
    predictions = process_sequences(
        root_dir=kitti_mots_root,
        model=model,
        processor=processor,
        device=device,
        sequences_to_process=test_sequences,
        output_root_dir=output_dir
    )
    
    # Build ground-truth COCO dictionary from KITTI-MOTS annotations
    coco_gt_dict = build_ground_truth_coco(kitti_mots_root, test_sequences)
    
    # Evaluate predictions using COCO metrics
    model_type = model_name.split("-")[-3]
    metrics_file_name = f"inference_{model_type}_metrics.csv"
    evaluate_coco(coco_gt_dict, predictions, output_csv=metrics_file_name)
