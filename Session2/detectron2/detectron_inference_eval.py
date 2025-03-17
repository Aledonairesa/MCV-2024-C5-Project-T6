import torch
import cv2
import numpy as np
import os
import io
import sys
import glob
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Create a mapping to ensure consistent image IDs
def create_image_id_mapping(root_dir, sequences):
    """Create a mapping from (sequence, frame) to unique image_id"""
    mapping = {}
    next_id = 0
    
    for seq in sequences:
        seq_inst_dir = os.path.join(root_dir, "instances", seq)
        img_files = sorted(glob.glob(os.path.join(seq_inst_dir, "*.png")))
        for img_file in img_files:
            base = os.path.basename(img_file)
            frame_num = int(os.path.splitext(base)[0])
            mapping[(seq, frame_num)] = next_id
            next_id += 1
    
    return mapping

# Inference function with proper image_id mapping
def run_inference(image_path, seq, image_id_mapping, output_dir="output", save_img=False):
    image = cv2.imread(image_path)
    outputs = predictor(image)
    
    # Filter only cars and pedestrians (COCO class IDs: 0=person, 2=car)
    instances = outputs["instances"]
    classes = instances.pred_classes.cpu().numpy()
    keep = (classes == 0) | (classes == 2)
    filtered_instances = instances[keep]
    
    if save_img:
        if len(filtered_instances) > 0:
            # Visualize the predictions
            v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            v = v.draw_instance_predictions(filtered_instances.to("cpu"))
            
            # Save the output image
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, os.path.basename(image_path))
            cv2.imwrite(output_path, v.get_image()[:, :, ::-1])
            print(f"Saved output to {output_path}")
    
    # Convert predictions to COCO format
    predictions = []
    scores = filtered_instances.scores.cpu().numpy()
    masks = filtered_instances.pred_masks.cpu().numpy()
    
    # Get the frame number from the image path
    frame_num = int(os.path.splitext(os.path.basename(image_path))[0])
    
    # Get the correct image_id from our mapping
    if (seq, frame_num) in image_id_mapping:
        correct_image_id = image_id_mapping[(seq, frame_num)]
    else:
        print(f"Warning: Could not find image_id for {seq}/{frame_num}")
        return []
    
    for i, mask in enumerate(masks):
        rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode('utf-8')
        category_id = 1 if classes[i] == 2 else 2
        bbox = maskUtils.toBbox(rle).tolist()
        area = float(maskUtils.area(rle))
        predictions.append({
            "image_id": correct_image_id,
            "category_id": category_id,
            "segmentation": rle,
            "score": float(scores[i]),
            "bbox": bbox,
            "area": area
        })
    return predictions

# Build ground truth in COCO format
def build_ground_truth_coco(root_dir, sequences, image_id_mapping):
    coco_gt = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "car"},
            {"id": 2, "name": "pedestrian"}
        ]
    }
    ann_id = 0
    
    for seq in sequences:
        txt_path = os.path.join(root_dir, "instances_txt", f"{seq}.txt")
        gt_mapping = {}
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                frame = int(parts[0])
                obj_id = int(parts[1])
                cat_id = obj_id // 1000
                if cat_id not in (1, 2):
                    continue
                if frame not in gt_mapping:
                    gt_mapping[frame] = {}
                gt_mapping[frame][obj_id] = cat_id

        seq_inst_dir = os.path.join(root_dir, "instances", seq)
        img_files = sorted(glob.glob(os.path.join(seq_inst_dir, "*.png")))
        for img_file in img_files:
            base = os.path.basename(img_file)
            frame_num = int(os.path.splitext(base)[0])
            
            # Get the correct image_id from our mapping
            if (seq, frame_num) in image_id_mapping:
                image_id = image_id_mapping[(seq, frame_num)]
            else:
                print(f"Warning: Could not find image_id for {seq}/{frame_num} in ground truth")
                continue
                
            gt_mask = np.array(Image.open(img_file))
            height, width = gt_mask.shape
            coco_gt["images"].append({
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": os.path.join(seq, base)
            })
            for inst_id in np.unique(gt_mask):
                if inst_id == 0:
                    continue
                if frame_num in gt_mapping and inst_id in gt_mapping[frame_num]:
                    mask = (gt_mask == inst_id).astype(np.uint8)
                    rle = maskUtils.encode(np.asfortranarray(mask))
                    area = float(maskUtils.area(rle))
                    bbox = maskUtils.toBbox(rle).tolist()
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
    return coco_gt

# Evaluate predictions with pycocotools COCO
def evaluate_coco(coco_gt_dict, predictions):
    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()
    
    coco_dt = coco_gt.loadRes(predictions)
    
    from pycocotools.cocoeval import COCOeval
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval

def save_coco_metrics(coco_eval, output_path="results"):
    # Extract metric names from COCOeval's summarize function
    metric_names = [
        "AP @ IoU=0.50:0.95 (all areas)",
        "AP @ IoU=0.50 (all areas)",
        "AP @ IoU=0.75 (all areas)",
        "AP @ IoU=0.50:0.95 (small objects)",
        "AP @ IoU=0.50:0.95 (medium objects)",
        "AP @ IoU=0.50:0.95 (large objects)",
        "AR @ IoU=0.50:0.95 (all areas, 1 detection)",
        "AR @ IoU=0.50:0.95 (all areas, 10 detections)",
        "AR @ IoU=0.50:0.95 (all areas, 100 detections)",
        "AR @ IoU=0.50:0.95 (small objects)",
        "AR @ IoU=0.50:0.95 (medium objects)",
        "AR @ IoU=0.50:0.95 (large objects)"
    ]

    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "coco_metrics.txt")
    # Save metrics to file with their corresponding names
    with open(output_file, "w") as f:
        for name, value in zip(metric_names, coco_eval.stats):
            f.write(f"{name}: {value:.4f}\n")
    
    print(f"Saved COCO evaluation metrics to {output_file}")

# Load model
model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
predictor = DefaultPredictor(cfg)

# Process all images in KITTI-MOTS training sequences 0016 to 0020
sequences = ["0016", "0017", "0018", "0019", "0020"]
base_path = "/ghome/c5mcv06/mcv/datasets/C5/KITTI-MOTS/training/image_02"
output_dir = "/ghome/c5mcv06/abril_working_dir/detectron2/visu_finetuning_aug"
gt_path = "/ghome/c5mcv06/mcv/datasets/C5/KITTI-MOTS"

# Create image ID mapping first
print("Creating image ID mapping...")
image_id_mapping = create_image_id_mapping(gt_path, sequences)
print(f"Created mapping for {len(image_id_mapping)} images")

# Run individual sequence evaluations
for seq in sequences:
    seq_path = os.path.join(base_path, seq)
    if os.path.exists(seq_path):
        print(f"Evaluating sequence {seq}...")
        seq_predictions = []
        for img_file in sorted(os.listdir(seq_path)):
            if img_file.endswith(".png"):
                img_path = os.path.join(seq_path, img_file)
                pred = run_inference(img_path, seq, image_id_mapping, os.path.join(output_dir, "output_img", seq), save_img=False)
                seq_predictions.extend(pred)
        
        # Filter the mapping to only include this sequence
        seq_mapping = {k: v for k, v in image_id_mapping.items() if k[0] == seq}
        seq_gt = build_ground_truth_coco(gt_path, [seq], seq_mapping)
        seq_eval = evaluate_coco(seq_gt, seq_predictions)
        save_coco_metrics(seq_eval, os.path.join(output_dir, "metriques/inference", seq))

# Now run combined evaluation with all sequences
print("Running combined evaluation for all sequences...")
all_predictions = []
for seq in sequences:
    seq_path = os.path.join(base_path, seq)
    if os.path.exists(seq_path):
        for img_file in sorted(os.listdir(seq_path)):
            if img_file.endswith(".png"):
                img_path = os.path.join(seq_path, img_file)
                pred = run_inference(img_path, seq, image_id_mapping, os.path.join(output_dir, "output_img", seq), save_img=True)
                all_predictions.extend(pred)

# Build ground truth for all sequences
combined_gt = build_ground_truth_coco(gt_path, sequences, image_id_mapping)

# Evaluate overall metrics
print("Evaluating combined metrics for all sequences...")
overall_eval_results = evaluate_coco(combined_gt, all_predictions)
save_coco_metrics(overall_eval_results, os.path.join(output_dir, "metriques/inference", "combined"))
print("Overall evaluation complete!")