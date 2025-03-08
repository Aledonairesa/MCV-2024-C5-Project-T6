import os
import torch
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
from tqdm import tqdm
import csv

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils_DETR import KITTIMOTSInferenceDataset, collate_fn_inference, visualize_predictions

print("Evaluating the final fine-tuned model")

# Paths
base_dir = "/ghome/c5mcv06/mcv/datasets/C5/KITTI-MOTS"
data_dir = base_dir  # Base directory containing the 'training' folder
annotation_dir = os.path.join(base_dir, "instances_txt")
output_dir = "./results_detr_kittimots_finetuning"

# Saved final model directory
model_dir = f"{output_dir}/final_model"
vis_output_dir = f"{output_dir}/visualizations"

# Load the fine-tuned DETR model and corresponding processor
processor = DetrImageProcessor.from_pretrained(model_dir)
model = DetrForObjectDetection.from_pretrained(model_dir)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Model and processor loaded successfully!")

# Create inference dataset
inference_dataset = KITTIMOTSInferenceDataset(
    data_dir=data_dir,
    annotation_dir=annotation_dir,
    processor=processor,
    split="test"
)

inference_dataloader = DataLoader(
    inference_dataset, 
    batch_size=1,
    shuffle=False,
    num_workers=2,
    collate_fn=collate_fn_inference
)

print(f"Inference dataset size: {len(inference_dataset)}")

# For ground truth & final COCO eval:
class_names = {
    1: "car",
    2: "pedestrian"
}

results = []               # For saving predictions/visualizations (KITTI format)
all_coco_predictions = []  # For COCO evaluation
all_gt_annotations = []    # For COCO evaluation
images_info_eval = {}      # For COCO evaluation
ann_id = 1

print("Starting inference...")
with torch.no_grad():
    for i, batch in enumerate(tqdm(inference_dataloader, desc="Inference")):
        pixel_values = batch["pixel_values"].to(device)
        outputs = model(pixel_values=pixel_values)

        for j in range(len(batch["image_id"])):
            image_id = batch["image_id"][j]
            original_size = batch["original_size"][j]  # (H, W)

            # Convert raw outputs to [x1, y1, x2, y2] boxes, plus label + score
            target_sizes = [original_size]  # list of (H, W) for each image
            results_processed = processor.post_process_object_detection(
                outputs, 
                target_sizes=target_sizes,
                threshold=0.5
            )[j]  # get the j-th image’s predictions

            # The DETR model was fine-tuned to produce label IDs in [0,1].
            # SHIFT to match KITTI = [1,2] for consistency with ground truth.
            shifted_labels = []
            for lbl in results_processed["labels"]:
                shifted_labels.append(lbl + 1)  # 0->1 (car), 1->2 (pedestrian)
            results_processed["labels"] = shifted_labels

            # --- Build predictions for KITTI text output ---
            kitti_predictions = []
            for box, label, score in zip(
                results_processed["boxes"], 
                results_processed["labels"], 
                results_processed["scores"]
            ):
                # Keep only if label is 1=car or 2=pedestrian
                label_id = label.item()
                if label_id in [1, 2]:
                    kitti_predictions.append({
                        "image_id": image_id,
                        "category_id": label_id, 
                        "bbox": box.tolist(),  # [x1, y1, x2, y2]
                        "score": score.item()
                    })

            # --- Build predictions for COCO eval ---
            # COCO needs "bbox" as [x, y, width, height], category_id in [1,2].
            coco_preds_this = []
            for box, label, score in zip(
                results_processed["boxes"], 
                results_processed["labels"], 
                results_processed["scores"]
            ):
                label_id = label.item()
                if label_id in [1, 2]:
                    x1, y1, x2, y2 = box.tolist()
                    w, h = x2 - x1, y2 - y1
                    coco_preds_this.append({
                        "image_id": image_id,
                        "category_id": label_id,  # 1 or 2
                        "bbox": [round(x1,2), round(y1,2), round(w,2), round(h,2)],
                        "score": round(score.item(),3)
                    })
            all_coco_predictions.extend(coco_preds_this)

            # --- Store ground truth for COCO eval ---
            # The dataset already has category_id in [1,2].
            for ann in batch["ground_truth"][j]:
                if ann["category_id"] in [1, 2]:
                    bbox = ann["bbox"]
                    area = bbox[2] * bbox[3]
                    all_gt_annotations.append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": ann["category_id"],
                        "bbox": bbox,      # [x, y, w, h]
                        "area": area,
                        "iscrowd": ann.get("iscrowd", 0)
                    })
                    ann_id += 1

            # Save image info (COCO expects width then height)
            if image_id not in images_info_eval:
                images_info_eval[image_id] = {
                    "id": image_id,
                    "width": original_size[1],  # W
                    "height": original_size[0]  # H
                }

            # Collect results for visualization
            results.append({
                "image_id": image_id,
                "predictions": kitti_predictions,
                "ground_truth": batch["ground_truth"][j],
                "original_size": original_size
            })

            # Visualize periodically
            if i % 100 == 0:
                visualize_predictions(
                    image=batch["original_image"][j],
                    predictions=results_processed,  # now labels are [1,2]
                    ground_truth=batch["ground_truth"][j],
                    class_names=class_names,
                    score_threshold=0.5,
                    output_dir=vis_output_dir,
                    image_id=image_id
                )

# --- Save results to a text file in KITTI-like format ---
result_file = os.path.join(output_dir, "inference_results.txt")
with open(result_file, 'w') as f:
    for result in results:
        image_id = result["image_id"]
        for pred in result["predictions"]:
            bbox = pred["bbox"]
            # DETR returns [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            line = (
                f"{image_id} {pred['category_id']} {pred['score']:.6f} "
                f"{x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n"
            )
            f.write(line)

print(f"Inference complete. Results saved to {result_file}")
print(f"Visualizations (sampled) saved to {vis_output_dir}")

# --- COCO Evaluation ---
coco_gt_dict = {
    "images": list(images_info_eval.values()),
    "annotations": all_gt_annotations,
    "categories": [
        {"id": 1, "name": "car"},
        {"id": 2, "name": "pedestrian"}
    ]
}

coco_gt = COCO()
coco_gt.dataset = coco_gt_dict
coco_gt.createIndex()

coco_dt = coco_gt.loadRes(all_coco_predictions)

coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

metric_names = [
    "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
    "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large"
]
metrics = {name: float(coco_eval.stats[i]) for i, name in enumerate(metric_names)}

csv_path = os.path.join(output_dir, "coco_metrics.csv")
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Metric", "Value"])
    for name, value in metrics.items():
        writer.writerow([name, value])

print("\nOfficial COCO Evaluation Metrics:")
for name, value in metrics.items():
    print(f"{name}: {value}")
