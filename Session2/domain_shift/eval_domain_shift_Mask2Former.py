import os
import glob
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from datasets import load_dataset

from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

BUILDINGS_LABELS_NAMES = {1: "building"}

import random

def visualize_results(idx, image, masks, boxes, scores, labels, output_dir, alpha=0.5):
    """
    Visualize instance segmentation results on a single image.

    Args:
        image  (PIL.Image): The original image (RGB).
        masks  (list[np.ndarray]): A list of binary masks (H x W), each for one instance.
        boxes  (list[list[int]]): A list of bounding boxes [x1, y1, x2, y2].
        scores (list[float])     : A list of confidence scores for each instance.
        labels (list[int])       : A list of labels (or class IDs) for each instance.
        alpha  (float)           : Mask transparency; defaults to 0.5.
    """
    # Convert PIL image to NumPy array for plotting
    image_np = np.array(image)

    # Create a Matplotlib figure
    fig, ax = plt.subplots()
    ax.imshow(image_np)
    ax.axis("off")

    for mask, box, score, label in zip(masks, boxes, scores, labels):
        # Random color for each instance
        color = (random.random(), random.random(), random.random())

        # Overlay the mask as a colored area
        overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
        overlay[..., :3] = color  # RGB
        overlay[..., 3] = mask.astype(np.float32) * alpha
        ax.imshow(overlay, interpolation='none')

        # Draw bounding box
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        rect = plt.Rectangle(
            (x1, y1), w, h,
            fill=False, edgecolor=color, linewidth=2
        )
        ax.add_patch(rect)

        # Annotate with label and score
        if label == 1:
            caption = f"building: {score:.2f}"
        else:
            caption = f"{label}: {score:.2f}"
        ax.text(
            x1, y1,
            caption,
            color="white", fontsize=8,
            bbox=dict(facecolor=color, alpha=alpha, pad=1, edgecolor="none")
        )

    plt.tight_layout()
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    plt.savefig(os.path.join(output_dir, "visualizations", f"{idx}.png"))


def parse_buildings_test_split(
    dataset,      # e.g. ds["test"] from keremberke/satellite-building-segmentation
    model,        # a trained model (Mask2Former, OneFormer, etc.)
    processor,    # corresponding processor (Mask2FormerImageProcessor, etc.)
    device="cuda",
    threshold=0.3,
    output_dir=None
):
    """
    Runs inference on the 'test' split of the buildings dataset and returns
    predictions in a COCO-like list of dicts (image_id, category_id, segmentation, score).
    Also visualizes results every 25 images, optionally saving them if output_dir is given.

    Args:
        dataset   : HF dataset split (ds["test"]).
        model     : PyTorch model with `.eval()` and `.to(device)`.
        processor : Processor with `.post_process_instance_segmentation(...)`.
        device    : "cpu" or "cuda".
        threshold : Score threshold for segmentation filtering.
        output_dir: If provided, saves every 25th visualization to this directory.

    Returns:
        predictions (list[dict]): Each dict is a single instance’s prediction:
          {
            "image_id": int,
            "category_id": int,
            "segmentation": RLE,
            "score": float
          }
    """
    predictions = []
    
    model.eval()
    model.to(device)

    for idx, example in enumerate(dataset):
        # Each example has a PIL image under "image"
        pil_image = example["image"]
        image_id = example["image_id"]

        # Prepare input
        inputs = processor(images=pil_image, return_tensors="pt").to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process to obtain instance segmentation maps & metadata
        instance_results = processor.post_process_instance_segmentation(
            outputs,
            target_sizes=[pil_image.size[::-1]],  # (height, width)
            threshold=threshold
        )[0]  # single element from the batch

        segments_info = instance_results.get("segments_info", [])
        instance_seg = instance_results.get("segmentation", None)

        if instance_seg is not None and len(segments_info) > 0:
            instance_seg = instance_seg.cpu().numpy()

            # We'll collect lists to optionally visualize
            masks = []
            boxes = []
            scores = []
            labels = []

            # Build the COCO-style predictions
            for seg_obj in segments_info:
                seg_id = seg_obj["id"]
                label_id = seg_obj["label_id"] + 1
                score = seg_obj.get("score", 1.0)

                # if label_id != 1:
                #     continue

                # Create the binary mask for this instance
                mask = (instance_seg == seg_id).astype(np.uint8)
                
                # Convert mask to RLE
                rle = maskUtils.encode(np.asfortranarray(mask))

                # Append to predictions
                predictions.append({
                    "image_id": image_id,
                    "category_id": label_id,   # or 1 if only one class for "building"
                    "segmentation": rle,
                    "score": float(score)
                })

                # For visualization: bounding box from mask
                y_indices, x_indices = np.where(mask > 0)
                if len(x_indices) > 0 and len(y_indices) > 0:
                    x1, y1 = x_indices.min(), y_indices.min()
                    x2, y2 = x_indices.max(), y_indices.max()
                else:
                    x1 = y1 = x2 = y2 = 0

                masks.append(mask)
                boxes.append([x1, y1, x2, y2])
                scores.append(score)
                labels.append(label_id)

            # Visualize every 25th image
            if idx % 25 == 0:
                visualize_results(idx, pil_image, masks, boxes, scores, labels, output_dir)

        print(f"Processed image {idx+1}/{len(dataset)} (image_id={image_id})")

    return predictions

def build_ground_truth_buildings_coco(hf_dataset):
    """
    Converts a split of the keremberke/satellite-building-segmentation dataset
    into a COCO-style dictionary suitable for instance segmentation evaluation.

    Args:
        hf_dataset (Dataset): A Hugging Face dataset split, e.g. ds["train"].

    Returns:
        dict: A dictionary with keys "images", "annotations", "categories" 
              in the COCO format.
    """

    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "building"}
        ]
    }

    annotation_id = 1  # Will increment for each object

    # Iterate through each example in the dataset
    for idx, example in enumerate(hf_dataset):
        # The dataset already has a unique image_id
        image_id = example["image_id"]
        width, height = example["width"], example["height"]

        # 1) Add the image entry
        coco_dict["images"].append({
            "id": image_id,
            "file_name": f"{image_id}.jpg",  # or any other scheme
            "width": width,
            "height": height
        })

        # 2) Add annotation entries for each object
        object_ids = example["objects"]["id"]
        bboxes = example["objects"]["bbox"]
        segmentations = example["objects"]["segmentation"]
        areas = example["objects"]["area"]
        cat_ids = example["objects"]["category"]  # Usually 0 => building

        for i in range(len(object_ids)):
            # The dataset's "bbox" is already [x, y, w, h]
            x, y, w, h = bboxes[i]
            area = areas[i]

            # The dataset's "segmentation" is a list (possibly nested),
            # typically of format [ [x1, y1, x2, y2, ...] ] 
            # You can pass it directly to COCO as "segmentation".
            seg = segmentations[i]

            # For a single building category, map 0 => category_id=1
            category_id = cat_ids[i] + 1

            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id, 
                "segmentation": seg,      # list of polygon(s)
                "bbox": [x, y, w, h],    # COCO expects [x, y, width, height]
                "area": float(area),     # or int(area)
                "iscrowd": 0             # set to 0 if not in crowd annotation
            }
            coco_dict["annotations"].append(annotation)
            annotation_id += 1

    return coco_dict


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

if __name__ == "__main__":
    # Output directory for visualizations
    output_dir = "./domain_shift_output/"
    
    # Load model and image processor
    processor = Mask2FormerImageProcessor.from_pretrained("./domain_shift_output/fine_tuned_domain_shift_model",
                                                          do_reduce_labels=True)
    model = Mask2FormerForUniversalSegmentation.from_pretrained("./domain_shift_output/fine_tuned_domain_shift_model")
    
    # Load data
    ds = load_dataset("keremberke/satellite-building-segmentation", name="full")
    test_raw_dataset = ds['test']

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Build ground-truth COCO dictionary from KITTI-MOTS annotations
    coco_gt_dict = build_ground_truth_buildings_coco(test_raw_dataset)
    
    # Process KITTI-MOTS sequences and collect predictions in COCO format
    predictions = parse_buildings_test_split(
        test_raw_dataset,
        model,
        processor,
        device=device,
        threshold=0.5,
        output_dir=output_dir
    )
    
    # Evaluate predictions using COCO metrics
    evaluate_coco(coco_gt_dict, predictions)
