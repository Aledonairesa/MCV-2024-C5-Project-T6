import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
import pycocotools.mask as mask_util
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class KITTIMOTSInferenceDataset(Dataset):
    def __init__(self, data_dir, annotation_dir, processor, split="test"):
        self.data_dir = data_dir
        self.annotation_dir = annotation_dir
        self.processor = processor
        self.split = split
        
        # Define sequence IDs for test split
        self.test_sequences = [f"{i:04d}" for i in range(16, 21)]  # 0016-0020

        # In KITTI-MOTS: 1=car, 2=pedestrian
        self.valid_class_ids = [1, 2]  
        
        # For ground truth / COCO evaluation
        self.class_names = {
            1: "car",
            2: "pedestrian"
        }

        # Parse the dataset
        self.images_info = []
        self.parse_dataset()

    def parse_dataset(self):
        print(f"Parsing KITTI-MOTS annotations for {self.split} split...")

        ann_files = [f for f in os.listdir(self.annotation_dir) if f.endswith('.txt')]
        ann_files = [f for f in ann_files if any(seq in f for seq in self.test_sequences)]

        annotations_by_frame = {}
        for ann_file in ann_files:
            sequence_id = ann_file.split('.')[0]
            with open(os.path.join(self.annotation_dir, ann_file), 'r') as f:
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
                if class_id not in self.valid_class_ids:
                    continue

                img_path = os.path.join(
                    self.data_dir, 
                    "training", 
                    "image_02", 
                    sequence_id, 
                    f"{frame_id:06d}.png"
                )

                image_id = f"{sequence_id}_{frame_id:06d}"
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

                mask = self.rle_to_mask(rle, height, width)
                bbox = self.mask_to_bbox(mask)

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

        self.images_info = list(annotations_by_frame.values())
        print(f"Found {len(self.images_info)} valid images with annotations for {self.split} split")

    def rle_to_mask(self, rle, height, width):
        try:
            if rle.startswith("WSV:"):
                rle = rle[4:]
            rle_dict = {"size": [height, width], "counts": rle.encode("utf-8")}
            mask = mask_util.decode(rle_dict)
            return mask
        except Exception as e:
            print(f"Error decoding RLE: {e}")
            return np.zeros((height, width), dtype=np.uint8)

    def mask_to_bbox(self, mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return [0, 0, 1, 1]
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return [int(x_min), int(y_min), int(x_max), int(y_max)]

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        img_info = self.images_info[idx]
        img_path = img_info["img_path"]

        # Read the image
        image = Image.open(img_path).convert("RGB")
        width, height = image.size
        original_size = (height, width)

        # Process the image (resize, normalize, etc.) with DETR processor
        inputs = self.processor(images=image, return_tensors="pt")

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "original_image": np.array(image),
            "image_id": img_info["image_id"],
            "original_size": original_size,
            "ground_truth": img_info["annotations"]
        }

def collate_fn_inference(batch):
    """
    Custom collate function to handle batching
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    return {
        "pixel_values": pixel_values,
        "original_image": [item["original_image"] for item in batch],
        "image_id": [item["image_id"] for item in batch],
        "original_size": [item["original_size"] for item in batch],
        "ground_truth": [item["ground_truth"] for item in batch]
    }

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

    # --- Plot Predictions (blue) ---
    for box, label, score in zip(
        predictions["boxes"], 
        predictions["labels"], 
        predictions["scores"]
    ):
        if score < score_threshold:
            continue

        box = box.cpu().tolist()
        label_id = label.cpu().item()  # This should already be in [1,2] after matching
        conf = score.cpu().item()

        # Convert box to [x, y, w, h]
        if len(box) == 4:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
        else:
            continue

        if label_id in class_names:
            class_name = class_names[label_id]
            rect = patches.Rectangle(
                (x1, y1), w, h,
                linewidth=1,
                edgecolor='#FF00DC',
                facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                x1, y1 + h + 10,
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
