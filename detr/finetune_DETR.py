import os
import torch
import pycocotools.mask as mask_util
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import DetrForObjectDetection, DetrImageProcessor
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import albumentations as A

class KITTIMOTSDataset(Dataset):
    def __init__(self, data_dir, annotation_dir, processor, split="train", transform=None):
        self.data_dir = data_dir
        self.annotation_dir = annotation_dir
        self.processor = processor
        self.transform = transform
        self.split = split
        
        # Sequence IDs for train/test split
        self.train_sequences = [f"{i:04d}" for i in range(16)]  # 0000-0015
        self.test_sequences = [f"{i:04d}" for i in range(16, 21)]  # 0016-0020
        
        # Select sequences based on split
        self.sequences = self.train_sequences if split == "train" else self.test_sequences
        
        # Original KITTI-MOTS class IDs in the annotation files are 1=car, 2=pedestrian
        # Keep track for filtering, later shift them to 0=car, 1=pedestrian
        self.valid_class_ids = [1, 2]
        
        # Parse the dataset
        self.images_info = []
        self.parse_dataset()
        
    def parse_dataset(self):
        print(f"Parsing KITTI-MOTS annotations for {self.split} split...")
        
        ann_files = [f for f in os.listdir(self.annotation_dir) if f.endswith('.txt')]
        ann_files = [f for f in ann_files if any(seq in f for seq in self.sequences)]
        
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
                class_id = int(parts[2])  # 1=car, 2=pedestrian in KITTI-MOTS
                height = int(parts[3])
                width = int(parts[4])
                rle = parts[5]
                
                # Filter out anything that's not car/pedestrian based on original IDs
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

                # IMPORTANT: Shift class_id from {1, 2} -> {0, 1}
                class_id_shifted = class_id - 1
                
                annotations_by_frame[image_id]["annotations"].append({
                    "id": obj_id,
                    "category_id": class_id_shifted,  # now 0=car, 1=pedestrian
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
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)
        
        # Get original annotations
        original_annotations = img_info["annotations"]
        bboxes = []
        category_ids = []
        
        # Filter out empty or invalid boxes
        for ann in original_annotations:
            x, y, w, h = ann["bbox"]
            # Make sure boxes have positive width and height
            if w > 0 and h > 0:
                bboxes.append(ann["bbox"])
                category_ids.append(ann["category_id"])
        
        # Apply transformations if any and if we have valid boxes
        if self.transform and bboxes:
            try:
                transformed = self.transform(
                    image=image_np,
                    bboxes=bboxes,
                    category_id=category_ids
                )
                image_transformed = transformed["image"]
                transformed_bboxes = transformed["bboxes"]
                transformed_category_ids = transformed["category_id"]
                
                # Create new annotations with transformed boxes
                transformed_annotations = []
                for i, (bbox, cat_id) in enumerate(zip(transformed_bboxes, transformed_category_ids)):
                    transformed_annotations.append({
                        "id": i,
                        "category_id": cat_id,
                        "bbox": list(bbox),
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0
                    })
                
                # Use transformed data
                image_to_process = image_transformed
                annotations_to_process = transformed_annotations
            except Exception as e:
                print(f"Transformation error: {e} - using original image")
                image_to_process = image_np
                annotations_to_process = original_annotations
        else:
            # Use original data
            image_to_process = image_np
            annotations_to_process = original_annotations
        
        # Process with DETR processor
        encoded = self.processor(
            images=image_to_process,
            annotations={"image_id": img_info["image_id"], "annotations": annotations_to_process},
            return_tensors="pt"
        )
        
        pixel_values = encoded["pixel_values"].squeeze(0)
        labels = encoded["labels"][0]

        return {"pixel_values": pixel_values, "labels": labels}

def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = [item["labels"] for item in batch]
    return {
        "pixel_values": pixel_values,
        "labels": labels
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    base_dir = "/ghome/c5mcv06/mcv/datasets/C5/KITTI-MOTS"
    data_dir = base_dir
    annotation_dir = os.path.join(base_dir, "instances_txt")
    output_dir = "./results_detr_kittimots_finetuning"
    
    os.makedirs(output_dir, exist_ok=True)
    
    model_name = "facebook/detr-resnet-50"
    
    # Label mappings
    id2label = {0: "car", 1: "pedestrian"}
    label2id = {"car": 0, "pedestrian": 1}

    # DETR processor with custom image size
    processor = DetrImageProcessor.from_pretrained(model_name, size=(480, 600))
    
    # Initialize the DETR model for 2 labels (car and pedestrian)
    model = DetrForObjectDetection.from_pretrained(
        model_name,
        ignore_mismatched_sizes=True,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id
    )
    
    print(f"Loaded model: {model_name}")
    
    # Transformations (Albumentations)
    train_transform = A.Compose([
        # Horizontal Flip
        A.HorizontalFlip(p=0.5),
        # Slight random shifts, scales, and rotations
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=7,
            p=0.5
        ),
        # Brightness / contrast
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.3
        ),
        # Color
        A.HueSaturationValue(
            hue_shift_limit=0.07,
            sat_shift_limit=0.07,
            val_shift_limit=0.07,
            p=0.2
        ),
        # Blur or Noise
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
        ], p=0.2),
        ], bbox_params=A.BboxParams(
        format='coco',      # Format [x_min, y_min, width, height]
        label_fields=['category_id'],
        min_area=1.0,       # Filter out tiny boxes (1 pixel minimum)
        min_visibility=0.1  # Filter boxes that are mostly cropped out
    ))
    
    train_dataset = KITTIMOTSDataset(
        data_dir=data_dir,
        annotation_dir=annotation_dir,
        processor=processor,
        split="train",
        transform=train_transform
    )
    
    test_dataset = KITTIMOTSDataset(
        data_dir=data_dir,
        annotation_dir=annotation_dir,
        processor=processor,
        split="test"
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=30,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        learning_rate=1e-5,
        weight_decay=1e-4,
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]  # Stop if no improvement for 5 epochs
    )
        
    print("Starting training...")
    trainer.train()
    
    model.save_pretrained(f"{output_dir}/final_model_FULLIMG_AUG")
    processor.save_pretrained(f"{output_dir}/final_model_FULLIMG_AUG")
    
    print(f"Fine-tuned model saved to {output_dir}/final_model_FULLIMG_AUG")

if __name__ == "__main__":
    main()
