import os
import torch
import numpy as np
from PIL import Image
import pycocotools.mask as mask_util
from torch.utils.data import Dataset
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import albumentations as A

# Define a custom dataset for fine-tuning Mask2Former on KITTI-MOTS.
class KITTIMOTSMask2FormerDataset(Dataset):
    def __init__(self, data_dir, annotation_dir, processor, split="train", transform=None):
        """
        Args:
            data_dir (str): KITTI-MOTS dataset root directory.
            annotation_dir (str): Directory with KITTI-MOTS annotation files (instances_txt).
            processor: Pre-loaded image processor (AutoImageProcessor) for Mask2Former.
            split (str): "train" or "test".
            transform: Albumentations transformations to apply (for both image and mask).
        """
        self.data_dir = data_dir
        self.annotation_dir = annotation_dir
        self.processor = processor
        self.transform = transform
        self.split = split

        # Define sequences: train = 0000-0015, test = 0016-0020
        if split == "train":
            self.sequences = [f"{i:04d}" for i in range(16)]  # 0000-0015
        else:
            self.sequences = [f"{i:04d}" for i in range(16, 21)]  # 0016-0020

        self.images_info = []
        self.parse_dataset()

    def parse_dataset(self):
        print(f"Parsing KITTI-MOTS annotations for {self.split} split...")
        for seq in self.sequences:
            # Parse annotation file for sequence (e.g. "0000.txt")
            ann_path = os.path.join(self.annotation_dir, f"{seq}.txt")
            gt_mapping = {}  # mapping: frame number -> {instance_id: category_id}
            if os.path.exists(ann_path):
                with open(ann_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 2:
                            continue
                        frame = int(parts[0])
                        obj_id = int(parts[1])
                        # In KITTI-MOTS, object id divided by 1000 gives the category id (1=car, 2=pedestrian)
                        cat_id = obj_id // 1000
                        # Only keep car (1) and pedestrian (2)
                        if cat_id not in (1, 2):
                            continue
                        if frame not in gt_mapping:
                            gt_mapping[frame] = {}
                        gt_mapping[frame][obj_id] = cat_id

            # List images for the current sequence (from "training/image_02")
            img_dir = os.path.join(self.data_dir, "training", "image_02", seq)
            inst_dir = os.path.join(self.data_dir, "instances", seq)
            if not os.path.isdir(img_dir):
                continue
            img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
            for img_file in img_files:
                # Expect file names like "000000.png"
                frame_num = int(os.path.splitext(img_file)[0])
                # Only add frames for which we have annotations
                if frame_num not in gt_mapping:
                    continue
                img_path = os.path.join(img_dir, img_file)
                inst_path = os.path.join(inst_dir, img_file)
                self.images_info.append({
                    "image_id": f"{seq}_{frame_num:06d}",
                    "img_path": img_path,
                    "inst_path": inst_path,
                    "frame": frame_num,
                    "gt_mapping": gt_mapping[frame_num]
                })
        print(f"Found {len(self.images_info)} images for {self.split} split.")

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        info = self.images_info[idx]
        # Load image and instance segmentation mask
        image = np.array(Image.open(info["img_path"]).convert("RGB"))
        class_and_instance_map = np.array(Image.open(info["inst_path"]))

        # Remove ignore regions (class ID 10)
        ignore_region_class_mask = (class_and_instance_map // 1000) == 10
        class_and_instance_map[ignore_region_class_mask] = 0

        # Extract the pixel wise instance id and category id maps
        instance_seg = class_and_instance_map % 1000
        class_id_map = class_and_instance_map // 1000
        class_labels = np.unique(class_id_map)

        # Build the instance to class dictionary
        inst2class = {}
        for label in class_labels:
            instance_ids = np.unique(instance_seg[class_id_map == label])
            inst2class.update({i: label for i in instance_ids})
        
        # Apply transformation (if provided) to both image and segmentation mask
        if self.transform:
            try:
                transformed = self.transform(image=image, mask=instance_seg)
                image, instance_seg = transformed['image'], transformed['mask']
            except Exception as e:
                print(f"Transform error: {e} - using original image and mask.")

        if class_labels.shape[0] == 1 and class_labels[0] == 0:
            # If the image has no objects then it is skipped
            inputs = self.processor([image], return_tensors="pt")
            inputs = {k:v.squeeze() for k,v in inputs.items()}
            inputs["class_labels"] = torch.tensor([0])
            inputs["mask_labels"] = torch.zeros(
                (0, inputs["pixel_values"].shape[-2], inputs["pixel_values"].shape[-1])
            )
        else:
            # Else use process the image with the segmentation maps
            inputs = self.processor(
                [image],
                [instance_seg],
                instance_id_to_semantic_id=inst2class,
                return_tensors="pt"
            )
            inputs = {
                k:v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k,v in inputs.items()
            }
        # Return the inputs
        return inputs

# Define a collate function for batching
def collate_fn(batch):
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    pixel_mask = torch.stack([example["pixel_mask"] for example in batch])
    class_labels = [example["class_labels"] for example in batch]
    mask_labels = [example["mask_labels"] for example in batch]
    return {"pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
            "class_labels": class_labels,
            "mask_labels": mask_labels}

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Directories (adjust these paths as needed)
    base_dir = "/ghome/c5mcv06/mcv/datasets/C5/KITTI-MOTS"
    data_dir = base_dir
    annotation_dir = os.path.join(base_dir, "instances_txt")
    output_dir = "./finetuning_output"
    os.makedirs(output_dir, exist_ok=True)

    # Model name and label mappings.
    model_name = "facebook/mask2former-swin-tiny-coco-instance"
    id2label = {1: "car", 2: "pedestrian"}
    label2id = {"car": 1, "pedestrian": 2}

    # Load processor and model.
    processor = Mask2FormerImageProcessor.from_pretrained(model_name, do_reduce_labels=True)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    model.to(device)
    print(f"Loaded model: {model_name}")

    # Define Albumentations transformations for training
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=7,
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.3
        ),
        A.HueSaturationValue(
            hue_shift_limit=0.07,
            sat_shift_limit=0.07,
            val_shift_limit=0.07,
            p=0.2
        ),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
        ], p=0.2)
    ], additional_targets={"mask": "mask"})

    # Create training and test datasets.
    train_dataset = KITTIMOTSMask2FormerDataset(
        data_dir=data_dir,
        annotation_dir=annotation_dir,
        processor=processor,
        split="train",
        # transform=train_transform
    )
    test_dataset = KITTIMOTSMask2FormerDataset(
        data_dir=data_dir,
        annotation_dir=annotation_dir,
        processor=processor,
        split="test",
        transform=None
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Set up training arguments.
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=20,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        learning_rate=5e-5,
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        remove_unused_columns=False,
    )

    # Create the Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
    )

    print("Starting training...")
    trainer.train()

    # Save the fine-tuned model and processor.
    model.save_pretrained(os.path.join(output_dir, "fine_tuned_model"))
    processor.save_pretrained(os.path.join(output_dir, "fine_tuned_model"))
    print(f"Fine-tuned model saved to {os.path.join(output_dir, 'fine_tuned_model')}")


if __name__ == "__main__":
    main()
