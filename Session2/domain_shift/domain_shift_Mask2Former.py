import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import albumentations as A

class SatelliteBuildingDataset(Dataset):
    def __init__(self, hf_dataset, processor=None, transform=None):
        """
        Args:
            hf_dataset  : A split from the Hugging Face dataset 
                          (e.g. load_dataset(...)[split]).
            processor   : A processor that expects an image and (optionally) masks,
                          such as a DetrImageProcessor or OneFormerProcessor, etc.
            transform   : Albumentations (or similar) transform to apply to both 
                          image and mask (optional).
        """
        self.hf_dataset = hf_dataset
        self.processor = processor
        self.transform = transform
        
    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]

        # 1) Read the image (already a PIL image)
        image = example["image"]  # e.g. <PIL.Image.Image image mode=RGB size=512x512>
        w, h = image.size

        # 2) Create an empty "class+instance" map
        #    Store class_id * 1000 + instance_id
        class_and_instance_map = np.zeros((h, w), dtype=np.int32)

        # 3) For each polygon, fill in the instance ID.
        #    Single class = 1.
        polygons = example["objects"]["segmentation"]
        instance_counter = 1

        # Separate image (grayscale) to do the polygon fill
        # then copy those pixel values over to the "class+instance" map.
        polygon_mask_img = Image.new(mode="I", size=(w, h), color=0)  
        draw = ImageDraw.Draw(polygon_mask_img)

        for poly_pts in polygons:
            if isinstance(poly_pts[0], list):  
                coords = poly_pts[0]
            else:
                coords = poly_pts

            # Convert to [(x, y), (x, y), ...]
            coords = list(zip(coords[0::2], coords[1::2]))

            # class = 1 => class_and_instance = 1*1000 + instance_counter
            draw.polygon(coords, fill=1000 + instance_counter)
            instance_counter += 1

        # Convert the polygon_mask_img back to a numpy array
        class_and_instance_map = np.array(polygon_mask_img, dtype=np.int32)

        # 4) Extract instance IDs and class IDs
        instance_seg = class_and_instance_map % 1000   # e.g. 1, 2, 3, ...
        class_id_map = class_and_instance_map // 1000  # e.g. 1 for buildings
        class_labels = np.unique(class_id_map)

        # Build a dict mapping each instance ID -> class ID
        inst2class = {}
        for c in class_labels:
            instance_ids = np.unique(instance_seg[class_id_map == c])
            for i_id in instance_ids:
                inst2class[i_id] = c

        # 5) Apply transformation (if provided) to both image and segmentation
        #    (e.g., an Albumentations transform that expects {"image", "mask"}).
        image_array = np.array(image.convert("RGB"))  # e.g. shape (H, W, 3)
        if self.transform:
            try:
                transformed = self.transform(image=image_array, mask=instance_seg)
                image_array, instance_seg = transformed["image"], transformed["mask"]
            except Exception as e:
                print(f"Transform error: {e} - using original image and mask.")

        # 6) Feed into the processor
        #    If there's no object instance at all, we do the "no objects" route.
        if len(class_labels) == 1 and class_labels[0] == 0:
            # i.e. no objects
            inputs = self.processor([image_array], return_tensors="pt")
            # Squeeze out the batch dimension from the processor
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            # Add the dummy "class_labels" and "mask_labels"
            inputs["class_labels"] = torch.tensor([0])
            inputs["mask_labels"] = torch.zeros(
                (0, inputs["pixel_values"].shape[-2], inputs["pixel_values"].shape[-1])
            )
        else:
            # Else, let the processor handle the segmentation
            inputs = self.processor(
                [image_array],
                [instance_seg],
                instance_id_to_semantic_id=inst2class,
                return_tensors="pt"
            )
            # Squeeze out the batch dimension
            inputs = {
                k: v.squeeze(0) if isinstance(v, torch.Tensor) else v[0]
                for k, v in inputs.items()
            }

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
    output_dir = "./domain_shift_output_aug"
    os.makedirs(output_dir, exist_ok=True)

    # Model name and label mappings.
    model_name = "facebook/mask2former-swin-tiny-coco-instance"
    id2label = {1: "building"}
    label2id = {"building": 1}

    # Load processor and model.
    processor = Mask2FormerImageProcessor.from_pretrained(model_name,
                                                          do_reduce_labels=True)
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
    ds = load_dataset("keremberke/satellite-building-segmentation", name="full")
    train_dataset = SatelliteBuildingDataset(ds['train'], processor=processor, transform=train_transform)
    test_dataset = SatelliteBuildingDataset(ds['test'], processor=processor)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Set up training arguments.
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=15,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("Starting training...")
    trainer.train()

    # Save the fine-tuned model and processor.
    model.save_pretrained(os.path.join(output_dir, "fine_tuned_domain_shift_model_aug"))
    processor.save_pretrained(os.path.join(output_dir, "fine_tuned_domain_shift_model_aug"))
    print(f"Fine-tuned model saved to {os.path.join(output_dir, 'fine_tuned_domain_shift_model_aug')}")

if __name__ == "__main__":
    main()
