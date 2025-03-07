import json
import os
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"
    
    dataset = list(dataset)  # Convert Hugging Face dataset to list
    train_data, temp_data = train_test_split(dataset, test_size=(1 - train_ratio), random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=(test_ratio / (test_ratio + val_ratio)), random_state=42)
    
    return {"train": train_data, "val": val_data, "test": test_data}

def convert_to_coco(hf_dataset, output_json="coco_format.json"):
    coco_format = {
        "licenses": [],
        "categories": [{"id": 1, "name": "object", "supercategory": "none"}],
        "images": [],
        "annotations": []
    }
    
    annotation_id = 1
    image_id = 1
    
    for split in ["train", "val", "test"]:
        dataset = hf_dataset[split]
        
        for example in tqdm(dataset, desc=f"Processing {split} data"):
            image = example["image"]
            image_filename = f"image_{image_id}.jpg"
            
            # Add image entry
            coco_format["images"].append({
                "id": image_id,
                "file_name": image_filename,
                "width": image.width,
                "height": image.height
            })
            
            # Extract bboxes correctly
            keys_bbox = ["Bbox [x", "y", "w", "h]"]
            bboxes = [list(map(example.get, keys_bbox))]
            labels = [example["labels"]]
            
            # Add annotations
            for label, bbox in zip(labels, bboxes):
                x, y, w, h = bbox
                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": label,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                annotation_id += 1
                
            image_id += 1
    
    # Save COCO JSON file
    with open(output_json, "w") as f:
        json.dump(coco_format, f, indent=4)
    
    print(f"COCO dataset saved to {output_json}")

# Load the dataset
dataset = load_dataset("Tsomaros/Chest_Xray_N_Object_Detection")
split_data = split_dataset(dataset["train"], train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
convert_to_coco(split_data, "chest_xray_coco.json")