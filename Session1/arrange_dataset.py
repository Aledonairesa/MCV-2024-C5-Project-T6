import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import pycocotools.mask as mask_util

def ensure_directory_exists(file_path):
    """Ensure the directory for the given file path exists."""
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

def parse_kitti_mots_annotation(annotation_file, image_id_map):
    annotations = []
    annotation_id = 0
    with open(annotation_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split(' ')
            time_frame = int(parts[0])
            obj_id = int(parts[1]) % 1000  # Ensure correct instance ID
            class_id = int(parts[2])
            
            if class_id == 10:
                continue  # Ignore objects with class_id 10
            
            img_height = int(parts[3])
            img_width = int(parts[4])
            rle = ' '.join(parts[5:])
            
            if time_frame not in image_id_map:
                continue
            
            image_id = image_id_map[time_frame]
            mask = rle_to_mask(rle, img_height, img_width)
            encoded_mask = mask_util.encode(np.asfortranarray(mask))
            encoded_mask['counts'] = encoded_mask['counts'].decode('utf-8')
            
            bbox = mask_to_bbox(mask)
            if bbox is None:
                continue
            area = bbox[2] * bbox[3]
            
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": bbox,
                "area": area,
                "segmentation": encoded_mask,
                "iscrowd": 0
            })
            annotation_id += 1
    return annotations

def rle_to_mask(rle, height, width):
    """Converts RLE to a binary mask."""
    rle_dict = {"size": [height, width], "counts": rle.encode("utf-8")}
    return mask_util.decode(rle_dict)

def mask_to_bbox(mask):
    """Computes bounding box from a binary mask."""
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return None  # No valid bbox
    x_min, y_min = np.min(x_indices), np.min(y_indices)
    x_max, y_max = np.max(x_indices), np.max(y_indices)
    return [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]

def convert_kitti_mots_to_coco(image_dir, annotation_dir, output_json):
    ensure_directory_exists(output_json)  # Ensure the directory exists before writing

    coco_data = {
        "info": {
            "year": "2025",
            "description": "KITTI-MOTS in COCO JSON format"
        },
        "licenses": [],
        "categories": [
            {"id": 1, "name": "car"},
            {"id": 2, "name": "pedestrian"}
        ],
        "images": [],
        "annotations": []
    }
    
    image_id = 0
    for seq_id in tqdm(range(21), desc="Processing sequences"):
        seq_name = f"{seq_id:04d}"
        image_seq_dir = os.path.join(image_dir, seq_name)
        annotation_file = os.path.join(annotation_dir, f"{seq_name}.txt")
        
        if not os.path.exists(image_seq_dir) or not os.path.exists(annotation_file):
            continue
        
        images = sorted(os.listdir(image_seq_dir))
        image_id_map = {}
        
        for img_name in images:
            img_path = os.path.join(image_seq_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue  # Skip corrupted images
            height, width = img.shape[:2]
            
            image_id_map[int(os.path.splitext(img_name)[0])] = image_id
            
            coco_data["images"].append({
                "id": image_id,
                "file_name": img_name,
                "height": height,
                "width": width
            })
            image_id += 1
            
        annotations = parse_kitti_mots_annotation(annotation_file, image_id_map)
        coco_data["annotations"].extend(annotations)
    
    with open(output_json, 'w') as f:
        json.dump(coco_data, f, indent=4)

image_directory = "/data/users/mireia/MCV/C5/KITTI-MOTS/training/image_02"
annotation_directory = "/data/users/mireia/MCV/C5/KITTI-MOTS/instances_txt"
output_file = "/data/users/mireia/MCV/C5/COCO-JSON-KITTI-MOTS/kitti-mots.json"
convert_kitti_mots_to_coco(image_directory, annotation_directory, output_file)
