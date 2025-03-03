import os
import shutil
import cv2
import numpy as np
import pycocotools.mask as mask_util

# Base Paths
KITTI_MOTS_PATH = "/data/users/mireia/MCV/C5/KITTI-MOTS"
YOLO_KITTI_MOTS_PATH = "/data/users/mireia/MCV/C5/YOLO-KITTI-MOTS"

# Derived Paths
KITTI_MOTS_IMAGES = os.path.join(KITTI_MOTS_PATH, "training/image_02")
KITTI_MOTS_LABELS = os.path.join(KITTI_MOTS_PATH, "instances_txt")
OUTPUT_IMAGES_TRAIN = os.path.join(YOLO_KITTI_MOTS_PATH, "images/train")
OUTPUT_IMAGES_VAL = os.path.join(YOLO_KITTI_MOTS_PATH, "images/val")
OUTPUT_LABELS_TRAIN = os.path.join(YOLO_KITTI_MOTS_PATH, "labels/train")
OUTPUT_LABELS_VAL = os.path.join(YOLO_KITTI_MOTS_PATH, "labels/val")

# Ensure output directories exist
for path in [OUTPUT_IMAGES_TRAIN, OUTPUT_IMAGES_VAL, OUTPUT_LABELS_TRAIN, OUTPUT_LABELS_VAL]:
    os.makedirs(path, exist_ok=True)

def rle_to_mask(rle, height, width):
    """Converts RLE to a binary mask."""
    rle_dict = {"size": [height, width], "counts": rle.encode("utf-8")}
    return mask_util.decode(rle_dict)

def mask_to_bbox(mask):
    """Converts a binary mask to bounding box in xywh format."""
    y_indices, x_indices = np.where(mask == 1)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None  # No valid bbox
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    width = x_max - x_min
    height = y_max - y_min
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    return x_center, y_center, width, height

def process_annotations():
    for subdir in sorted(os.listdir(KITTI_MOTS_IMAGES)):
        subdir_path = os.path.join(KITTI_MOTS_IMAGES, subdir)
        label_file = os.path.join(KITTI_MOTS_LABELS, f"{subdir}.txt")
        
        if not os.path.isdir(subdir_path) or not os.path.exists(label_file):
            continue
        
        with open(label_file, "r") as f:
            annotations = f.readlines()
        
        image_files = sorted(os.listdir(subdir_path))
        frame_annotations = {}
        for ann in annotations:
            parts = ann.strip().split(" ")
            frame, _, class_id, img_h, img_w, rle = int(parts[0]), int(parts[1]) % 1000, int(parts[2]), int(parts[3]), int(parts[4]), parts[5]
            
            if class_id not in [1, 2]:  # Skip annotations with class_id other than 1 or 2
                continue
            
            class_id -= 1  # Adjust class_id (1 -> 0, 2 -> 1)
            
            if frame not in frame_annotations:
                frame_annotations[frame] = []
            frame_annotations[frame].append((class_id, img_h, img_w, rle))
        
        for img_file in image_files:
            frame_number = int(os.path.splitext(img_file)[0])
            new_img_name = f"{subdir}_{img_file}"
            img_src = os.path.join(subdir_path, img_file)
            label_dst_dir = OUTPUT_LABELS_TRAIN if int(subdir) <= 15 else OUTPUT_LABELS_VAL
            img_dst_dir = OUTPUT_IMAGES_TRAIN if int(subdir) <= 15 else OUTPUT_IMAGES_VAL
            img_dst = os.path.join(img_dst_dir, new_img_name)
            label_dst = os.path.join(label_dst_dir, new_img_name.replace(".png", ".txt"))
            
            shutil.copy(img_src, img_dst)
            img_h, img_w, _ = cv2.imread(img_src).shape
            
            # Ensure an empty label file is created
            with open(label_dst, "w") as label_file:
                if frame_number in frame_annotations:
                    for class_id, ann_h, ann_w, rle in frame_annotations[frame_number]:
                        mask = rle_to_mask(rle, ann_h, ann_w)
                        bbox = mask_to_bbox(mask)
                        if bbox:
                            x, y, w, h = bbox
                            x /= img_w
                            y /= img_h
                            w /= img_w
                            h /= img_h
                            label_file.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

if __name__ == "__main__":
    process_annotations()
    print("Dataset conversion completed.")