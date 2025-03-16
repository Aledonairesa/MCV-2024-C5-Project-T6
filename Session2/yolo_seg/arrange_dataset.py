import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import pycocotools.mask as mask_util

# Base Paths
KITTI_MOTS_PATH = "/ghome/c5mcv06/mcv/datasets/C5/KITTI-MOTS"
YOLO_KITTI_MOTS_PATH = "/ghome/c5mcv06/biel_working_dir/YOLO-KITTI-MOTS"

# Derived Paths
KITTI_MOTS_IMAGES = os.path.join(KITTI_MOTS_PATH, "training/image_02")
KITTI_MOTS_LABELS = os.path.join(KITTI_MOTS_PATH, "instances_txt")
OUTPUT_IMAGES_TRAIN = os.path.join(YOLO_KITTI_MOTS_PATH, "images/train")
OUTPUT_IMAGES_VAL = os.path.join(YOLO_KITTI_MOTS_PATH, "images/val")
OUTPUT_LABELS_TRAIN = os.path.join(YOLO_KITTI_MOTS_PATH, "labels/train")
OUTPUT_LABELS_VAL = os.path.join(YOLO_KITTI_MOTS_PATH, "labels/val")
# Path for visualization
OUTPUT_VIS_TRAIN = os.path.join(YOLO_KITTI_MOTS_PATH, "visualization/train")
OUTPUT_VIS_VAL = os.path.join(YOLO_KITTI_MOTS_PATH, "visualization/val")

# Ensure output directories exist
for path in [OUTPUT_IMAGES_TRAIN, OUTPUT_IMAGES_VAL, OUTPUT_LABELS_TRAIN, OUTPUT_LABELS_VAL,
             OUTPUT_VIS_TRAIN, OUTPUT_VIS_VAL]:
    os.makedirs(path, exist_ok=True)

def rle_to_mask(rle, height, width):
    """Converts RLE to a binary mask."""
    rle_dict = {"size": [height, width], "counts": rle.encode("utf-8")}
    return mask_util.decode(rle_dict)

def extract_contour_points(mask):
    """Extract contour points from binary mask."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (in case there are multiple)
    if not contours:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Simplify the contour to reduce number of points
    epsilon = 0.005 * cv2.arcLength(largest_contour, True)
    approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Convert to normalized format for YOLO
    contour_points = approx_contour.reshape(-1, 2)
    return contour_points

def create_colored_visualization(img, masks, class_ids, instance_ids):
    """Create a visualization of segmentation masks overlaid on the image."""
    vis_img = img.copy()
    overlay = np.zeros_like(img)
    
    # Define colors for different classes (BGR format for OpenCV)
    colors = {
        0: (0, 0, 255),  # Red for pedestrians (class 0)
        1: (255, 0, 0)   # Blue for cars (class 1)
    }
    
    for mask, class_id, instance_id in zip(masks, class_ids, instance_ids):
        # Base color from class
        color = list(colors[class_id])
        
        # Vary saturation based on instance ID
        factor = 0.5 + ((instance_id % 5) / 10.0)
        color = [int(c * factor) for c in color]
        
        # Apply color to mask area
        for c in range(3):
            overlay[:, :, c] = np.where(mask == 1, color[c], overlay[:, :, c])
    
    # Blend with original image (50% transparency)
    alpha = 0.5
    vis_img = cv2.addWeighted(vis_img, 1 - alpha, overlay, alpha, 0)
    
    return vis_img

def process_annotations():
    for subdir in tqdm(sorted(os.listdir(KITTI_MOTS_IMAGES)), desc="Processing sequences:"):
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
            frame = int(parts[0])
            instance_id = int(parts[1]) % 1000
            class_id = int(parts[2])
            img_h, img_w = int(parts[3]), int(parts[4])
            rle = parts[5]
            
            if class_id not in [1, 2]:  # Skip annotations with class_id other than 1 or 2
                continue
            
            class_id -= 1  # Adjust class_id (1 -> 0, 2 -> 1)
            
            if frame not in frame_annotations:
                frame_annotations[frame] = []
            frame_annotations[frame].append((class_id, instance_id, img_h, img_w, rle))
        
        for img_file in image_files:
            frame_number = int(os.path.splitext(img_file)[0])
            new_img_name = f"{subdir}_{img_file}"
            img_src = os.path.join(subdir_path, img_file)
            
            # Determine output directories based on sequence number
            is_train = int(subdir) <= 15
            label_dst_dir = OUTPUT_LABELS_TRAIN if is_train else OUTPUT_LABELS_VAL
            img_dst_dir = OUTPUT_IMAGES_TRAIN if is_train else OUTPUT_IMAGES_VAL
            vis_dst_dir = OUTPUT_VIS_TRAIN if is_train else OUTPUT_VIS_VAL
            
            # Set destination paths
            img_dst = os.path.join(img_dst_dir, new_img_name)
            label_dst = os.path.join(label_dst_dir, new_img_name.replace(".png", ".txt"))
            vis_dst = os.path.join(vis_dst_dir, new_img_name)
            
            # Copy original image
            shutil.copy(img_src, img_dst)
            
            # Read image for dimensions and visualization
            img = cv2.imread(img_src)
            img_h, img_w, _ = img.shape
            
            # Process annotations for this frame
            masks = []
            class_ids = []
            instance_ids = []
            
            with open(label_dst, "w") as label_file:
                if frame_number in frame_annotations:
                    for class_id, instance_id, ann_h, ann_w, rle in frame_annotations[frame_number]:
                        # Convert RLE to mask
                        mask = rle_to_mask(rle, ann_h, ann_w)
                        
                        # Extract contour points from mask
                        contour_points = extract_contour_points(mask)
                        
                        if contour_points is not None and len(contour_points) >= 3:  # Need at least 3 points to form a polygon
                            # Save for visualization
                            masks.append(mask)
                            class_ids.append(class_id)
                            instance_ids.append(instance_id)
                            
                            # Format for YOLO segmentation:
                            # class x1 y1 x2 y2 ... xn yn
                            line = f"{class_id}"
                            for pt in contour_points:
                                # Normalize coordinates
                                x, y = pt
                                x_norm = x / img_w
                                y_norm = y / img_h
                                line += f" {x_norm:.6f} {y_norm:.6f}"
                            label_file.write(line + "\n")
                
            # Create visualization if masks are present
            if masks and os.path.exists(img_src):
                vis_img = create_colored_visualization(img, masks, class_ids, instance_ids)
                os.makedirs(os.path.dirname(vis_dst), exist_ok=True)
                cv2.imwrite(vis_dst, vis_img)
            elif not os.path.exists(vis_dst_dir):
                os.makedirs(vis_dst_dir, exist_ok=True)

if __name__ == "__main__":
    process_annotations()
    print("KITTI-MOTS to YOLO segmentation conversion completed.")