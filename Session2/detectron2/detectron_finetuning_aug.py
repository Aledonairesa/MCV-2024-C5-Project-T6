import os
import torch
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper, build_detection_train_loader, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import detectron2.data.transforms as T

import albumentations as A
from albumentations.pytorch import ToTensorV2

import json
import numpy as np
import cv2
import pycocotools.mask as mask_util
from glob import glob
from tqdm import tqdm

def convert_kitti_mots_to_coco(kitti_mots_base_dir, train_output_json_path, test_output_json_path, split_idx=15):
    """
    Convert KITTI MOTS dataset to COCO format for Detectron2 training.
    Splits data into train (sequences 0000-0015) and test (sequences 0016-0020).
    Ignores class 10 (person) as requested.
    
    Args:
        kitti_mots_base_dir (str): Path to the base directory of KITTI MOTS dataset.
        train_output_json_path (str): Path to save the training COCO JSON file.
        test_output_json_path (str): Path to save the testing COCO JSON file.
    
    Returns:
        tuple: COCO data for training and testing
    """
    print("Converting KITTI MOTS to COCO format with train/test split...")
    
    # Define the KITTI MOTS class mapping
    # Class 1: Car, Class 2: Pedestrian, Class 10: Person (Ignored)
    class_mapping = {
        1: 1,  # Car -> Car
        2: 2,  # Pedestrian -> Pedestrian
        # Class 10 (Person) is ignored
    }
    
    # Define the COCO categories
    categories = [
        {"id": 1, "name": "car", "supercategory": "vehicle"},
        {"id": 2, "name": "pedestrian", "supercategory": "person"},
    ]
    
    # Initialize COCO format structures for train and test
    train_coco_data = {
        "info": {
            "description": "KITTI MOTS Training Dataset in COCO Format (Sequences 0000-0015)",
            "version": "1.0",
            "year": 2020,
            "contributor": "Converted from KITTI MOTS",
            "date_created": "2025-03-15",
        },
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "categories": categories,
        "images": [],
        "annotations": [],
    }
    
    test_coco_data = {
        "info": {
            "description": "KITTI MOTS Test Dataset in COCO Format (Sequences 0016-0020)",
            "version": "1.0",
            "year": 2020,
            "contributor": "Converted from KITTI MOTS",
            "date_created": "2025-03-15",
        },
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "categories": categories,
        "images": [],
        "annotations": [],
    }
    
    # Find all image files
    image_paths = sorted(glob(os.path.join(kitti_mots_base_dir, "training/image_02", "**", "*.png"), recursive=True))
    
    if not image_paths:
        raise ValueError(f"No images found in {os.path.join(kitti_mots_base_dir, 'training/image_02')}")
    
    train_annotation_id = 1
    test_annotation_id = 1
    train_image_id = 1
    test_image_id = 1
    
    for image_path in tqdm(image_paths, total=len(image_paths)):
        # Get image dimensions
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}, skipping...")
            continue
            
        height, width = img.shape[:2]
        
        # Get sequence and frame number from image path
        # Example: /path/to/kitti_mots/training/image_02/0002/000005.png
        sequence = os.path.basename(os.path.dirname(image_path))
        frame = os.path.splitext(os.path.basename(image_path))[0]
        
        # Determine if this sequence belongs to train or test set
        is_train = int(sequence) <= split_idx  # Sequences 0000-0015 for training
        
        # Select the appropriate dataset and IDs
        if is_train:
            coco_data = train_coco_data
            image_id = train_image_id
            annotation_id = train_annotation_id
        else:
            coco_data = test_coco_data
            image_id = test_image_id
            annotation_id = test_annotation_id
        
        # Add image to COCO data
        image_info = {
            "id": image_id,
            "file_name": os.path.relpath(image_path, kitti_mots_base_dir),
            "width": width,
            "height": height,
            "date_captured": "",
            "license": 1,
            "sequence": sequence,
            "frame": frame,
        }
        coco_data["images"].append(image_info)
        
        # Path to corresponding instance file
        instance_path = os.path.join(
            kitti_mots_base_dir, 
            "instances", 
            sequence, 
            f"{frame}.png"
        )
        
        if not os.path.exists(instance_path):
            # If the instance file doesn't exist, continue to the next image
            continue
        
        # Read instance mask
        instance_mask = cv2.imread(instance_path, cv2.IMREAD_UNCHANGED)
        if instance_mask is None:
            print(f"Warning: Could not read instance mask {instance_path}, skipping...")
            continue
        
        # Process instance mask
        # In KITTI MOTS, the instance ID is encoded as:
        # (class_id * 1000 + instance_id)
        # So to get the class_id, we divide by 1000 and take the integer part
        instance_ids = np.unique(instance_mask)
        
        for instance_id in instance_ids:
            if instance_id == 0:  # Background
                continue
            
            class_id = instance_id // 1000
            obj_id = instance_id % 1000
            
            # Skip class 10 as requested
            if class_id == 10:
                continue
                
            # Check if the class is in our mapping
            if class_id not in class_mapping:
                continue
            
            # Extract binary mask for this instance
            binary_mask = (instance_mask == instance_id).astype(np.uint8)
            
            # Calculate bounding box from mask
            positions = np.where(binary_mask)
            if len(positions[0]) == 0:  # Skip if mask is empty
                continue
                
            x_min = np.min(positions[1])
            y_min = np.min(positions[0])
            x_max = np.max(positions[1])
            y_max = np.max(positions[0])
            
            width = x_max - x_min + 1
            height = y_max - y_min + 1
            
            # Convert binary mask to polygon format
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Skip instances with no valid contours
            if not contours:
                continue
                
            # Convert contours to COCO polygon format
            segmentation = []
            for contour in contours:
                # Flatten the contour and convert to float
                contour = contour.flatten().tolist()
                # Only add contour if it has enough points (min 6 points for 3 coordinates)
                if len(contour) >= 6:
                    segmentation.append(contour)
            
            # Skip if no valid segmentation
            if not segmentation:
                continue
                
            # Create annotation
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_mapping[class_id],
                "segmentation": segmentation,
                "area": int(cv2.contourArea(contours[0])),  # Area from largest contour
                "bbox": [int(x_min), int(y_min), int(width), int(height)],
                "iscrowd": 0,
                "attributes": {"track_id": int(obj_id)},
            }
            
            coco_data["annotations"].append(annotation)
            annotation_id += 1
        
        # Increment appropriate image ID counter
        if is_train:
            train_image_id += 1
            train_annotation_id = annotation_id
        else:
            test_image_id += 1
            test_annotation_id = annotation_id
    
    # Save COCO format JSONs
    with open(train_output_json_path, "w") as f:
        json.dump(train_coco_data, f)
    
    with open(test_output_json_path, "w") as f:
        json.dump(test_coco_data, f)
    
    print(f"Conversion complete.")
    print(f"Training data (sequences 0000-0015) saved to {train_output_json_path}")
    print(f"Test data (sequences 0016-0020) saved to {test_output_json_path}")
    print(f"Training images: {len(train_coco_data['images'])}")
    print(f"Training annotations: {len(train_coco_data['annotations'])}")
    print(f"Test images: {len(test_coco_data['images'])}")
    print(f"Test annotations: {len(test_coco_data['annotations'])}")
    
    return train_coco_data, test_coco_data

# Custom trainer class that includes evaluation
class KITTIMOTSTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "evaluation")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

class AugmentedKITTIMOTSTrainer(KITTIMOTSTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        augmentations = [
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),  # Horizontal flip
            T.RandomRotation(angle=[-7, 7]),  # Small rotations
            T.RandomBrightness(0.85, 1.15),  # Brightness ±15%
            T.RandomContrast(0.85, 1.15),  # Contrast ±15%
            T.ResizeShortestEdge(short_edge_length=(576, 880), max_size=1333, sample_style="choice"),  # Approximate scaling
        ]
        
        return build_detection_train_loader(
            cfg,
            mapper=DatasetMapper(cfg, is_train=True, augmentations=augmentations)
        )


def train_kitti_mots_detectron2():
    """
    Function to finetune Detectron2 on KITTI MOTS dataset
    with train/test split: train on sequences 0000-0015, test on 0016-0020
    """
    # Set paths
    kitti_mots_dir = "/ghome/c5mcv06/mcv/datasets/C5/KITTI-MOTS"  # Update this path
    output_dir = "./finetuning_output_aug_stest"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert KITTI MOTS to COCO format if needed
    train_coco_json_path = os.path.join(output_dir, "kitti_mots_train_coco.json")
    test_coco_json_path = os.path.join(output_dir, "kitti_mots_test_coco.json")

    split_idx = 15 # 0000-00split_idx --> train  00split_idx+1-0020 --> test
    
    if not (os.path.exists(train_coco_json_path) and os.path.exists(test_coco_json_path)):
        print("Converting KITTI MOTS to COCO format with train/test split...")
        convert_kitti_mots_to_coco(kitti_mots_dir, train_coco_json_path, test_coco_json_path, split_idx)
    
    # Register datasets with Detectron2
    register_coco_instances(
        "kitti_mots_train", 
        {}, 
        train_coco_json_path, 
        kitti_mots_dir
    )
    
    register_coco_instances(
        "kitti_mots_test", 
        {}, 
        test_coco_json_path, 
        kitti_mots_dir
    )
    
    # Set class names for visualization
    for dataset_name in ["kitti_mots_train", "kitti_mots_test"]:
        metadata = MetadataCatalog.get(dataset_name)
        metadata.thing_classes = ["car", "pedestrian"]
    
    # Configure Detectron2
    cfg = get_cfg()
    
    # Use Mask R-CNN with ResNet50 backbone
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    
    # Load pre-trained weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    
    # Dataset settings
    cfg.DATASETS.TRAIN = ("kitti_mots_train",)
    cfg.DATASETS.TEST = ("kitti_mots_test",)
    
    # Set number of classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # car, pedestrian
    
    # Training parameters - adjust based on your GPU
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.002
    cfg.SOLVER.MAX_ITER = 50000  # Increased for better convergence
    cfg.SOLVER.STEPS = (1000, 1500)  # Adjust learning rate at these iterations
    cfg.SOLVER.CHECKPOINT_PERIOD = 10000
    
    # Add evaluation during training
    cfg.TEST.EVAL_PERIOD = 5000  # Evaluate every 200 iterations
    
    # Use GPU if available
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Output directory for saving model
    cfg.OUTPUT_DIR = output_dir
    
    # Train the model
    print("Starting training on sequences 0000-0015...")
    trainer = AugmentedKITTIMOTSTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    print(f"Training complete. Model saved to {output_dir}")
    
    # Run final evaluation on test set
    print("Running final evaluation on test set (sequences 0016-0020)...")
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("kitti_mots_test", cfg, False, output_dir=os.path.join(output_dir, "final_evaluation"))
    
    from detectron2.data import build_detection_test_loader
    val_loader = build_detection_test_loader(cfg, "kitti_mots_test")
    print(evaluator.evaluate(val_loader))
    #print(inference_on_dataset(predictor.model, val_loader, evaluator))

    #torch.save(trainer.model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "model_final.pth"))

if __name__ == "__main__":
    train_kitti_mots_detectron2()