import os
import cv2
import numpy as np
import torch
import json
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
import pickle

def evaluate_kitti_mots_model(model_path, sequences_dir, sequence_ids=["0016"], 
                             output_dir="./visualization", score_threshold=0.5):
    """
    Evaluate model on specific KITTI MOTS sequences and save visualizations.
    
    Args:
        model_path (str): Path to the model weights file. If None, use pretrained weights.
        sequences_dir (str): Path to directory containing sequence folders
        sequence_ids (list): List of specific sequence IDs to evaluate
        output_dir (str): Directory to save visualization results
        score_threshold (float): Confidence threshold for detections
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up configuration
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    
    # Determine if we're using custom weights or pretrained weights
    using_custom_weights = model_path and os.path.exists(model_path)
    
    if using_custom_weights:
        # For custom weights, we can use 2 classes as they were trained that way
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # car, pedestrian
        cfg.MODEL.WEIGHTS = model_path
        print(f"Using custom weights from {model_path}")
    else:
        # For pretrained weights, keep the original 80 COCO classes
        # Don't modify NUM_CLASSES when using pretrained weights
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        print("Using pretrained weights with original COCO classes.")
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device
    print(f"Using device: {device}")
    
    # Set up predictor
    predictor = DefaultPredictor(cfg)
    
    # Set up metadata for visualization
    if using_custom_weights:
        # Custom metadata for custom model
        metadata = MetadataCatalog.get("kitti_mots_vis")
        if not hasattr(metadata, "thing_classes"):
            metadata.thing_classes = ["car", "pedestrian"]
    else:
        # Use COCO metadata for pretrained model
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0] if len(cfg.DATASETS.TRAIN) else "coco_2017_val")
    
    # Process each specified sequence
    for sequence_id in sequence_ids:
        sequence_path = os.path.join(sequences_dir, sequence_id)
        
        if not os.path.exists(sequence_path):
            print(f"Warning: Sequence {sequence_id} not found at {sequence_path}")
            continue
            
        sequence_output_dir = os.path.join(output_dir, sequence_id)
        os.makedirs(sequence_output_dir, exist_ok=True)
        
        print(f"Processing sequence: {sequence_id}")
            
        # Find all images
        image_files = sorted([f for f in os.listdir(sequence_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Create a results dictionary to store metrics
        results = {
            "sequence": sequence_id,
            "frames": len(image_files),
            "detections": 0,
            "car_detections": 0,
            "pedestrian_detections": 0,
            "average_confidence": 0.0
        }
        
        total_confidence = 0.0
        
        # Process each image in the sequence
        for image_file in tqdm(image_files, desc=f"Sequence {sequence_id}"):
            image_path = os.path.join(sequence_path, image_file)
            
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image {image_path}")
                continue
                
            # Run inference
            outputs = predictor(img)
            
            # Get instances
            instances = outputs["instances"].to("cpu")
            
            if not using_custom_weights:
                # For pretrained weights, filter for cars (2) and persons (0) in COCO
                classes = instances.pred_classes.numpy()
                keep = (classes == 0) | (classes == 2)  # COCO: 0=person, 2=car
                instances = instances[keep]
                
                # Remap COCO classes to KITTI classes
                # Create a new tensor for remapped classes
                new_classes = torch.zeros_like(instances.pred_classes)
                # COCO car (2) -> KITTI car (0), COCO person (0) -> KITTI pedestrian (1)
                new_classes[instances.pred_classes == 2] = 0  # car
                new_classes[instances.pred_classes == 0] = 1  # pedestrian
                instances.pred_classes = new_classes
            
            num_instances = len(instances)
            
            # Update metrics
            results["detections"] += num_instances
            
            if num_instances > 0:
                classes = instances.pred_classes.numpy()
                scores = instances.scores.numpy()
                
                # Count detections by class (using remapped classes)
                results["car_detections"] += np.sum(classes == 0)  # Car class
                results["pedestrian_detections"] += np.sum(classes == 1)  # Pedestrian class
                
                # Update total confidence
                total_confidence += np.sum(scores)
            
            # Create visualization
            v = Visualizer(
                img[:, :, ::-1],  # Convert BGR to RGB
                metadata=metadata,
                scale=1.0,
                instance_mode=ColorMode.SEGMENTATION
            )
            
            # For visualization with pretrained weights, we need to handle the class mapping
            if not using_custom_weights:
                # Create custom metadata for visualization with correct class names
                vis_metadata = MetadataCatalog.get("kitti_mots_vis")
                vis_metadata.thing_classes = ["car", "pedestrian"]
                #vis_metadata.thing_colors = [(0, 0, 255), (0, 255, 0)]  # BGR format
                v = Visualizer(
                    img[:, :, ::-1],
                    metadata=vis_metadata,
                    scale=1.0,
                    instance_mode=ColorMode.SEGMENTATION
                )
            
            # Draw instance predictions with segmentation masks
            result = v.draw_instance_predictions(instances)
            result_img = result.get_image()[:, :, ::-1]  # Convert RGB back to BGR
            
            # Save visualization
            output_path = os.path.join(sequence_output_dir, image_file)
            cv2.imwrite(output_path, result_img)
        
        # Calculate average confidence
        if results["detections"] > 0:
            results["average_confidence"] = float(total_confidence / results["detections"])
        
        print(f"Sequence {sequence_id} results:")
        print(f"  Total frames: {results['frames']}")
        print(f"  Total detections: {results['detections']}")
        print(f"  Car detections: {results['car_detections']}")
        print(f"  Pedestrian detections: {results['pedestrian_detections']}")
        print(f"  Average confidence: {results['average_confidence']:.4f}")
        print()
    
    print(f"All specified sequences processed. Visualizations saved to {output_dir}")


def main():
    """
    Main function to run evaluation and visualization
    """
    # Path to your finetuned model, None for pretrained weights
    # Uncomment the one you want to use
    
    # For pretrained weights:
    model_path = None
    
    # For custom weights:
    # model_path = "/ghome/c5mcv06/abril_working_dir/detectron2/weights/R-50.pkl"
    
    # Path to KITTI MOTS sequences directory
    # This should contain sequence folders like 0000, 0001, etc.
    sequences_dir = "/ghome/c5mcv06/mcv/datasets/C5/KITTI-MOTS/training/image_02"
    
    # Specific sequence IDs to evaluate
    sequence_ids = ["0017"]
    
    # Output directory for visualizations
    output_dir = "./visualizations/all_classes"
    
    # Detection confidence threshold
    confidence_threshold = 0.5
    
    # Run evaluation and visualization on specific sequences
    evaluate_kitti_mots_model(
        model_path,
        sequences_dir,
        sequence_ids,
        output_dir,
        confidence_threshold
    )

if __name__ == "__main__":
    main()