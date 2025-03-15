# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import os
import tempfile
import time
import warnings
import pickle

import cv2
import numpy as np
import tqdm
import csv

from natsort import natsorted

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

# Import COCO evaluation tools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)


    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
        args.confidence_threshold
    )
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="/home/mmajo/MCV/C5/MCV-2024-C5-Project-T6/detectron2/config_files/Custom-Faster-RCNN.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--webcam", action="store_true", help="Take inputs from webcam."
    )
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--csv-path",
        help="Path of the csv file to save the results",
        default="output.csv",
    )
    parser.add_argument(
        "--annotation-dir",
        help="Path to ground truth annotation directory",
        default=None,
    )
    parser.add_argument(
        "--test-sequences",
        nargs="+",
        help="List of test sequence IDs",
        default=["0016", "0017", "0018", "0019", "0020"],
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


# Function to convert RLE to mask using COCO mask_util
def rle_to_mask(rle, height, width):
    try:
        if rle.startswith("WSV:"):
            rle = rle[4:]
        rle_dict = {"size": [height, width], "counts": rle.encode("utf-8")}
        mask = mask_util.decode(rle_dict)
        return mask
    except Exception as e:
        print(f"Error decoding RLE: {e}")
        return np.zeros((height, width), dtype=np.uint8)


# Function to convert mask to bbox
def mask_to_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return [0, 0, 1, 1]  # Return 1x1 box instead of 0x0 to avoid errors
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


# Parse the KITTI-MOTS annotations
def parse_annotations(annotation_dir, test_sequences, valid_class_ids=[1, 2]):
    print("Parsing KITTI-MOTS annotations...")
    
    ann_files = [f for f in os.listdir(annotation_dir) if f.endswith('.txt')]
    ann_files = [f for f in ann_files if any(seq in f for seq in test_sequences)]
    
    annotations_by_frame = {}
    
    for ann_file in tqdm.tqdm(ann_files, desc="Parsing annotation files"):
        sequence_id = ann_file.split('.')[0]
        with open(os.path.join(annotation_dir, ann_file), 'r') as f:
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
            if class_id not in valid_class_ids:
                continue

            # Create unique image_id
            image_id = f"{sequence_id}_{frame_id:06d}"
            
            # Find the corresponding image path
            img_path = os.path.join(
                os.path.dirname(annotation_dir), 
                "training", 
                "image_02", 
                sequence_id, 
                f"{frame_id:06d}.png"
            )
            
            # Skip if image doesn't exist on disk
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

            # Get bounding box from RLE (if possible) or parse from annotation
            try:
                from pycocotools import mask as mask_util
                mask = rle_to_mask(rle, height, width)
                bbox = mask_to_bbox(mask)
            except:
                # If mask_util not available, use provided coordinates
                # Note: KITTI-MOTS uses RLE format, so this is a fallback
                bbox = [0, 0, width, height]  # Placeholder
            
            x_min, y_min, x_max, y_max = bbox
            width_box = x_max - x_min
            height_box = y_max - y_min

            annotations_by_frame[image_id]["annotations"].append({
                "id": obj_id,
                "category_id": class_id,  # Keep as [1,2] for ground truth
                "bbox": [x_min, y_min, width_box, height_box],  # [x, y, width, height] for COCO
                "area": width_box * height_box,
                "iscrowd": 0
            })
    
    return list(annotations_by_frame.values())


def main() -> None:
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    # Create output directories
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        os.makedirs(os.path.join(args.output, "visualizations"), exist_ok=True)

    # --- COCO Evaluation Setup ---
    all_coco_predictions = []
    all_gt_annotations = []
    images_info = []
    ann_id = 1
    class_names = {1: "car", 2: "pedestrian"}
    valid_class_ids = [1, 2]  # car, pedestrian
    
    # Parse ground truth annotations if annotation directory is provided
    if args.annotation_dir:
        print("Parsing ground truth annotations...")
        annotations_data = parse_annotations(args.annotation_dir, args.test_sequences, valid_class_ids)
        print(f"Found {len(annotations_data)} images with annotations")
        
        # Create COCO ground truth structure
        for idx, img_info in enumerate(annotations_data):
            image_id = img_info["image_id"]
            
            # Add image info
            images_info.append({
                "id": image_id,
                "file_name": os.path.basename(img_info["img_path"]),
                "width": img_info["width"],
                "height": img_info["height"]
            })
            
            # Add annotations
            for ann in img_info["annotations"]:
                all_gt_annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": ann["category_id"],
                    "bbox": ann["bbox"],
                    "area": ann["area"],
                    "iscrowd": ann["iscrowd"]
                })
                ann_id += 1

    if args.input:
        image_mapping = {}  # To map image path to image_id
        if args.annotation_dir:
            # Create a mapping from image path to image_id
            for img_info in annotations_data:
                image_mapping[os.path.basename(img_info["img_path"])] = img_info["image_id"]

        with open(args.csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file, delimiter=";")
            writer.writerow(["image", "bbox", "score", "class"])

            if len(args.input) == 1:
                args.input = glob.glob(os.path.expanduser(args.input[0]))
                assert args.input, "The input path(s) was not found"
            else:
                args.input = natsorted(args.input)
                
            for path in tqdm.tqdm(args.input, disable=not args.output):
                # use PIL, to be consistent with evaluation
                img = read_image(path, format="BGR")
                start_time = time.time()
                predictions, visualized_output = demo.run_on_image(img)
                logger.info(
                    "{}: {} in {:.2f}s".format(
                        path,
                        "detected {} instances".format(len(predictions["instances"]))
                        if "instances" in predictions
                        else "finished",
                        time.time() - start_time,
                    )
                )

                # Save visualization
                if args.output:
                    if os.path.isdir(args.output):
                        out_filename = os.path.join(args.output, "visualizations", os.path.basename(path))
                    else:
                        assert (
                            len(args.input) == 1
                        ), "Please specify a directory with args.output"
                        out_filename = args.output
                    visualized_output.save(out_filename)
                else:
                    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                    cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                    if cv2.waitKey(0) == 27:
                        break  # esc to quit

                predictions = predictions['instances']
 
                boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
                scores = predictions.scores if predictions.has("scores") else None
                classes = predictions.pred_classes if predictions.has("pred_classes") else None

                # Get image_id from filename if we have annotations
                image_id = None
                basename = os.path.basename(path)
                if basename in image_mapping:
                    image_id = image_mapping[basename]
                else:
                    # Extract sequence and frame if possible from filename (assuming format like sequence_frame.png)
                    parts = os.path.splitext(basename)[0].split('_')
                    if len(parts) >= 2:
                        try:
                            sequence_id = parts[0]
                            frame_id = int(parts[1])
                            image_id = f"{sequence_id}_{frame_id:06d}"
                        except:
                            image_id = os.path.splitext(basename)[0]
                    else:
                        image_id = os.path.splitext(basename)[0]

                if boxes:
                    for box, score, cls in zip(boxes, scores, classes):
                        box = box.cpu().numpy()
                        cls_id = int(cls.item())
                        
                        # Map Detectron2/COCO class IDs to KITTI IDs if needed
                        # COCO: 0=person, 2=car; KITTI: 1=car, 2=pedestrian
                        kitti_cls = cls_id
                        if cls_id == 0:  # person in COCO
                            kitti_cls = 2  # pedestrian in KITTI
                        elif cls_id == 2:  # car in COCO
                            kitti_cls = 1  # car in KITTI
                        
                        # For CSV output: x1,y1,x2,y2
                        box_str = f'{box[1]},{box[0]},{box[3]},{box[2]}'
                        writer.writerow([basename, box_str, score.item(), kitti_cls])
                        
                        # For COCO evaluation: [x, y, width, height]
                        x1, y1, x2, y2 = box
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Add to COCO predictions for evaluation
                        if image_id:
                            all_coco_predictions.append({
                                "image_id": image_id,
                                "category_id": kitti_cls,
                                "bbox": [round(x1, 2), round(y1, 2), round(width, 2), round(height, 2)],
                                "score": round(float(score.item()), 3)
                            })
                else:
                    writer.writerow([basename, None, None, None])

    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv")
            if test_opencv_video_format("x264", ".mkv")
            else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()

    # --- COCO Evaluation ---
    if args.annotation_dir and all_coco_predictions:
        print("Performing COCO evaluation...")
        
        # Create COCO ground truth format
        coco_gt_dict = {
            "images": images_info,
            "annotations": all_gt_annotations,
            "categories": [
                {"id": 1, "name": "car"},
                {"id": 2, "name": "pedestrian"}
            ]
        }

        # Save all results to pickle
        if args.output:
            results_dict = {
                "predictions": all_coco_predictions,
                "ground_truth": coco_gt_dict
            }
            with open(os.path.join(args.output, "predictions.pkl"), "wb") as f:
                pickle.dump(results_dict, f)

        # Initialize COCO API for ground truth
        coco_gt = COCO()
        coco_gt.dataset = coco_gt_dict
        coco_gt.createIndex()

        try:
            # Load predictions
            coco_dt = coco_gt.loadRes(all_coco_predictions)
            
            # Run evaluation
            coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Store metrics
            metric_names = [
                "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
                "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large"
            ]
            metrics = {name: float(coco_eval.stats[i]) for i, name in enumerate(metric_names)}
            
            # Save metrics to CSV
            if args.output:
                csv_path = os.path.join(args.output, "coco_metrics.csv")
                with open(csv_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Metric", "Value"])
                    for name, value in metrics.items():
                        writer.writerow([name, value])
            
            print("\nOfficial COCO Evaluation Metrics:")
            for name, value in metrics.items():
                print(f"{name}: {value:.4f}")
            
            # Per-class metrics
            print("\nPer-class AP50:")
            for cat_id in [1, 2]:  # car, pedestrian
                # Set evaluation parameters for this class
                coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
                coco_eval.params.catIds = [cat_id]
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                print(f"Class {class_names[cat_id]}: AP50 = {coco_eval.stats[1]:.4f}")
                
        except Exception as e:
            print(f"Error during COCO evaluation: {e}")
            # Save the problematic data for debugging
            if args.output:
                with open(os.path.join(args.output, "coco_gt.pkl"), "wb") as f:
                    pickle.dump(coco_gt_dict, f)
                with open(os.path.join(args.output, "coco_preds.pkl"), "wb") as f:
                    pickle.dump(all_coco_predictions, f)


if __name__ == "__main__":
    main()  # pragma: no cover
