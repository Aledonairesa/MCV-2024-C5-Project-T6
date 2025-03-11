from detectron2.engine.hooks import HookBase
from detectron2.engine.defaults import DefaultTrainer
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_train_loader, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
import datetime
import numpy as np
import logging
import wandb
from typing import Mapping
from detectron2.data import detection_utils as utils
import albumentations as A
from detectron2.data import transforms as T
from collections import OrderedDict
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    DatasetMapper,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)

def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(
                dataset_name, evaluator_type
            )
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)

class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        all_metrics_dict = []
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            losses_dict = self._get_loss(inputs)
            all_metrics_dict.append(losses_dict)  # list of dictionaries
            losses.append(sum(losses_dict.values()))  # list of floats (total loss per image)

        metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
        }
        total_losses_reduced = np.mean(losses)
        metrics_dict = {f'validation_{k}': v for k, v in metrics_dict.items()}
        metrics_dict['validation_total_loss'] = total_losses_reduced

        wandb.log(metrics_dict)

        self.trainer.storage.put_scalar('validation_loss', total_losses_reduced)
        comm.synchronize()

        print('Validation completed.\n')

        return total_losses_reduced
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        return metrics_dict
        
        
    def after_step(self):
        
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter

        if is_final or (self._period > 0 and next_iter % self._period == 0):

            total_losses_reduced = self._do_loss_eval()
            if total_losses_reduced < self.trainer.best_val_loss:
                self.trainer.best_val_loss = total_losses_reduced
                self.trainer.checkpointer.save(f'model_best')
                print('Model saved as best model.')
        self.trainer.storage.put_scalars(timetest=12)


class AlbumentationsMapper(DatasetMapper):
    """
    A custom dataset mapper that uses Albumentations augmentations
    """
    def __init__(self, cfg, is_train=True):
        # Initialize with the parent class constructor
        super().__init__(cfg, is_train=is_train)
        
        # Define albumentations transforms only for training
        if is_train:
            self.transform = A.Compose([
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
                format='coco',  # Format [x_min, y_min, width, height]
                label_fields=['category_ids'],
                min_area=1.0,  # Filter out tiny boxes (1 pixel minimum)
                min_visibility=0.1  # Filter boxes that are mostly cropped out
            ))
        else:
            self.transform = None

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that built-in models in detectron2 accept
        """
        dataset_dict = dataset_dict.copy()
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        
        if self.transform and self.is_train:
            # Extract annotations for albumentations
            boxes = []
            category_ids = []
            
            if "annotations" in dataset_dict:
                for annotation in dataset_dict["annotations"]:
                    # Get bbox in COCO format [x, y, width, height]
                    bbox = annotation["bbox"]
                    boxes.append(bbox)
                    category_ids.append(annotation["category_id"])
                    
            # Apply albumentations transforms
            if boxes:
                transformed = self.transform(
                    image=image,
                    bboxes=boxes,
                    category_ids=category_ids
                )
                
                image = transformed["image"]
                transformed_boxes = transformed["bboxes"]
                transformed_category_ids = transformed["category_ids"]
                
                # Update the annotations with the transformed boxes
                for i, annotation in enumerate(dataset_dict["annotations"]):
                    if i < len(transformed_boxes):
                        annotation["bbox"] = list(transformed_boxes[i])
                        annotation["category_id"] = transformed_category_ids[i]
            else:
                # If there are no boxes, just transform the image
                transformed = self.transform(image=image)
                image = transformed["image"]
        
        # Continue with the regular DatasetMapper processing
        utils.check_image_size(dataset_dict, image)
        
        # Apply other transforms from the base mapper
        image_shape = image.shape[:2]  # h, w
        
        # Apply base detectron2 augmentations if any
        if self.is_train:
            # Apply standard Detectron2 image transforms
            aug_input = T.AugInput(image)
            transforms = self.augmentations(aug_input)
            image = aug_input.image
        
        # Format image for model input
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        
        # Handle annotations
        if not self.is_train:
            # don't need annotations in validation
            dataset_dict.pop("annotations", None)
            return dataset_dict
            
        if "annotations" in dataset_dict:
            # Transform annotations after augmentations
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )
            
            # Filter out any invalid boxes after transforms
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
            
        return dataset_dict


class CustomTrainer(DefaultTrainer):
    """
    Custom trainer deriving from the DefaultTrainer.

    Overloads build_hooks to add a hook to calculate loss on the test (val) set during training.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.best_val_loss = float("inf")  # Initialize best validation loss
        wandb.init(project="detectron_loss_trial", name="finetuning")  # Initialize WandB
    
    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        It now calls `build_detection_train_loader` with a customized DatasetMapper
        that applies albumentations augmentations.
        """
        mapper = AlbumentationsMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        It now calls `build_detection_test_loader` with a customized DatasetMapper
        """
        mapper = AlbumentationsMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @staticmethod
    def write_metrics(
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        cur_iter: int,
        prefix: str = "",
    ) -> None:
        """
        Log the metrics (including losses) to WandB.

        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
            cur_iter (int): current iteration
            prefix (str): prefix for logging keys
        """
        # Convert the losses to CPU and detach from the computation graph
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time

        storage = get_event_storage()
        # Keep track of data time per rank
        storage.put_scalar("rank_data_time", data_time, cur_iter=cur_iter)

        # Gather metrics across all workers (for DDP-style training)
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            # data_time among workers can have high variance, so we take the max
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time, cur_iter=cur_iter)

            # Average the remaining metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={cur_iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar(
                "{}total_loss".format(prefix), total_losses_reduced, cur_iter=cur_iter
            )

            # Log the metrics to WandB
            if comm.is_main_process():
                # You can use `wandb.log` to log the training losses and other metrics.
                wandb.log(metrics_dict, step=cur_iter)

            # Log all metrics using the storage method for detectron2 (for event tracking)
            if len(metrics_dict) > 1:
                storage.put_scalars(cur_iter=cur_iter, **metrics_dict)
        
    # Inside `run_step` method
    def run_step(self):
        self._trainer.iter = self.iter
        self._trainer.run_step()  # Runs the default training step

        # Log training loss at every N steps (set your custom logging frequency)
        if self.iter % 20 == 0:  # Log every 100 steps (or your preferred frequency)
            storage = get_event_storage()
            loss_dict = {k: v[0] if isinstance(v, tuple) else v for k, v in storage.latest().items() if "loss" in k}

            # Log to WandB
            if comm.is_main_process():
                wandb.log({"training_total_loss": sum(loss_dict.values()), **loss_dict}, step=self.iter)



    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            20,  # validation period
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            )
        ))
        return hooks
    
    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res