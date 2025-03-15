#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
import wandb
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    DatasetMapper,
)
from detectron2.engine import (
    default_argument_parser,
    default_setup,
    DefaultTrainer,
    hooks,
    launch,
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
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data.datasets import register_coco_instances

from custom_training_utils import CustomTrainer, build_evaluator


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

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
    
def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def register_dataset():
    register_coco_instances(
        "KITTI-MOTS_train",
        {},
        "/home/mmajo/MCV/C5/MCV-2024-C5-Project-T6/detectron/kitti-mots-coco-train.json",
        "/data/users/mireia/MCV/C5/COCO-JSON-KITTI-MOTS/train"
    )

    register_coco_instances(
        "KITTI-MOTS_val",
        {},
        "/home/mmajo/MCV/C5/MCV-2024-C5-Project-T6/detectron/kitti-mots-coco-val.json",
        "/data/users/mireia/MCV/C5/COCO-JSON-KITTI-MOTS/val"
    )

# def register_dataset():
#     register_coco_instances(
#         "domain-shift_train",
#         {},
#         "/home/mmajo/MCV/C5/MCV-2024-C5-Project-T6/detectron/chest_xray_train.json",
#         "/data/users/mireia/MCV/C5/domain-shift-dataset/train_images"
#     )

#     register_coco_instances(
#         "domain-shift_val",
#         {},
#         "/home/mmajo/MCV/C5/MCV-2024-C5-Project-T6/detectron/chest_xray_val.json",
#         "/data/users/mireia/MCV/C5/domain-shift-dataset/val_images"
#     )

#     register_coco_instances(
#         "domain-shift_test",
#         {},
#         "/home/mmajo/MCV/C5/MCV-2024-C5-Project-T6/detectron/chest_xray_test.json",
#         "/data/users/mireia/MCV/C5/domain-shift-dataset/test_images"
#     )

def main(args):
    register_dataset()
    cfg = setup(args)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Initialize Weights and Biases
    wandb.init(project="detectron2_finetuning", name="domain-shift", config=args)

    if args.eval_only:
        model = CustomTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = CustomTrainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(CustomTrainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.build_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


def invoke_main() -> None:
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
