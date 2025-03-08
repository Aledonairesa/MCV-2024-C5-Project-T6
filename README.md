# Session 1 - Object Detection

## Contents
- [Introduction](#introduction)
- [Quick Set-Up](#quick-set-up)
- [Datasets and Metrics](#datasets-and-metrics)
- [Running Inference with Pre-Trained Models (Task C)](#running-inference-with-pre-trained-models-task-c)
- [Evaluating Pre-Trained Models (Task D)](#evaluating-pre-trained-models-task-d)
- [Fine-Tuning the Models on KITTI-MOTS (Task E)](#fine-tuning-the-models-on-kitti-mots-task-e)
- [Fine-Tuning on Chest X-Ray Dataset (Domain Shift) (Task F)](#fine-tuning-on-chest-x-ray-dataset-domain-shift-task-f)



## Introduction

In this Session we experiment with different object detection architectures. In particular, we work with three different models from three different frameworks:
- Faster R-CNN from [Detectron2](https://ai.meta.com/tools/detectron2/)
- DETR from [Hugging Face](https://huggingface.co/)
- YOLO from [Ultralytics](https://www.ultralytics.com/)

For each of these architectures, we do the following:
- Perform inference without fine-tuning.
- Fine-tune on a specific dataset.
- Extract qualitative and quantitative results, comparing them.

Additionally, we:
- Fine-tune on a domain shifted dataset.
- Globally compare the different architectures.

The models are pre-trained on the COCO dataset, and the dataset we use for inference and fine-tuning is the KITTI-MOTS. The domain shifted dataset is the Chest X-Ray dataset from Hugging Face. For evaluation, we use the official COCO metrics. More information about the dataset and metrics in Section [Datasets and Metrics](#datasets-and-metrics).



## Quick Set-Up

To perform the experiments, we assume the KITTI-MOTS dataset is structured as follows:

```
.
└── KITTI-MOTS/
    ├── instances_txt/
    │   ├── 0000.txt
    │   ├── 0001.txt
    │   └── ...
    └── training/
        └── image_02/
            ├── 0000/
            │   ├── 000000.png
            │   ├── 000001.png
            │   └── ...
            ├── 0001/
            │   ├── 000000.png
            │   ├── 000001.png
            │   └── ...
            └── ...
```

Regarding the scripts, they are all in Python. We have performed experiments separately for each differen architecture/framework. Thus, we specify next three different set-ups:

### Running Faster R-CNN (Detectron2) experiments

### Running DETR (Hugging Face) experiments

The scripts related to DETR are in the `/huggingface` folder and are the following:
- `inference_DETR`

### Running YOLO (Ultralytics) experiments



## Datasets and Metrics



## Running Inference with Pre-Trained Models (Task C)



## Evaluating Pre-Trained Models (Task D)



## Fine-Tuning the Models on KITTI-MOTS (Task E)



## Fine-Tuning on Chest X-Ray Dataset (Domain Shift) (Task F)


