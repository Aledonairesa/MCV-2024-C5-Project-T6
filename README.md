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

The scripts related to Faster R-CNN are in the `/detectron` folder and are the following:
TO DO
TO DO
TO DO

### Running DETR (Hugging Face) experiments

The scripts related to DETR are in the `/detr` folder and are the following:
- `inference_DETR.py`: performs inference without fine tuning + plots prediction examples + calculates COCO metrics.
- `finetune_DETR.py`: fine-tunes DETR to coco and saves the model.
- `eval_finetune_DETR.py`: with the saved fine-tuned model, plots prediction examples + calculates COCO metrics.
- `utils.py`: utilities for the previous DETR scripts.
- `requirements.txt`: pip requirements to execute the DETR scripts.

The scripts don't require any arguments and can be executed directly with `python <script>.py`. More information about the scripts in the following sections.

### Running YOLO (Ultralytics) experiments

The scripts related to YOLO are in the `/huggingface` folder and are the following:
TO DO
TO DO
TO DO



## Datasets and Metrics

In this Session there are three datasets involved:
- [**COCO (Common Objects in Context)**](https://cocodataset.org/): It's the dataset the models we use are pre-trained with. It's is a widely used large-scale dataset for object detection, segmentation, keypoint detection, and image captioning. It contains over 330,000 images, with 1.5 million object instances spanning 80 object categories.
- [**KITTI-MOTS**](https://www.cvlibs.net/datasets/kitti/): It's the dataset we use for inference and fine-tuning. It's an extension of the KITTI dataset, designed for multi-object tracking and segmentation (MOTS) in autonomous driving scenarios. It provides instance-level segmentation masks for cars and pedestrians across 2,000 frames from real-world street scenes. We only use the official training partition, sub-dividing it in our own train and test sets (sequences 0-15 and 16-20, respectively).
- [**Chest X-Ray**](https://huggingface.co/datasets/Tsomaros/Chest_Xray_N_Object_Detection): It's the dataset we use for fine-tuning with domain shift. It features different conditions, containing 8 different classes to detect. Only has one bounding box per image, and the train split (that we use) contains 880 images.

The metrics that we use are the official COCO metrics:
- **AP**: AP at IoU=50:05:95  
- **AP<sub>IoU=.50</sub>**: AP at IoU=0.50  
- **AP<sub>IoU=.75</sub>**: AP at IoU=0.75  
- **AP<sub>small</sub>**: AP for small objects (area < 32²)  
- **AP<sub>medium</sub>**: AP for medium objects (32² < area < 96²)  
- **AP<sub>large</sub>**: AP for large objects (area > 96²)  
- **AR<sub>max=1</sub>**: AR for 1 detection per image  
- **AR<sub>max=10</sub>**: AR for 10 detections per image  
- **AR<sub>max=100</sub>**: AR for 100 detections per image  
- **AR<sub>small</sub>**: AR for small objects (area < 32²)  
- **AR<sub>medium</sub>**: AR for medium objects (32² < area < 96²)  
- **AR<sub>large</sub>**: AR for large objects (area > 96²)

We calculate them using the `COCOeval()` function from the `pycocotools` library.

## Running Inference with Pre-Trained Models (Task C)

### Faster R-CNN from Detectron2

### DETR from Hugging Face

For this task, first we load the pre-trained DETR model “facebook/detr-resnet-50” from Hugging Face, along with the corresponding image processor. Then, we preprocess the test data with a custom PyTorch DataLoader, which uses a tailored collate function and the image processor. We then loop over the dataset, filtering and matching COCO to KITTI-MOTS classes. This is encapsulated in the `inference_DETR.py` script, which also plots some example prediction images, and calculates the official COCO metrics. Results are saved to the folder `results_detr_kittimots_inference`, which is created automatically if it doesn't exist.

Here's an example inference image:

### YOLO from Ultralytics



## Evaluating Pre-Trained Models (Task D)



## Fine-Tuning the Models on KITTI-MOTS (Task E)



## Fine-Tuning on Chest X-Ray Dataset (Domain Shift) (Task F)


