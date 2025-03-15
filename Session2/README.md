# Session 2 - Object Segmentation

Project Presentation Link: [.](.)


## Contents
- [Introduction](#introduction)
- [Quick Set-Up](#quick-set-up)
- [Datasets and Metrics](#datasets-and-metrics)
- [Object Segmentation Examples](#object-segmentation-examples)



## Introduction

In this Session, we experiment with different **object segmentation** architectures. In particular, we work with three different models from three different frameworks:
- Mask R-CNN from [Detectron2](https://ai.meta.com/tools/detectron2/)
- Mask2Former from [Hugging Face](https://huggingface.co/)
- YOLO-SEG from [Ultralytics](https://www.ultralytics.com/)

For each of these architectures, we do the following:
- Perform inference, off-the-shelf.
- Fine-tune on a specific dataset.
- In both cases, extract qualitative and quantitative results, comparing them.

Additionally, we:
- Fine-tune on a domain shifted dataset.
- Globally compare the different architectures.

The models are pre-trained on the **COCO dataset**, and the dataset we use for inference and fine-tuning is the **KITTI-MOTS dataset**. The domain shifted dataset is the **Satellite Building Segmentation dataset** from Hugging Face. For evaluation, we use the official COCO metrics. More information about the dataset and metrics in Section [Datasets and Metrics](#datasets-and-metrics).



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

The Satellite Building Segmentation dataset is automatically loaded using the Hugging Face API.

### Running the Mask R-CNN experiments

-


### Running the Mask2Former experiments

The scripts related to Mask2Former (off-the-shelf + fine-tuning) are in the `/mask2former` folder and are the following:
- `inference_Mask2Former`: performs inference on KITTI-MOTS (preprocessed) with the off-the-shelf Mask2Former (tiny), visualizes the predictions and calculates performance metrics.
- `finetune_Mask2Former`: fine-tunes Mask2Former (tiny) on KITTI-MOTS (preprocessed) and saves the resulting model.
- `eval_finetune_Mask2Former`: loads a fine-tuned model, performs inference, visualizes the predictions and calculates performance metrics.
- `requirements.txt`: pip requirements to execute the Mask2Former scripts.

The scripts don't require any arguments and can be executed directly with `python <script>.py`. Adapt the KITTI-MOTS dataset path as needed (`base_dir` variable), as well as the output folder (`output_dir` variable).


### Running the YOLO-SEG experiments

-


### Running the domain-shift experiments with Mask2Former

The scripts related to the domain-shift with Mask2Former can be found in the `domain_shift` folder, and are the following:
- `domain_shift_Mask2Former.py`: loads and preprocess the Satellite Building Segmentation dataset and fine-tunes Mask2Former (tiny), then saves the fine-tuned model.
- `eval_domain_shift_Mask2Former.py`: loads a fine-tuned model on the Satellite Building Segmentation dataset, performs inference, visualizes the results and calculates performance metrics.
- `requirements.txt`: pip requirements to execute the domain-shift with Mask2Former scripts. The requirements are the same as the Mask2Former experiments requirements.

The scripts don't require any arguments and can be executed directly with `python <script>.py`. Adapt the output folder (`output_dir` variable) as needed.



## Datasets and Metrics

In this Session, there are three datasets involved:
- [**COCO (Common Objects in Context)**](https://cocodataset.org/): It's the dataset the models we use are pre-trained with. It's a widely used large-scale dataset for different tasks, including object segmentation. It contains over 330,000 images, with 1.5 million object instances spanning 80 object categories.
- [**KITTI-MOTS**](https://www.cvlibs.net/datasets/kitti/): It's the dataset we use for inference and fine-tuning. It's an extension of the KITTI dataset, designed for multi-object tracking and segmentation (MOTS) in autonomous driving scenarios. It provides instance-level segmentation masks for cars and pedestrians across 2,000 frames. We only use the official training partition, sub-dividing it in our own train and test sets (sequences 0-15 and 16-20, respectively).
- [**Satellite Building Segmentation**](https://huggingface.co/datasets/keremberke/satellite-building-segmentation): It's the dataset we use for fine-tuning with domain shift. It features 9665 satellite images, showing one or multiple buildings. The task is to segment the buildings. There is one class (building), and each image usually contains multiple instances in very different sizes. The dataset is partitioned in train, validation and test. We only use the train partition for training, and test for evaluaton.

The metrics that we use are the official COCO metrics:
- **AP**: AP at IoU=50:05:95  
- **AP<sub>IoU=.50</sub>**: AP at IoU=0.50  
- **AP<sub>IoU=.75</sub>**: AP at IoU=0.75  
- **AP<sub>S</sub>**: AP for small objects (area < 32²)  
- **AP<sub>M</sub>**: AP for medium objects (32² < area < 96²)  
- **AP<sub>L</sub>**: AP for large objects (area > 96²)  
- **AR<sub>1</sub>**: AR for 1 detection per image  
- **AR<sub>10</sub>**: AR for 10 detections per image  
- **AR<sub>100</sub>**: AR for 100 detections per image  
- **AR<sub>S</sub>**: AR for small objects (area < 32²)  
- **AR<sub>M</sub>**: AR for medium objects (32² < area < 96²)  
- **AR<sub>L</sub>**: AR for large objects (area > 96²)

We calculate them using the `COCOeval()` function from the `pycocotools` library.



## Object Segmentation Examples
