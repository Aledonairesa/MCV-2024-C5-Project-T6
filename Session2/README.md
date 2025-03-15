# Session 2 - Object Segmentation

Project Presentation Link: [.](.)


## Contents
- [Introduction](#introduction)
- [Quick Set-Up](#quick-set-up)
- [Datasets and Metrics](#datasets-and-metrics)



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



## Datasets and Metrics
