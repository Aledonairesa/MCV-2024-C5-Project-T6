#!/bin/bash

CURRENTDATE=$(date +"%m-%d_%H:%M:%S")
RUN_NAME="faster-rcnn-pred-domain-shift2"
CUDA_VISIBLE_DEVICES=5 nohup \
    python inference.py \
        --config-file config_files/Custom-Faster-RCNN_domain-shift.yaml \
        --input /data/users/mireia/MCV/C5/domain-shift-dataset/test_images/* \
        --output runs/inference/${RUN_NAME} \
        --confidence-threshold 0.1 \
        --csv-path runs/inference/${RUN_NAME}/${RUN_NAME}.csv \
        --opts MODEL.WEIGHTS output_files/faster_rcnn_finetuning_domain-shift2/model_best.pth \
    > output_files/inference/${RUN_NAME}_inference_${CURRENTDATE}.out &
