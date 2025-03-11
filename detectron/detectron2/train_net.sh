#!/bin/bash

CURRENTDATE=$(date +"%m-%d_%H:%M:%S")
RUN_NAME="train-faster-aug1"
CUDA_VISIBLE_DEVICES=4 nohup \
    python train_net.py \
        --num-gpus 1 \
        --config-file config_files/Custom-Faster-RCNN.yaml \
    > output_files/train/${RUN_NAME}_${CURRENTDATE}.out &
