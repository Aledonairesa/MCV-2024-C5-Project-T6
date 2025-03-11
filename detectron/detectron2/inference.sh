#!/bin/bash

CURRENTDATE=$(date +"%m-%d_%H:%M:%S")
RUN_NAME="faster-rcnn-pred-abril"
CUDA_VISIBLE_DEVICES=4 nohup \
    python eval_detectron.py \
        --config-file config_files/Base-RCNN-FPN.yaml \
        --input /data/users/mireia/MCV/C5/YOLO-KITTI-MOTS/all_frames/* \
        --output runs/inference/${RUN_NAME} \
        --confidence-threshold 0.5 \
        --csv-path runs/inference/${RUN_NAME}/${RUN_NAME}.csv \
        --opts MODEL.WEIGHTS pretrained_weights/mask_rcnn_R_50_FPN_3x.pkl \
    > output_files/inference/${RUN_NAME}_inference_${CURRENTDATE}.out &
