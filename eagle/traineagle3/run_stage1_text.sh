#!/bin/bash

# Stage 1: Text Pre-training
# Adjust these paths as needed
BASE_MODEL_PATH="/workspace/Models/Qwen2.5-VL-7B-Instruct"
TEXT_DATA_PATH="/workspace/prepared_datasets/ultrachat_200k_json/regenerated_complete_train_T00.jsonl"
SAVE_DIR="./checkpoints/stage1_text"

# Create save directory
mkdir -p $SAVE_DIR

# Set CUDA devices (adjust as needed)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Run training
# Note: Using DeepSpeed launcher. Ensure deepspeed is installed.
deepspeed --include localhost:0,1,2,3,4,5,6,7 eagle/traineagle3/train_text_qwen.py \
    --basepath $BASE_MODEL_PATH \
    --trainpath $TEXT_DATA_PATH \
    --savedir $SAVE_DIR \
    --deepspeed_config eagle/traineagle3/ds_config.json \
    --num_epochs 1
