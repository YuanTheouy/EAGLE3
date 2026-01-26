#!/bin/bash

# Stage 2: VQA Fine-tuning
# Adjust these paths as needed
BASE_MODEL_PATH="/workspace/Models/Qwen2.5-VL-7B-Instruct"
VQA_DATA_PATH="/workspace/datasets/VQAv2/vqa_v2_validation_answer.jsonl"
VQA_TEST_PATH="/workspace/datasets/VQAv2/vqa_v2_validation_answer.jsonl" 
SAVE_DIR="/workspace/ckpts/stage2_vl_2"
STAGE1_CHECKPOINT="/workspace/ckpts/stage1_text_2/state_10" # Example: load state_0 from stage 1

# Create save directory
mkdir -p $SAVE_DIR

# Set CUDA devices (adjust as needed)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Run training
# Note: We pass --text_model_path to load the pre-trained EAGLE head from Stage 1
deepspeed --include localhost:0,1,2,3,4,5,6,7 eagle/traineagle3/train_vl_qwen.py \
    --basepath $BASE_MODEL_PATH \
    --trainpath $VQA_DATA_PATH \
    --savedir $SAVE_DIR \
    --testpath $VQA_TEST_PATH \
    --deepspeed_config eagle/traineagle3/ds_config.json \
    --text_model_path $STAGE1_CHECKPOINT \
    --num_epochs 20
