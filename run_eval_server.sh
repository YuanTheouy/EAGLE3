#!/bin/bash

# 运行 VQA 投机采样评测
# 请确保 vqa_v2_val_sampled_200_converted.jsonl 已上传到服务器

CUDA_VISIBLE_DEVICES=2 python eagle/evaluation/gen_ea_answer_qwen25vl.py \
 --base-model-path /workspace/Models/Qwen2.5-VL-7B-Instruct \
 --ea-model-path /workspace/ckpts/stage1_text/state_19 \
 --question-file vqa_v2_val_sampled_200_converted.jsonl \
 --answer-file vqa_predictions.jsonl \
 --max-new-tokens 512 \
 --temperature 0.0
