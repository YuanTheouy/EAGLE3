#!/bin/bash

# 正确的命令格式，确保反斜杠后面没有空格
python3 eagle/evaluation/gen_baseline_answer_qwen25vl.py \
--model-path /workspace/Models/Qwen2.5-VL-7B-Instruct \
--bench-name vqa \
--max-new-token 100 \
--temperature 0.0 \
--num-choices 1

# 或者使用一行命令
# python3 eagle/evaluation/gen_baseline_answer_qwen25vl.py --model-path /workspace/Models/Qwen2.5-VL-7B-Instruct --bench-name vqa --max-new-token 100 --temperature 0.0 --num-choices 1