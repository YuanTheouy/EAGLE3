#!/bin/bash

# Default Configuration
# Adjust these variables or override them via environment variables
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/workspace/Models/Llama-3.1-8B-Instruct}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/workspace/prepared_datasets/ultrachat_200k_json/regenerated_complete_train_T00.jsonl}"
TEST_DATA_PATH="${TEST_DATA_PATH:-/workspace/prepared_datasets/ultrachat_200k_json/regenerated_complete_test_T00.jsonl}"
SAVE_DIR="${SAVE_DIR:-/workspace/Models/EAGLE-LLama-3.1-v3}"

# Training resources
INCLUDE_GPUS="${INCLUDE_GPUS:-localhost:0,1,2,3,4,5,6,7}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-60100}"

# Project setup
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS_DIR="$PROJECT_ROOT/logs"
WANDB_DIR="$LOGS_DIR"  # wandb will create a 'wandb' subdirectory here

# Ensure log directory exists
mkdir -p "$WANDB_DIR"

echo "=================================================="
echo "Starting EAGLE Training"
echo "Project Root: $PROJECT_ROOT"
echo "Logs Directory: $LOGS_DIR"
echo "WandB Directory: $WANDB_DIR"
echo "Base Model: $BASE_MODEL_PATH"
echo "Save Directory: $SAVE_DIR"
echo "GPUs: $INCLUDE_GPUS"
echo "=================================================="

# Export WandB environment variables
export WANDB_DIR="$WANDB_DIR"
export WANDB_MODE="${WANDB_MODE:-online}"
# export WANDB_API_KEY="" # Set this if needed

# Navigate to the training directory to ensure relative imports work
cd "$PROJECT_ROOT/eagle/traineagle3"

# Run DeepSpeed
deepspeed \
    --include="$INCLUDE_GPUS" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    main.py \
    --deepspeed_config ds_config.json \
    --basepath "$BASE_MODEL_PATH" \
    --trainpath "$TRAIN_DATA_PATH" \
    --testpath "$TEST_DATA_PATH" \
    --savedir "$SAVE_DIR"

echo "Training finished."
