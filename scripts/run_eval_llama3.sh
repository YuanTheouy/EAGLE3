#!/bin/bash

# Default paths - modify these as needed or pass as environment variables
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/workspace/Models/Llama-3.1-8B-Instruct}"
EA_MODEL_PATH="${EA_MODEL_PATH:-/workspace/Models/EAGLE-LLama-3.1-v2/state_5}"
BENCH_NAME="${BENCH_NAME:-alpaca}"
GPU_ID="${GPU_ID:-0}"

# Get the project root directory (assuming script is in scripts/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Define output file
OUTPUT_DIR="$PROJECT_ROOT/results/$BENCH_NAME"
mkdir -p "$OUTPUT_DIR"
ANSWER_FILE="$OUTPUT_DIR/llama3_eagle3.jsonl"

echo "Running evaluation with:"
echo "  Base Model: $BASE_MODEL_PATH"
echo "  EA Model: $EA_MODEL_PATH"
echo "  Benchmark: $BENCH_NAME"
echo "  Output: $ANSWER_FILE"
echo "  GPU: $GPU_ID"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

python3 -m eagle.evaluation.gen_ea_answer_llama3chat \
  --base-model-path "$BASE_MODEL_PATH" \
  --ea-model-path "$EA_MODEL_PATH" \
  --use_eagle3 \
  --bench-name "$BENCH_NAME" \
  --answer-file "$ANSWER_FILE"

echo "Done. Results saved to $ANSWER_FILE"
