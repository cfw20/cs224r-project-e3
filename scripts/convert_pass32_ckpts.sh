#!/bin/bash
# One-off conversion of both Qwen3-0.6B GSM8K checkpoints to HuggingFace format.
# Run this FIRST before firing the parallel evals.
#
# Usage:
#   conda activate cs224r-hw3
#   bash scripts/convert_pass32_ckpts.sh

set -euo pipefail

BASE_MODEL="Qwen/Qwen3-0.6B"

for exp in qwen3-0p6b-gsm8k-grpo-clean qwen3-0p6b-gsm8k-grpo-mixed qwen3-0p6b-gsm8k-grpo-gibberish; do
  echo "=== Converting $exp step 150 ==="
  modal run modal_convert_ckpt.py \
    --exp-name "$exp" \
    --step 150 \
    --base-model "$BASE_MODEL"
done

echo "=== Both checkpoints converted. HF dirs under /data/ckpts/${exp}_hf_step150 ==="
