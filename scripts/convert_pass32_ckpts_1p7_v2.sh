#!/bin/bash
# One-off conversion of both Qwen3-1.7B v2 GSM8K checkpoints to HuggingFace format.
# Run this FIRST before firing the parallel evals.
#
# These are the "variance-check" v2 reruns of the original 1.7B Track A/B,
# trained to step 400.
#
# Usage:
#   conda activate cs224r-hw3
#   bash scripts/convert_pass32_ckpts_1p7_v2.sh

set -euo pipefail

BASE_MODEL="Qwen/Qwen3-1.7B"

for exp in qwen3-1p7b-gsm8k-grpo-clean-v2 qwen3-1p7b-gsm8k-grpo-mixed-v2; do
  echo "=== Converting $exp step 400 ==="
  modal run modal_convert_ckpt.py \
    --exp-name "$exp" \
    --step 400 \
    --base-model "$BASE_MODEL"
done

echo "=== Both checkpoints converted. HF dirs under /data/ckpts/<exp>_hf_step400 ==="
