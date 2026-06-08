#!/bin/bash
# Run the four 32-sample GSM8K evals IN PARALLEL on separate Modal instances.
#
# IMPORTANT: run scripts/convert_pass32_ckpts.sh FIRST (one-off, ~4 min) so
# the HF checkpoints exist under /data/ckpts/<exp>_hf_step150.
#
# Outputs land on the Modal volume 'e3-generation-vol' under /data/pass32_0p6b/:
#   per_problem_gsm8k_<track>_step<N>.csv   (n_correct / n_samples per question)
#   metrics_gsm8k_<track>_step<N>.json      (pass@1,2,4,8,16,32)
#
# The notebook Section 9 downloads those per_problem CSVs and computes the
# pass@k sweep itself.
#
# Usage:
#   conda activate cs224r-hw3
#   bash scripts/run_pass32_sweep.sh
#
# Notes:
#   - These evals are now independent; open three terminals or background
#     them with "&" to run truly in parallel.
#   - --max-response-length 1024 matches the training-time generation budget.
#   - modal_eval_general.py now uses H200 for faster inference.

set -euo pipefail

DATASET="gsm8k"
N_SAMPLES=32
MAX_RESP=1024
OUTPUT_DIR="/data/pass32_0p6b"

echo "=== Eval 1/4: Base model (step 0) ==="
modal run modal_eval_general.py \
  --dataset "$DATASET" \
  --model-path "Qwen/Qwen3-0.6B" \
  --model base \
  --n-samples "$N_SAMPLES" \
  --max-response-length "$MAX_RESP" \
  --output-dir "$OUTPUT_DIR" \
  --output-tag step0

echo "=== Eval 2/4: Track A step 150 ==="
modal run modal_eval_general.py \
  --dataset "$DATASET" \
  --model-path "/data/ckpts/qwen3-0p6b-gsm8k-grpo-clean_hf_step150" \
  --model track_a \
  --n-samples "$N_SAMPLES" \
  --max-response-length "$MAX_RESP" \
  --output-dir "$OUTPUT_DIR" \
  --output-tag step150

echo "=== Eval 3/4: Track B step 150 ==="
modal run modal_eval_general.py \
  --dataset "$DATASET" \
  --model-path "/data/ckpts/qwen3-0p6b-gsm8k-grpo-mixed_hf_step150" \
  --model track_b \
  --n-samples "$N_SAMPLES" \
  --max-response-length "$MAX_RESP" \
  --output-dir "$OUTPUT_DIR" \
  --output-tag step150

echo "=== Eval 4/4: Track D step 150 ==="
modal run modal_eval_general.py \
  --dataset "$DATASET" \
  --model-path "/data/ckpts/qwen3-0p6b-gsm8k-grpo-gibberish_hf_step150" \
  --model track_d \
  --n-samples "$N_SAMPLES" \
  --max-response-length "$MAX_RESP" \
  --output-dir "$OUTPUT_DIR" \
  --output-tag step150

echo "=== All evals launched. Artifacts on volume under ${OUTPUT_DIR}/ ==="
