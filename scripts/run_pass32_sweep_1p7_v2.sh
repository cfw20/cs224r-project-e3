#!/bin/bash
# Run the two 32-sample GSM8K evals IN PARALLEL on separate Modal instances for
# the Qwen3-1.7B v2 variance-check runs (Track A clean vs Track B mixed).
#
# IMPORTANT: run scripts/convert_pass32_ckpts_1p7_v2.sh FIRST (one-off) so
# the HF checkpoints exist under /data/ckpts/<exp>_hf_step400.
#
# Outputs land on the Modal volume 'e3-generation-vol' under /data/pass32_1p7b_v2/:
#   per_problem_gsm8k_<track>_step<N>.csv   (n_correct / n_samples per question)
#   metrics_gsm8k_<track>_step<N>.json      (pass@1,2,4,8,16,32)
#
# We deliberately do NOT eval the base model (saves time); we only compare
# Track A vs Track B at step 400. The --model tags are suffixed with _v2 so the
# output filenames do not collide with any existing 1.7B (non-v2) evals.
#
# The notebook Section 9 downloads those per_problem CSVs and computes the
# pass@k sweep itself.
#
# Usage:
#   conda activate cs224r-hw3
#   bash scripts/run_pass32_sweep_1p7_v2.sh
#
# Notes:
#   - These evals are independent; open two terminals or background them with
#     "&" to run truly in parallel.
#   - --max-response-length 1024 matches the training-time generation budget
#     (data.max_response_length=1024 in scripts/grpo/grpo_gsm8k_a100.sh).
#   - modal_eval_general.py uses H200 for faster inference.

set -euo pipefail

DATASET="gsm8k"
N_SAMPLES=32
MAX_RESP=1024
OUTPUT_DIR="/data/pass32_1p7b_v2"

echo "=== Eval 1/2: Track A v2 step 400 ==="
modal run modal_eval_general.py \
  --dataset "$DATASET" \
  --model-path "/data/ckpts/qwen3-1p7b-gsm8k-grpo-clean-v2_hf_step400" \
  --model track_a_v2 \
  --n-samples "$N_SAMPLES" \
  --max-response-length "$MAX_RESP" \
  --output-dir "$OUTPUT_DIR" \
  --output-tag step400

echo "=== Eval 2/2: Track B v2 step 400 ==="
modal run modal_eval_general.py \
  --dataset "$DATASET" \
  --model-path "/data/ckpts/qwen3-1p7b-gsm8k-grpo-mixed-v2_hf_step400" \
  --model track_b_v2 \
  --n-samples "$N_SAMPLES" \
  --max-response-length "$MAX_RESP" \
  --output-dir "$OUTPUT_DIR" \
  --output-tag step400

echo "=== All evals launched. Artifacts on volume under ${OUTPUT_DIR}/ ==="
