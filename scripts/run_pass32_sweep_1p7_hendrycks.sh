#!/bin/bash
# Run the two 32-sample MATH-500 evals IN PARALLEL on separate Modal instances
# for the Qwen3-1.7B Hendrycks MATH runs (Track A clean vs Track B mixed).
#
# NO conversion step is needed: the step-400 HF checkpoints already exist on the
# Modal volume 'e3-generation-vol' at:
#   /data/ckpts/qwen3-1p7b-hendrycks-grpo-clean_hf   (converted 2026-05-29, after step 400)
#   /data/ckpts/qwen3-1p7b-hendrycks-grpo-mixed_hf
# (Both were converted from the final/step-400 FSDP checkpoint.)
#
# Outputs land on the Modal volume under /data/pass32_1p7b_hendrycks/:
#   per_problem_math_<track>_step<N>.csv   (n_correct / n_samples per question)
#   metrics_math_<track>_step<N>.json      (pass@1,2,4,8,16,32)
#
# We deliberately do NOT eval the base model (saves time); we only compare
# Track A vs Track B at step 400. The --model tags are suffixed with _hendrycks
# so the output filenames do not collide with any existing MATH evals.
#
# The notebook Section 9 downloads those per_problem CSVs and computes the
# pass@k sweep itself.
#
# Usage:
#   conda activate cs224r-hw2
#   bash scripts/run_pass32_sweep_1p7_hendrycks.sh
#
# Notes:
#   - These evals are independent; open two terminals or background them with
#     "&" to run truly in parallel.
#   - --dataset math => MATH-500 (HuggingFaceH4/MATH-500), scorer=math (SymPy).
#   - --max-response-length 2048 matches the training-time generation budget
#     (data.max_response_length=2048 in scripts/grpo/grpo_hendrycks_a100.sh).
#   - modal_eval_general.py uses H200 for faster inference.

set -euo pipefail

DATASET="math"
N_SAMPLES=32
MAX_RESP=2048
OUTPUT_DIR="/data/pass32_1p7b_hendrycks"

echo "=== Eval 1/2: Track A (clean) hendrycks step 400 ==="
modal run modal_eval_general.py \
  --dataset "$DATASET" \
  --model-path "/data/ckpts/qwen3-1p7b-hendrycks-grpo-clean_hf" \
  --model track_a_hendrycks \
  --n-samples "$N_SAMPLES" \
  --max-response-length "$MAX_RESP" \
  --output-dir "$OUTPUT_DIR" \
  --output-tag step400

echo "=== Eval 2/2: Track B (mixed) hendrycks step 400 ==="
modal run modal_eval_general.py \
  --dataset "$DATASET" \
  --model-path "/data/ckpts/qwen3-1p7b-hendrycks-grpo-mixed_hf" \
  --model track_b_hendrycks \
  --n-samples "$N_SAMPLES" \
  --max-response-length "$MAX_RESP" \
  --output-dir "$OUTPUT_DIR" \
  --output-tag step400

echo "=== All evals launched. Artifacts on volume under ${OUTPUT_DIR}/ ==="
