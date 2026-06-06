#!/bin/bash
# Convert every saved Qwen3-0.6B GSM8K checkpoint to HF, then run a 32-sample
# (pass@32) GSM8K eval for each checkpoint of both tracks on Modal.
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
#   - Step 0 = base model Qwen/Qwen3-0.6B (no checkpoint); evaluated once and
#     reused for both tracks in the notebook.
#   - --max-response-length 1024 matches the training-time generation budget.

set -euo pipefail

BASE_MODEL="Qwen/Qwen3-0.6B"
DATASET="gsm8k"
N_SAMPLES=32
MAX_RESP=1024
OUTPUT_DIR="/data/pass32_0p6b"
STEPS=(150)

declare -A TRACK_EXP=(
  [track_a]="qwen3-0p6b-gsm8k-grpo-clean"
  [track_b]="qwen3-0p6b-gsm8k-grpo-mixed"
)

echo "=== Step 0: base model eval (shared by both tracks) ==="
modal run modal_eval_general.py \
  --dataset "$DATASET" \
  --model-path "$BASE_MODEL" \
  --model base \
  --n-samples "$N_SAMPLES" \
  --max-response-length "$MAX_RESP" \
  --output-dir "$OUTPUT_DIR" \
  --output-tag step0

for track in track_a track_b; do
  exp="${TRACK_EXP[$track]}"
  for step in "${STEPS[@]}"; do
    echo "=== Convert $track ($exp) step $step ==="
    modal run modal_convert_ckpt.py \
      --exp-name "$exp" \
      --step "$step" \
      --base-model "$BASE_MODEL"

    echo "=== Eval $track step $step (n=$N_SAMPLES) ==="
    modal run modal_eval_general.py \
      --dataset "$DATASET" \
      --model-path "/data/ckpts/${exp}_hf_step${step}" \
      --model "$track" \
      --n-samples "$N_SAMPLES" \
      --max-response-length "$MAX_RESP" \
      --output-dir "$OUTPUT_DIR" \
      --output-tag "step${step}"
  done
done

echo "=== Sweep complete. Artifacts on volume under ${OUTPUT_DIR}/ ==="
