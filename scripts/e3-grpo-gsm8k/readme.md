# Adapting the e3 Recipe for GSM8K on Qwen3-1.7B

## Why it works
GSM8K is easier than AIME, but Qwen3-1.7B still only solves ~30% without RL. The e3 recipe applies because the base model's **generation vs. verification asymmetry** still exists — it can check an arithmetic answer more reliably than it can produce one from scratch. Training with negative gradients and a coupled curriculum teaches the model to exploit this by chaining trial -> verify -> refine.

## The adapted recipe

### 1. Base model: Qwen3-1.7B
Use the same base model path as the existing e3 scripts. Its inherent verification-generation asymmetry is task-agnostic.

### 2. Keep negative gradients
Set the same non-standard hyperparameters the e3 scripts use:
- `actor_rollout_ref.actor.only_train_on_positive=False` — do **not** mask out wrong answers; let their negative advantages produce gradients.
- `actor_rollout_ref.actor.clip_ratio_low=0.2` and `actor_rollout_ref.actor.clip_ratio_high=0.5` (or similar) — loosen the upper bound so the model can increase probability on rare exploration tokens more aggressively.

### 3. Two-stage curriculum (data difficulty + token budget)
Split GSM8K into an **easy** subset and a **hard** subset (e.g., by base-model pass rate or problem complexity).

| Stage | Data | `max_response_length` | Starting checkpoint |
|-------|------|----------------------|---------------------|
| 1 | `easy` | 512 | Qwen3-1.7B base |
| 2 | `hard` | 1024 | Stage 1 checkpoint |

For each stage, set `data.max_extrapolation_length = 2 * max_response_length` so validation tests longer context.

### 4. Evaluation
After Stage 2, evaluate on the full GSM8K test set with:
- **In-distribution**: `max_response_length=1024` (matches training budget)
- **Extrapolation**: `max_response_length=2048` (2x training budget)

This tests whether the model learned to keep exploring when given more tokens than it was trained on — the core e3 property.

## Important nuance to watch
For GSM8K, 512 tokens might already be plenty for the easy subset. The curriculum only works if the token budget is the **smallest** one where the model is still *positively rewarded* for chaining an extra verification step. If the model solves easy problems in 100 tokens flat, lower the budget or enrich the easy set with problems that actually need the full 512 to get right. Otherwise the model has no incentive to learn exploration.

## Experiment Setup
All training runs for this experiment will be executed on Modal AI instances.

## Implementation Plan

### 1. Produce easy/hard split via Qwen3-1.7B eval

Run `modal_eval_general.py` on the **GSM8K train set** with Qwen3-1.7B to get per-problem pass rates. We use the **strict** `#### N` scorer (matching the training reward) at a `512`-token budget (the stage-1 easy budget), and direct all output to this experiment's own directory via `--output-dir`:

```bash
modal run modal_eval_general.py --dataset gsm8k --model qwen --split train \
    --scorer gsm8k_strict --max-response-length 512 \
    --output-dir /data/e3_gsm8k --output-tag train_strict_l512
```

This writes `per_problem_gsm8k_qwen_train_strict_l512.csv` to `/data/e3_gsm8k/` on the `e3-generation-vol` volume. Each row has `problem_idx`, `n_correct`, `accuracy` (pass rate over `n_samples`). We sort by `accuracy` ascending and partition:
- **easy** (~70%): highest-accuracy problems (Qwen3 solves them reliably)
- **hard** (~30%): lowest-accuracy problems (Qwen3 struggles)

> **Note on the scorer.** The default GSM8K eval scorer in `modal_eval_general.py` is `gsm8k_flexible` (last-number-in-response) at a long budget. For difficulty splitting we deliberately switch to `gsm8k_strict` so the measured difficulty matches what the model is actually rewarded for during e3 training. A test-split flexible eval already exists on the volume (`per_problem_gsm8k_qwen_n4_l32768.csv`); only this **train-split, strict, 512-token** eval is new.

A new notebook, **`split_gsm8k_dataset.ipynb`**, will be created in this directory (`/home/chung/cs224r-project-e3/scripts/e3-grpo-gsm8k/`). It will:
1. Read the eval CSV from `/data/e3_gsm8k/` and generate the six split parquets (`train_easy_clean.parquet`, `train_hard_clean.parquet`, `train_easy_mixed.parquet`, `train_hard_mixed.parquet`, `train_easy_trivia.parquet`, `train_hard_trivia.parquet`).
2. Let the user **spot-check** sample easy questions vs. hard questions, along with the base model's generated answers, to sanity-check the split quality.

All artifacts for this experiment are kept under `/data/e3_gsm8k/` on the Modal volume to avoid conflicts with other experiments.

### 2. Upload split parquets to Modal volume
Use `scripts/data_upload_e3_gsm8k.py` (modeled on `scripts/data_upload_gsm8k.py`) to upload the six split parquets plus a clean `test.parquet` to `/data/e3_gsm8k/` on the same Modal volume.

### 3. Three e3 training tracks: Track C, Track D and Track E
Following the data-augmentation experiment design (Track A = clean, Track B = mixed), we run three independent e3 training tracks on the GSM8K curriculum:

| Track | Training data | Description |
|-------|--------------|-------------|
| **C (clean)** | `train_easy_clean.parquet` + `train_hard_clean.parquet` | Standard e3 curriculum on clean GSM8K splits. |
| **D (mixed)** | `train_easy_mixed.parquet` + `train_hard_mixed.parquet` | Same curriculum, but each split includes clean originals **plus** padded clones with prepended trivia. |
| **E (trivia-only)** | `train_easy_trivia.parquet` + `train_hard_trivia.parquet` | Same curriculum on **only** trivia-padded questions (no clean originals). Trivia facts are drawn independently from `TRIVIA_FACTS`. |

All tracks evaluate on the **same clean test split** (`test.parquet`). Training is launched via `modal_train_e3_gsm8k.py --track {c,d,e} --stage {1,2}`, writing checkpoints to `ckpts/qwen3-1p7b-gsm8k-e3-{flavor}-stage{N}` where `flavor` is `clean` (c), `mixed` (d), or `trivia` (e).

Training uses `scripts/grpo/grpo_gsm8k_e3.sh` — a parametrized copy of `grpo_gsm8k_a100.sh` that takes `MAX_RESPONSE_LENGTH`, `MAX_EXTRAPOLATION_LENGTH`, and `MODEL_PATH` via env vars. It keeps the e3 hyperparameters:

```bash
actor_rollout_ref.actor.clip_ratio_low=0.2 \
actor_rollout_ref.actor.clip_ratio_high=0.5 \
actor_rollout_ref.actor.only_train_on_positive=False \
custom_reward_function.path=verl/utils/reward_score/gsm8k_custom.py \
```

Each track has a two-stage curriculum:

| Stage | Data | `max_response_length` | Starting checkpoint |
|-------|------|----------------------|---------------------|
| 1 | `easy` (clean or mixed) | 512 | Qwen3-1.7B base |
| 2 | `hard` (clean or mixed) | 1024 | Converted stage-1 HF checkpoint |

`modal_train_e3_gsm8k.py` sets `data.max_extrapolation_length = 2 * max_response_length` so validation tests longer context.

> **Resuming between stages.** verl's FSDP checkpoints are sharded `.pt` files; the in-loop `huggingface/` subdir only stores the config and tokenizer, **not** the weights. So stage 2 cannot point directly at the stage-1 checkpoint dir. Between stages you must merge the shards into a real HF model with `modal_convert_ckpt.py` (the same step used to prepare checkpoints for eval). `modal_train_e3_gsm8k.py` then auto-resolves the stage-2 start model to `/data/ckpts/<stage1-exp>_hf`.

### End-to-end run sequence (Track C example)
```bash
# 0. Difficulty eval (writes per-problem CSV to /data/e3_gsm8k/)
modal run modal_eval_general.py --dataset gsm8k --model qwen --split train \
    --scorer gsm8k_strict --max-response-length 512 \
    --output-dir /data/e3_gsm8k --output-tag train_strict_l512

# 1. Generate splits locally + spot-check, then upload
#    (run scripts/e3-grpo-gsm8k/split_gsm8k_dataset.ipynb)
modal run scripts/data_upload_e3_gsm8k.py

# 2. Stage 1 (easy, 512)
modal run --detach modal_train_e3_gsm8k.py --track c --stage 1

# 3. Convert stage-1 checkpoint to HF so stage 2 can resume
modal run modal_convert_ckpt.py --exp-name qwen3-1p7b-gsm8k-e3-clean-stage1

# 4. Stage 2 (hard, 1024) — resumes from the converted stage-1 model
modal run --detach modal_train_e3_gsm8k.py --track c --stage 2
```
For Track D, swap `--track c` for `--track d` and use `qwen3-1p7b-gsm8k-e3-mixed-stage1` in the convert step.

For Track E (trivia-only), swap `--track c` for `--track e` and use `qwen3-1p7b-gsm8k-e3-trivia-stage1` in the convert step:
```bash
# Track E Stage 1 (easy, 512)
modal run --detach modal_train_e3_gsm8k.py --track e --stage 1

# Convert stage-1 checkpoint to HF so stage 2 can resume
modal run modal_convert_ckpt.py --exp-name qwen3-1p7b-gsm8k-e3-trivia-stage1

# Track E Stage 2 (hard, 1024) — resumes from the converted stage-1 model
modal run --detach modal_train_e3_gsm8k.py --track e --stage 2
```

### Important nuance
The eval must run on `--split train`, not the default test split, because we need per-problem accuracy on the **training** data to decide which problems are easy vs. hard. The default in `modal_eval_general.py` for GSM8K is `default_split="test"`, so `--split train` is required.

Also, `grpo_gsm8k_e3.sh` uses the GSM8K custom scorer (`gsm8k_custom.py` with `method="strict"`), which is the right reward function for e3 training because it enforces the `#### N` format. Keep it exactly as-is.
