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
Run `modal_eval_general.py` on the **GSM8K train set** with Qwen3-1.7B to get per-problem pass rates:

```bash
modal run modal_eval_general.py --dataset gsm8k --model qwen --split train
```

This writes a `per_problem_gsm8k_qwen_{tag}.csv` to `/data/aime_eval/` on the `e3-generation-vol` volume. Each row has `problem_idx`, `n_correct`, `accuracy` (pass rate over `n_samples`). We sort by `accuracy` ascending and partition:
- **easy** (~70%): highest-accuracy problems (Qwen3 solves them reliably)
- **hard** (~30%): lowest-accuracy problems (Qwen3 struggles)

**Check result:** No existing GSM8K eval artifacts were found in `/data/aime_eval/` on the Modal volume, so this eval must be run first.

### 2. Upload split parquets to Modal volume
Adapt the pattern from `scripts/data_upload_gsm8k.py` to create and upload the two new parquets (`train_easy.parquet`, `train_hard.parquet`) to a path like `/data/gsm8k_e3/` on the same Modal volume.

### 3. Two e3 training scripts for the curriculum
Use `scripts/grpo/grpo_gsm8k_a100.sh` as the template. It already has the correct e3 hyperparameters:

```bash
actor_rollout_ref.actor.clip_ratio_low=0.2 \
actor_rollout_ref.actor.clip_ratio_high=0.5 \
actor_rollout_ref.actor.only_train_on_positive=False \
custom_reward_function.path=verl/utils/reward_score/gsm8k_custom.py \
```

We write two scripts:

| Script | Data | `max_response_length` | Starting model |
|--------|------|----------------------|----------------|
| `stage1_easy_512.sh` | `train_easy.parquet` | 512 | Qwen3-1.7B base |
| `stage2_hard_1024.sh` | `train_hard.parquet` | 1024 | Best Stage 1 checkpoint |

Each script sets `data.max_extrapolation_length = 2 * max_response_length` so validation tests longer context.

### Important nuance
The eval must run on `--split train`, not the default test split, because we need per-problem accuracy on the **training** data to decide which problems are easy vs. hard. The default in `modal_eval_general.py` for GSM8K is `default_split="test"`, so `--split train` is required.

Also, `grpo_gsm8k_a100.sh` already uses the GSM8K custom scorer (`gsm8k_custom.py` with `method="strict"`), which is the right reward function for e3 training because it enforces the `#### N` format. Keep it exactly as-is.
