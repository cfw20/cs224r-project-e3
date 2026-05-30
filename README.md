## Setup 
1. Follow instructions here to setup verl: https://verl.readthedocs.io/en/latest/README_vllm0.8.html
2. `conda activate verl`
3. `pip install seaborn`

## Data
Our training and test data are stored on Hugging Face:
- Training data, stage 1: https://huggingface.co/datasets/CMU-AIRe/e3-math-easy
- Training data, stage 2: https://huggingface.co/datasets/CMU-AIRe/e3-math-medhard
- Test data: https://huggingface.co/datasets/CMU-AIRe/hmmt-aime-2025

To set up data for training,
1. Create a local directory to store data
2. Run `python examples/data_preprocess/math/generate_dataset.py --local_dir $dir --remote_dir $hf_dir --split $split`
3. Ensure that `data.train_files` and `data.val_files` in your scripts (e.g., `scripts/grpo/grpo_16k.sh`) point to the downloaded data


## Eval
For eval you can run `bash scripts/eval.sh`

## Eval on Modal AI
The `modal_eval_general.py` script runs evaluation on a Modal GPU (A100-80GB) via a three-phase pipeline:

1. **Dataset preparation** — Loads AIME, MATH, or GSM8K from HuggingFace and formats prompts into a parquet file.
2. **Generation** — Runs `verl.trainer.main_generation` with vLLM on a single A100. It batches multiple questions together (batch_size = min(n_problems, 24)) and generates `n_samples` responses per question sequentially.
3. **Scoring** — Compares generated answers against ground truth using dataset-specific scorers and computes pass@k metrics.

**Supported datasets and models**
- Datasets: `aime`, `math`, `gsm8k`
- Models: `qwen` (Qwen3-1.7B), `e3` (CMU-AIRe/e3-1.7B)

**Performance notes**
- Default `n_samples` is 4 for GSM8K, 1 for MATH, and 16 for AIME.
- Default `max_response_length` is 32768 for all datasets.
- Higher `n_samples` or longer responses increase runtime significantly on a single A100.

**Smoke tests**
```bash
modal run modal_eval_general.py --dataset gsm8k --num-problems 5 --n-samples 1 --output-tag smoke
modal run modal_eval_general.py --dataset math  --num-problems 5 --n-samples 1 --max-response-length 2048 --output-tag smoke
modal run modal_eval_general.py --dataset aime  --num-problems 2 --n-samples 2 --max-response-length 2048 --output-tag smoke
```

**Full evaluations**
```bash
modal run modal_eval_general.py --dataset gsm8k --model qwen
modal run modal_eval_general.py --dataset math  --model qwen
modal run modal_eval_general.py --dataset aime  --model e3
```

## Experiments

We run **two distinct experiments**, each with its own tracks:

1. **Noise-control generalization (RLAD)** — Tests whether training with trivia-augmented prompts improves generalization on clean data. Run on both GSM8K and Hendrycks MATH.
2. **E3 curriculum on GSM8K** — Adapts the e3 training recipe (negative gradients + asymmetric clipping + curriculum) to GSM8K with a two-stage easy→hard curriculum.

### Track summary

| Track | Experiment | Dataset | Data | Description | GPU | Wandb project |
|-------|------------|---------|------|-------------|-----|---------------|
| **A** | Noise-control | GSM8K | clean | Standard GRPO, clean data | H100 | `rlad-noise-control` |
| **B** | Noise-control | GSM8K | mixed | Standard GRPO, trivia-augmented | H100 | `rlad-noise-control` |
| **A** | Noise-control | Hendrycks | clean | Standard GRPO, clean data | H200 | `rlad-hendrycks` |
| **B** | Noise-control | Hendrycks | mixed | Standard GRPO, trivia-augmented | H200 | `rlad-hendrycks` |
| **C** | E3 curriculum | GSM8K | clean | E3 recipe: easy→hard curriculum | H100 | `e3-gsm8k` |
| **D** | E3 curriculum | GSM8K | mixed | E3 recipe + trivia noise, easy→hard | H100 | `e3-gsm8k` |

> **Why H200 for Hendrycks?** Hendrycks MATH uses `max_response_length=2048` (vs. 1024 for GSM8K) and `ppo_max_token_len_per_gpu=32768` (vs. 16384), requiring more VRAM. All other tracks use H100.

All checkpoints land under `/data/ckpts/` on the `e3-generation-vol` Volume. Data for each experiment lives in its own directory to avoid clashes:
- Noise-control GSM8K: `/data/gsm8k_padded/`
- Noise-control Hendrycks: `/data/hendrycks_math/`
- E3 curriculum GSM8K: `/data/e3_gsm8k/`

---

## Experiment 1: Noise-control generalization (RLAD)

### Theoretical motivation

This experiment tests a **noise-control generalization hypothesis** inspired by curriculum-learning literature: *does exposing a model to irrelevant but factual noise during RL training force it to learn more robust extraction and reasoning strategies, thereby improving generalization on clean data?*

The core idea is that models trained solely on clean, tightly-coupled prompt-reward distributions may overfit to surface-level correlations. By deliberately injecting **semantically unrelated but grammatically coherent text** into a subset of training prompts, we force the policy to learn to *ignore* distractors and attend to the task-relevant signal. If the model successfully learns this filtering behavior, it should generalize *better* on the clean test distribution than a model trained exclusively on clean data (Track A), because it has been regularized against spurious correlations.

### How the augmented dataset is created

The augmentation logic is dataset-specific but follows the same pattern in both cases. For GSM8K, see `examples/data_preprocess/gsm8k_padded.py`. For Hendrycks MATH, see `examples/data_preprocess/hendrycks_padded.py`.

#### 1. Trivia fact selection

We maintain a fixed list of 20 short trivia facts (see `TRIVIA_FACTS` in each script). Each fact is chosen to satisfy three constraints:

- **No numerals**: Facts contain no Arabic numerals (e.g., "Octopuses have three hearts" is avoided; we use "Octopuses have several hearts"). This prevents the model from accidentally extracting a number from the trivia and matching it to the ground truth.
- **Short length**: Each fact is ≤15 tokens, minimizing the prompt-length overhead.
- **Factual but irrelevant**: The facts are true statements about the world but bear no logical connection to the math problems.

#### 2. Row construction

For **Track A** (`--mode clean`), each row contains the original question, a task-specific instruction, and the ground-truth answer:

**GSM8K** (`data_source="openai/gsm8k"`):
```json
{
  "data_source": "openai/gsm8k",
  "prompt": [{"role": "user", "content": "{question} Let's think step by step and output the final answer after \"####\"."}],
  "ability": "math",
  "level": "unknown",
  "reward_model": {"style": "rule", "ground_truth": "{answer}"},
  "extra_info": {"split": "train", "index": i, "answer": ..., "question": ...}
}
```

**Hendrycks MATH** (`data_source="DigitalLearningGmbH/MATH-lighteval"`):
```json
{
  "data_source": "DigitalLearningGmbH/MATH-lighteval",
  "prompt": [{"role": "user", "content": "{question} Let's think step by step and output the final answer within \\boxed{}."}],
  "ability": "math",
  "level": "unknown",
  "reward_model": {"style": "rule", "ground_truth": "{boxed_answer}"},
  "extra_info": {"split": "train", "index": i, "answer": ..., "question": ...}
}
```

For **Track B** (`--mode mixed`), each script first emits the same clean rows, then appends **padded clones**. Each clone:
- Selects one trivia fact uniformly at random from the 20 facts.
- Prepends that fact plus a space to the original question text: `"{fact} {question}"`.
- Keeps the **same ground truth** as the original row.
- Receives a continued index (`M + i`) so every row is unique.

The test set is **always generated clean**, regardless of mode, to guarantee a fair comparison.

#### 3. Reward scoring

**GSM8K — strict scorer**

The reward function is defined in `verl/utils/reward_score/gsm8k_custom.py`. It wraps the standard GSM8K scorer but **hard-codes `method='strict'`**:

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    return _gsm8k_compute_score(
        solution_str=solution_str,
        ground_truth=str(ground_truth),
        method="strict",
        format_score=0.0,
        score=1.0,
    )
```

In the original verl scorer, `method='flexible'` uses a regex that extracts the last number in the response. If we used flexible scoring, a model could earn reward on Track B by simply echoing a number from the prepended trivia fact (e.g., if the trivia mentioned "three" and the answer happened to be 3). The strict scorer requires the model to explicitly terminate its response with `#### N`, ensuring that the model must actually solve the math problem rather than exploit surface-level number matches in the noise.

**Hendrycks MATH — boxed extractor**

Hendrycks MATH uses the default verl math scorer (`verl/utils/reward_score/math.py`). It extracts the final `\boxed{...}` expression from the response and compares the unwrapped answer against the ground truth with mathematical equivalence checking. Because the answer must be explicitly wrapped in `\boxed{}`, the model cannot gain reward by quoting numbers from the prepended trivia — it must produce a properly formatted boxed answer. This makes the boxed format naturally robust to the same contamination risk that the strict scorer addresses for GSM8K.

#### 4. Training hyperparameters

Both tracks for a given dataset use the same GRPO configuration (defined in a dataset-specific shell script and launched via the corresponding `modal_train_*.py`):

**Shared across both datasets:**
- **Base model**: `Qwen/Qwen3-1.7B`
- **Algorithm**: GRPO (`adv_estimator=grpo`)
- **Train batch size**: 64 prompts per step
- **Rollouts per prompt**: 8 (`rollout.n=8`) for training, 4 for validation
- **Max prompt length**: 512 tokens
- **Learning rate**: 1e-6
- **KL coefficient**: 0.001
- **Entropy coefficient**: 0.001
- **PPO clipping**: low=0.2, high=0.5
- **PPO mini/micro batch size**: 32
- **Gradient checkpointing**: enabled
- **FSDP param offload (reference)**: enabled (saves memory)
- **Total steps**: 400
- **Validation frequency**: every 100 steps
- **Checkpoint frequency**: every 100 steps

**Dataset-specific differences:**

| Parameter | GSM8K | Hendrycks MATH |
|---|---|---|
| `data.max_response_length` | 1024 | 2048 |
| `test_freq` | every 100 steps | every 25 steps |
| `actor.ppo_max_token_len_per_gpu` | 16384 | 32768 |
| `rollout.max_num_batched_tokens` | 16384 | 32768 |
| `rollout.gpu_memory_utilization` | 0.6 | 0.8 |
| Custom reward function | `gsm8k_custom.py` (strict) | Default `math.py` (boxed) |

The longer response and batch-token budgets for Hendrycks MATH accommodate the significantly longer reasoning chains required by competition-level problems.

---

### Noise-control GSM8K — Tracks A & B

#### Step 1: Generate and upload data

```bash
modal run scripts/data_upload_gsm8k.py
```

Writes `train_clean.parquet`, `train_mixed.parquet`, `test.parquet` to `/data/gsm8k_padded/` on the Volume.

#### Step 2: Train

```bash
# Track A (clean)
modal run --detach modal_train_gsm8k.py --track a

# Track B (mixed)
modal run --detach modal_train_gsm8k.py --track b
```

#### Step 3: Convert checkpoints

```bash
modal run modal_convert_ckpt.py --track a
modal run modal_convert_ckpt.py --track b
```

#### Step 4: Evaluate on clean test set

```bash
# Using lm-eval (industry-standard GSM8K benchmark)
modal run modal_eval_general.py --dataset gsm8k --model track_a --use-lm-eval
modal run modal_eval_general.py --dataset gsm8k --model track_b --use-lm-eval
```

#### Smoke test

```bash
modal run modal_train_gsm8k.py --track a --total-steps 2
```

---

### Noise-control Hendrycks MATH — Tracks A & B

#### Step 1: Generate and upload data

```bash
modal run scripts/data_upload_hendrycks.py
```

Writes `train_clean.parquet`, `train_mixed.parquet`, and overwrites `test.parquet` with the canonical MATH-500 validation set (500 problems) to `/data/hendrycks_math/` on the Volume.

#### Step 2: Train

```bash
# Track A (clean)
modal run --detach modal_train_hendrycks.py --track a

# Track B (mixed)
modal run --detach modal_train_hendrycks.py --track b
```

#### Step 3: Convert checkpoints

```bash
modal run modal_convert_ckpt.py --track a --dataset math
modal run modal_convert_ckpt.py --track b --dataset math
```

#### Step 4: Evaluate on clean test set

```bash
modal run modal_eval_general.py --dataset math --model track_a
modal run modal_eval_general.py --dataset math --model track_b
```

#### Smoke test

```bash
modal run modal_train_hendrycks.py --track a --total-steps 2
```

---

## Experiment 2: E3 curriculum on GSM8K

This experiment adapts the **e3 training recipe** to GSM8K. The e3 recipe has three key ingredients:
1. **Asymmetric PPO clipping** (`clip_ratio_low=0.2`, `clip_ratio_high=0.5`)
2. **Negative gradients** (`only_train_on_positive=False`)
3. **Curriculum training** — start with easy problems and a short token budget, then graduate to harder problems and a longer budget.

We run this on both clean and mixed GSM8K data to isolate the interaction between curriculum training and noise augmentation.

### Curriculum design

| Stage | Data | `max_response_length` | Starting checkpoint |
|-------|------|----------------------|---------------------|
| 1 | easy split (clean or mixed) | 512 | Qwen3-1.7B base |
| 2 | hard split (clean or mixed) | 1024 | Converted stage-1 HF checkpoint |

Validation always uses `max_extrapolation_length = 2 * max_response_length` to test whether the model extrapolates to longer contexts than it was trained on.

> **Resuming between stages.** verl's FSDP checkpoints are sharded `.pt` files. Between stages you must merge them into a real HF model with `modal_convert_ckpt.py`. The in-loop `huggingface/` subdir only stores config + tokenizer, **not** the weights.

### How the easy/hard split is produced

1. Run a **strict** `#### N` scorer eval on the GSM8K **train** split at a 512-token budget (matching the stage-1 training reward):
   ```bash
   modal run modal_eval_general.py --dataset gsm8k --model qwen --split train \
       --scorer gsm8k_strict --max-response-length 512 \
       --output-dir /data/e3_gsm8k --output-tag train_strict_l512
   ```
2. Use `scripts/e3-grpo-gsm8k/split_gsm8k_dataset.ipynb` to:
   - Read the per-problem accuracy CSV from the Volume.
   - Partition ~70% highest-accuracy problems as **easy**, ~30% lowest as **hard**.
   - Generate four training parquets (`train_easy_clean`, `train_hard_clean`, `train_easy_mixed`, `train_hard_mixed`) and one clean `test.parquet`.
   - Spot-check sample easy vs. hard questions with base-model answers.
3. Upload the split parquets to the Volume:
   ```bash
   modal run scripts/data_upload_e3_gsm8k.py
   ```

### E3 GSM8K — Track C (clean)

#### Stage 1: easy, 512 tokens

```bash
modal run --detach modal_train_e3_gsm8k.py --track c --stage 1
```

#### Convert stage-1 checkpoint

```bash
modal run modal_convert_ckpt.py --exp-name qwen3-1p7b-gsm8k-e3-clean-stage1
```

#### Stage 2: hard, 1024 tokens

```bash
modal run --detach modal_train_e3_gsm8k.py --track c --stage 2
```

#### Evaluate final checkpoint

```bash
modal run modal_convert_ckpt.py --exp-name qwen3-1p7b-gsm8k-e3-clean-stage2
modal run modal_eval_general.py --dataset gsm8k --model qwen \
    --model-path /data/ckpts/qwen3-1p7b-gsm8k-e3-clean-stage2_hf --use-lm-eval
```

> `--model-path` bypasses the `MODEL_IDS` lookup and points directly at the converted HF checkpoint.

---

### E3 GSM8K — Track D (mixed)

#### Stage 1: easy mixed, 512 tokens

```bash
modal run --detach modal_train_e3_gsm8k.py --track d --stage 1
```

#### Convert stage-1 checkpoint

```bash
modal run modal_convert_ckpt.py --exp-name qwen3-1p7b-gsm8k-e3-mixed-stage1
```

#### Stage 2: hard mixed, 1024 tokens

```bash
modal run --detach modal_train_e3_gsm8k.py --track d --stage 2
```

#### Evaluate final checkpoint

```bash
modal run modal_convert_ckpt.py --exp-name qwen3-1p7b-gsm8k-e3-mixed-stage2
modal run modal_eval_general.py --dataset gsm8k --model qwen \
    --model-path /data/ckpts/qwen3-1p7b-gsm8k-e3-mixed-stage2_hf --use-lm-eval
```

---

## Evaluating arbitrary checkpoints

`modal_eval_general.py` supports two ways to specify the model to evaluate:

1. **Named model** (`--model qwen`, `--model track_a`, `--model e3`) — looks up the path in the internal `MODEL_IDS` table.
2. **Arbitrary path** (`--model-path /data/ckpts/...`) — bypasses the lookup entirely. Use this for e3 stages or any custom checkpoint.

Example:
```bash
modal run modal_eval_general.py --dataset gsm8k --model qwen \
    --model-path /data/ckpts/my-custom-experiment_hf --use-lm-eval
```

---

## Analyzing results

After tracks finish training, compare them using:

- `notebooks/rollouts/` — Per-step rollout JSONs and analysis for each track.
- `analyze_rollouts.py` — Standalone script for computing accuracy, response length, and answer-position statistics across saved rollouts.

Key metrics to compare across tracks:
- **Validation accuracy** at each checkpoint
- **Average response length** (shorter responses often indicate more confident, direct reasoning)
- **Answer position** (earlier placement of the final answer suggests more efficient reasoning chains)
- **Extrapolation performance** (e3 only): accuracy at `2x` the training response length
