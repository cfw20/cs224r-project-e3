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
- Default `n_samples` is 4 for GSM8K/MATH and 16 for AIME.
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
modal run modal_eval_general.py --dataset math  --model qwen --subset 500
modal run modal_eval_general.py --dataset aime  --model e3
```

## Data augmentation experiment

### Theoretical motivation

This experiment tests a **noise-control generalization hypothesis** inspired by curriculum-learning literature: *does exposing a model to irrelevant but factual noise during RL training force it to learn more robust extraction and reasoning strategies, thereby improving generalization on clean data?*

The core idea is that models trained solely on clean, tightly-coupled prompt-reward distributions may overfit to surface-level correlations. By deliberately injecting **semantically unrelated but grammatically coherent text** into a subset of training prompts, we force the policy to learn to *ignore* distractors and attend to the task-relevant signal. If the model successfully learns this filtering behavior, it should generalize *better* on the clean test distribution than a model trained exclusively on clean data (Track A), because it has been regularized against spurious correlations.

We operationalize this on two math benchmarks:

- **GSM8K** (grade-school math): Answers are strictly numerical, making reward computation unambiguous. Questions are short and self-contained, so prepended trivia does not semantically interfere with the problem logic.
- **Hendrycks MATH** (competition math): Problems require multi-step symbolic reasoning and terminate answers in `\boxed{}`. The longer reasoning chains and more complex answer formats provide a stronger test of whether the model can maintain structured reasoning despite noise.

Both provide controlled settings where we can precisely attribute performance differences to the noise-augmented training regimen.

### Experiment design

We run **two independent GRPO training tracks** on Qwen3-1.7B for each dataset, using identical hyperparameters and evaluation protocols within that dataset:

#### GSM8K

| Track | Training data | Size | Description |
|---|---|---|---|
| **A (clean)** | `train_clean.parquet` | 7,473 rows | Standard GSM8K train split. |
| **B (mixed)** | `train_mixed.parquet` | 14,946 rows | 7,473 originals + 7,473 copies with prepended trivia. |

Both tracks evaluate on the **same clean test split** (`test.parquet`, 1,319 rows).

#### Hendrycks MATH

| Track | Training data | Size | Description |
|---|---|---|---|
| **A (clean)** | `train_clean.parquet` | ~7,500 rows | Standard Hendrycks MATH train split. |
| **B (mixed)** | `train_mixed.parquet` | ~15,000 rows | ~7,500 originals + ~7,500 copies with prepended trivia. |

Both tracks evaluate on the **same clean test split** (`test.parquet`, ~5,000 rows).

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
- **GPU**: H100 (single node, 1 GPU)
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
- **vLLM GPU memory utilization**: 0.6 (leaves headroom for PyTorch)
- **Total steps**: 400
- **Validation frequency**: every 100 steps
- **Checkpoint frequency**: every 100 steps

**Dataset-specific differences:**

| Parameter | GSM8K | Hendrycks MATH |
|---|---|---|
| `data.max_response_length` | 1024 | 2048 |
| `actor.ppo_max_token_len_per_gpu` | 16384 | 32768 |
| `rollout.max_num_batched_tokens` | 16384 | 32768 |
| Custom reward function | `gsm8k_custom.py` (strict) | Default `math.py` (boxed) |
| Wandb project | `rlad-noise-control` | `rlad-hendrycks` |

The longer response and batch-token budgets for Hendrycks MATH accommodate the significantly longer reasoning chains required by competition-level problems.

### Running the end-to-end experiment (GSM8K)

#### Step 1: Generate and upload data to Modal Volume

```bash
modal run scripts/data_upload_gsm8k.py
```

This single command runs `gsm8k_padded.py` twice inside a Modal container (once for `--mode clean`, once for `--mode mixed`) and writes all three parquets (`train_clean.parquet`, `train_mixed.parquet`, `test.parquet`) to the shared `e3-generation-vol` Volume.

#### Step 2: Train Track A (clean)

```bash
modal run --detach modal_train_gsm8k.py --track a
```

This trains on `train_clean.parquet` for 400 steps with validation every 100 steps. Checkpoints and wandb logs are automatically saved. The `--detach` flag returns your terminal immediately while the job runs in the cloud.

#### Step 3: Train Track B (mixed)

```bash
modal run --detach modal_train_gsm8k.py --track b
```

This trains on `train_mixed.parquet` with identical hyperparameters. You can run this in parallel with Track A if you have separate Modal GPU quota, or run it sequentially after Track A finishes.

#### Step 4: Evaluate both checkpoints

After training completes, convert the latest verl FSDP checkpoint to HuggingFace format (this is handled by `modal_convert_ckpt.py`):

```bash
modal run modal_convert_ckpt.py --track a
modal run modal_convert_ckpt.py --track b
```

Then evaluate each converted checkpoint on the clean GSM8K test set:

```bash
modal run modal_eval_general.py --dataset gsm8k --model track_a
modal run modal_eval_general.py --dataset gsm8k --model track_b
```

#### Smoke test (2 steps)

Before committing to the full 400-step run, verify the pipeline works:

```bash
modal run modal_train_gsm8k.py --track a --total-steps 2
```

This completes in ~5-10 minutes on an H100 and validates that data loading, vLLM generation, reward scoring, and wandb logging all function correctly.

### Running the end-to-end experiment (Hendrycks MATH)

The Hendrycks MATH pipeline is structurally identical to the GSM8K pipeline, but uses dataset-specific scripts and hyperparameters.

#### Step 1: Generate and upload data to Modal Volume

```bash
modal run scripts/data_upload_hendrycks.py
```

This runs `hendrycks_padded.py` twice (clean and mixed) and writes the parquets to `/data/hendrycks_math/` on the `e3-generation-vol` Volume.

#### Step 2: Train both tracks

```bash
# Track A (clean)
modal run --detach modal_train_hendrycks.py --track a

# Track B (mixed)
modal run --detach modal_train_hendrycks.py --track b
```

Both use `scripts/grpo/grpo_hendrycks_a100.sh` with the longer response budgets described above. The default is 400 steps; override with `--total-steps N` if needed.

#### Step 3: Evaluate both checkpoints

Convert checkpoints:

```bash
modal run modal_convert_ckpt.py --track a
modal run modal_convert_ckpt.py --track b
```

Evaluate on the clean Hendrycks MATH test set:

```bash
modal run modal_eval_general.py --dataset math --model track_a
modal run modal_eval_general.py --dataset math --model track_b
```

#### Smoke test

```bash
modal run modal_train_hendrycks.py --track a --total-steps 2
```

### Analyzing results

After both tracks finish training, you can compare them using the analysis notebooks and scripts in the repository:

- `notebooks/rollouts/` — Contains per-step rollout JSONs and analysis for each track.
- `analyze_rollouts.py` — Standalone script for computing accuracy, response length, and answer-position statistics across saved rollouts.

Key metrics to compare across tracks:
- **Validation accuracy** at each checkpoint
- **Average response length** (shorter responses often indicate more confident, direct reasoning)
- **Answer position** (earlier placement of the final answer suggests more efficient reasoning chains)
