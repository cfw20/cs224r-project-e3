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

We operationalize this on GSM8K (a grade-school math benchmark) because:
- The answers are strictly numerical, making reward computation unambiguous.
- The questions are short and self-contained, so prepended trivia does not semantically interfere with the problem logic.
- It provides a controlled setting where we can precisely attribute performance differences to the noise-augmented training regimen.

### Experiment design

We run **two independent GRPO training tracks** on Qwen3-1.7B, using identical hyperparameters and evaluation protocols:

| Track | Training data | Size | Description |
|---|---|---|---|
| **A (clean)** | `train_clean.parquet` | 7,473 rows | Standard GSM8K train split. |
| **B (mixed)** | `train_mixed.parquet` | 14,946 rows | 7,473 originals + 7,473 copies with prepended trivia. |

Both tracks evaluate on the **same clean test split** (`test.parquet`, 1,319 rows) to ensure comparability.

### How the augmented dataset is created

The augmentation is implemented in `examples/data_preprocess/gsm8k_padded.py`.

#### 1. Trivia fact selection

We maintain a fixed list of 20 short trivia facts (see `TRIVIA_FACTS` in the script). Each fact is chosen to satisfy three constraints:

- **No numerals**: Facts contain no Arabic numerals (e.g., "Octopuses have three hearts" is avoided; we use "Octopuses have several hearts"). This prevents the model from accidentally extracting a number from the trivia and matching it to the ground truth.
- **Short length**: Each fact is ≤15 tokens, minimizing the prompt-length overhead.
- **Factual but irrelevant**: The facts are true statements about the world but bear no logical connection to grade-school math problems.

#### 2. Row construction

For **Track A** (`--mode clean`), each row is:
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

For **Track B** (`--mode mixed`), the script first emits the same 7,473 clean rows, then appends 7,473 **padded clones**. Each clone:
- Selects one trivia fact uniformly at random from the 20 facts.
- Prepends that fact plus a space to the original question text: `"{fact} {question}"`.
- Keeps the **same ground truth** as the original row.
- Receives a continued index (`M + i`) so every row is unique.

The test set is **always generated clean**, regardless of mode, to guarantee a fair comparison.

#### 3. Why strict reward scoring is critical

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

#### 4. Training hyperparameters

Both tracks use the same GRPO configuration (defined in `scripts/grpo/grpo_gsm8k_a100.sh` and launched via `modal_train_gsm8k.py`):

- **Base model**: `Qwen/Qwen3-1.7B`
- **GPU**: H100 (single node, 1 GPU)
- **Algorithm**: GRPO (`adv_estimator=grpo`)
- **Train batch size**: 64 prompts per step
- **Rollouts per prompt**: 8 (`rollout.n=8`) for training, 4 for validation
- **Max prompt length**: 512 tokens
- **Max response length**: 1024 tokens
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

### Running the end-to-end experiment

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
