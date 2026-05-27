# Data Augmentation Experiment: Code Flow Guide

> **Goal:** Understand what happens, in what order, and why, when you run the "data augmentation experiment" described in the README.
> This doc is written for someone new to the codebase.

---

## The Big Picture

This experiment tests a simple but powerful idea:

> **If we train a model on math problems that have random trivia sentences prepended to them, does the model learn to "ignore noise" and actually get *better* at solving clean math problems?**

We run **two training tracks** on the same base model (`Qwen3-1.7B`) and compare them:

| Track | Training Data | What's Special? |
|---|---|---|
| **A (clean)** | 7,473 original GSM8K problems | Baseline — no modifications |
| **B (mixed)** | 7,473 originals + 7,473 copies with trivia prepended | Model must learn to ignore noise |

Both tracks are evaluated on the **same clean test set** to see if Track B generalizes better.

---

## End-to-End Flow Diagram

Here is the entire pipeline from raw data to final metrics:

```
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1: Prepare Data (run once)                                   │
│  scripts/data_upload_gsm8k.py                                       │
│     │                                                               │
│     ▼                                                               │
│  examples/data_preprocess/gsm8k_padded.py                           │
│     ├─ mode=clean  → train_clean.parquet  (7,473 rows)             │
│     ├─ mode=mixed  → train_mixed.parquet  (14,946 rows)            │
│     └─ always      → test.parquet         (1,319 rows)             │
│     │                                                               │
│     ▼                                                               │
│  Saved to Modal Volume: /data/gsm8k_padded/                         │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2: Train (run twice, once per track)                          │
│  modal_train_gsm8k.py --track a                                     │
│  modal_train_gsm8k.py --track b                                     │
│     │                                                               │
│     ▼                                                               │
│  scripts/grpo/grpo_gsm8k_a100.sh  (launches verl GRPO trainer)      │
│     │                                                               │
│     ▼                                                               │
│  verl.trainer.main_ppo  (GRPO algorithm)                           │
│     ├─ loads train_clean.parquet  (track a)                        │
│     ├─ loads train_mixed.parquet  (track b)                        │
│     ├─ uses custom reward: verl/utils/reward_score/gsm8k_custom.py │
│     └─ saves checkpoints every 100 steps                            │
│     │                                                               │
│     ▼                                                               │
│  Checkpoints saved to: /data/ckpts/<experiment_name>/                 │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3: Convert Checkpoints (run twice)                            │
│  modal_convert_ckpt.py --track a                                    │
│  modal_convert_ckpt.py --track b                                    │
│     │                                                               │
│     ▼                                                               │
│  Merges FSDP shards into standard HuggingFace format                │
│  Saved to: /data/ckpts/<experiment_name>_hf/                         │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 4: Evaluate (run twice)                                       │
│  modal_eval_general.py --dataset gsm8k --model track_a              │
│  modal_eval_general.py --dataset gsm8k --model track_b              │
│     │                                                               │
│     ▼                                                               │
│  Phase 1: Prepare dataset → parquet                                 │
│  Phase 2: Generate responses → vLLM inference                      │
│  Phase 3: Score responses → pass@k metrics                          │
│     │                                                               │
│     ▼                                                               │
│  Results: metrics JSON + per-problem CSV                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Data Preparation

**File:** `examples/data_preprocess/gsm8k_padded.py`

This is where the "magic" happens for Track B. It loads the standard `openai/gsm8k` dataset from HuggingFace and builds the training files.

### What it does

1. Loads the GSM8K dataset (~7,500 train problems, ~1,300 test problems)
2. Extracts the ground truth answer from each problem's solution text using a regex:
   ```python
   # The GSM8K dataset has answers like: "...some reasoning... #### 72"
   solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
   ```
3. Builds each row into a standard format that the training code (`verl`) expects:
   ```python
   {
       "data_source": "openai/gsm8k",
       "prompt": [{"role": "user", "content": "{question} Let's think step by step..."}],
       "ability": "math",
       "reward_model": {"style": "rule", "ground_truth": "72"},
       "extra_info": {...}
   }
   ```

### Track A vs Track B

**Track A (`--mode clean`):**
- Just writes the original 7,473 rows as-is.

**Track B (`--mode mixed`):**
- Writes the original 7,473 rows **plus** 7,473 **padded clones**.
- Each clone prepends a random trivia fact to the question:
  ```python
  fact = rng.choice(TRIVIA_FACTS)  # e.g. "Honey never spoils."
  padded_q = f"{fact} {q}"          # "Honey never spoils. Janet has 3 apples..."
  ```

**Why trivia?** The facts are:
- **Short** (≤ 15 tokens)
- **Factual but irrelevant** to math
- **Contain no numerals** (so the model can't accidentally match a number from trivia to the answer)

**File:** `scripts/data_upload_gsm8k.py`

This is a thin Modal wrapper. It runs `gsm8k_padded.py` inside a cloud container and writes the files directly to a shared cloud volume (`e3-generation-vol`).

---

## Step 2: Training with GRPO

**File:** `modal_train_gsm8k.py`

This is the launcher. Think of it as a "remote controller" that sends your training job to a cloud GPU (H100).

```python
@app.local_entrypoint()
def main(track: str = "a", total_steps: int = 400):
    result = run_train.remote(track=track, ...)  # <-- dispatches to Modal GPU
```

The actual training happens inside `run_train()`, which runs a bash script:

```python
cmd = ["bash", "scripts/grpo/grpo_gsm8k_a100.sh"]
subprocess.run(cmd, ...)
```

**File:** `scripts/grpo/grpo_gsm8k_a100.sh`

This script launches the `verl` training framework with the **GRPO** algorithm. Key settings:

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \          # <-- Group Relative Policy Optimization
    data.train_files="$TRAIN_PARQUET" \     # <-- track a = clean, track b = mixed
    data.val_files="$VAL_PARQUET" \         # <-- always test.parquet (clean)
    data.train_batch_size=64 \               # 64 prompts per training step
    actor_rollout_ref.rollout.n=8 \         # 8 rollouts per prompt during training
    actor_rollout_ref.rollout.val_kwargs.n=4 \  # 4 rollouts per prompt during validation
    custom_reward_function.path=verl/utils/reward_score/gsm8k_custom.py \
    ...
```

### What is GRPO? (In One Sentence)

Instead of training the model to predict the next token like a normal language model, GRPO **generates multiple answers** for the same problem, **scores them** with a rule-based reward function, and **updates the model** to make high-reward answers more likely.

### The Custom Reward Function

**File:** `verl/utils/reward_score/gsm8k_custom.py`

This is critical. During training, the model only gets a reward if it produces the answer in the exact format `#### 72`.

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    return _gsm8k_compute_score(
        solution_str=solution_str,
        ground_truth=str(ground_truth),
        method="strict",      # <-- MUST produce "#### N" to get reward
        format_score=0.0,     # <-- wrong format = zero reward
        score=1.0,
    )
```

**Why strict?** If we used "flexible" scoring (which just finds the last number anywhere in the response), a model could get lucky by copying a number from the prepended trivia fact. Strict scoring forces the model to actually solve the math problem.

---

## Step 3: Checkpoint Conversion

**File:** `modal_convert_ckpt.py`

After training, `verl` saves checkpoints in a distributed format called **FSDP** (Fully Sharded Data Parallel). Each checkpoint is split across multiple "shard" files. To evaluate the model with standard tools, we need to merge these shards back into a normal HuggingFace model directory.

```
Input:  /data/ckpts/qwen3-1p7b-gsm8k-grpo-clean/global_step_400/actor/
        ├─ model_world_size_1_rank_0.pt
        └─ ...

Output: /data/ckpts/qwen3-1p7b-gsm8k-grpo-clean_hf/
        ├─ config.json
        ├─ model.safetensors
        └─ ...
```

The script finds the latest checkpoint automatically (`--step=-1`) or uses a specific step.

---

## Step 4: Evaluation

**File:** `modal_eval_general.py`

This is a three-phase pipeline. It takes a trained model, runs it on the test set, and computes accuracy metrics.

### Phase 1: Dataset Preparation

```python
def _prepare_dataset(dataset_key, ...):
    # 1. Load openai/gsm8k from HuggingFace
    ds = load_dataset("openai/gsm8k", "main")
    # 2. Extract ground truth answers with regex: "#### 72"
    gt = _extract_answer(row["answer"], "hash")
    # 3. Format into RLHF parquet with prompts
    out_df.to_parquet(out_path)
```

### Phase 2: Generation

```python
def _run_generation(...):
    cmd = [
        "python3", "-m", "verl.trainer.main_generation",
        f"data.path={data_path}",
        f"model.path={model_id}",
        f"data.n_samples={n_samples}",  # e.g. 4 responses per problem
        ...
    ]
```

This launches `verl`'s generation engine, which uses **vLLM** (a fast inference library) to generate multiple answers for each test problem.

### Phase 3: Scoring

```python
def _score_outputs(...):
    for each problem:
        for each generated response:
            score = score_fn(response, ground_truth)  # 1.0 = correct, 0.0 = wrong
```

For GSM8K evaluation, the scorer uses **`method='flexible'`** — it extracts the last number from the response. This is more forgiving than the strict scorer used during training, allowing us to measure real-world performance even if the model forgets the `####` format.

### Metrics Computed

The evaluation outputs:
- **`pass@1`**: What % of problems did the model get right on its first try?
- **`pass@k`**: Did the model get at least one correct answer out of `k` attempts? (uses an unbiased estimator)
- **`extract_failures`**: How many responses had no parseable number at all?

---

## Code-to-Concept Map

| File | Role in the Story |
|---|---|
| `scripts/data_upload_gsm8k.py` | Uploads data to the cloud volume |
| `examples/data_preprocess/gsm8k_padded.py` | Creates clean and noise-augmented datasets |
| `modal_train_gsm8k.py` | Sends training jobs to Modal GPUs |
| `scripts/grpo/grpo_gsm8k_a100.sh` | Configures the GRPO hyperparameters |
| `verl/utils/reward_score/gsm8k_custom.py` | Strict reward function (prevents cheating on trivia) |
| `modal_convert_ckpt.py` | Merges distributed checkpoints into a single model |
| `modal_eval_general.py` | Runs the full eval pipeline (prepare → generate → score) |

---

## Key Takeaways

1. **Track A** is the baseline. **Track B** is the experiment. The only difference is that Track B's training data has trivia prepended to half the examples.
2. **Strict scoring during training** is essential. Without it, the model could "cheat" by extracting numbers from the trivia instead of solving the math problem.
3. **Flexible scoring during evaluation** gives us a more realistic sense of whether the model actually learned to reason, even if it doesn't perfectly format its answer.
4. The entire pipeline is orchestrated via **Modal**, which means you can run GPU jobs from your laptop without managing cloud infrastructure.

---

*This document is a living guide. We will expand sections (e.g., diving into the GRPO algorithm, visualizing training curves, or analyzing per-problem failures) as needed.*
