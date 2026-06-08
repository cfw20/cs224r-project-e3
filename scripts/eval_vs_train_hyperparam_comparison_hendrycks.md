# Eval vs. Training Hyperparameter Comparison — Hendrycks MATH Pass@k Sweep

**Training config**: `scripts/grpo/grpo_hendrycks_a100.sh` (invoked via `modal_train_hendrycks.py`, runs on Modal H200)
**Eval config**   : `scripts/run_pass32_sweep_1p7_hendrycks.sh` (invokes `modal_eval_general.py`, runs on Modal H200)
**Target step**   : 400 (final checkpoint)

---

## 1. Core Generation / Inference Settings

| Parameter | Training (GRPO) | Eval (pass@32 sweep) | Match? | Significant? | Notes |
|---|---|---|---|---|---|
| **GPU** | H200 | H200 | ✅ Yes | No | Script name says `a100.sh`, but actual Modal decorator uses H200. |
| **max_response_length** | 2048 | 2048 | ✅ Yes | **No** | Same cap. MATH reasoning needs longer than GSM8K. |
| **max_prompt_length** | 512 | 1024 | ❌ No | **Minor** | Eval defaults to 1024 in `modal_eval_general.py`; prompts are short anyway (~500 chars). |
| **temperature** | 0.6 (rollout + val_kwargs) | 0.6 | ✅ Yes | **No** | Same sampling temperature. |
| **do_sample** | True (rollout + val_kwargs) | True | ✅ Yes | **No** | Same sampling mode. |
| **top_k** | -1 (default, not overridden) | 20 | ❌ No | **YES** ⚠️ | Training disables top_k filtering; eval restricts to top-20 tokens. Alters sampling distribution. |
| **top_p** | 1.0 (default, not overridden) | 0.95 | ❌ No | **YES** ⚠️ | Training disables nucleus sampling; eval uses p=0.95. Alters sampling distribution. |
| **gpu_memory_utilization** | 0.8 | 0.9 | ❌ No | Minor | Eval bumps to 0.9 for larger batch (32×500). Still safe on H200 (141 GB). |
| **max_num_batched_tokens** | 32768 | 100000 | ❌ No | Minor | Eval raises limit to fit 32 samples × 500 problems in one batch. |
| **tensor_model_parallel_size** | 1 | 1 | ✅ Yes | No | Single-GPU for both. |
| **dtype** | bfloat16 (default) | bfloat16 (default) | ✅ Yes | No | Same precision. |
| **enforce_eager** | False | False | ✅ Yes | No | CUDA graph enabled on both. |

---

## 2. Sampling Budget (n)

| Parameter | Training (GRPO) | Eval (pass@32 sweep) | Match? | Significant? | Notes |
|---|---|---|---|---|---|
| **rollout.n** (train-time samples) | 8 | — | — | — | Per-question samples used for GRPO advantage estimation. |
| **val_kwargs.n** (validation during training) | 4 | — | — | — | Validation rollouts saved to WandB at `test_freq` steps. |
| **n_samples** (eval) | — | 32 | — | — | Dedicated eval for pass@k up to 32. Intentionally higher. |

> **Verdict**: The eval intentionally generates 32 samples per problem vs. 4 during training validation. This is **by design** (more samples → better pass@k estimate), not a mismatch.

---

## 3. Model / Checkpoint

| Parameter | Training (GRPO) | Eval (pass@32 sweep) | Match? | Significant? | Notes |
|---|---|---|---|---|---|
| **Base architecture** | Qwen/Qwen3-1.7B | Qwen/Qwen3-1.7B | ✅ Yes | No | Same base model. |
| **Checkpoint loaded** | FSDP shard (live) | `/data/ckpts/qwen3-1p7b-hendrycks-grpo-{clean,mixed}_hf` | — | — | Eval loads the HF-converted step-400 checkpoint (converted 2026-05-29). |
| **Gradient checkpointing** | Enabled | — | — | — | Training only (eval is inference-only). |
| **FSDP / sharding** | Enabled | — | — | — | Training only. |

---

## 4. Dataset & Scoring

| Parameter | Training (GRPO) | Eval (pass@32 sweep) | Match? | Significant? | Notes |
|---|---|---|---|---|---|
| **Test dataset** | HuggingFaceH4/MATH-500 (via `test.parquet`) | HuggingFaceH4/MATH-500 | ✅ Yes | **No** | Same 500-problem validation set. |
| **Answer format** | `\boxed{}` | `\boxed{}` | ✅ Yes | **No** | Both expect boxed notation. |
| **Instruction appended** | "Let's think step by step and output the final answer within `\boxed{}`." | "Let's think step by step and output the final answer within `\boxed{}`." | ✅ Yes | **No** | Identical prompt suffix. |
| **Scorer** | `verl/utils/reward_score/math.py` (`compute_score`) | `verl/utils/reward_score/math.py` (`compute_score`) | ✅ Yes | **No** | Same scorer on both. |
| **num_problems** | All 500 | All 500 | ✅ Yes | **No** | Full test set evaluated. |

---

## 5. Training-Only / Eval-Only Hyperparameters

These are expected to differ (not a mismatch):

| Parameter | Training | Eval | Why different? |
|---|---|---|---|
| **train_batch_size** | 64 | — | Eval is inference-only; no training batch. |
| **ppo_mini_batch_size** | 32 | — | GRPO optimizer batch. Irrelevant for eval. |
| **optimizer lr** | 1e-6 | — | Only relevant for training updates. |
| **clip_ratio** | [0.2, 0.5] | — | GRPO policy clip bounds. |
| **kl_loss_coef** | 0.001 | — | KL regularization during GRPO. |
| **entropy_coeff** | 0.001 | — | Entropy bonus during GRPO. |
| **total_training_steps** | 400 | — | Training budget. |
| **save_freq** | 100 | — | Rollouts saved every 100 steps. |
| **test_freq** | 25 | — | Validation run every 25 steps. |
| **batch_size** (inference) | — | 500 (= n_problems) | Eval processes all 500 problems at once. |
| **output_tag** | — | `step400` | Artifact tag for the sweep. |

---

## 6. Risk Assessment Summary

| Risk | Severity | Mitigation |
|---|---|---|
| **top_k / top_p mismatch** (train: -1/1.0, eval: 20/0.95) | 🔴 Medium | The sampling distribution differs between training rollouts and eval. This could inflate or deflate pass@k estimates relative to what the model saw during training. To compare apples-to-apples, set eval `top_k=-1` and `top_p=1.0` to mirror training defaults. |
| **max_prompt_length mismatch** (train: 512, eval: 1024) | 🟢 Low | Prompts are ~500 chars / ~100 tokens. The 512 vs 1024 difference is irrelevant in practice—no prompt is long enough to hit either limit. |
| **gpu_memory_utilization** (0.8 → 0.9) | 🟢 Low | H200 has 141 GB; 0.9 utilization leaves ~14 GB headroom. Safe. |
| **max_num_batched_tokens** (32768 → 100000) | 🟢 Low | Eval needs higher throughput for 32×500 samples. The 1.7B model comfortably fits. |

---

*Generated: 2026-06-07*
