# Modal AI Volume Artifacts Summary

This document tracks all output artifacts produced by experiments in this repo and persisted to the Modal AI volume `e3-generation-vol` (mounted at `/data` in containers).

---

## Experiment Overview

| Experiment | Tracks | Dataset | Curriculum | Max Response Length | Train Batch Size | Test Freq | Reward Function | What Makes It Different |
|------------|--------|---------|------------|--------------------|------------------|-----------|-----------------|------------------------|
| `grpo_gsm8k_e3.sh` | c (clean), d (mixed), e (trivia) | GSM8K | Two-stage: easy → hard | 512 (stage 1) / 1024 (stage 2) | 128 | 100 | Custom (`gsm8k_custom.py`) | Parameterized copy of `grpo_gsm8k_a100.sh`; adds two-stage curriculum, larger batch size (128), and `val_kwargs.n=1` |
| `grpo_gsm8k_a100.sh` | a (clean), b (mixed) | GSM8K | None (single stage) | 1024 | 64 | 25 | Custom (`gsm8k_custom.py`) | Standard GRPO baseline on GSM8K; `val_kwargs.n=4` |
| `grpo_hendrycks_a100.sh` | a (clean), b (mixed) | Hendrycks MATH | None (single stage) | 2048 | 64 | 25 | Default (`math.py`) | Same base GRPO recipe as `grpo_gsm8k_a100.sh` but on MATH; longer responses (2048); default verl reward routing; `val_kwargs.n=4` |

---

## All Experiments — Checkpoint Directory Overview (as of 2026-06-02)

| Experiment | Directory | Status | Notes |
|------------|-----------|--------|-------|
| `modal_train_e3_gsm8k.py` | `qwen3-1p7b-gsm8k-e3-clean-stage1` | Complete | Steps 2, 100, 200, 300 + rollouts |
| `modal_train_e3_gsm8k.py` | `qwen3-1p7b-gsm8k-e3-clean-stage1_hf` | Converted | HF-compatible checkpoint |
| `modal_train_e3_gsm8k.py` | `qwen3-1p7b-gsm8k-e3-mixed-stage1` | Complete | Steps 100, 200, 300 + rollouts |
| `modal_train_e3_gsm8k.py` | `qwen3-1p7b-gsm8k-e3-mixed-stage1_hf` | Converted | HF-compatible checkpoint |
| `modal_train_e3_gsm8k.py` | `qwen3-1p7b-gsm8k-e3-trivia-stage1` | Complete | Steps 2, 100, 200, 300 + rollouts |
| `modal_train_e3_gsm8k.py` | `qwen3-1p7b-gsm8k-e3-trivia-stage1_hf` | Converted | HF-compatible checkpoint |
| `modal_train_e3_gsm8k.py` | `qwen3-1p7b-gsm8k-e3-clean-stage2` | Complete | Steps 4, 100, 200, 300 + rollouts |
| `modal_train_e3_gsm8k.py` | `qwen3-1p7b-gsm8k-e3-mixed-stage2` | Complete | Steps 4, 100, 200, 300 + rollouts |
| `modal_train_e3_gsm8k.py` | `qwen3-1p7b-gsm8k-e3-trivia-stage2` | In-progress | Steps 100, 200 + rollouts; missing step 300 |
| `modal_train_gsm8k.py` | `qwen3-1p7b-gsm8k-grpo-clean` | Complete | Steps 2, 3, 100, 200, 300, 400 + rollouts |
| `modal_train_gsm8k.py` | `qwen3-1p7b-gsm8k-grpo-clean_hf` | Converted | HF-compatible checkpoint |
| `modal_train_gsm8k.py` | `qwen3-1p7b-gsm8k-grpo-mixed` | Complete | Steps 100, 200, 300, 400 + rollouts |
| `modal_train_gsm8k.py` | `qwen3-1p7b-gsm8k-grpo-mixed_hf` | Converted | HF-compatible checkpoint |
| `modal_train_hendrycks.py` | `qwen3-1p7b-hendrycks-grpo-clean` | Complete | Steps 1, 100, 200, 300, 400 + rollouts |
| `modal_train_hendrycks.py` | `qwen3-1p7b-hendrycks-grpo-clean_hf` | Converted | HF-compatible checkpoint |
| `modal_train_hendrycks.py` | `qwen3-1p7b-hendrycks-grpo-mixed` | Complete | Steps 100, 200, 300, 400 + rollouts |
| `modal_train_hendrycks.py` | `qwen3-1p7b-hendrycks-grpo-mixed_hf` | Converted | HF-compatible checkpoint |

### Rollout Generation Parameters by Experiment

| Experiment | `val_kwargs.n` | `temperature` | `do_sample` | `max_response_length` | `test_freq` | `val_before_train` |
|------------|----------------|---------------|-------------|----------------------|-------------|-------------------|
| `grpo_gsm8k_e3.sh` | 1 | 0.6 | True | 512 (stage 1) / 1024 (stage 2) | 100 | True |
| `grpo_gsm8k_a100.sh` | 4 | 0.6 | True | 1024 | 25 | True |
| `grpo_hendrycks_a100.sh` | 4 | 0.6 | True | 2048 | 25 | True |

- **`val_kwargs.n`**: Number of responses generated per validation problem. The e3 recipe uses `1`; standard GSM8K and Hendrycks use `4`.
- **`temperature`**: Sampling temperature for validation rollouts (`0.6` across all experiments).
- **`do_sample`**: Whether to use random sampling instead of greedy decoding (`True` across all experiments).
- **`max_response_length`**: Maximum tokens the model can output during validation. Hendrycks uses `2048` because MATH problems require longer reasoning chains.
- **`test_freq`**: How often (in training steps) validation and rollout saving occur. E3 uses `100`; standard experiments use `25`.
- **`val_before_train`**: Whether to run validation (and save step-0 rollouts) before the first training step (`True` across all experiments).

---

## E3 GSM8K Curriculum Experiments (`modal_train_e3_gsm8k.py`)

Two-stage GRPO training on GSM8K with the e3 recipe (negative gradients + asymmetric clipping). Tracks: **c (clean)**, **d (mixed)**, **e (trivia)**.

### FSDP Training Checkpoints

Produced by `verl.trainer.main_ppo` at intervals set by `trainer.save_freq` (default: 100 steps).

| Property | Detail |
|----------|--------|
| **Location** | `/data/ckpts/qwen3-1p7b-gsm8k-e3-{flavor}-stage{1,2}/` |
| **Per-step structure** | `global_step_N/` containing: |
| | - `actor/model_world_size_1_rank_0.pt` (~7.6 GiB) — model weights |
| | - `actor/optim_world_size_1_rank_0.pt` (~12.8 GiB) — optimizer state |
| | - `actor/extra_state_world_size_1_rank_0.pt` (~14 KiB) — extra training state |
| | - `data.pt` (~1.5 KiB) — metadata |
| | - `latest_checkpointed_iteration.txt` — pointer to latest saved step |

### HuggingFace-Converted Checkpoints

Produced by `modal_convert_ckpt.py`. Required as the stage-2 starting model (the in-loop `huggingface/` dir only stores config + tokenizer, not weights).

| Property | Detail |
|----------|--------|
| **Location** | `/data/ckpts/qwen3-1p7b-gsm8k-e3-{flavor}-stage{1,2}_hf/` |
| **Key files** | `model.safetensors` (~3.2 GiB), `config.json`, tokenizer files (`tokenizer.json`, `vocab.json`, `merges.txt`, etc.), `generation_config.json` |

### Rollout JSON Files

Saved at `trainer.test_freq` intervals (default: 100 steps). Contains generated sequences/rollouts for validation and analysis.

| Property | Detail |
|----------|--------|
| **Location** | Inside each checkpoint dir: `/data/ckpts/.../{N}_rollouts.json` |
| **Size** | ~2.5–2.7 MiB per file |
| **Examples** | `0_rollouts.json`, `2_rollouts.json`, `100_rollouts.json`, `200_rollouts.json`, `300_rollouts.json` |

### Input Data

The following are uploaded via `scripts/data_upload_e3_gsm8k.py` and consumed as inputs:

| File | Purpose |
|------|---------|
| `/data/e3_gsm8k/train_easy_clean.parquet` | Stage 1 clean split |
| `/data/e3_gsm8k/train_hard_clean.parquet` | Stage 2 clean split |
| `/data/e3_gsm8k/train_easy_mixed.parquet` | Stage 1 mixed split |
| `/data/e3_gsm8k/train_hard_mixed.parquet` | Stage 2 mixed split |
| `/data/e3_gsm8k/train_easy_trivia.parquet` | Stage 1 trivia split |
| `/data/e3_gsm8k/train_hard_trivia.parquet` | Stage 2 trivia split |
| `/data/e3_gsm8k/test.parquet` | Shared validation set |

---

## Standard GRPO on GSM8K (`modal_train_gsm8k.py`)

Standard GRPO training on GSM8K (non-curriculum). Tracks: **a (clean)**, **b (mixed)**.
Base model: `Qwen/Qwen3-1.7B`. Checkpoints under `/data/ckpts/qwen3-1p7b-gsm8k-grpo-{clean,mixed}/`.

*(Detailed artifact breakdown TBD.)*

---

## Hendrycks MATH (`modal_train_hendrycks.py`)

GRPO training on Hendrycks MATH dataset. Tracks: **a (clean)**, **b (mixed)**.
Base model: `Qwen/Qwen3-1.7B`. Checkpoints under `/data/ckpts/qwen3-1p7b-hendrycks-grpo-{clean,mixed}/`.

*(Detailed artifact breakdown TBD.)*

---

## Other Experiments (TBD)

- [ ] `modal_eval_aime.py` / `modal_eval_general.py` — evaluation runs
- [ ] `modal_generation_test.py` — generation tests
- [ ] `modal_convert_ckpt.py` — checkpoint conversion utility

