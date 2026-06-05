# GSM8K Track A Original Run Anomaly Investigation

## Scope

This investigation checks whether the first successful GSM8K Track A run (`qwen3-1p7b-gsm8k-grpo-clean`) may reflect an operational artifact rather than a clean experimental trajectory.

All downloaded files were copied locally under `investigation_artifacts/gsm8k_track_a_anomaly/`. No remote WandB or Modal artifacts were modified.

## Primary artifacts inspected

### WandB runs

- Original Track A:
  - `go4zdpgb`: crashed, created `2026-05-22T18:08:36Z`, summary step `243`
  - `a72jwsb8`: crashed, created `2026-05-24T18:22:03Z`, summary step `324`
  - `fgw6prw4`: finished, created `2026-05-24T22:06:43Z`, summary step `400`
- Later Track A v2:
  - `pc68soex`: finished, created `2026-06-03T17:57:59Z`, summary step `400`
- Track B references:
  - `3thxksh1`: original mixed, finished, summary step `400`
  - `3ygfhphf`: mixed v2, finished, summary step `400`

### Modal volume paths

- Original Track A: `ckpts/qwen3-1p7b-gsm8k-grpo-clean`
- Later Track A v2: `ckpts/qwen3-1p7b-gsm8k-grpo-clean-v2`
- Track B references also downloaded for comparison.

## Key findings

### 1. Original Track A is a stitched three-run trajectory

The final combined history is not from one uninterrupted training process. It consists of:

| Run ID | Retained step range | Notes |
|---|---:|---|
| `go4zdpgb` | 3-199 | first crashed run |
| `a72jwsb8` | 200-299 | first resume, crashed |
| `fgw6prw4` | 300-400 | second resume, finished |

The final run log confirms it loaded from `/data/ckpts/qwen3-1p7b-gsm8k-grpo-clean/global_step_300`.

### 2. Resume overlap steps were replayed and were not deterministic

The first crash logged through step 243, but the next run resumed from checkpoint step 200. Likewise, the second crash logged through step 324, but the final run resumed from checkpoint step 300.

The overlapping replay regions differ materially:

| Overlap | Metric | Mean abs diff | Max abs diff |
|---|---|---:|---:|
| `go4zdpgb` vs `a72jwsb8`, steps 200-243 | `response_length/mean` | 44.17 | 146.77 |
| `go4zdpgb` vs `a72jwsb8`, steps 200-243 | `response_length/clip_ratio` | 0.111 | 0.391 |
| `a72jwsb8` vs `fgw6prw4`, steps 300-324 | `response_length/mean` | 26.94 | 78.03 |
| `a72jwsb8` vs `fgw6prw4`, steps 300-324 | `response_length/clip_ratio` | 0.061 | 0.188 |

This is strong evidence that the original Track A curve contains operational replay effects. The retained combined history keeps the later replayed steps, which is reasonable for reconstructing the final checkpoint path, but it is not equivalent to a single deterministic uninterrupted run.

### 3. No obvious training-data or major hyperparameter mismatch found

Across the three original Track A segments:

- `data.train_files = /data/gsm8k_padded/train_clean.parquet`
- `data.val_files = /data/gsm8k_padded/test.parquet`
- `data.max_response_length = 1024`
- `actor_rollout_ref.rollout.n = 8`
- `actor_rollout_ref.rollout.val_kwargs.n = 4`
- `actor_rollout_ref.rollout.temperature = 0.6`
- `trainer.resume_mode = auto`
- `trainer.save_freq = 100`
- `trainer.test_freq = 100`
- `trainer.val_before_train = True`

The original Track A segments had no meaningful config differences among themselves.

### 4. Original Track A and clean-v2 differ in one WandB config field found so far

The only meaningful config difference detected between original Track A and `qwen3-1p7b-gsm8k-grpo-clean-v2` was:

| Field | Original Track A | Track A v2 |
|---|---:|---:|
| `actor_rollout_ref.rollout.gpu_memory_utilization` | 0.6 | 0.7 |

This could affect vLLM batching/performance and possibly operational stability, but by itself it does not prove a semantic training bug.

### 5. Finished-run logs do not show obvious OOM or Python exceptions

Downloaded `output.log` files for finished runs show:

- no `Traceback`
- no `Error`
- no `Exception`
- no `out of memory`
- no `OOM`
- no `Killed`

However, WandB did not expose `output.log` files for the two crashed original Track A runs, so the direct crash cause is still unknown.

### 6. Modal checkpoint markers look normal at step 400

Both original Track A and Track A v2 have:

- `latest_checkpointed_iteration.txt = 400`
- `global_step_400/data.pt`
- `global_step_400/actor`

Original Track A additionally has early `global_step_2`, `global_step_3`, `2_rollouts.json`, and `3_rollouts.json`; v2 starts with the expected `0_rollouts.json`, then `100/200/300/400`.

### 6.1 Checkpoint `data.pt` metadata looks normal

Downloaded and inspected `global_step_200/data.pt`, `global_step_300/data.pt`, and `global_step_400/data.pt` for original Track A and Track A v2.

The files are small PyTorch ZIP saves containing dataloader snapshot metadata. They were parsed locally from `data/data.pkl`.

Original Track A and Track A v2 match exactly on the relevant checkpoint counters at each step:

| Step | `_snapshot_step` | `samples_yielded` | `_sampler_iter_yielded` | `_last_yielded_worker_id` | `_num_workers` |
|---:|---:|---:|---:|---:|---:|
| 200 | 84 | 5376 | 84 | 3 | 8 |
| 300 | 68 | 4352 | 68 | 3 | 8 |
| 400 | 52 | 3328 | 52 | 3 | 8 |

The only original-vs-v2 difference at the same checkpoint step is `_base_seed`, which is expected for independent runs.

The decreasing `samples_yielded` values across `200 -> 300 -> 400` are consistent with epoch cycling rather than corruption: with train batch size 64 and ~116-117 batches per epoch, steps 200/300/400 land at different positions within the current epoch.

### 7. Rollout artifacts confirm the response-length anomaly is present in primary Modal outputs

The original Track A rollout JSONs are longer than v2 at the same validation checkpoints, not merely an artifact of the derived CSVs.

| Experiment | Step | Mean output chars | Score mean | Hash rate |
|---|---:|---:|---:|---:|
| original clean | 100 | 2525.3 | 0.8391 | 0.9177 |
| clean v2 | 100 | 2188.8 | 0.8548 | 0.9238 |
| original clean | 200 | 2801.4 | 0.8635 | 0.9395 |
| clean v2 | 200 | 2039.3 | 0.8679 | 0.9587 |
| original clean | 300 | 2669.8 | 0.8711 | 0.9483 |
| clean v2 | 300 | 2022.7 | 0.8753 | 0.9642 |
| original clean | 400 | 2331.1 | 0.8751 | 0.9524 |
| clean v2 | 400 | 2097.2 | 0.8815 | 0.9716 |

## Current interpretation

The original Track A behavior appears real in the sense that it is present in primary WandB histories and primary Modal rollout JSONs. It is not just a notebook/CSV derivation error.

However, it is not clean evidence of a stable Track A-vs-Track B effect, because original Track A was interrupted twice and replayed from checkpoints. The replayed overlaps are non-identical, and the later uninterrupted Track A v2 behaves differently. Therefore, the original Track A should be treated as operationally confounded until the crash causes and resume mechanics are better understood.

## Local outputs generated

- `analysis_tools/investigate_gsm8k_track_a_anomaly.py`
- `investigation_artifacts/gsm8k_track_a_anomaly/wandb_runs/`
- `investigation_artifacts/gsm8k_track_a_anomaly/wandb_files/`
- `investigation_artifacts/gsm8k_track_a_anomaly/modal_volume/`
- `investigation_artifacts/gsm8k_track_a_anomaly/modal_metadata/`
- `investigation_artifacts/gsm8k_track_a_anomaly/reports/anomaly_report.json`
- `investigation_artifacts/gsm8k_track_a_anomaly/reports/history_summary.csv`
- `investigation_artifacts/gsm8k_track_a_anomaly/reports/rollout_summary.csv`
- `investigation_artifacts/gsm8k_track_a_anomaly/reports/original_resume_boundary_rows.csv`
- `investigation_artifacts/gsm8k_track_a_anomaly/reports/checkpoint_data_pt_flat.csv`
- `investigation_artifacts/gsm8k_track_a_anomaly/reports/response_length_mean.png`
- `investigation_artifacts/gsm8k_track_a_anomaly/reports/clip_ratio.png`
- `investigation_artifacts/gsm8k_track_a_anomaly/reports/time_per_step.png`
- `investigation_artifacts/gsm8k_track_a_anomaly/reports/val_accuracy.png`

## Remaining gaps

- Direct Modal app logs for the two crashed runs were not available by app name in the current Modal environment.
- WandB did not expose `output.log` for the two crashed runs.
- Checkpoint `data.pt` metadata did not reveal abnormal counters, so it does not explain the response-length anomaly.
