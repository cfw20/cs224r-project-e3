# Recipe: Create a Track A vs Track B Pass@k Analysis Pipeline

This recipe describes how to replicate the end-to-end pipeline used for the Qwen3-1.7B v2 (step 400) Track A vs Track B analysis. Future AI agents can follow this to set up the same workflow for **any other Track A / Track B pair** in the project.

---

## 1. Understand the Source Notebook

The canonical analysis notebook is `notebooks/0p6_analysis_complete_run.ipynb`. It contains:

- **Sections 1–7:** WandB history fetch, validation accuracy curves, timing, response length, training dynamics, memory, summary tables.
- **Section 8:** Per-question rollout analysis using `{step}_rollouts.json` files from Modal checkpoints.
- **Section 9:** Pass@k analysis using dedicated 32-sample eval CSVs (`per_problem_*.csv`) produced by `modal_eval_general.py`.

**Strategy:** Copy `0p6_analysis_complete_run.ipynb` and adapt the source cells. Do not try to build a new notebook from scratch.

---

## 2. Discover the New Runs

Before writing any code, confirm the exact identifiers for the new runs.

### 2a. WandB Run IDs and Experiment Names

Use the existing tool `analysis_tools/investigate_gsm8k_track_a_anomaly.py` (or WandB web UI) to find:

- `RUN_ID_CLEAN` — the WandB run ID for the clean Track A run.
- `RUN_ID_MIXED` — the WandB run ID for the mixed Track B run.
- Corresponding `experiment_name` values from WandB config.

Example (1.7B v2):
```
RUN_ID_CLEAN  = "pc68soex"   # exp: qwen3-1p7b-gsm8k-grpo-clean-v2
RUN_ID_MIXED  = "3ygfhphf"   # exp: qwen3-1p7b-gsm8k-grpo-mixed-v2
```

### 2b. Modal Volume Artifacts

Check what exists on the Modal volume `e3-generation-vol`:

```bash
conda activate cs224r-hw2  # env with modal CLI
modal volume ls e3-generation-vol ckpts/<experiment_name>
```

Look for:
- `global_step_N/` directories (to know which checkpoints exist).
- `{step}_rollouts.json` files (to know which rollout steps are available for Section 8).
- Whether HF conversions already exist (`<experiment_name>_hf_step<N>`).

**Critical:** If rollouts were saved at `save_freq=100`, the available steps may be `[0, 100, 200, 300, 400]`, not every 25. Section 8 must use the **actual** available steps.

---

## 3. Create the Checkpoint Conversion Script

If HF model directories (`<exp>_hf_step<N>`) do not yet exist, create a one-off conversion script based on `scripts/convert_pass32_ckpts.sh` or `scripts/convert_pass32_ckpts_1p7_v2.sh`.

**Template:**
```bash
#!/bin/bash
set -euo pipefail
BASE_MODEL="<HF_BASE_MODEL>"   # e.g., Qwen/Qwen3-1.7B
for exp in <EXP_A> <EXP_B>; do
  modal run modal_convert_ckpt.py \
    --exp-name "$exp" \
    --step <TARGET_STEP> \
    --base-model "$BASE_MODEL"
done
```

**Key rules:**
- Use `--step <N>` to convert a **specific** checkpoint step (not `--step -1` which picks the latest).
- Output lands under `/data/ckpts/<exp>_hf_step<N>` on the Modal volume.

---

## 4. Create the Eval Sweep Script

Create a sweep script based on `scripts/run_pass32_sweep.sh` or `scripts/run_pass32_sweep_1p7_v2.sh`.

**Template:**
```bash
#!/bin/bash
set -euo pipefail
DATASET="gsm8k"
N_SAMPLES=32
MAX_RESP=<MATCHES_TRAINING>   # e.g., 1024 if data.max_response_length=1024
OUTPUT_DIR="/data/pass32_<pair_name>"

modal run modal_eval_general.py \
  --dataset "$DATASET" \
  --model-path "/data/ckpts/<EXP_A>_hf_step<N>" \
  --model <UNIQUE_TAG_A> \
  --n-samples "$N_SAMPLES" \
  --max-response-length "$MAX_RESP" \
  --output-dir "$OUTPUT_DIR" \
  --output-tag step<N>

modal run modal_eval_general.py \
  --dataset "$DATASET" \
  --model-path "/data/ckpts/<EXP_B>_hf_step<N>" \
  --model <UNIQUE_TAG_B> \
  ...
```

**Critical rules:**
- Always pass `--model-path` explicitly. Do **not** rely on `MODEL_IDS` inside `modal_eval_general.py`, because those are hardcoded to older runs and may return the wrong checkpoints.
- Use a **unique `--model` tag** (e.g., `track_a_v2`, `track_b_e3`) so output filenames (`per_problem_gsm8k_<model>_step<N>.csv`) do not collide with existing evals on the volume.
- `max-response-length` must match the **training-time** `data.max_response_length`.
- Skip the base model eval if you only care about the trained-checkpoint comparison (saves time/credits).

---

## 5. Copy and Adapt the Analysis Notebook

### 5a. Copy

```bash
cp notebooks/0p6_analysis_complete_run.ipynb notebooks/<NEW_NAME>.ipynb
```

### 5b. Adapt Source Cells (Python JSON Edit)

Because `.ipynb` files cannot be edited with text-replace tools in this IDE, load the notebook JSON in Python and apply string replacements to each cell's source list.

**Group 1 — Identity / WandB:**
- Title markdown: model size, step count, "Analysis"
- `RUN_ID_CLEAN`, `RUN_ID_MIXED`
- Experiment names in `print(...)` and `TRACK_CONFIGS`

**Group 2 — Rollouts (Section 8):**
- `ROLLOUT_DIR`: `"rollouts_0p6b"` → `"rollouts_<pair_name>"`
- `TRACK_CONFIGS` exp names
- `STEPS`: must match **actual** available rollout steps on the volume (NOT the 0.6B `[0, 25, 50, ...]`).
- All hardcoded step selectors in qualitative cells: `== 150`, `[0, 150]` → target final step (e.g., `== 400`, `[0, 400]`).
- Markdown text mentioning step ranges (e.g., "steps 0, 25, 50, 75, 100, 125, and 150").

**Group 3 — Pass@k (Section 9):**
- `PASS_DIR`: `"pass32_0p6b"` → `"pass32_<pair_name>"`
- `PASS_STEPS`: match the eval steps you ran (e.g., `[400]` if only one checkpoint was evaluated). Remove base-model `step == 0` branch if base model was skipped.
- `remote_csv_name()`: remove the base-model special case if not evaluating base.
- `for track in ("track_a", "track_b"):` → use the **same unique tags** you passed to `--model` in the sweep script (e.g., `track_a_v2`, `track_b_v2`).
- `for track, label in ...`: same tag substitution.
- Section 9 markdown header: update script name reference, base-model note.

**Group 4 — Output Artifacts:**
- All `plt.savefig("0p6b_...")` → new prefix
- All `.to_csv("0p6b_...")` → new prefix
- Final summary print statements listing generated artifacts

### 5c. Validate After Editing

Run a Python sanity check:
```python
import json, ast
nb = json.load(open("notebooks/<NEW_NAME>.ipynb"))
# Check JSON validity
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        code = "".join(cell["source"])
        # Strip IPython magics before parse
        ast.parse("\n".join(l for l in code.split("\n") if not l.lstrip().startswith(("!", "%"))))
# Confirm no stale references remain
stale = ["0p6b_", "kvbep5gl", "zh9cbjbh", "qwen3-0p6b", "== 150"]
```

---

## 6. Execution Order

1. **Convert checkpoints** (one-off, ~4 min each):
   ```bash
   bash scripts/convert_pass32_ckpts_<pair>.sh
   ```
2. **Run eval sweep** (parallel Modal instances, ~20–30 min each):
   ```bash
   bash scripts/run_pass32_sweep_<pair>.sh
   ```
3. **Verify artifacts on volume:**
   ```bash
   modal volume ls e3-generation-vol pass32_<pair_name>/
   ```
   You should see:
   - `per_problem_gsm8k_<tag_a>_step<N>.csv`
   - `per_problem_gsm8k_<tag_b>_step<N>.csv`
   - `metrics_gsm8k_<tag>_step<N>.json`
4. **Open notebook and run cells** top-to-bottom.

---

## 7. Common Gotchas

| Issue | Mitigation |
|---|---|
| `MODEL_IDS` lookup in `modal_eval_general.py` silently picks the wrong model | Always pass `--model-path` explicitly. |
| Output filename collision with existing evals | Use a unique `--model` tag (suffix with `_v2`, `_e3`, etc.). |
| Rollout `STEPS` mismatch | Query the volume first (`modal volume ls ...`) and set `STEPS` to actual saved steps. |
| Base model included in `PASS_STEPS` but not evaluated | Remove `step == 0` branch from `remote_csv_name()` and set `PASS_STEPS` to only the evaluated steps. |
| Qualitative cells hardcode `step 150` | Search for `== 150` and `[0, 150]` in source and replace with the actual final step. |
| `.ipynb` cannot be edited with standard text tools | Use Python `json.load` / `json.dump` to manipulate cell sources. |
| Conda env mismatch | `modal` CLI lives in `cs224r-hw2` (no torch); `verl` imports need the training env. Keep them separate. |

---

## 8. Minimal Checklist for New Agents

Before saying "done", verify:

- [ ] WandB run IDs and experiment names are correct.
- [ ] `global_step_<N>` and `{step}_rollouts.json` exist on the Modal volume.
- [ ] HF conversion script uses `--step <N>` (not `-1`).
- [ ] Eval sweep uses explicit `--model-path` and unique `--model` tags.
- [ ] Notebook `STEPS` matches actual rollout steps on the volume.
- [ ] Notebook `PASS_STEPS` matches actual eval steps.
- [ ] All hardcoded step selectors (`== 150`, `[0, 150]`) updated to target step.
- [ ] All output filenames use a unique prefix.
- [ ] No stale `0p6b_`, old run IDs, or old exp names remain in **source** cells.
- [ ] Notebook JSON is valid and all code cells parse without syntax errors.
