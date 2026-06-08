#!/usr/bin/env python3
"""
Append "Section 11: Pass@k Sampling-Budget Comparison" to
notebooks/rlad_analysis_hendrycks.ipynb.

This mirrors Section 9 of notebooks/0p6_analysis_complete_run.ipynb but is
scoped to the step-400 Hendrycks pass@32 evals (Track A vs Track B), since no
step-0 32-sample baseline exists.

Usage:
    conda run -n cs224r-hw2 python scripts/add_passk_section_hendrycks.py
"""

import os
import json
import uuid

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_PATH = os.path.join(REPO_ROOT, "notebooks", "rlad_analysis_hendrycks.ipynb")


MD_HEADER = """\
## 11. Pass@k Sampling-Budget Comparison (up to pass@32)

The Section 8 rollouts were saved during training with only **4 samples per question** (`val_kwargs.n=4`), so they can show at most pass@4, and the Section 9 post-training eval used a single sample (pass@1). To study how each track scales with a larger sampling budget, we ran a dedicated evaluation that generates **32 samples per question** on the full MATH-500 test set for the final (step 400) checkpoint of both tracks (see `scripts/run_pass32_sweep_1p7_hendrycks.sh`).

Each eval writes a `per_problem_*.csv` (with `n_correct` out of `n_samples=32` per question). From those counts we compute the **unbiased pass@k estimator** (Chen et al., HumanEval) for `k in {1, 2, 4, 8, 16, 32}` and compare Track A (clean) vs Track B (mixed).

Only the **step 400** checkpoint was evaluated with 32 samples, so this is a single-checkpoint comparison."""


CODE_DOWNLOAD = '''\
# Download the 32-sample per-problem eval CSVs from the Modal volume.
# These are produced by scripts/run_pass32_sweep_1p7_hendrycks.sh and live under
# /data/pass32_1p7b_hendrycks/ on volume 'e3-generation-vol'.
import os
import subprocess

PASS_DIR = "pass32_1p7b_hendrycks"   # local cache
PASS_STEPS = [400]
KS = [1, 2, 4, 8, 16, 32]
os.makedirs(PASS_DIR, exist_ok=True)


def remote_csv_name(track, step):
    return f"per_problem_math_{track}_hendrycks_step{step}.csv"


missing = []
for track in ("track_a", "track_b"):
    for step in PASS_STEPS:
        fname = remote_csv_name(track, step)
        local_path = os.path.join(PASS_DIR, fname)
        if not os.path.exists(local_path):
            missing.append((fname, local_path))

if missing:
    print(f"Downloading {len(missing)} eval CSVs from Modal volume 'e3-generation-vol'...")
    for fname, local_path in missing:
        remote_path = f"pass32_1p7b_hendrycks/{fname}"
        cmd = ["modal", "volume", "get", "e3-generation-vol", remote_path]
        print(f"  {remote_path} -> {local_path}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PASS_DIR)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr.strip()}")
            raise RuntimeError(f"Failed to download {remote_path}. Has the sweep finished?")
    print("Download complete.")
else:
    print("All pass@k eval CSVs already present locally.")

# Verify presence
for track in ("track_a", "track_b"):
    for step in PASS_STEPS:
        p = os.path.join(PASS_DIR, remote_csv_name(track, step))
        assert os.path.exists(p), f"Missing: {p}"
print("All pass@k eval CSVs ready.")'''


CODE_COMPUTE = '''\
# Compute the unbiased pass@k for k in {1,2,4,8,16,32} from per-problem n_correct.
import numpy as np
import pandas as pd


def pass_at_k(n, c, k):
    """Unbiased pass@k estimator (Chen et al., HumanEval).

    n = samples per question, c = number correct, k = budget.
    Probability that at least one of k drawn samples is correct.
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


rows = []
for track_label, track_key in (("A", "track_a"), ("B", "track_b")):
    for step in PASS_STEPS:
        df = pd.read_csv(os.path.join(PASS_DIR, remote_csv_name(track_key, step)))
        n_samples = int(df["n_samples"].iloc[0])
        num_problems = len(df)
        for k in KS:
            vals = [pass_at_k(int(r.n_samples), int(r.n_correct), k) for r in df.itertuples()]
            rows.append({
                "track": track_label,
                "step": step,
                "k": k,
                "pass_at_k": float(np.mean(vals)),
                "n_samples": n_samples,
                "num_problems": num_problems,
            })

df_passk = pd.DataFrame(rows)

# Sanity check: pass@k must be non-decreasing in k for each (track, step).
for (t, s), g in df_passk.groupby(["track", "step"]):
    g = g.sort_values("k")
    assert (g["pass_at_k"].diff().dropna() >= -1e-9).all(), f"pass@k not monotonic for track {t} step {s}"

print(f"Computed pass@k for {df_passk[['track','step']].drop_duplicates().shape[0]} (track, step) pairs.")
print(f"Samples per question: {sorted(df_passk['n_samples'].unique())}, "
      f"problems: {sorted(df_passk['num_problems'].unique())}")
df_passk'''


CODE_PLOT = '''\
# Pass@k curves at step 400: Track A vs Track B.
import matplotlib.pyplot as plt

colors = {"A": "#d62728", "B": "#2ca02c"}
step = 400

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: line plot of pass@k vs k
ax = axes[0]
for label in ("A", "B"):
    d = df_passk[(df_passk["track"] == label) & (df_passk["step"] == step)].sort_values("k")
    ax.plot(d["k"], d["pass_at_k"] * 100, "o-", color=colors[label],
            linewidth=2, markersize=7, label=f"Track {label}")
ax.set_xscale("log", base=2)
ax.set_xticks(KS)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xlabel("k (samples)")
ax.set_ylabel("pass@k (%)")
ax.set_title(f"Pass@k Curve (MATH-500, step {step})")
ax.grid(True, alpha=0.3)
ax.legend()

# Right: grouped bar chart of pass@k by k
ax = axes[1]
x = np.arange(len(KS))
width = 0.38
for i, label in enumerate(("A", "B")):
    d = df_passk[(df_passk["track"] == label) & (df_passk["step"] == step)].sort_values("k")
    ax.bar(x + (i - 0.5) * width, d["pass_at_k"].values * 100, width,
           color=colors[label], alpha=0.85, label=f"Track {label}")
ax.set_xticks(x)
ax.set_xticklabels(KS)
ax.set_xlabel("k (samples)")
ax.set_ylabel("pass@k (%)")
ax.set_title(f"Pass@k by Budget (MATH-500, step {step})")
ax.grid(True, alpha=0.3, axis="y")
ax.legend()

plt.suptitle("Pass@k Sampling-Budget Comparison: Track A vs Track B (Hendrycks MATH)",
             fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("1p7b_hendrycks_pass_at_k.png", dpi=150, bbox_inches="tight")
plt.show()'''


CODE_TABLE = '''\
# Summary table: pass@k columns x track rows, plus A-vs-B gap at pass@32.
pivot = (
    df_passk
    .pivot_table(index=["step", "track"], columns="k", values="pass_at_k")
    .mul(100)
    .round(2)
)
pivot.columns = [f"pass@{k}" for k in pivot.columns]

pivot.to_csv("1p7b_hendrycks_pass_at_k.csv")
print("Saved pass@k table to 1p7b_hendrycks_pass_at_k.csv\\n")
print(pivot)

# Gap at pass@32 (Track B - Track A) in percentage points.
gap = (
    df_passk[df_passk["k"] == 32]
    .pivot_table(index="step", columns="track", values="pass_at_k")
    .mul(100)
)
gap["gap_B_minus_A"] = (gap["B"] - gap["A"]).round(2)
print("\\npass@32 gap (Track B - Track A), percentage points:")
print(gap["gap_B_minus_A"])'''


def _split_source(text):
    """Split text into a list of lines, each ending in \\n except the last."""
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def _md_cell(text):
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": _split_source(text),
    }


def _code_cell(text):
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _split_source(text),
    }


def main():
    with open(NB_PATH) as f:
        nb = json.load(f)

    new_cells = [
        _md_cell(MD_HEADER),
        _code_cell(CODE_DOWNLOAD),
        _code_cell(CODE_COMPUTE),
        _code_cell(CODE_PLOT),
        _code_cell(CODE_TABLE),
    ]

    nb["cells"].extend(new_cells)

    with open(NB_PATH, "w") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")

    print(f"Appended {len(new_cells)} cells (Section 11) to {NB_PATH}")
    print(f"Notebook now has {len(nb['cells'])} cells.")


if __name__ == "__main__":
    main()
