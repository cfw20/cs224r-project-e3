#!/usr/bin/env python3
"""
Modal evaluation: run verl main_generation on AIME 2025 + score with the
exact same `curriculum_math/compute_score.py` used by `scripts/eval.sh`.

Usage:
    modal run modal_eval_aime.py --model qwen
    modal run modal_eval_aime.py --model e3 --n-samples 16 --num-problems 30
    modal run modal_eval_aime.py --model qwen --num-problems 2 --n-samples 2 \\
        --max-response-length 2048 --output-tag smoke

Methodology mirrors `scripts/eval.sh`:
  - dataset:     CMU-AIRe/hmmt-aime-2025
  - instruction: "Let's think step by step and output the final answer within \\boxed{}."
  - scoring:    verl.utils.reward_score.curriculum_math.compute_score.compute_score
  - rollout:    temp=0.6, top_k=20, top_p=0.95, do_sample=True
  - lengths:    prompt=1024, response=32768
The only differences from eval.sh are those required to run on a single
Modal GPU (main_generation instead of main_ppo; in-process scoring instead
of the PPO reward manager).
"""

import modal

app = modal.App("e3-eval-aime")

image = (
    modal.Image.from_dockerfile("docker/Dockerfile.ngc.vllm0.8.noverl")
    .add_local_dir(".", "/root/e3", copy=True)
    .run_commands("pip install -e /root/e3")
)

vol = modal.Volume.from_name("e3-generation-vol", create_if_missing=True)

REPO_PATH = "/root/e3"
DATA_DIR = "/data/aime_eval"

MODEL_IDS = {
    "qwen": "Qwen/Qwen3-1.7B",
    "e3": "CMU-AIRe/e3-1.7B",
}

INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."


def _prepare_dataset(num_problems, competition, tag):
    """Download CMU-AIRe/hmmt-aime-2025, filter, format as RLHF parquet."""
    import os
    import pandas as pd
    from datasets import load_dataset

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, f"aime2025_{tag}.parquet")

    ds = load_dataset("CMU-AIRe/hmmt-aime-2025")
    # pick the only / first split available
    split_name = list(ds.keys())[0]
    print(f"[data] Using split '{split_name}' with {len(ds[split_name])} rows")
    ds = ds[split_name]

    df = ds.to_pandas()
    print(f"[data] Columns: {list(df.columns)}")

    # Try to locate the competition field and filter
    comp_col = None
    for c in ["source", "competition", "dataset", "origin"]:
        if c in df.columns:
            comp_col = c
            break
    if comp_col is not None and competition != "all":
        mask = df[comp_col].astype(str).str.lower().str.contains(competition.lower())
        print(
            f"[data] Filtering by {comp_col} contains '{competition}': "
            f"{int(mask.sum())}/{len(df)} rows kept"
        )
        df = df[mask].reset_index(drop=True)

    # Locate problem + answer columns
    prob_col = next(c for c in ["problem", "question", "prompt"] if c in df.columns)
    ans_col = next(c for c in ["answer", "solution", "ground_truth"] if c in df.columns)
    print(f"[data] problem_col={prob_col}, answer_col={ans_col}")

    if num_problems is not None:
        df = df.head(num_problems).reset_index(drop=True)
        print(f"[data] Truncated to {len(df)} problems")

    rows = []
    for idx, row in df.iterrows():
        question = f"{row[prob_col]} {INSTRUCTION}"
        rows.append(
            {
                "data_source": "math",
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": str(row[ans_col])},
                "extra_info": {"split": "test", "index": int(idx)},
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(out_path, index=False)
    print(f"[data] Wrote {len(out_df)} prompts -> {out_path}")
    return out_path, len(out_df)


def _run_generation(
    data_path,
    output_path,
    model_id,
    n_samples,
    batch_size,
    max_prompt_length,
    max_response_length,
):
    """Invoke verl.trainer.main_generation as a subprocess."""
    import subprocess

    cmd = [
        "python3", "-m", "verl.trainer.main_generation",
        "trainer.nnodes=1",
        "trainer.n_gpus_per_node=1",
        f"data.path={data_path}",
        "data.prompt_key=prompt",
        f"data.n_samples={n_samples}",
        f"data.batch_size={batch_size}",
        f"data.output_path={output_path}",
        f"model.path={model_id}",
        "+model.trust_remote_code=True",
        # mirror eval.sh val_kwargs
        "rollout.temperature=0.6",
        "rollout.top_k=20",
        "rollout.top_p=0.95",
        "rollout.do_sample=True",
        f"rollout.prompt_length={max_prompt_length}",
        f"rollout.response_length={max_response_length}",
        "rollout.tensor_model_parallel_size=1",
        "rollout.gpu_memory_utilization=0.9",
        "rollout.enforce_eager=False",
        "rollout.free_cache_engine=False",
        "rollout.max_num_batched_tokens=50000",
        # required workarounds (keys missing from generation.yaml)
        "+rollout.extrapolation_val=False",
        "+rollout.extrapolation_length=0",
    ]
    print("[gen] Running:")
    print("    " + " \\\n    ".join(cmd))
    subprocess.run(cmd, check=True)


def _pass_at_k(n, c, k):
    """Unbiased pass@k estimator from Chen et al. (HumanEval)."""
    if n - c < k:
        return 1.0
    # 1 - C(n-c, k) / C(n, k), computed stably
    import numpy as np
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def _score_outputs(output_path, n_samples, model_tag, tag):
    """Score generated responses using the exact eval.sh compute_score."""
    import os
    import json
    import pandas as pd
    import numpy as np

    # Import the same scoring function eval.sh uses
    from verl.utils.reward_score.curriculum_math.compute_score import compute_score

    df = pd.read_parquet(output_path)
    num_problems = len(df)
    print(f"[score] Loaded {num_problems} problems from {output_path}")

    correctness = np.zeros((num_problems, n_samples), dtype=np.int32)
    extract_failures = 0
    per_problem_rows = []

    for i, row in df.iterrows():
        gt = row["reward_model"]["ground_truth"]
        responses = row["responses"]
        # responses is a list-like of length n_samples
        sample_scores = []
        for j, resp in enumerate(responses):
            try:
                score = compute_score(
                    data_source="math",
                    solution_str=resp,
                    ground_truth=gt,
                    extra_info=None,
                )
            except Exception as e:
                print(f"[score] compute_score raised on problem {i} sample {j}: {e}")
                score = 0.0
            if score == 0.0 and "\\boxed" not in resp:
                extract_failures += 1
            correctness[i, j] = int(score == 1.0)
            sample_scores.append(int(score == 1.0))

        per_problem_rows.append(
            {
                "problem_idx": int(i),
                "ground_truth": str(gt),
                "n_samples": n_samples,
                "n_correct": int(correctness[i].sum()),
                "accuracy": float(correctness[i].mean()),
            }
        )

    # Metrics
    metrics = {
        "model": model_tag,
        "tag": tag,
        "num_problems": int(num_problems),
        "n_samples": int(n_samples),
        "extract_failures": int(extract_failures),
        "pass@1_mean": float(correctness.mean()),
    }
    for k in (1, 4, 8, 16):
        if k <= n_samples:
            vals = [_pass_at_k(n_samples, int(correctness[i].sum()), k) for i in range(num_problems)]
            metrics[f"pass@{k}"] = float(np.mean(vals))

    # Save
    metrics_path = os.path.join(DATA_DIR, f"metrics_{model_tag}_{tag}.json")
    per_problem_path = os.path.join(DATA_DIR, f"per_problem_{model_tag}_{tag}.csv")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(per_problem_rows).to_csv(per_problem_path, index=False)

    print("\n=== Eval Summary ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\n[score] metrics  -> {metrics_path}")
    print(f"[score] per-prob -> {per_problem_path}")

    return metrics


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/data": vol},
    timeout=4 * 3600,
)
def run_eval(
    model: str,
    n_samples: int,
    num_problems,
    max_response_length: int,
    max_prompt_length: int,
    competition: str,
    output_tag: str,
):
    import os
    import subprocess

    os.environ["HF_HOME"] = "/data/hf_cache"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if model not in MODEL_IDS:
        raise ValueError(f"Unknown --model={model!r}; choose from {list(MODEL_IDS)}")
    model_id = MODEL_IDS[model]
    tag = output_tag or f"n{n_samples}_l{max_response_length}"

    # 1. Install local verl source (same pattern as modal_generation_test.py)
    # print(f"[setup] pip install -e {REPO_PATH}")
    # subprocess.run(["pip", "install", "-e", REPO_PATH], check=True)

    # 2. Prepare AIME parquet
    data_path, n_problems = _prepare_dataset(
        num_problems=num_problems, competition=competition, tag=tag,
    )

    # 3. Run generation
    output_path = os.path.join(DATA_DIR, f"aime2025_{model}_{tag}_outputs.parquet")
    # batch_size: keep modest so one batch fits in VRAM at 32k tokens
    batch_size = min(n_problems, 8)
    _run_generation(
        data_path=data_path,
        output_path=output_path,
        model_id=model_id,
        n_samples=n_samples,
        batch_size=batch_size,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
    )

    # 4. Score
    metrics = _score_outputs(
        output_path=output_path,
        n_samples=n_samples,
        model_tag=model,
        tag=tag,
    )

    vol.commit()
    return metrics


@app.local_entrypoint()
def main(
    model: str = "qwen",
    n_samples: int = 16,
    num_problems: int = -1,
    max_response_length: int = 32768,
    max_prompt_length: int = 1024,
    competition: str = "aime",
    output_tag: str = "",
):
    """Local entrypoint. `num_problems=-1` means use all problems."""
    np = None if num_problems is None or num_problems < 0 else int(num_problems)
    metrics = run_eval.remote(
        model=model,
        n_samples=n_samples,
        num_problems=np,
        max_response_length=max_response_length,
        max_prompt_length=max_prompt_length,
        competition=competition,
        output_tag=output_tag,
    )
    print("\nFinal metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
