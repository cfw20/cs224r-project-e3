#!/usr/bin/env python3
"""
Prepares the canonical MATH-500 validation set as a verl parquet.

Usage:
    python3 examples/data_preprocess/math500_prep.py \
        --local_dir /data/hendrycks_math \
        --output_name test.parquet
"""

import argparse
import os

import datasets

from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


DATA_SOURCE = "HuggingFaceH4/MATH-500"
INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."


def extract_solution(solution_str: str) -> str:
    boxed = last_boxed_only_string(solution_str)
    assert boxed is not None, f"MATH-500 answer extraction failed: {solution_str!r}"
    return remove_boxed(boxed)


def build_row(problem: str, solution: str, idx: int):
    gt = extract_solution(solution)
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": f"{problem} {INSTRUCTION}"}],
        "ability": "math",
        "level": "unknown",
        "reward_model": {"style": "rule", "ground_truth": gt},
        "extra_info": {
            "split": "test",
            "index": idx,
            "answer": solution,
            "question": problem,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data_hendrycks_math")
    parser.add_argument("--output_name", default="test.parquet")
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)

    ds = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")
    print(f"[math500_prep] Loaded {len(ds)} problems")

    rows = [build_row(ex["problem"], ex["solution"], idx) for idx, ex in enumerate(ds)]

    out_path = os.path.join(args.local_dir, args.output_name)
    datasets.Dataset.from_list(rows).to_parquet(out_path)
    print(f"[math500_prep] Wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
