#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

PROJECT = "cfw20-stanford-university/rlad-noise-control"
ORIGINAL_TRACK_A_RUN_IDS = ["go4zdpgb", "a72jwsb8", "fgw6prw4"]
DEFAULT_EXP_NAMES = [
    "qwen3-1p7b-gsm8k-grpo-clean",
    "qwen3-1p7b-gsm8k-grpo-clean-v2",
    "qwen3-1p7b-gsm8k-grpo-mixed",
]
DEFAULT_STEPS = [0, 100, 200, 300, 400]


def outdir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_json(obj):
    return json.loads(json.dumps(obj, default=str))


def get_wandb_api():
    import wandb

    return wandb.Api()


def run_identity(run):
    return {
        "id": run.id,
        "name": run.name,
        "display_name": getattr(run, "display_name", None),
        "state": run.state,
        "created_at": str(getattr(run, "created_at", "")),
        "updated_at": str(getattr(run, "updated_at", "")),
        "url": run.url,
        "summary_step": run.summary.get("_step", None),
        "summary_runtime": run.summary.get("_runtime", None),
        "summary_timestamp": run.summary.get("_timestamp", None),
        "tags": list(run.tags),
    }


def list_wandb(args):
    api = get_wandb_api()
    rows = []
    for run in api.runs(args.project):
        if args.name_contains and args.name_contains not in run.name:
            continue
        item = run_identity(run)
        rows.append(item)
    rows.sort(key=lambda r: (r.get("name") or "", r.get("created_at") or ""))
    print(json.dumps(rows, indent=2))


def download_wandb(args):
    api = get_wandb_api()
    base = outdir(args.output_dir)
    runs_dir = outdir(base / "wandb_runs")

    selected = {}
    for rid in args.run_ids:
        try:
            run = api.run(f"{args.project}/{rid}")
            selected[run.id] = run
        except Exception as exc:
            print(f"WARN failed explicit run {rid}: {exc}")

    all_runs = list(api.runs(args.project))
    for name in args.exp_names:
        candidates = [r for r in all_runs if r.name == name]
        candidates.sort(key=lambda r: str(getattr(r, "created_at", "")))
        for run in candidates:
            selected[run.id] = run

    manifest = []
    for run_id, run in selected.items():
        run_dir = outdir(runs_dir / f"{run.name}__{run.id}")
        print(f"Downloading WandB run {run.name} {run.id} -> {run_dir}")
        history = run.history(samples=args.samples, pandas=True)
        history.to_csv(run_dir / "history.csv", index=False)
        metadata = run_identity(run)
        metadata["config"] = safe_json(dict(run.config))
        metadata["summary"] = safe_json(dict(run.summary))
        with open(run_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        manifest.append(metadata)

    with open(base / "wandb_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {len(manifest)} WandB runs to {runs_dir}")


def modal_volume_ls(args):
    paths = args.paths or ["ckpts"]
    for path in paths:
        cmd = ["modal", "volume", "ls", args.volume, path]
        print("$", " ".join(cmd))
        result = subprocess.run(cmd, text=True, capture_output=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)


def download_modal(args):
    base = outdir(args.output_dir)
    modal_dir = outdir(base / "modal_volume")
    for exp_name in args.exp_names:
        exp_dir = outdir(modal_dir / exp_name)
        for step in args.steps:
            remote_path = f"ckpts/{exp_name}/{step}_rollouts.json"
            target = exp_dir / f"{step}_rollouts.json"
            if target.exists() and not args.force:
                print(f"SKIP existing {target}")
                continue
            cmd = ["modal", "volume", "get", args.volume, remote_path, str(target)]
            print("$", " ".join(cmd))
            result = subprocess.run(cmd, text=True, capture_output=True)
            if result.stdout:
                print(result.stdout)
            if result.returncode != 0:
                print(result.stderr)


def load_histories(base):
    rows = []
    for hist_path in sorted((base / "wandb_runs").glob("*/history.csv")):
        meta_path = hist_path.parent / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        df = pd.read_csv(hist_path)
        df["_run_id"] = meta["id"]
        df["_run_name"] = meta["name"]
        df["_created_at"] = meta.get("created_at")
        rows.append(df)
    return rows


def summarize_history(df, label):
    step = df["_step"] if "_step" in df else pd.Series(dtype=float)
    out = {
        "label": label,
        "rows": int(len(df)),
        "step_min": float(step.min()) if len(step.dropna()) else None,
        "step_max": float(step.max()) if len(step.dropna()) else None,
        "duplicate_steps": int(step.duplicated().sum()) if len(step) else 0,
    }
    for col in [
        "response_length/mean",
        "response_length/clip_ratio",
        "perf/time_per_step",
        "timing_s/gen",
        "timing_s/update_actor",
        "timing_s/ref",
        "critic/rewards/mean",
        "actor/entropy",
        "val/openai/gsm8k/reward/mean",
    ]:
        if col in df.columns:
            s = df[col].dropna()
            if len(s):
                out[col] = {
                    "first": float(s.iloc[0]),
                    "last": float(s.iloc[-1]),
                    "mean": float(s.mean()),
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "nonnull": int(len(s)),
                }
    return out


def combine_original_track_a(histories):
    pieces = [df for df in histories if str(df["_run_id"].iloc[0]) in ORIGINAL_TRACK_A_RUN_IDS]
    if not pieces:
        return None
    combined = pd.concat(pieces, ignore_index=True)
    combined = combined.sort_values(["_step", "_created_at"]).drop_duplicates(subset=["_step"], keep="last")
    return combined


def rollout_summary(path):
    with open(path) as f:
        data = json.load(f)
    lengths = []
    scores = []
    hash_flags = []
    idxs = []
    inputs_prefix = []
    for r in data:
        out = str(r.get("output", ""))
        inp = str(r.get("input", ""))
        lengths.append(len(out))
        scores.append(float(r.get("score", 0.0)))
        hash_flags.append("####" in out)
        idxs.append(r.get("index"))
        inputs_prefix.append(inp[:100])
    s = pd.Series(lengths, dtype="float64")
    return {
        "path": str(path),
        "samples": len(data),
        "unique_indices": len(set(idxs)),
        "length_mean_chars": float(s.mean()) if len(s) else None,
        "length_p50_chars": float(s.quantile(0.5)) if len(s) else None,
        "length_p90_chars": float(s.quantile(0.9)) if len(s) else None,
        "length_max_chars": float(s.max()) if len(s) else None,
        "score_mean": float(pd.Series(scores).mean()) if scores else None,
        "has_hash_rate": float(pd.Series(hash_flags).mean()) if hash_flags else None,
        "input_prefix_examples": sorted(set(inputs_prefix))[:3],
    }


def analyze(args):
    base = Path(args.output_dir)
    report_dir = outdir(base / "reports")
    histories = load_histories(base)
    report = {"history_summaries": [], "rollout_summaries": [], "overlap_checks": []}

    for df in histories:
        label = f"{df['_run_name'].iloc[0]}__{df['_run_id'].iloc[0]}"
        report["history_summaries"].append(summarize_history(df, label))

    combined = combine_original_track_a(histories)
    if combined is not None:
        combined.to_csv(report_dir / "original_track_a_combined_history.csv", index=False)
        report["history_summaries"].append(summarize_history(combined, "original_track_a_combined_by_known_resume_ids"))
        if "_run_id" in combined.columns and "_step" in combined.columns:
            step_ranges = combined.groupby("_run_id")["_step"].agg(["min", "max", "count"]).reset_index()
            report["original_track_a_combined_run_step_ranges"] = step_ranges.to_dict(orient="records")

    for roll_path in sorted((base / "modal_volume").glob("*/*_rollouts.json")):
        try:
            report["rollout_summaries"].append(rollout_summary(roll_path))
        except Exception as exc:
            report["rollout_summaries"].append({"path": str(roll_path), "error": str(exc)})

    with open(report_dir / "anomaly_report.json", "w") as f:
        json.dump(report, f, indent=2)

    rows = []
    for item in report["history_summaries"]:
        flat = {k: v for k, v in item.items() if not isinstance(v, dict)}
        for k, v in item.items():
            if isinstance(v, dict):
                flat[f"{k}__mean"] = v.get("mean")
                flat[f"{k}__first"] = v.get("first")
                flat[f"{k}__last"] = v.get("last")
                flat[f"{k}__max"] = v.get("max")
        rows.append(flat)
    if rows:
        pd.DataFrame(rows).to_csv(report_dir / "history_summary.csv", index=False)

    if report["rollout_summaries"]:
        pd.DataFrame(report["rollout_summaries"]).to_csv(report_dir / "rollout_summary.csv", index=False)

    print(f"Wrote report files under {report_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="investigation_artifacts/gsm8k_track_a_anomaly")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list-wandb")
    p.add_argument("--project", default=PROJECT)
    p.add_argument("--name-contains", default="qwen3-1p7b-gsm8k-grpo")
    p.set_defaults(func=list_wandb)

    p = sub.add_parser("download-wandb")
    p.add_argument("--project", default=PROJECT)
    p.add_argument("--run-ids", nargs="*", default=ORIGINAL_TRACK_A_RUN_IDS)
    p.add_argument("--exp-names", nargs="*", default=DEFAULT_EXP_NAMES)
    p.add_argument("--samples", type=int, default=20000)
    p.set_defaults(func=download_wandb)

    p = sub.add_parser("list-modal")
    p.add_argument("--volume", default="e3-generation-vol")
    p.add_argument("--paths", nargs="*", default=["ckpts", "ckpts/qwen3-1p7b-gsm8k-grpo-clean", "ckpts/qwen3-1p7b-gsm8k-grpo-clean-v2"])
    p.set_defaults(func=modal_volume_ls)

    p = sub.add_parser("download-modal")
    p.add_argument("--volume", default="e3-generation-vol")
    p.add_argument("--exp-names", nargs="*", default=DEFAULT_EXP_NAMES)
    p.add_argument("--steps", nargs="*", type=int, default=DEFAULT_STEPS)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=download_modal)

    p = sub.add_parser("analyze")
    p.set_defaults(func=analyze)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
