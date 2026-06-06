#!/usr/bin/env python3
"""
Modal helper to convert verl FSDP checkpoints into HF model directories for
post-training evaluation with modal_eval_general.py.

Locates the latest `global_step_*/actor` directory under
/data/ckpts/<experiment_name>/ and writes the merged HF model to
/data/ckpts/<experiment_name>_hf/.

Usage:
    modal run modal_convert_ckpt.py --track a
    modal run modal_convert_ckpt.py --track b
    modal run modal_convert_ckpt.py --track a --step 500   # explicit step

    # Convert any experiment directly by name (e.g. e3 curriculum stages):
    modal run modal_convert_ckpt.py --exp-name qwen3-1p7b-gsm8k-e3-clean-stage1
"""

import modal

app = modal.App("rlad-noise-control-convert")

image = (
    modal.Image.from_dockerfile("docker/Dockerfile.ngc.vllm0.8.noverl")
    .add_local_dir(".", "/root/e3", copy=True)
    .run_commands("pip install -e /root/e3", "pip install seaborn", "pip install fire")
)

vol = modal.Volume.from_name("e3-generation-vol", create_if_missing=True)

REPO_PATH = "/root/e3"
CKPT_ROOT = "/data/ckpts"
BASE_MODEL = "Qwen/Qwen3-1.7B"

TRACK_TO_EXP_NAME = {
    "gsm8k": {
        "a": "qwen3-1p7b-gsm8k-grpo-clean",
        "b": "qwen3-1p7b-gsm8k-grpo-mixed",
    },
    "math": {
        "a": "qwen3-1p7b-hendrycks-grpo-clean",
        "b": "qwen3-1p7b-hendrycks-grpo-mixed",
    }
}


def _find_latest_step_dir(exp_dir: str) -> str:
    import os
    import re

    candidates = []
    for name in os.listdir(exp_dir):
        m = re.match(r"global_step_(\d+)$", name)
        if m:
            candidates.append((int(m.group(1)), name))
    if not candidates:
        raise FileNotFoundError(f"No global_step_* dirs in {exp_dir}")
    candidates.sort()
    return os.path.join(exp_dir, candidates[-1][1])


def _detect_world_size(actor_dir: str) -> int:
    import os
    import re

    sizes = set()
    for name in os.listdir(actor_dir):
        m = re.match(r"model_world_size_(\d+)_rank_\d+\.pt$", name)
        if m:
            sizes.add(int(m.group(1)))
    if not sizes:
        raise FileNotFoundError(
            f"No model_world_size_*_rank_*.pt shards in {actor_dir}"
        )
    if len(sizes) > 1:
        raise RuntimeError(f"Multiple world sizes detected in {actor_dir}: {sizes}")
    return sizes.pop()


@app.function(
    image=image,
    gpu="A100-80GB",  # need GPU to load model weights; CPU often OOMs on assembly
    volumes={"/data": vol},
    timeout=2 * 3600,
)
def run_convert(track: str, dataset: str, step: int, exp_name: str, base_model: str):
    import os
    import re
    import subprocess

    os.environ["HF_HOME"] = "/data/hf_cache"

    # --exp-name takes precedence and bypasses the track table (used for e3 stages).
    if not exp_name:
        if dataset not in TRACK_TO_EXP_NAME:
            raise ValueError(f"Unknown --dataset={dataset!r}; choose from {list(TRACK_TO_EXP_NAME)}")
        dataset_tracks = TRACK_TO_EXP_NAME[dataset]
        if track not in dataset_tracks:
            raise ValueError(f"Unknown --track={track!r} for dataset={dataset!r}; choose from {list(dataset_tracks)}")
        exp_name = dataset_tracks[track]

    exp_dir = os.path.join(CKPT_ROOT, exp_name)
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f"No checkpoint dir for track {track} on dataset {dataset}: {exp_dir}")

    if step > 0:
        step_dir = os.path.join(exp_dir, f"global_step_{step}")
        if not os.path.isdir(step_dir):
            raise FileNotFoundError(f"Requested step dir missing: {step_dir}")
    else:
        step_dir = _find_latest_step_dir(exp_dir)
    print(f"[convert] using step dir: {step_dir}")

    # Resolve the concrete step number so each checkpoint gets its own HF dir.
    m = re.search(r"global_step_(\d+)$", step_dir)
    resolved_step = int(m.group(1)) if m else step

    actor_dir = os.path.join(step_dir, "actor")
    if not os.path.isdir(actor_dir):
        raise FileNotFoundError(f"Expected actor dir at {actor_dir}")

    world_size = _detect_world_size(actor_dir)
    print(f"[convert] detected world_size={world_size}")

    output_path = os.path.join(CKPT_ROOT, f"{exp_name}_hf_step{resolved_step}")
    os.makedirs(output_path, exist_ok=True)

    cmd = [
        "python3",
        os.path.join(REPO_PATH, "convert_fsdp_to_hf.py"),
        "--fsdp_checkpoint_path", actor_dir,
        "--huggingface_model_path", base_model,
        "--output_path", output_path,
        "--world_size", str(world_size),
    ]
    print(f"[convert] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO_PATH)
    vol.commit()
    print(f"[convert] HF model written to {output_path}")
    return {"track": track, "dataset": dataset, "step": resolved_step, "step_dir": step_dir, "hf_path": output_path}


@app.local_entrypoint()
def main(track: str = "a", dataset: str = "gsm8k", step: int = -1, exp_name: str = "", base_model: str = BASE_MODEL):
    """--step=-1 means latest available global_step_* dir.
    --exp-name converts that experiment directly, ignoring --track/--dataset.
    --base-model sets the HF architecture to load weights into (default Qwen/Qwen3-1.7B;
    pass Qwen/Qwen3-0.6B for the 0.6B runs)."""
    result = run_convert.remote(track=track, dataset=dataset, step=step, exp_name=exp_name, base_model=base_model)
    print(f"[main] done: {result}")
