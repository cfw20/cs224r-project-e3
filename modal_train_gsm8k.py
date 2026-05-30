#!/usr/bin/env python3
"""
Modal training entrypoint for the RLAD noise-control GSM8K experiment.

Runs verl GRPO on a single A100-80GB for either:
  - track a (clean)  : uses /data/gsm8k_padded/train_clean.parquet
  - track b (mixed)  : uses /data/gsm8k_padded/train_mixed.parquet

Both tracks share:
  - base model      Qwen/Qwen3-1.7B
  - val parquet     /data/gsm8k_padded/test.parquet
  - step budget     --total-steps (default 400)

Usage:
    modal run --detach modal_train_gsm8k.py --track a
    modal run --detach modal_train_gsm8k.py --track b
    modal run --detach modal_train_gsm8k.py --track a --total-steps 200
"""

import modal

app = modal.App("rlad-noise-control-train")

image = (
    modal.Image.from_dockerfile("docker/Dockerfile.ngc.vllm0.8.noverl")
    .add_local_dir(".", "/root/e3", copy=True)
    .run_commands("pip install -e /root/e3", "pip install seaborn")
)

vol = modal.Volume.from_name("e3-generation-vol", create_if_missing=True)

REPO_PATH = "/root/e3"
DATA_DIR = "/data/gsm8k_padded"           # parquets live here on the Volume
CKPT_ROOT = "/data/ckpts"                 # checkpoints land here on the Volume
BASE_MODEL = "Qwen/Qwen3-1.7B"

TRACK_TO_TRAIN_PARQUET = {
    "a": "train_clean.parquet",
    "b": "train_mixed.parquet",
}
TRACK_TO_EXP_NAME = {
    "a": "qwen3-1p7b-gsm8k-grpo-clean",
    "b": "qwen3-1p7b-gsm8k-grpo-mixed",
}


@app.function(
    image=image,
    gpu="H100",
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=24 * 3600,
)
def run_train(track: str, total_steps: int, save_freq: int, test_freq: int):
    import os
    import subprocess

    if track not in TRACK_TO_TRAIN_PARQUET:
        raise ValueError(f"Unknown --track={track!r}; choose from {list(TRACK_TO_TRAIN_PARQUET)}")

    os.environ["HF_HOME"] = "/data/hf_cache"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # wandb-secret is expected to expose WANDB_API_KEY in the env

    train_parquet = os.path.join(DATA_DIR, TRACK_TO_TRAIN_PARQUET[track])
    val_parquet = os.path.join(DATA_DIR, "test.parquet")
    exp_name = TRACK_TO_EXP_NAME[track]
    ckpt_dir = os.path.join(CKPT_ROOT, exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    for p in (train_parquet, val_parquet):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing parquet on Volume: {p}. Did you run scripts/data_upload_gsm8k.py?"
            )

    env = os.environ.copy()
    env.update({
        "TRAIN_PARQUET": train_parquet,
        "VAL_PARQUET": val_parquet,
        "BASE_MODEL": BASE_MODEL,
        "CKPT_DIR": ckpt_dir,
        "EXPERIMENT_NAME": exp_name,
        "TOTAL_STEPS": str(total_steps),
        "SAVE_FREQ": str(save_freq),
        "TEST_FREQ": str(test_freq),
        "WANDB_PROJECT": "rlad-noise-control",
    })

    cmd = ["bash", os.path.join(REPO_PATH, "scripts/grpo/grpo_gsm8k_a100.sh")]
    print(f"[modal_train] track={track} exp={exp_name} steps={total_steps}")
    print(f"[modal_train] train_parquet={train_parquet}")
    print(f"[modal_train] val_parquet={val_parquet}")
    print(f"[modal_train] ckpt_dir={ckpt_dir}")
    try:
        subprocess.run(cmd, check=True, cwd=REPO_PATH, env=env)
    finally:
        # Always commit the volume so partial checkpoints survive timeouts/crashes.
        vol.commit()

    return {"track": track, "experiment_name": exp_name, "ckpt_dir": ckpt_dir}


@app.local_entrypoint()
def main(
    track: str = "a",
    total_steps: int = 400,
    save_freq: int = 100,
    test_freq: int = 100,
):
    print(f"[main] track={track} total_steps={total_steps} save_freq={save_freq} test_freq={test_freq}")
    result = run_train.remote(
        track=track,
        total_steps=total_steps,
        save_freq=save_freq,
        test_freq=test_freq,
    )
    print(f"[main] done: {result}")
