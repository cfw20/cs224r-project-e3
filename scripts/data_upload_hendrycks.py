#!/usr/bin/env python3
"""
One-shot Modal helper: generate clean + mixed Hendrycks MATH parquets and
upload them to the e3-generation-vol Volume at /data/hendrycks_math/.

Runs hendrycks_padded.py inside a Modal container (so the parquets are written
directly to the Volume, no local upload required).

Usage:
    modal run scripts/data_upload_hendrycks.py
    modal run scripts/data_upload_hendrycks.py --seed 123
"""

import modal

app = modal.App("rlad-hendrycks-data-upload")

image = (
    modal.Image.from_dockerfile("docker/Dockerfile.ngc.vllm0.8.noverl")
    .add_local_dir(".", "/root/e3", copy=True)
    .run_commands("pip install -e /root/e3", "pip install seaborn")
)

vol = modal.Volume.from_name("e3-generation-vol", create_if_missing=True)

REPO_PATH = "/root/e3"
DATA_DIR = "/data/hendrycks_math"


@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=30 * 60,
)
def run_prepare(seed: int):
    import os
    import subprocess

    os.environ["HF_HOME"] = "/data/hf_cache"
    os.makedirs(DATA_DIR, exist_ok=True)

    script = os.path.join(REPO_PATH, "examples/data_preprocess/hendrycks_padded.py")

    for mode in ("clean", "mixed"):
        cmd = [
            "python3", script,
            "--mode", mode,
            "--local_dir", DATA_DIR,
            "--seed", str(seed),
        ]
        print(f"[data_upload] running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=REPO_PATH)

    print(f"[data_upload] listing {DATA_DIR}:")
    for name in sorted(os.listdir(DATA_DIR)):
        full = os.path.join(DATA_DIR, name)
        size = os.path.getsize(full) if os.path.isfile(full) else "-"
        print(f"  {name}  ({size} bytes)")

    vol.commit()
    return {"data_dir": DATA_DIR}


@app.local_entrypoint()
def main(seed: int = 42):
    result = run_prepare.remote(seed=seed)
    print(f"[main] done: {result}")
