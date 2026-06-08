#!/usr/bin/env python3
"""
One-shot Modal helper: generate clean + mixed GSM8K parquets and upload them
to the e3-generation-vol Volume at /data/gsm8k_padded/.

Runs gsm8k_padded.py inside a Modal container (so the parquets are written
directly to the Volume, no local upload required).

Usage:
    modal run scripts/data_upload_gsm8k.py
    modal run scripts/data_upload_gsm8k.py --seed 123
"""

import modal

app = modal.App("rlad-noise-control-data-upload")

image = (
    modal.Image.from_dockerfile("docker/Dockerfile.ngc.vllm0.8.noverl")
    .add_local_dir(".", "/root/e3", copy=True)
    .run_commands("pip install -e /root/e3", "pip install seaborn")
)

vol = modal.Volume.from_name("e3-generation-vol", create_if_missing=True)

REPO_PATH = "/root/e3"
DATA_DIR = "/data/gsm8k_padded"


@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=30 * 60,
)
def run_prepare(seed: int):
    import os
    import subprocess
    import tempfile
    import shutil
    import filecmp

    os.environ["HF_HOME"] = "/data/hf_cache"
    os.makedirs(DATA_DIR, exist_ok=True)

    staging_dir = tempfile.mkdtemp(prefix="gsm8k_staging_")
    script = os.path.join(REPO_PATH, "examples/data_preprocess/gsm8k_padded.py")

    results = []

    for mode in ("clean", "mixed", "trivia"):
        cmd = [
            "python3", script,
            "--mode", mode,
            "--local_dir", staging_dir,
            "--seed", str(seed),
        ]
        print(f"[data_upload] generating: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=REPO_PATH)

        train_name = {
            "clean": "train_clean.parquet",
            "mixed": "train_mixed.parquet",
            "trivia": "train_trivia.parquet",
        }[mode]

        staged_path = os.path.join(staging_dir, train_name)
        vol_path = os.path.join(DATA_DIR, train_name)

        if os.path.exists(vol_path):
            if filecmp.cmp(staged_path, vol_path, shallow=False):
                results.append(f"{train_name}: IDENTICAL (skipped)")
                print(f"[data_upload] {train_name}: identical to Volume copy, skipping")
            else:
                results.append(f"{train_name}: DIFFERENT (updated)")
                print(f"[data_upload] {train_name}: differs from Volume copy, updating")
                shutil.copy2(staged_path, vol_path)
        else:
            results.append(f"{train_name}: NEW (uploaded)")
            print(f"[data_upload] {train_name}: new file, copying to Volume")
            shutil.copy2(staged_path, vol_path)

    print(f"[data_upload] listing {DATA_DIR}:")
    for name in sorted(os.listdir(DATA_DIR)):
        full = os.path.join(DATA_DIR, name)
        size = os.path.getsize(full) if os.path.isfile(full) else "-"
        print(f"  {name}  ({size} bytes)")

    vol.commit()
    shutil.rmtree(staging_dir)
    return {"data_dir": DATA_DIR, "results": results}


@app.local_entrypoint()
def main(seed: int = 42):
    result = run_prepare.remote(seed=seed)
    print(f"[main] done: {result}")
