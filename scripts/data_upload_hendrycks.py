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
    import tempfile
    import shutil
    import filecmp

    os.environ["HF_HOME"] = "/data/hf_cache"
    os.makedirs(DATA_DIR, exist_ok=True)

    staging_dir = tempfile.mkdtemp(prefix="hendrycks_staging_")
    script = os.path.join(REPO_PATH, "examples/data_preprocess/hendrycks_padded.py")

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

    # Overwrite test.parquet with the canonical MATH-500 validation set
    math500_script = os.path.join(REPO_PATH, "examples/data_preprocess/math500_prep.py")
    math500_cmd = ["python3", math500_script, "--local_dir", staging_dir, "--output_name", "test.parquet"]
    print(f"[data_upload] running: {' '.join(math500_cmd)}")
    subprocess.run(math500_cmd, check=True, cwd=REPO_PATH)

    staged_test = os.path.join(staging_dir, "test.parquet")
    vol_test = os.path.join(DATA_DIR, "test.parquet")

    if os.path.exists(vol_test):
        if filecmp.cmp(staged_test, vol_test, shallow=False):
            results.append("test.parquet: IDENTICAL (skipped)")
            print("[data_upload] test.parquet: identical to Volume copy, skipping")
        else:
            results.append("test.parquet: DIFFERENT (updated)")
            print("[data_upload] test.parquet: differs from Volume copy, updating")
            shutil.copy2(staged_test, vol_test)
    else:
        results.append("test.parquet: NEW (uploaded)")
        print("[data_upload] test.parquet: new file, copying to Volume")
        shutil.copy2(staged_test, vol_test)

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
