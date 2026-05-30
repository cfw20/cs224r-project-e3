#!/usr/bin/env python3
"""
Upload the e3 GSM8K curriculum parquets to the e3-generation-vol Volume at
/data/e3_gsm8k/.

The parquets are produced locally by scripts/e3-grpo-gsm8k/split_gsm8k_dataset.ipynb
(default output dir: scripts/e3-grpo-gsm8k/data_e3_gsm8k). This script uploads them
to the Volume so modal_train_e3_gsm8k.py can read them.

Expected files in --local-dir:
    train_easy_clean.parquet
    train_hard_clean.parquet
    train_easy_mixed.parquet
    train_hard_mixed.parquet
    test.parquet

Usage:
    modal run scripts/data_upload_e3_gsm8k.py
    modal run scripts/data_upload_e3_gsm8k.py --local-dir scripts/e3-grpo-gsm8k/data_e3_gsm8k
"""

import os

import modal

app = modal.App("e3-gsm8k-data-upload")

vol = modal.Volume.from_name("e3-generation-vol", create_if_missing=True)

REMOTE_DIR = "e3_gsm8k"  # -> /data/e3_gsm8k on the Volume

EXPECTED_FILES = [
    "train_easy_clean.parquet",
    "train_hard_clean.parquet",
    "train_easy_mixed.parquet",
    "train_hard_mixed.parquet",
    "test.parquet",
]


@app.local_entrypoint()
def main(local_dir: str = "scripts/e3-grpo-gsm8k/data_e3_gsm8k"):
    local_dir = os.path.abspath(local_dir)
    if not os.path.isdir(local_dir):
        raise FileNotFoundError(f"local-dir not found: {local_dir}")

    present = []
    missing = []
    for name in EXPECTED_FILES:
        path = os.path.join(local_dir, name)
        (present if os.path.exists(path) else missing).append(name)

    if not present:
        raise FileNotFoundError(
            f"None of the expected parquets were found in {local_dir}.\n"
            f"Run scripts/e3-grpo-gsm8k/split_gsm8k_dataset.ipynb first."
        )
    if missing:
        print(f"[upload] WARNING: missing (will skip): {missing}")

    print(f"[upload] uploading {len(present)} files from {local_dir} -> /data/{REMOTE_DIR}/")
    with vol.batch_upload(force=True) as batch:
        for name in present:
            local_path = os.path.join(local_dir, name)
            remote_path = f"{REMOTE_DIR}/{name}"
            print(f"  + {name}  ->  /data/{remote_path}")
            batch.put_file(local_path, remote_path)

    print("[upload] done.")
