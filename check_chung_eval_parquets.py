"""
check_chung_eval_parquets.py  —  FOR CHUNG TO RUN on his own machine.

Purpose: confirm whether the eval *output* files (the generated solutions, saved
as .parquet) actually exist on your Modal volume. W&B only stores the scalar
scores; the per-sample generations needed for cosine-similarity / MATH output
analysis live on the volume, if they were committed at all.

This script ONLY LISTS files. It does not download, delete, or change anything.

------------------------------------------------------------------------------
SETUP (you're already logged into your own Modal, so nothing to change):
    pip install modal          # if not already installed

USAGE:
    python check_chung_eval_parquets.py

If your volume isn't named "cs224r-trivia-vol", either edit VOLUME_NAME below,
or just run the bare CLI command at the bottom of this file with the right name.
------------------------------------------------------------------------------
"""

import subprocess
import sys


# Edit if your volume has a different name (run `modal volume list` to see them).
VOLUME_NAME = "e3-generation-vol"

# Directories where eval outputs are written, per the eval scripts.
EVAL_DIRS = [
    "/data/eval",
    "/data/eval/aime",
]


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        print("  ERROR: 'modal' CLI not found. Run:  pip install modal")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("  ERROR: timed out talking to Modal.")
        return ""
    if out.stdout:
        print(out.stdout.rstrip())
    if out.returncode != 0:
        # modal prints the useful error to stderr (e.g. missing path / wrong volume)
        print((out.stderr or "").rstrip())
    return out.stdout


def main():
    print(f"Checking volume: {VOLUME_NAME}")
    print("(listing only — nothing is downloaded or modified)")

    # 1. Confirm the volume exists / show all volumes for reference.
    run(["modal", "volume", "list"])

    # 2. List each eval directory.
    all_listings = ""
    for d in EVAL_DIRS:
        all_listings += run(["modal", "volume", "ls", VOLUME_NAME, d])

    # 3. Plain-English verdict.
    print("\n" + "=" * 60)
    has_parquet = ".parquet" in all_listings
    has_outputs = "_outputs" in all_listings
    if has_parquet and has_outputs:
        print("RESULT: output parquet files ARE present. Good to download.")
        print("To send them to Anna, run:")
        print(f"    modal volume get {VOLUME_NAME} /data/eval/ ./chung_eval/")
        print("then share the ./chung_eval/ folder (Drive, or commit to the repo).")
    elif has_parquet:
        print("RESULT: some .parquet files found, but none named '*_outputs.parquet'.")
        print("Those may be input/prepared data, not generations. Check the names above —")
        print("the generations are the files ending in '_outputs.parquet'.")
    else:
        print("RESULT: no .parquet files found in the eval dirs.")
        print("Likely meaning: the eval generations were never committed to the volume,")
        print("so they'd need to be re-generated. Send the listing above to Anna to confirm.")
    print("=" * 60)


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------------------
# Don't want to run the script? These two CLI commands do the core check by hand
# (replace the volume name if different):
#
#     modal volume list
#     modal volume ls cs224r-trivia-vol /data/eval/
#
# Look for files ending in  _outputs.parquet  — those are the generations.
# ------------------------------------------------------------------------------
