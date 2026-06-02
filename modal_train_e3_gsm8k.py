#!/usr/bin/env python3
"""
Modal training entrypoint for the e3 GSM8K curriculum experiment (Track C / D / E).

Runs verl GRPO with the e3 recipe (negative gradients + asymmetric clipping) on a
single H100/A100-80GB, in a two-stage curriculum:

  Stage 1: easy split,  max_response_length=512,  start from Qwen/Qwen3-1.7B base
  Stage 2: hard split,  max_response_length=1024, start from the stage-1 checkpoint

Tracks:
  --track c : clean GSM8K splits        (train_easy_clean / train_hard_clean)
  --track d : trivia-mixed splits       (train_easy_mixed / train_hard_mixed)
  --track e : trivia-only splits        (train_easy_trivia / train_hard_trivia)

Data lives under /data/e3_gsm8k/ on the e3-generation-vol Volume (upload it first
with scripts/data_upload_e3_gsm8k.py). Checkpoints land in
/data/ckpts/qwen3-1p7b-gsm8k-e3-{clean,mixed,trivia}-stage{1,2}.

IMPORTANT: between stage 1 and stage 2 you must convert the stage-1 FSDP checkpoint
to an HF model so stage 2 can resume from real weights (the in-loop `huggingface/`
dir only stores config + tokenizer, not weights):

    modal run modal_train_e3_gsm8k.py --track c --stage 1
    modal run modal_convert_ckpt.py --exp-name qwen3-1p7b-gsm8k-e3-clean-stage1
    modal run modal_train_e3_gsm8k.py --track c --stage 2

Usage:
    modal run --detach modal_train_e3_gsm8k.py --track c --stage 1
    modal run --detach modal_train_e3_gsm8k.py --track c --stage 2 --total-steps 400
"""

import modal

app = modal.App("e3-gsm8k-train")

image = (
    modal.Image.from_dockerfile("docker/Dockerfile.ngc.vllm0.8.noverl")
    .add_local_dir(".", "/root/e3", copy=True)
    .run_commands("pip install -e /root/e3", "pip install seaborn")
)

vol = modal.Volume.from_name("e3-generation-vol", create_if_missing=True)

REPO_PATH = "/root/e3"
DATA_DIR = "/data/e3_gsm8k"                # split parquets live here
CKPT_ROOT = "/data/ckpts"                  # checkpoints land here
BASE_MODEL = "Qwen/Qwen3-1.7B"

# track -> "clean" | "mixed" | "trivia"
TRACK_TO_FLAVOR = {"c": "clean", "d": "mixed", "e": "trivia"}

# stage -> (split, max_response_length)
STAGE_TO_CONFIG = {
    1: {"split": "easy", "max_response_length": 512},
    2: {"split": "hard", "max_response_length": 1024},
}


def _exp_name(flavor: str, stage: int) -> str:
    return f"qwen3-1p7b-gsm8k-e3-{flavor}-stage{stage}"


@app.function(
    image=image,
    gpu="H200",
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=24 * 3600,
)
def run_train(track: str, stage: int, total_steps: int, save_freq: int, test_freq: int, model_path: str):
    import os
    import subprocess

    if track not in TRACK_TO_FLAVOR:
        raise ValueError(f"Unknown --track={track!r}; choose from {list(TRACK_TO_FLAVOR)}")
    if stage not in STAGE_TO_CONFIG:
        raise ValueError(f"Unknown --stage={stage!r}; choose from {list(STAGE_TO_CONFIG)}")

    os.environ["HF_HOME"] = "/data/hf_cache"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # wandb-secret is expected to expose WANDB_API_KEY in the env

    flavor = TRACK_TO_FLAVOR[track]
    stage_cfg = STAGE_TO_CONFIG[stage]
    max_response_length = stage_cfg["max_response_length"]

    train_parquet = os.path.join(DATA_DIR, f"train_{stage_cfg['split']}_{flavor}.parquet")
    val_parquet = os.path.join(DATA_DIR, "test.parquet")
    exp_name = _exp_name(flavor, stage)
    ckpt_dir = os.path.join(CKPT_ROOT, exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Resolve the starting model.
    if model_path:
        start_model = model_path
    elif stage == 1:
        start_model = BASE_MODEL
    else:
        # Stage 2 resumes from the converted stage-1 HF checkpoint.
        prev_exp = _exp_name(flavor, stage - 1)
        start_model = os.path.join(CKPT_ROOT, f"{prev_exp}_hf")

    for p in (train_parquet, val_parquet):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing parquet on Volume: {p}. Did you run scripts/data_upload_e3_gsm8k.py?"
            )
    if stage == 2 and not model_path and not os.path.isdir(start_model):
        raise FileNotFoundError(
            f"Stage-2 start model not found: {start_model}\n"
            f"Convert the stage-1 checkpoint first:\n"
            f"    modal run modal_convert_ckpt.py --exp-name {_exp_name(flavor, 1)}"
        )

    env = os.environ.copy()
    env.update({
        "TRAIN_PARQUET": train_parquet,
        "VAL_PARQUET": val_parquet,
        "MODEL_PATH": start_model,
        "CKPT_DIR": ckpt_dir,
        "EXPERIMENT_NAME": exp_name,
        "TOTAL_STEPS": str(total_steps),
        "MAX_RESPONSE_LENGTH": str(max_response_length),
        "MAX_EXTRAPOLATION_LENGTH": str(max_response_length * 2),
        "SAVE_FREQ": str(save_freq),
        "TEST_FREQ": str(test_freq),
        "WANDB_PROJECT": "e3-gsm8k",
    })

    cmd = ["bash", os.path.join(REPO_PATH, "scripts/grpo/grpo_gsm8k_e3.sh")]
    print(f"[modal_train_e3] track={track} flavor={flavor} stage={stage} exp={exp_name} steps={total_steps}")
    print(f"[modal_train_e3] train_parquet={train_parquet}")
    print(f"[modal_train_e3] val_parquet={val_parquet}")
    print(f"[modal_train_e3] start_model={start_model}")
    print(f"[modal_train_e3] max_response_length={max_response_length}")
    print(f"[modal_train_e3] ckpt_dir={ckpt_dir}")
    try:
        subprocess.run(cmd, check=True, cwd=REPO_PATH, env=env)
    finally:
        # Always commit the volume so partial checkpoints survive timeouts/crashes.
        vol.commit()

    return {"track": track, "stage": stage, "experiment_name": exp_name, "ckpt_dir": ckpt_dir}


@app.local_entrypoint()
def main(
    track: str = "c",
    stage: int = 1,
    total_steps: int = 400,
    save_freq: int = 100,
    test_freq: int = 100,
    model_path: str = "",
):
    """--model-path overrides the starting checkpoint (otherwise base for stage 1,
    converted stage-1 HF dir for stage 2)."""
    print(f"[main] track={track} stage={stage} total_steps={total_steps} "
          f"save_freq={save_freq} test_freq={test_freq} model_path={model_path or '<auto>'}")
    result = run_train.remote(
        track=track,
        stage=stage,
        total_steps=total_steps,
        save_freq=save_freq,
        test_freq=test_freq,
        model_path=model_path,
    )
    print(f"[main] done: {result}")
