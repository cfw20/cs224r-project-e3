#!/usr/bin/env python3
"""
Modal test: run verl's main_generation on a small model + tiny math dataset.

Usage:
    pip install modal
    modal run modal_generation_test.py

This spins up one Modal container with a single GPU, runs inference on 2 tiny
math prompts using a 1.3B parameter model, and saves the generated answers.
"""

import modal
import os

# ---------------------------------------------------------------------------
# 1. Define the remote image
# ---------------------------------------------------------------------------
app = modal.App("e3-generation-test")

image = (
    modal.Image.debian_slim(python_version="3.10")
    # System deps for building flash-attn (optional) and general tooling
    .apt_install("git", "build-essential", "wget", "pkg-config")
    # Core Python deps — pinned to match the project's Dockerfile
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "torchaudio==2.6.0",
        "vllm==0.8.2",
        "transformers>=4.49.0",
        "accelerate",
        "datasets",
        "peft",
        "hf-transfer",
        "ray[default]",
        "codetiming",
        "hydra-core",
        "pandas",
        "pyarrow>=15.0.0",
        "pylatexenc",
        "qwen-vl-utils",
        "wandb",
        "dill",
        "pybind11",
        "liger-kernel",
        "mathruler",
        "pytest",
        "yapf",
        "py-spy",
        "pyext",
        "tensordict<=0.6.2",
        "torchdata",
        "numpy",
        "omegaconf",
    )
    # Install the verl package from the local code (editable, inside the image)
    .copy_local_dir(os.path.dirname(__file__), "/root/e3")
    .run_commands("cd /root/e3 && pip install -e .")
)

# Persistent volume for model cache and outputs
# (so repeated runs don't re-download the base model)
vol = modal.Volume.from_name("e3-generation-vol", create_if_missing=True)

# ---------------------------------------------------------------------------
# 2. Helper: build a tiny test dataset
# ---------------------------------------------------------------------------
def _make_tiny_dataset() -> str:
    """Create a 2-row parquet of math prompts and return its path."""
    import pandas as pd
    import os

    data_dir = "/data/test_math"
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "tiny_math.parquet")

    prompts = [
        [{"role": "user", "content": "What is 15 + 27?"}],
        [{"role": "user", "content": "Solve for x: 2x + 5 = 13"}],
    ]
    df = pd.DataFrame({"prompt": prompts})
    df.to_parquet(path, index=False)
    return path

# ---------------------------------------------------------------------------
# 3. Remote function — the actual generation job
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A10g",               # cheapest GPU with enough VRAM for 1.3B model
    # gpu="A100",             # upgrade here if you want faster / bigger models
    gpu_count=1,              # single GPU is plenty for a 1.3B model
    volumes={"/data": vol},
    timeout=1800,             # 30 min — mostly model download on first run
)
def run_generation():
    import subprocess
    import os

    # HuggingFace cache lives on the persistent volume
    os.environ["HF_HOME"] = "/data/hf_cache"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Build the tiny dataset
    data_path = _make_tiny_dataset()
    output_path = "/data/tiny_math_outputs.parquet"

    # Choose a small model that fits comfortably on one GPU.
    # deepseek-coder-1.3b-instruct is ~2.6 GB in bf16.
    model_path = "deepseek-ai/deepseek-coder-1.3b-instruct"

    cmd = [
        "python3", "-m", "verl.trainer.main_generation",
        "trainer.nnodes=1",
        "trainer.n_gpus_per_node=1",
        f"data.path={data_path}",
        "data.prompt_key=prompt",
        "data.n_samples=1",
        f"data.output_path={output_path}",
        f"model.path={model_path}",
        "+model.trust_remote_code=True",
        "rollout.temperature=0.7",
        "rollout.top_k=50",
        "rollout.top_p=0.9",
        "rollout.prompt_length=512",
        "rollout.response_length=256",
        "rollout.tensor_model_parallel_size=1",
        "rollout.gpu_memory_utilization=0.9",
        "rollout.enforce_eager=True",
        "data.batch_size=2",
    ]

    print("Running command:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd="/root/e3", check=True)

    # Read and print results
    import pandas as pd
    df = pd.read_parquet(output_path)
    print("\n=== Generated outputs ===")
    for idx, row in df.iterrows():
        print(f"\nPrompt {idx}: {row['prompt']}")
        print(f"Response: {row['responses']}")

    vol.commit()  # flush writes to the persistent volume
    return output_path

# ---------------------------------------------------------------------------
# 4. Local entrypoint — fires the remote job from your laptop
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    out_path = run_generation.remote()
    print(f"\nOutputs saved in Modal volume at: {out_path}")
