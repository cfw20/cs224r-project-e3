#!/usr/bin/env python3
"""
Modal test: run verl's main_generation on a small model + tiny math dataset.

Usage:
    # 1. Create a GitHub PAT (Settings -> Developer settings -> Personal access tokens)
    # 2. Store it as a Modal Secret:
    #    modal secret create github-token GITHUB_TOKEN=ghp_xxxxxxxx
    # 3. Run:
    #    modal run modal_generation_test.py
"""

import modal
import os

app = modal.App("e3-generation-test")

# Load the GitHub token from a Modal Secret (created via CLI ahead of time).
# If running locally and the env var exists, fall back to that for convenience.
if modal.is_local() and os.environ.get("GITHUB_TOKEN"):
    github_secret = modal.Secret.from_dict({"GITHUB_TOKEN": os.environ["GITHUB_TOKEN"]})
else:
    github_secret = modal.Secret.from_name("github-token")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "build-essential", "wget", "pkg-config")
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
        "flash-attn>=2.5.8",
    )
)

vol = modal.Volume.from_name("e3-generation-vol", create_if_missing=True)


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


@app.function(
    image=image,
    gpu="A10",
    secrets=[github_secret],
    volumes={"/data": vol},
    timeout=1800,
)
def run_generation():
    import subprocess
    import os

    token = os.environ["GITHUB_TOKEN"]
    repo_path = "/root/e3"

    # Clone (and install) at runtime so the secret is available
    if not os.path.exists(repo_path):
        subprocess.run(
            ["git", "clone", f"https://{token}@github.com/cfw20/cs224r-project-e3.git", repo_path],
            check=True,
        )
        subprocess.run(["pip", "install", "-e", repo_path], check=True)

    os.environ["HF_HOME"] = "/data/hf_cache"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    data_path = _make_tiny_dataset()
    output_path = "/data/tiny_math_outputs.parquet"
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
    subprocess.run(cmd, cwd=repo_path, check=True)

    import pandas as pd
    df = pd.read_parquet(output_path)
    print("\n=== Generated outputs ===")
    for idx, row in df.iterrows():
        print(f"\nPrompt {idx}: {row['prompt']}")
        print(f"Response: {row['responses']}")

    vol.commit()
    return output_path


@app.local_entrypoint()
def main():
    out_path = run_generation.remote()
    print(f"\nOutputs saved in Modal volume at: {out_path}")
