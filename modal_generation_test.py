#!/usr/bin/env python3
"""
Modal test: run verl's main_generation on a small model + tiny math dataset.

Usage:
    modal run modal_generation_test.py
"""

import modal
import os

app = modal.App("e3-generation-test")

image = modal.Image.from_dockerfile("docker/Dockerfile.ngc.vllm0.8.noverl").add_local_dir(".", "/root/e3")
# image = modal.Image.from_dockerfile("docker/Dockerfile.ngc.vllm").add_local_dir(".", "/root/e3")


vol = modal.Volume.from_name("e3-generation-vol", create_if_missing=True)

repo_path = "/root/e3"

cmd = [
    "python3", "-m", "verl.trainer.main_generation",
    "trainer.nnodes=1",
    "trainer.n_gpus_per_node=1",
    "data.path=/data/test_math/tiny_math.parquet",
    "data.prompt_key=prompt",
    "data.n_samples=1",
    "data.output_path=/data/tiny_math_outputs.parquet",
    "model.path=deepseek-ai/deepseek-coder-1.3b-instruct",
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
    "+rollout.extrapolation_val=False",
    "+rollout.extrapolation_length=0",
]


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
    volumes={"/data": vol},
    timeout=1800,
)
def run_generation():
    import subprocess
    import os

    os.environ["HF_HOME"] = "/data/hf_cache"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    _make_tiny_dataset()  # creates /data/test_math/tiny_math.parquet
    output_path = "/data/tiny_math_outputs.parquet"

    # Install local verl source in editable mode
    print(f"Installing local verl from {repo_path} ...")
    subprocess.run(["pip", "install", "-e", repo_path], check=True)

    # Run generation using the local verl package
    print("Running verl generation:")
    subprocess.run(cmd, check=True)

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
