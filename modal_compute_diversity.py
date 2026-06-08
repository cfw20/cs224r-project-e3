#!/usr/bin/env python3
"""
Modal script to compute semantic diversity of generated solutions using
Qwen3-Embedding-4B embeddings and pairwise cosine similarity.

Usage:
    modal run --detach modal_compute_diversity.py

Reads:
    /data/pass32_0p6b/gsm8k_track_a_step150_outputs.parquet
    /data/pass32_0p6b/gsm8k_track_b_step150_outputs.parquet

Writes:
    /data/diversity_0p6b/embedding_diversity_results.csv
    /data/diversity_0p6b/embedding_diversity_summary.json
"""

import modal

app = modal.App("rlad-compute-diversity")

image = (
    modal.Image.from_dockerfile("docker/Dockerfile.ngc.vllm0.8.noverl")
    .add_local_dir(".", "/root/e3", copy=True)
    .run_commands(
        "pip install -e /root/e3",
        "pip install seaborn",
    )
)

vol = modal.Volume.from_name("e3-generation-vol", create_if_missing=True)

PASS32_DIR = "/data/pass32_0p6b"
OUTPUT_DIR = "/data/diversity_0p6b"
MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
N_QUESTIONS = 100
SEED = 42


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks and extra whitespace."""
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("</think>", "").replace("<think>", "")
    return text.strip()


def _mean_pool(last_hidden, attention_mask):
    """Mean-pool token embeddings using the attention mask."""
    import torch
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    masked = last_hidden * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@app.function(
    image=image,
    gpu="H200",
    volumes={"/data": vol},
    timeout=2 * 3600,
)
def compute_diversity():
    import os
    import json
    import random
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    os.environ["HF_HOME"] = "/data/hf_cache"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # ------------------------------------------------------------------
    # 1. Load parquets
    # ------------------------------------------------------------------
    a_path = os.path.join(PASS32_DIR, "gsm8k_track_a_step150_outputs.parquet")
    b_path = os.path.join(PASS32_DIR, "gsm8k_track_b_step150_outputs.parquet")

    df_a = pd.read_parquet(a_path)
    df_b = pd.read_parquet(b_path)
    assert len(df_a) == len(df_b), "Mismatched parquet lengths"
    n_total = len(df_a)
    print(f"[diversity] Loaded {n_total} questions per track")

    # ------------------------------------------------------------------
    # 2. Sample 100 questions (same indices for both tracks)
    # ------------------------------------------------------------------
    rng = random.Random(SEED)
    sampled_indices = sorted(rng.sample(range(n_total), N_QUESTIONS))
    print(f"[diversity] Sampled {N_QUESTIONS} question indices: {sampled_indices[:10]}...")

    df_a_sample = df_a.iloc[sampled_indices].reset_index(drop=True)
    df_b_sample = df_b.iloc[sampled_indices].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. Load embedding model
    # ------------------------------------------------------------------
    print(f"[diversity] Loading embedding model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = model.cuda().eval()
    print("[diversity] Model loaded on GPU")

    # ------------------------------------------------------------------
    # 4. Encode responses
    # ------------------------------------------------------------------
    def encode_texts(texts, batch_size=64):
        """Encode a list of strings into normalized embedding vectors."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            emb = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)
            all_embeddings.append(emb.cpu().numpy())
        return np.concatenate(all_embeddings, axis=0)

    def process_track(df_sample, track_label):
        rows = []
        for q_idx in range(len(df_sample)):
            # Extract the 32 response strings
            responses = df_sample.iloc[q_idx]["responses"]
            if len(responses) != 32:
                print(f"[diversity] WARNING: Track {track_label} question {q_idx} has {len(responses)} responses, expected 32")

            # Clean text
            cleaned = [_strip_think_tags(str(r)) for r in responses]

            # Encode all 32 responses
            embeddings = encode_texts(cleaned, batch_size=32)  # 32 at once is fine
            assert embeddings.shape == (len(responses), embeddings.shape[1]), f"Unexpected shape: {embeddings.shape}"

            # Pairwise cosine similarity: dot product of normalized vectors
            sim_matrix = embeddings @ embeddings.T  # (n, n)
            n = sim_matrix.shape[0]
            # Extract upper triangle (excluding diagonal)
            triu_idx = np.triu_indices(n, k=1)
            similarities = sim_matrix[triu_idx]
            mean_sim = float(similarities.mean())
            std_sim = float(similarities.std())

            rows.append({
                "track": track_label,
                "question_idx": int(df_sample.iloc[q_idx]["extra_info"]["index"]),
                "mean_similarity": mean_sim,
                "std_similarity": std_sim,
                "n_pairs": len(similarities),
            })

            if (q_idx + 1) % 10 == 0:
                print(f"[diversity] Track {track_label}: processed {q_idx + 1}/{len(df_sample)} questions")

        return pd.DataFrame(rows)

    print("[diversity] Encoding Track A...")
    df_results_a = process_track(df_a_sample, "A")
    print("[diversity] Encoding Track B...")
    df_results_b = process_track(df_b_sample, "B")

    df_results = pd.concat([df_results_a, df_results_b], ignore_index=True)

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "embedding_diversity_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"[diversity] Saved CSV -> {csv_path}")

    summary = {
        "model": MODEL_NAME,
        "n_questions": N_QUESTIONS,
        "seed": SEED,
        "tracks": {},
    }
    for track in ["A", "B"]:
        subset = df_results[df_results["track"] == track]["mean_similarity"]
        summary["tracks"][track] = {
            "mean_similarity": float(subset.mean()),
            "std_similarity": float(subset.std()),
            "min_similarity": float(subset.min()),
            "max_similarity": float(subset.max()),
        }

    json_path = os.path.join(OUTPUT_DIR, "embedding_diversity_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[diversity] Saved JSON -> {json_path}")

    # Commit volume
    vol.commit()
    print("[diversity] Volume committed")

    return summary


@app.local_entrypoint()
def main():
    import json
    result = compute_diversity.remote()
    print("\n=== Diversity Summary ===")
    print(json.dumps(result, indent=2))
