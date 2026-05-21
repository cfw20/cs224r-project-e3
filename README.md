## Setup 
1. Follow instructions here to setup verl: https://verl.readthedocs.io/en/latest/README_vllm0.8.html
2. `conda activate verl`
3. `pip install seaborn`

## Data
Our training and test data are stored on Hugging Face:
- Training data, stage 1: https://huggingface.co/datasets/CMU-AIRe/e3-math-easy
- Training data, stage 2: https://huggingface.co/datasets/CMU-AIRe/e3-math-medhard
- Test data: https://huggingface.co/datasets/CMU-AIRe/hmmt-aime-2025

To set up data for training,
1. Create a local directory to store data
2. Run `python examples/data_preprocess/math/generate_dataset.py --local_dir $dir --remote_dir $hf_dir --split $split`
3. Ensure that `data.train_files` and `data.val_files` in your scripts (e.g., `scripts/grpo/grpo_16k.sh`) point to the downloaded data


## Eval
For eval you can run `bash scripts/eval.sh`

## Eval on Modal AI
The `modal_eval_general.py` script runs evaluation on a Modal GPU (A100-80GB) via a three-phase pipeline:

1. **Dataset preparation** — Loads AIME, MATH, or GSM8K from HuggingFace and formats prompts into a parquet file.
2. **Generation** — Runs `verl.trainer.main_generation` with vLLM on a single A100. It batches multiple questions together (batch_size = min(n_problems, 24)) and generates `n_samples` responses per question sequentially.
3. **Scoring** — Compares generated answers against ground truth using dataset-specific scorers and computes pass@k metrics.

**Supported datasets and models**
- Datasets: `aime`, `math`, `gsm8k`
- Models: `qwen` (Qwen3-1.7B), `e3` (CMU-AIRe/e3-1.7B)

**Performance notes**
- Default `n_samples` is 4 for GSM8K/MATH and 16 for AIME.
- Default `max_response_length` is 32768 for all datasets.
- Higher `n_samples` or longer responses increase runtime significantly on a single A100.

**Smoke tests**
```bash
modal run modal_eval_general.py --dataset gsm8k --num-problems 5 --n-samples 1 --output-tag smoke
modal run modal_eval_general.py --dataset math  --num-problems 5 --n-samples 1 --max-response-length 2048 --output-tag smoke
modal run modal_eval_general.py --dataset aime  --num-problems 2 --n-samples 2 --max-response-length 2048 --output-tag smoke
```

**Full evaluations**
```bash
modal run modal_eval_general.py --dataset gsm8k --model qwen
modal run modal_eval_general.py --dataset math  --model qwen --subset 500
modal run modal_eval_general.py --dataset aime  --model e3
```
