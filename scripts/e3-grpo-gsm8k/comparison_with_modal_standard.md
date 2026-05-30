# Comparison: Our e3 GSM8K Setup vs. Modal's "Standard" GRPO+verl Example

## Background

The Modal AI documentation includes a tutorial ([grpo_verl](https://modal.com/docs/examples/grpo_verl)) demonstrating how to train a small Qwen model on GSM8K using GRPO and the verl framework. This tutorial represents the "standard" or "vanilla" approach: it uses off-the-shelf hyperparameters, a small base model (Qwen2-0.5B), and a straightforward reward function with no curriculum or special training tricks. Our project, by contrast, adapts the **e3 training recipe** (learning to explore with negative gradients and curriculum training) to GSM8K on the larger Qwen3-1.7B model, and includes a data-augmentation track with prepended trivia noise. Comparing these two setups is instructive because it highlights exactly which ingredients in our recipe are genuinely novel versus which are inherited from the standard verl+GRPO pipeline. This document catalogs the similarities and differences in detail, with a particular focus on the reward scoring logic.

---

## What's the Same

| Aspect | Modal Example | Our Script (`scripts/grpo/grpo_gsm8k_a100.sh`) |
|--------|-------------|-----------------------------------|
| **Framework** | `verl.trainer.main_ppo` with `algorithm.adv_estimator=grpo` | Same |
| **Dataset** | GSM8K | Same |
| **Algorithm** | GRPO (Group Relative Policy Optimization) | Same |
| **KL loss** | Enabled (`use_kl_loss=True`, `kl_loss_coef=0.001`, `low_var_kl`) | Same |
| **Learning rate** | `1e-6` | Same |
| **Response length** | `1024` | Same |
| **Reward signal** | Binary correct/incorrect (1.0 or 0.0) | Same |
| **Format enforcement** | Requires `#### N` in the response | Same |
| **Modal deployment** | Runs on Modal GPU with Volume for data/checkpoints | Same |

---

## What's Different

| Aspect | Modal Example | Our Script |
|--------|-------------|-------------|
| **Base model** | `Qwen/Qwen2-0.5B` (~0.5B params) | `Qwen/Qwen3-1.7B` (~1.7B params) |
| **GPU** | `H100:2` (2 GPUs) | `A100-80GB:1` (1 GPU) |
| **Train batch size** | `128` | `64` |
| **PPO mini batch** | `128` | `32` |
| **Max prompt length** | `64` | `512` |
| **Rollout samples (`n`)** | `5` per prompt | `8` per prompt |
| **Validation samples** | Not specified | `4` per prompt |
| **Temperature** | Not specified | `0.6` |
| **Remove padding** | `False` | `True` |
| **Tensor model parallel** | `2` | `1` |
| **GPU memory util** | `0.4` | `0.6` |
| **Entropy coefficient** | `0` | `0.001` |
| **Total steps** | `1` (smoke test) | `400`–`500` |
| **PPO clipping** | Default symmetric (`0.2`) | Asymmetric (`low=0.2`, `high=0.5`) |
| **Negative gradients** | Default behavior | Explicitly kept (`only_train_on_positive=False`) |
| **Curriculum** | None | Two-stage (easy→hard) |

---

## Scoring: The Key Difference

### Modal Blog's Scorer

The Modal example copies the **standard verl GSM8K scorer** verbatim into their script. It has two modes:

**`strict` mode** — looks for the exact `#### N` pattern:

```python
def extract_solution(solution_str, method='strict'):
    if method == "strict":
        # This also tests the formatting of the model
        solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if solution is None:
            final_answer = None
        else:
            final_answer = solution.group(0)
            final_answer = final_answer.split("#### ")[1].replace(",", "").replace("$", "")
```

**`flexible` mode** — looks for the last number anywhere in the response:

```python
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        if len(answer) == 0:
            pass  # no reward
        else:
            invalid_str = ['', '.']
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
```

The Modal blog **only uses `strict` mode** in their `compute_reward`:

```python
def compute_reward(data_source, solution_str, ground_truth, extra_info):
    answer = extract_solution(solution_str=solution_str, method="strict")
    if answer is None:
        return 0.0
    else:
        if answer == ground_truth:
            return 1.0
        else:
            return 0.0
```

This is a **simple binary reward**: either the model outputs `#### {correct_number}` and gets `1.0`, or it doesn't and gets `0.0`.

### Our Scorer (`gsm8k_custom.py`)

Our script uses `gsm8k_custom.py`, which is a **thin adapter** around the same underlying verl scorer:

```python
# verl/utils/reward_score/gsm8k_custom.py
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    return _gsm8k_compute_score(
        solution_str=solution_str,
        ground_truth=str(ground_truth),
        method="strict",
        format_score=0.0,
        score=1.0,
    )
```

On the surface this looks identical: both use `strict` mode, both return `1.0` for correct and `0.0` for wrong. But there are two subtle differences:

#### Difference 1: The `format_score` parameter

The underlying verl `compute_score` has a `format_score` argument:

```python
# verl/utils/reward_score/gsm8k.py
def compute_score(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return score
        else:
            return format_score
```

The Modal blog's `compute_reward` hardcodes `0.0` for both the "no answer" case and the "wrong answer" case. Our `gsm8k_custom.py` explicitly passes `format_score=0.0`, which means the same thing: **no partial credit for formatting**. But the verl framework *supports* giving partial credit (e.g., `format_score=0.5` for getting the format right even if the number is wrong). Our script explicitly pins this to `0.0`.

#### Difference 2: Why our scorer exists at all

Our `gsm8k_custom.py` wasn't created for scoring logic differences — it was created for **interface compatibility**. The verl framework expects:

```python
compute_score(data_source, solution_str, ground_truth, extra_info)
```

But the original verl `gsm8k.compute_score` signature is:

```python
compute_score(solution_str, ground_truth, method='strict', format_score=0., score=1.)
```

Our custom wrapper adapts the call signature. More importantly, the comment in the file reveals the deeper reason:

```python
"""
This module is a thin adapter that hard-codes method='strict' so the reward
signal cannot be satisfied by random numbers leaking in from prepended trivia
facts (Track B). The model MUST emit "#### N" to earn reward.
"""
```

So while both setups use `strict` mode, **our setup was specifically designed to be robust against noise-augmented training**. If you used `flexible` mode with Track B (where trivia facts like "Octopuses have three hearts" are prepended), the model could accidentally get reward by echoing the number `3` from the trivia. The `strict` mode prevents this because the model must explicitly terminate with `#### N`.

The Modal example doesn't have this concern because they don't use trivia-augmented data. Their copied scorer is functionally equivalent to ours for a "clean" training run, but ours is architecturally hardened against the mixed-data experiment.

---

## Edge Case Handling

Both scorers handle the same edge cases identically:

| Edge case | How both handle it |
|-----------|-------------------|
| No `####` found | Returns `None` → reward `0.0` |
| `####` found but no number after it | Regex doesn't match → reward `0.0` |
| Number has commas (e.g., `#### 1,234`) | Strips commas → compares `1234` |
| Number has `$` (e.g., `#### $50`) | Strips `$` → compares `50` |
| Model outputs multiple `####` lines | Only the **first** match matters (regex `search`, not `findall`) |
| Negative numbers (e.g., `#### -42`) | Supported: regex includes `\\-?` |
| Decimal numbers (e.g., `#### 3.14`) | Supported: regex includes `\\.` |
| Wrong number after `####` | Returns the extracted number → compares to ground truth → reward `0.0` |

One subtle gotcha: the regex `\\-?[0-9\\.\\,]+` will greedily match things like `3.14.15` as a single token. Both scorers pass this through as-is and compare string equality against the ground truth, so if the ground truth is `3.14` but the model says `3.14.15`, it gets `0.0`.

---

## Bottom Line

For a **clean GSM8K run** (no trivia augmentation), the Modal example and our script produce **identical rewards**. The differences are all in the training recipe (model size, clip ratios, negative gradients, curriculum) and infrastructure (1 vs 2 GPUs, batch sizes). The scoring itself is the same binary `#### N` matcher.

Where they diverge is **intent**: the Modal example is a minimal getting-started tutorial. Our script is a research-grade setup that hardcodes `strict` scoring specifically to make the mixed-data Track D experiment valid.
