#!/bin/bash
# e3-recipe GRPO launcher for the GSM8K curriculum (Track C / Track D), tuned for
# a single A100/H100-80GB on Modal. Parametrized copy of grpo_gsm8k_a100.sh that
# exposes the per-stage token budget and the starting checkpoint via env vars.
#
# Used by modal_train_e3_gsm8k.py for the two curriculum stages:
#   Stage 1: easy split, MAX_RESPONSE_LENGTH=512,  MODEL_PATH=Qwen/Qwen3-1.7B (base)
#   Stage 2: hard split, MAX_RESPONSE_LENGTH=1024, MODEL_PATH=<stage-1 HF checkpoint>
#
# Keeps the e3-specific hyperparameters:
#   clip_ratio_low=0.2, clip_ratio_high=0.5, only_train_on_positive=False
#   custom reward = verl/utils/reward_score/gsm8k_custom.py (strict #### N)
#
# Required env vars:
#   TRAIN_PARQUET            absolute path to training parquet (easy/hard, clean/mixed)
#   VAL_PARQUET              absolute path to clean GSM8K test parquet
#   MODEL_PATH               HF id or local path to starting model (base or stage-1 ckpt)
#   CKPT_DIR                 absolute path under Modal Volume for checkpoints
#   EXPERIMENT_NAME          wandb experiment name
#   TOTAL_STEPS              total training steps
#   MAX_RESPONSE_LENGTH      512 (stage 1) or 1024 (stage 2)
#
# Optional:
#   MAX_EXTRAPOLATION_LENGTH defaults to 2 * MAX_RESPONSE_LENGTH ("train short, eval long")
#   WANDB_PROJECT            defaults to "e3-gsm8k"
#   SAVE_FREQ                defaults to 100
#   TEST_FREQ                defaults to 100

set -euo pipefail

: "${TRAIN_PARQUET:?TRAIN_PARQUET is required}"
: "${VAL_PARQUET:?VAL_PARQUET is required}"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${CKPT_DIR:?CKPT_DIR is required}"
: "${EXPERIMENT_NAME:?EXPERIMENT_NAME is required}"
: "${TOTAL_STEPS:?TOTAL_STEPS is required}"
: "${MAX_RESPONSE_LENGTH:?MAX_RESPONSE_LENGTH is required}"

MAX_EXTRAPOLATION_LENGTH="${MAX_EXTRAPOLATION_LENGTH:-$((MAX_RESPONSE_LENGTH * 2))}"
WANDB_PROJECT="${WANDB_PROJECT:-e3-gsm8k}"
SAVE_FREQ="${SAVE_FREQ:-100}"
TEST_FREQ="${TEST_FREQ:-100}"

echo "[grpo_gsm8k_e3] TRAIN_PARQUET=$TRAIN_PARQUET"
echo "[grpo_gsm8k_e3] VAL_PARQUET=$VAL_PARQUET"
echo "[grpo_gsm8k_e3] MODEL_PATH=$MODEL_PATH"
echo "[grpo_gsm8k_e3] CKPT_DIR=$CKPT_DIR"
echo "[grpo_gsm8k_e3] EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "[grpo_gsm8k_e3] TOTAL_STEPS=$TOTAL_STEPS"
echo "[grpo_gsm8k_e3] MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH"
echo "[grpo_gsm8k_e3] MAX_EXTRAPOLATION_LENGTH=$MAX_EXTRAPOLATION_LENGTH"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_PARQUET" \
    data.val_files="$VAL_PARQUET" \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.max_extrapolation_length="$MAX_EXTRAPOLATION_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    +actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.5 \
    actor_rollout_ref.actor.only_train_on_positive=False \
    actor_rollout_ref.actor.remove_truncated=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=verl/utils/reward_score/gsm8k_custom.py \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.total_training_steps="$TOTAL_STEPS" \
    trainer.total_epochs=20 \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.resume_mode=auto \
    "${@}"
