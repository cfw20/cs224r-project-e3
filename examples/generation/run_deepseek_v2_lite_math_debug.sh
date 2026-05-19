set -x

echo "=== Diagnostic checks ==="
which python3
python3 -c "import sys; print(sys.executable)"
python3 -m pip list
which pip
python3 -c "import verl; print('verl found at:', verl.__file__)" || echo "verl NOT found"
echo "=== End diagnostics ==="

data_path=/data/test_math/tiny_math.parquet
save_path=/data/tiny_math_outputs.parquet
model_path=deepseek-ai/deepseek-coder-1.3b-instruct

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=$save_path \
    model.path=$model_path \
    +model.trust_remote_code=True \
    rollout.temperature=0.7 \
    rollout.top_k=50 \
    rollout.top_p=0.9 \
    rollout.prompt_length=512 \
    rollout.response_length=256 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.9 \
    rollout.enforce_eager=True \
    data.batch_size=2
