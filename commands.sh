#1
#train-baseline
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate embeddings_hub
sleep 3
export WANDB_MODE=offline
export NCCL_NVLS_ENABLE=0
bash scripts/train_qwen3_0.6b_baseline.sh
