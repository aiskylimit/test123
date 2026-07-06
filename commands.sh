#1 +60+a
#probe-linear-ablation
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate embeddings_hub
sleep 3

nvidia-smi
sleep 3

export WANDB_MODE=offline

python run_probe_tests.py \
    --gpus 0 1 2 3 4 5 6 7 \
    --baseline /opt/dlami/nvme/smoke_test_outputs/baseline/checkpoint-6500 \
    --ckpt-base /opt/dlami/nvme/smoke_test_outputs_v3 \
    --arms linear_ablation \
    --steps 1500 3250 5500 6500 \
    --output-dir /opt/dlami/nvme/probe_results
