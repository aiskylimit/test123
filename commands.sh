#1 +60+a
#train-linear-ablation
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate embeddings_hub
sleep 3

nvidia-smi
sleep 3

export WANDB_MODE=offline

python run_smoke_tests_v3.py --arms linear_ablation --stop-at-step 6500
