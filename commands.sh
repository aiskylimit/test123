#1
#run-eval-tests
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
    --arms V6f_128 V5_mid10 V3_emb V2_emb V2c_tail_emb \
    --steps 1500 3250 5500 6500 \
    --output-dir /opt/dlami/nvme/probe_results
