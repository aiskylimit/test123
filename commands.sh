#1
#train-v3-variants
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate embeddings_hub
sleep 3
export WANDB_MODE=offline

nvidia-smi
sleep 3

python run_smoke_tests_v3.py --arms V6f_128 V5_mid10 V3_emb V2_emb --save-token-ids --stop-at-step 6500
