#1
#train-v3-variants

# Kill any leftover training processes
pkill -f "smoke_train" || true
pkill -f "accelerate" || true

pkill -9 -f "smoke_train" || true
pkill -9 -f "accelerate" || true
sleep 300

echo "=== Verify accelerate config ==="
diff ~/.cache/huggingface/accelerate/default_config.yaml resources/accelerate_config.yaml && echo "Config matches" || echo "CONFIG MISMATCH — re-copying"
diff ~/.cache/huggingface/accelerate/default_config.yaml resources/accelerate_config.yaml > /dev/null 2>&1 || cp resources/accelerate_config.yaml ~/.cache/huggingface/accelerate/default_config.yaml

echo "=== GPU status before training ==="
nvidia-smi
sleep 3

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate embeddings_hub
sleep 3
export WANDB_MODE=offline

python run_smoke_tests_v3.py --arms V6f_128 V5_mid10 V3_emb V2_emb --save-token-ids --stop-at-step 6500
