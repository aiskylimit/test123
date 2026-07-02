#1 +60+a
#kill-and-cleanup
pkill -f "smoke_train_v3.py" || true
pkill -f "accelerate" || true
sleep 30

# Remove all V3 output folders
rm -rf /opt/dlami/nvme/smoke_test_outputs_v3/V6f_128
rm -rf /opt/dlami/nvme/smoke_test_outputs_v3/V6f_128.log
rm -rf /opt/dlami/nvme/smoke_test_outputs_v3/V5_mid10
rm -rf /opt/dlami/nvme/smoke_test_outputs_v3/V5_mid10.log
rm -rf /opt/dlami/nvme/smoke_test_outputs_v3/V3_emb
rm -rf /opt/dlami/nvme/smoke_test_outputs_v3/V3_emb.log
rm -rf /opt/dlami/nvme/smoke_test_outputs_v3/V2_emb
rm -rf /opt/dlami/nvme/smoke_test_outputs_v3/V2_emb.log

# Remove HF cache (downloaded models, datasets cache, tokenizer cache)
# Keep accelerate config — re-copy it after
cp ~/.cache/huggingface/accelerate/default_config.yaml /tmp/accelerate_config_backup.yaml 2>/dev/null
rm -rf ~/.cache/huggingface
mkdir -p ~/.cache/huggingface/accelerate
cp /tmp/accelerate_config_backup.yaml ~/.cache/huggingface/accelerate/default_config.yaml 2>/dev/null

# Show dataset files BEFORE cleanup
echo "=== Dataset files BEFORE cleanup ==="
for d in /opt/dlami/nvme/embhub_data/Qwen_Qwen3-0.6B/train/*/; do
  echo "--- $d ---"
  ls "$d" | head -20
  echo "total: $(ls "$d" | wc -l) files"
done

# Remove tokenization/grouping cache (cache-*.arrow, NOT data-*.arrow)
find /opt/dlami/nvme/embhub_data/ -name "cache-*.arrow" -delete 2>/dev/null
find /opt/dlami/nvme/embhub_data/ -name "tmp*" -type f -delete 2>/dev/null

# Show dataset files AFTER cleanup
echo ""
echo "=== Dataset files AFTER cleanup ==="
for d in /opt/dlami/nvme/embhub_data/Qwen_Qwen3-0.6B/train/*/; do
  echo "--- $d ---"
  ls "$d" | head -20
  echo "total: $(ls "$d" | wc -l) files"
done

echo "=== Cleaned ==="
ls /opt/dlami/nvme/smoke_test_outputs_v3/ 2>/dev/null || echo "Output dir empty or removed"
nvidia-smi
