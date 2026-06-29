#1
#verify-results
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate embeddings_hub
sleep 3

echo "=== Training configs ==="

echo ""
echo "--- Baseline train_config.json ---"
python3 -c "
import json
with open('/opt/dlami/nvme/smoke_test_outputs/baseline/checkpoint-6500/train_config.json') as f:
    d = json.load(f)
print('no_embhub:', d['embhub']['no_embhub'])
print('seed:', d['training']['seed'])
print('lr:', d['training']['learning_rate'])
print('batch:', d['training']['per_device_train_batch_size'])
print('grad_accum:', d['training']['gradient_accumulation_steps'])
print('block_size:', d['data']['block_size'])
"

echo ""
echo "--- S3_a015 train_config.json ---"
python3 -c "
import json
with open('/opt/dlami/nvme/smoke_test_outputs/S3_a015/checkpoint-6500/train_config.json') as f:
    d = json.load(f)
print('no_embhub:', d['embhub']['no_embhub'])
print('alpha:', d['embhub']['alpha'])
print('scale_init:', d['embhub']['scale_init'])
print('scale_lr_mult:', d['embhub']['scale_lr_mult'])
print('scale_no_wd:', d['embhub']['scale_no_wd'])
print('seed:', d['training']['seed'])
print('lr:', d['training']['learning_rate'])
print('batch:', d['training']['per_device_train_batch_size'])
print('grad_accum:', d['training']['gradient_accumulation_steps'])
"

echo ""
echo "--- S3_a02 train_config.json ---"
python3 -c "
import json
with open('/opt/dlami/nvme/smoke_test_outputs/S3_a02/checkpoint-6500/train_config.json') as f:
    d = json.load(f)
print('no_embhub:', d['embhub']['no_embhub'])
print('alpha:', d['embhub']['alpha'])
print('scale_init:', d['embhub']['scale_init'])
print('scale_lr_mult:', d['embhub']['scale_lr_mult'])
print('scale_no_wd:', d['embhub']['scale_no_wd'])
print('seed:', d['training']['seed'])
print('lr:', d['training']['learning_rate'])
print('batch:', d['training']['per_device_train_batch_size'])
print('grad_accum:', d['training']['gradient_accumulation_steps'])
"

echo ""
echo "=== Verify probe results match local copies ==="

echo ""
echo "--- Baseline probe step 6500 ---"
grep -A8 "Test B" /opt/dlami/nvme/smoke_test_outputs/baseline/probe2_single_step6500_RESULTS.md | grep "| 0.00"

echo ""
echo "--- S3_a015 probe step 6500 ---"
grep -A8 "Test B" /opt/dlami/nvme/smoke_test_outputs/S3_a015/probe2_single_step6500_RESULTS.md | grep "| 0.00"

echo ""
echo "--- S3_a02 probe step 6500 ---"
grep -A8 "Test B" /opt/dlami/nvme/smoke_test_outputs/S3_a02/probe2_single_step6500_RESULTS.md | grep "| 0.00"

echo ""
echo "=== Loss at step 6500 ==="

for ARM in baseline S3_a015 S3_a02; do
  python3 -c "
import json
with open('/opt/dlami/nvme/smoke_test_outputs/${ARM}/checkpoint-6500/trainer_state.json') as f:
    d = json.load(f)
logs = [l for l in d['log_history'] if 'loss' in l]
closest = min(logs, key=lambda l: abs(l['step'] - 6500))
print(f'${ARM} step {closest[\"step\"]}: loss={closest[\"loss\"]:.4f}')
"
done

echo ""
echo "=== MD5 of probe result files ==="
md5sum /opt/dlami/nvme/smoke_test_outputs/baseline/probe2_single_step6500_RESULTS.md
md5sum /opt/dlami/nvme/smoke_test_outputs/S3_a015/probe2_single_step6500_RESULTS.md
md5sum /opt/dlami/nvme/smoke_test_outputs/S3_a02/probe2_single_step6500_RESULTS.md
