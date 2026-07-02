#1
#kill-and-cleanup
# Kill any running training processes
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

echo "=== Cleaned ==="
ls /opt/dlami/nvme/smoke_test_outputs_v3/ 2>/dev/null || echo "Output dir empty or removed"
nvidia-smi
