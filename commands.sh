#1
#collect-and-pull-ablation
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate embeddings_hub
sleep 3

OUT=/opt/dlami/nvme/probe_results
CKPT_BASE=/opt/dlami/nvme/smoke_test_outputs_v3

# Copy smoke metrics
cp "$CKPT_BASE/linear_ablation/smoke_metrics.csv" "$OUT/smoke_metrics_linear_ablation.csv" 2>/dev/null

# Copy trainer_state per step
for S in 1500 3250 5500 6500; do
    SRC="$CKPT_BASE/linear_ablation/checkpoint-$S/trainer_state.json"
    if [ -f "$SRC" ]; then
        cp "$SRC" "$OUT/trainer_state_linear_ablation_step${S}.json"
    fi
done

ls -la $OUT/*linear_ablation*
echo "done"
