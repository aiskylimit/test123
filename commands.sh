#1
#collect-and-pull
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate embeddings_hub
sleep 3

CKPT_BASE=/opt/dlami/nvme/smoke_test_outputs_v3
BASELINE_BASE=/opt/dlami/nvme/smoke_test_outputs/baseline
OUT=/opt/dlami/nvme/probe_results
ARMS="V6f_128 V5_mid10 V3_emb V2_emb V2c_tail_emb"
STEPS="1500 3250 5500 6500"

# Copy smoke_metrics.csv per arm
for ARM in $ARMS; do
    if [ -f "$CKPT_BASE/$ARM/smoke_metrics.csv" ]; then
        cp "$CKPT_BASE/$ARM/smoke_metrics.csv" "$OUT/smoke_metrics_${ARM}.csv"
    fi
done

# Copy trainer_state.json per arm per step
for ARM in $ARMS; do
    for S in $STEPS; do
        SRC="$CKPT_BASE/$ARM/checkpoint-$S/trainer_state.json"
        if [ -f "$SRC" ]; then
            cp "$SRC" "$OUT/trainer_state_${ARM}_step${S}.json"
        fi
    done
done

# Baseline trainer_state
if [ -f "$BASELINE_BASE/checkpoint-6500/trainer_state.json" ]; then
    cp "$BASELINE_BASE/checkpoint-6500/trainer_state.json" "$OUT/trainer_state_baseline_step6500.json"
fi

ls -la $OUT/
echo "done"
