#!/bin/bash
#SBATCH --account=numedai-scale
#SBATCH --partition=booster
#SBATCH --job-name=horizon-push-v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --output=/e/scratch/numedai-scale/mitura1/horizon_logs/push_v2_%j.out
#SBATCH --error=/e/scratch/numedai-scale/mitura1/horizon_logs/push_v2_%j.err
# ═══════════════════════════════════════════════════════════════════════════════
# Push Phase B & C occupancy with architectural improvements:
#
#   Phase B improvements:
#     - Depthwise-separable final_conv: im2col workspace 20.25 GiB → 0.63 GiB
#     - Gradient checkpointing on decoder stages
#     - Now try l2_channels=512 (original arch) with higher batch
#     - Sweep: l2=512 batch=6 → batch=4 → l2=256 batch=8 → batch=6
#
#   Phase C: batch=64 (confirmed working, 90.39 GB / 95.1%)
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

module purge 2>/dev/null || true
module load Stages/2026 2>/dev/null || true
module load GCCcore/14.3.0 2>/dev/null || true
module load Python 2>/dev/null || true

REPO="/e/project1/numedai-scale/mitura1/horizon_disentangled"
cd "$REPO"

if [ -f "/e/project1/numedai-scale/mitura1/venvs/activate_lymph.sh" ]; then
    source /e/project1/numedai-scale/mitura1/venvs/activate_lymph.sh
fi

export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-72}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

JOB_ID="${SLURM_JOB_ID:-manual_$(date +%s)}"
OUTPUT_BASE="/e/scratch/numedai-scale/mitura1/horizon_profiling/push_v2_${JOB_ID}"
mkdir -p "$OUTPUT_BASE"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
unset CUDA_VISIBLE_DEVICES

COMMON_TORCHRUN="torchrun --nnodes=1 --nproc_per_node=4 --master_addr=${MASTER_ADDR} --rdzv_backend=c10d"
BENCH="scripts/training/profile_scaling_benchmark.py"

echo "════════════════════════════════════════════════════════════════"
echo "Horizon — Push Occupancy v2 (DW-sep conv + grad checkpoint)"
echo "Job ID:       $JOB_ID"
echo "Node:         $(hostname)"
echo "GPUs:         $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n1 || echo 'N/A')"
echo "Output:       $OUTPUT_BASE"
echo "════════════════════════════════════════════════════════════════"

# ─── Phase B: l2_channels=512 (original arch), DW-sep + grad ckpt ────────────
# Try batch=6 → 4 with l2=512 (im2col now ~1.26 GiB instead of 40.5 GiB)
PORT=29501
B_DONE=0

for BATCH in 6 4; do
    PORT=$((PORT + 1))
    echo ""
    echo ">>> Phase B: l2=512 batch=${BATCH}/gpu (DW-sep + grad checkpoint)"
    set +e
    $COMMON_TORCHRUN --master_port=$PORT --rdzv_id="${JOB_ID}_B512_b${BATCH}" --rdzv_endpoint="${MASTER_ADDR}:${PORT}" \
        $BENCH --ngpus 4 --phases B \
        --latent-input-shape 16,64,64,48 --l2-channels 512 --batch-b $BATCH --grad-accum-b 4 \
        --max-steps 8 --use-fsdp --no-compile-b \
        --output-dir "$OUTPUT_BASE" --run-label "phase_b_l512_b${BATCH}"
    EXIT=$?
    set -euo pipefail
    if [ $EXIT -eq 0 ]; then
        echo ">>> Phase B l2=512 batch=${BATCH} SUCCEEDED"
        B_DONE=1
        break
    fi
    echo ">>> Phase B l2=512 batch=${BATCH} failed, trying next..."
done

# If l2=512 all failed, try l2=256 with higher batch
if [ $B_DONE -eq 0 ]; then
    for BATCH in 8 6; do
        PORT=$((PORT + 1))
        echo ""
        echo ">>> Phase B: l2=256 batch=${BATCH}/gpu (DW-sep + grad checkpoint)"
        set +e
        $COMMON_TORCHRUN --master_port=$PORT --rdzv_id="${JOB_ID}_B256_b${BATCH}" --rdzv_endpoint="${MASTER_ADDR}:${PORT}" \
            $BENCH --ngpus 4 --phases B \
            --latent-input-shape 16,64,64,48 --l2-channels 256 --batch-b $BATCH --grad-accum-b 4 \
            --max-steps 8 --use-fsdp --no-compile-b \
            --output-dir "$OUTPUT_BASE" --run-label "phase_b_l256_b${BATCH}"
        EXIT=$?
        set -euo pipefail
        if [ $EXIT -eq 0 ]; then
            echo ">>> Phase B l2=256 batch=${BATCH} SUCCEEDED"
            break
        fi
        echo ">>> Phase B l2=256 batch=${BATCH} failed, trying next..."
    done
fi

# ─── Phase C: confirmed batch=64 (90.39 GB / 95.1%) ─────────────────────────
PORT=$((PORT + 1))
echo ""
echo ">>> Phase C: Spectral GBM batch=64/gpu on 4mm (48,56,40) — 95.1% occupancy"
set +e
$COMMON_TORCHRUN --master_port=$PORT --rdzv_id="${JOB_ID}_C64" --rdzv_endpoint="${MASTER_ADDR}:${PORT}" \
    $BENCH --ngpus 4 --phases C \
    --pde-shape 48,56,40 --dx-mm 4.0 --batch-c 64 \
    --max-steps 8 --use-spectral \
    --output-dir "$OUTPUT_BASE" --run-label "phase_c_batch64"
set -euo pipefail

# ─── Phase C: 2mm ultra-fine grid (96,112,80) ────────────────────────────────
PORT=$((PORT + 1))
echo ""
echo ">>> Phase C: 2mm grid (96,112,80), batch=6/gpu"
set +e
$COMMON_TORCHRUN --master_port=$PORT --rdzv_id="${JOB_ID}_C2mm" --rdzv_endpoint="${MASTER_ADDR}:${PORT}" \
    $BENCH --ngpus 4 --phases C \
    --pde-shape 96,112,80 --dx-mm 2.0 --batch-c 6 \
    --max-steps 8 --use-spectral \
    --output-dir "$OUTPUT_BASE" --run-label "phase_c_2mm_b6"
C2MM_EXIT=$?
set -euo pipefail

if [ $C2MM_EXIT -ne 0 ]; then
    PORT=$((PORT + 1))
    echo ">>> Phase C 2mm batch=6 failed, trying batch=4..."
    set +e
    $COMMON_TORCHRUN --master_port=$PORT --rdzv_id="${JOB_ID}_C2mm4" --rdzv_endpoint="${MASTER_ADDR}:${PORT}" \
        $BENCH --ngpus 4 --phases C \
        --pde-shape 96,112,80 --dx-mm 2.0 --batch-c 4 \
        --max-steps 8 --use-spectral \
        --output-dir "$OUTPUT_BASE" --run-label "phase_c_2mm_b4"
    set -euo pipefail
fi

echo ""
echo "✓ Push-occupancy v2 benchmark complete!"
echo "Results in: $OUTPUT_BASE"
for f in "$OUTPUT_BASE"/*/scaling_results.json; do
    echo "--- $(basename $(dirname $f)) ---"
    cat "$f" 2>/dev/null || true
    echo
done
