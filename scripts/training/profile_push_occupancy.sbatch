#!/bin/bash
#SBATCH --account=numedai-scale
#SBATCH --partition=booster
#SBATCH --job-name=horizon-push-occ
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --output=/e/scratch/numedai-scale/mitura1/horizon_logs/push_occ_%j.out
#SBATCH --error=/e/scratch/numedai-scale/mitura1/horizon_logs/push_occ_%j.err
# ═══════════════════════════════════════════════════════════════════════════════
# Push Phase B & C occupancy higher on 4× GH200 (95 GB usable)
#
#   Phase B sweeps: batch=6 → 5 (l2_channels=256, FSDP, no-compile)
#   Phase C sweeps: batch=72 → 64 (spectral, 4mm 48×56×40)
#   Phase C bonus:  2mm grid (96,112,80), batch=8  — 8× finer PDE
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
OUTPUT_BASE="/e/scratch/numedai-scale/mitura1/horizon_profiling/push_occ_${JOB_ID}"
LOG_DIR="/e/scratch/numedai-scale/mitura1/horizon_logs"
mkdir -p "$OUTPUT_BASE" "$LOG_DIR"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
unset CUDA_VISIBLE_DEVICES

COMMON_TORCHRUN="torchrun --nnodes=1 --nproc_per_node=4 --master_addr=${MASTER_ADDR} --rdzv_backend=c10d"

echo "════════════════════════════════════════════════════════════════"
echo "Horizon — Push Occupancy Higher (Phase B & C)"
echo "Job ID:       $JOB_ID"
echo "Node:         $(hostname)"
echo "GPUs:         $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n1 || echo 'N/A')"
echo "Output:       $OUTPUT_BASE"
echo "════════════════════════════════════════════════════════════════"

# ─── Phase B: try batch=6, then batch=5 ──────────────────────────────────────
PORT_B=29502
echo ""
echo ">>> Phase B: CausalVAE batch=6/gpu (target ~89 GB, ~93%)"
set +e
$COMMON_TORCHRUN --master_port=$PORT_B --rdzv_id="${JOB_ID}_B6" --rdzv_endpoint="${MASTER_ADDR}:${PORT_B}" \
    scripts/training/profile_scaling_benchmark.py \
    --ngpus 4 --phases B \
    --latent-input-shape 16,64,64,48 --l2-channels 256 --batch-b 6 --grad-accum-b 4 \
    --max-steps 8 --use-fsdp --no-compile-b \
    --output-dir "$OUTPUT_BASE" --run-label "phase_b_batch6"
B6_EXIT=$?
set -euo pipefail

if [ $B6_EXIT -ne 0 ]; then
    echo ">>> Phase B batch=6 failed (likely OOM), trying batch=5..."
    PORT_B=$((PORT_B + 10))
    set +e
    $COMMON_TORCHRUN --master_port=$PORT_B --rdzv_id="${JOB_ID}_B5" --rdzv_endpoint="${MASTER_ADDR}:${PORT_B}" \
        scripts/training/profile_scaling_benchmark.py \
        --ngpus 4 --phases B \
        --latent-input-shape 16,64,64,48 --l2-channels 256 --batch-b 5 --grad-accum-b 4 \
        --max-steps 8 --use-fsdp --no-compile-b \
        --output-dir "$OUTPUT_BASE" --run-label "phase_b_batch5"
    B5_EXIT=$?
    set -euo pipefail
fi

# ─── Phase C 4mm: try batch=72, then batch=64 ────────────────────────────────
PORT_C=29503
echo ""
echo ">>> Phase C: Spectral GBM batch=72/gpu on 4mm (48,56,40) — target ~90 GB"
set +e
$COMMON_TORCHRUN --master_port=$PORT_C --rdzv_id="${JOB_ID}_C72" --rdzv_endpoint="${MASTER_ADDR}:${PORT_C}" \
    scripts/training/profile_scaling_benchmark.py \
    --ngpus 4 --phases C \
    --pde-shape 48,56,40 --dx-mm 4.0 --batch-c 72 \
    --max-steps 8 --use-spectral \
    --output-dir "$OUTPUT_BASE" --run-label "phase_c_batch72"
C72_EXIT=$?
set -euo pipefail

if [ $C72_EXIT -ne 0 ]; then
    echo ">>> Phase C batch=72 failed (likely OOM), trying batch=64..."
    PORT_C=$((PORT_C + 10))
    set +e
    $COMMON_TORCHRUN --master_port=$PORT_C --rdzv_id="${JOB_ID}_C64" --rdzv_endpoint="${MASTER_ADDR}:${PORT_C}" \
        scripts/training/profile_scaling_benchmark.py \
        --ngpus 4 --phases C \
        --pde-shape 48,56,40 --dx-mm 4.0 --batch-c 64 \
        --max-steps 8 --use-spectral \
        --output-dir "$OUTPUT_BASE" --run-label "phase_c_batch64"
    C64_EXIT=$?
    set -euo pipefail
fi

# ─── Phase C 2mm: ultra-fine grid (96,112,80), batch=8 ───────────────────────
PORT_C2=29504
echo ""
echo ">>> Phase C: Spectral GBM 2mm grid (96,112,80), batch=8/gpu — 8× finer PDE"
set +e
$COMMON_TORCHRUN --master_port=$PORT_C2 --rdzv_id="${JOB_ID}_C2mm" --rdzv_endpoint="${MASTER_ADDR}:${PORT_C2}" \
    scripts/training/profile_scaling_benchmark.py \
    --ngpus 4 --phases C \
    --pde-shape 96,112,80 --dx-mm 2.0 --batch-c 8 \
    --max-steps 8 --use-spectral \
    --output-dir "$OUTPUT_BASE" --run-label "phase_c_2mm"
C2MM_EXIT=$?
set -euo pipefail

if [ $C2MM_EXIT -ne 0 ]; then
    echo ">>> Phase C 2mm batch=8 failed (likely OOM), trying batch=4..."
    PORT_C2=$((PORT_C2 + 10))
    set +e
    $COMMON_TORCHRUN --master_port=$PORT_C2 --rdzv_id="${JOB_ID}_C2mm4" --rdzv_endpoint="${MASTER_ADDR}:${PORT_C2}" \
        scripts/training/profile_scaling_benchmark.py \
        --ngpus 4 --phases C \
        --pde-shape 96,112,80 --dx-mm 2.0 --batch-c 4 \
        --max-steps 8 --use-spectral \
        --output-dir "$OUTPUT_BASE" --run-label "phase_c_2mm_b4"
    set -euo pipefail
fi

echo ""
echo "✓ Push-occupancy benchmark complete!"
echo "Results in: $OUTPUT_BASE"
ls -la "$OUTPUT_BASE"/*/scaling_results.json 2>/dev/null || true
for f in "$OUTPUT_BASE"/*/scaling_results.json; do
    echo "--- $(dirname $f | xargs basename) ---"
    cat "$f" 2>/dev/null || true
done
