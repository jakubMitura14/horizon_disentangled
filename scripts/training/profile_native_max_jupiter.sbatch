#!/bin/bash
#SBATCH --account=numedai-scale
#SBATCH --partition=booster
#SBATCH --job-name=horizon-native-max
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --output=/e/scratch/numedai-scale/mitura1/horizon_logs/native_max_%j.out
#SBATCH --error=/e/scratch/numedai-scale/mitura1/horizon_logs/native_max_%j.err
# ═══════════════════════════════════════════════════════════════════════════════
# Horizon Disentangled — Zero-Cropping & High-Occupancy 4× GH200 Benchmark
#
# Eliminates sub-volume cropping and saturates 95 GB GH200 VRAM:
#   Phase A: VoComni-H (feature_size=192, ~1.03B params) + FSDP FULL_SHARD
#            Full native 3D volume (4, 256, 256, 192), batch=2/gpu (8 global)
#   Phase B: 756M CausalVAE + FSDP FULL_SHARD
#            Full native 4x latent (16, 64, 64, 48), batch=10/gpu (40 global)
#   Phase C: Spectral cuFFT GBM Solver on 4mm high-res grid (48, 56, 40)
#            batch=40/gpu (160 trajectories/step across 4 GPUs)
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
OUTPUT_BASE="/e/scratch/numedai-scale/mitura1/horizon_profiling/native_max_${JOB_ID}"
LOG_DIR="/e/scratch/numedai-scale/mitura1/horizon_logs"
mkdir -p "$OUTPUT_BASE" "$LOG_DIR"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29501

echo "════════════════════════════════════════════════════════════════"
echo "Horizon — Zero-Cropping & High-Occupancy 4× GH200 Benchmark"
echo "Job ID:       $JOB_ID"
echo "Node:         $(hostname)"
echo "GPUs:         $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n1 || echo 'N/A')"
echo "Output:       $OUTPUT_BASE"
echo "════════════════════════════════════════════════════════════════"

unset CUDA_VISIBLE_DEVICES

torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --rdzv_id="$JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    scripts/training/profile_scaling_benchmark.py \
    --ngpus 4 \
    --phases A,B,C \
    --feature-size 192 \
    --vol-shape 4,256,256,192 \
    --batch-a 2 \
    --latent-input-shape 16,64,64,48 \
    --batch-b 4 \
    --grad-accum-b 4 \
    --no-compile-b \
    --pde-shape 48,56,40 \
    --dx-mm 4.0 \
    --batch-c 40 \
    --max-steps 8 \
    --use-fsdp \
    --use-compile \
    --use-spectral \
    --output-dir "$OUTPUT_BASE" \
    --run-label "4gpu_native_max"

echo ""
echo "✓ Zero-cropping max-occupancy benchmark complete!"
cat "$OUTPUT_BASE/4gpu_native_max/scaling_results.json" || true
