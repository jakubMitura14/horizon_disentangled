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
# Runs each phase in its own torchrun process (matching phased production training)
# to eliminate cross-phase CUDA allocator fragmentation:
#   Phase A: VoComni-H (feature_size=192, 998.9M params) + FSDP FULL_SHARD + compile
#            Full native 3D volume (4, 256, 256, 192), batch=2/gpu (8 global)
#   Phase B: 756M CausalVAE + FSDP FULL_SHARD
#            Full native 4x latent (16, 64, 64, 48)->(256, 256, 192), batch=2/gpu, accum=8
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

COMMON_TORCHRUN="torchrun --nnodes=1 --nproc_per_node=4 --master_addr=${MASTER_ADDR} --rdzv_backend=c10d"

echo ">>> Phase A: VoComni-H (998.9M) on full native (4, 256, 256, 192)"
$COMMON_TORCHRUN --master_port=29501 --rdzv_id="${JOB_ID}_A" --rdzv_endpoint="${MASTER_ADDR}:29501" \
    scripts/training/profile_scaling_benchmark.py \
    --ngpus 4 --phases A \
    --feature-size 192 --vol-shape 4,256,256,192 --batch-a 2 \
    --max-steps 8 --use-fsdp --use-compile \
    --output-dir "$OUTPUT_BASE" --run-label "phase_a_native"

echo ">>> Phase B: 756M CausalVAE on full native (16, 64, 64, 48) -> (256, 256, 192)"
$COMMON_TORCHRUN --master_port=29502 --rdzv_id="${JOB_ID}_B" --rdzv_endpoint="${MASTER_ADDR}:29502" \
    scripts/training/profile_scaling_benchmark.py \
    --ngpus 4 --phases B \
    --latent-input-shape 16,64,64,48 --l2-channels 256 --batch-b 3 --grad-accum-b 4 \
    --max-steps 8 --use-fsdp --no-compile-b \
    --output-dir "$OUTPUT_BASE" --run-label "phase_b_native"

echo ">>> Phase C: Spectral GBM Solver on 4mm high-res (48, 56, 40), batch=40/gpu"
$COMMON_TORCHRUN --master_port=29503 --rdzv_id="${JOB_ID}_C" --rdzv_endpoint="${MASTER_ADDR}:29503" \
    scripts/training/profile_scaling_benchmark.py \
    --ngpus 4 --phases C \
    --pde-shape 48,56,40 --dx-mm 4.0 --batch-c 40 \
    --max-steps 8 --use-spectral \
    --output-dir "$OUTPUT_BASE" --run-label "phase_c_native"

echo ""
echo "✓ Zero-cropping max-occupancy benchmark complete!"
cat "$OUTPUT_BASE/phase_a_native/scaling_results.json" || true
cat "$OUTPUT_BASE/phase_b_native/scaling_results.json" || true
cat "$OUTPUT_BASE/phase_c_native/scaling_results.json" || true
