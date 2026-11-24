#!/bin/bash
#SBATCH --job-name=horizon-profile-1epoch
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/profile_1epoch_%j.out
#SBATCH --error=logs/profile_1epoch_%j.err
# ──────────────────────────────────────────────────────────────────────────────
# Horizon Disentangled — 1-Epoch Multi-Phase Profiling SLURM Job
#
# Profiles Phase A (Vision Encoder), Phase B (199M Causal VAE), and Phase C
# (Vectorised NJDE RK4 dynamics) for exactly 1 epoch using PyTorch Profiler
# (Chrome traces) or NVIDIA Nsight Systems.
#
# Features:
#   1. Automatic local NVMe/SSD data staging ($SLURM_TMPDIR or /tmp)
#   2. Automatic XNAT dataset download if local HDF5 files are not present
#   3. PyTorch Profiler Chrome trace generation (*.json.gz for Perfetto)
#   4. Hardware telemetry (peak VRAM, GPU compute throughput, operator timing)
#
# Submission:
#   sbatch scripts/training/profile_1epoch_all_phases.sbatch
#
# Interactive debugging:
#   srun --partition=gpu --gres=gpu:1 --cpus-per-task=16 --mem=128G --time=01:00:00 --pty bash
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ─── Environment Setup ───────────────────────────────────────────────────────
if command -v module &> /dev/null; then
    module purge || true
    module load cuda/12.4 || module load cuda || true
fi

export CONDA_PREFIX="${CONDA_PREFIX:-/DATA/conda/envs/nukl-to-ct}"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd /DATA/horizon_disentangled
PYTHON="$CONDA_PREFIX/bin/python"

# ─── CUDA & PyTorch Optimization Flags ──────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# ─── Directories & Staging ──────────────────────────────────────────────────
JOB_ID="${SLURM_JOB_ID:-manual_$(date +%s)}"
OUTPUT_DIR="experiments/profiling_1epoch_${JOB_ID}"
mkdir -p "$OUTPUT_DIR" logs

# Local NVMe scratch directory for fast I/O
NODE_SCRATCH="${SLURM_TMPDIR:-/tmp/horizon_hdf5_${JOB_ID}}"
mkdir -p "$NODE_SCRATCH"

echo "========================================================================"
echo "Horizon Disentangled — 1-Epoch Profiling Run"
echo "Job ID:    $JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1)"
echo "Scratch:   $NODE_SCRATCH"
echo "Output:    $OUTPUT_DIR"
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "========================================================================"

# ─── Data Staging ────────────────────────────────────────────────────────────
# 1. Check if shared storage has HDF5 datasets
SHARED_HDF5="/DATA/horizon_disentangled/hdf5"

if [ -f "$SHARED_HDF5/breast_mincrop_s00.h5" ] || [ -f "$SHARED_HDF5/encoder_features.h5" ]; then
    echo ">>> Staging dataset from shared storage ($SHARED_HDF5) to local NVMe ($NODE_SCRATCH)..."
    # Copy essential HDF5 files to local scratch SSD
    cp -u "$SHARED_HDF5"/*.h5 "$NODE_SCRATCH/" 2>/dev/null || true
    echo "✓ Local NVMe staging complete: $(du -sh "$NODE_SCRATCH")"
else
    echo ">>> Local HDF5 not found on shared storage. Downloading from XNAT..."
    $PYTHON scripts/data/xnat_download.py \
        --download-dataset \
        --project temp \
        --resource-label HORIZON_HDF5 \
        --output "$NODE_SCRATCH"
fi

# ─── Profiling Execution ─────────────────────────────────────────────────────
echo ">>> Launching 1-Epoch Profiler across Phases A, B, and C..."

# Mode 1: PyTorch Profiler (Generates Chrome traces for ui.perfetto.dev)
$PYTHON scripts/training/profile_1epoch_all_phases.py \
    --data-dir "$NODE_SCRATCH" \
    --output-dir "$OUTPUT_DIR" \
    --profile-type torch \
    --phases A,B,C \
    --batch-b 32 \
    --batch-c 24 \
    --max-steps 10 \
    --num-workers 4

# Mode 2 (Optional): NVIDIA Nsight Systems CLI Profiler
# To run Nsight Systems instead, uncomment the following block:
# echo ">>> Running under NVIDIA Nsight Systems..."
# nsys profile \
#     -t cuda,nvtx,osrt,cudnn,cublas \
#     -s none \
#     --capture-range=nvtx \
#     --nvtx-capture=PhaseB_Step_2 \
#     -o "$OUTPUT_DIR/nsys_horizon_1epoch" \
#     --force-overwrite true \
#     $PYTHON scripts/training/profile_1epoch_all_phases.py \
#         --data-dir "$NODE_SCRATCH" \
#         --output-dir "$OUTPUT_DIR" \
#         --profile-type nsys \
#         --phases B,C \
#         --batch-b 32 \
#         --batch-c 24 \
#         --max-steps 5

# ─── Summary & Cleanup ───────────────────────────────────────────────────────
echo "========================================================================"
echo "Profiling Complete!"
echo "Profile summary saved to: $OUTPUT_DIR/profile_summary.json"
if [ -f "$OUTPUT_DIR/profile_summary.json" ]; then
    cat "$OUTPUT_DIR/profile_summary.json"
fi
echo "========================================================================"

# Clean up local scratch if created in /tmp
if [[ "$NODE_SCRATCH" == /tmp/* ]]; then
    echo "Cleaning up temporary scratch $NODE_SCRATCH..."
    rm -rf "$NODE_SCRATCH"
fi

echo "Job finished successfully at $(date)"
