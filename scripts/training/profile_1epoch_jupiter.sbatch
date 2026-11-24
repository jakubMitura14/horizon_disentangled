#!/bin/bash
#SBATCH --account=numedai-scale
#SBATCH --partition=booster
#SBATCH --job-name=horizon-profile-1epoch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --output=/e/scratch/numedai-scale/mitura1/horizon_logs/profile_1epoch_%j.out
#SBATCH --error=/e/scratch/numedai-scale/mitura1/horizon_logs/profile_1epoch_%j.err
# ═══════════════════════════════════════════════════════════════════════════
# Horizon Disentangled — 1-Epoch Multi-Phase Profiling on JUPITER Booster
#
# Cluster: JUPITER Booster (FZJ) — 4× NVIDIA GH200 Grace-Hopper per node
# GPU Memory: 95 GB HBM3 per GH200 GPU (120 GB model, ~95 GB usable CUDA)
# Interconnect: NVLink-C2C (900 GB/s bidirectional per GPU)
#
# Profiles Phase A (3D Vision Encoder), Phase B (199M Causal VAE), and
# Phase C (Vectorised NJDE RK4 ODE dynamics) using PyTorch Profiler.
#
# Submission:
#   sbatch scripts/training/profile_1epoch_jupiter.sbatch
#
# Interactive debugging (single-GPU, 60 min):
#   salloc -A numedai-scale -p booster -N 1 --ntasks=1 --cpus-per-task=72 -t 01:00:00
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ─── JUPITER Lmod Module Stack ───────────────────────────────────────────
module purge 2>/dev/null || true
module load Stages/2026 2>/dev/null || true
module load GCCcore/14.3.0 2>/dev/null || true
module load Python 2>/dev/null || true

# ─── Environment Activation ─────────────────────────────────────────────
REPO="/e/project1/numedai-scale/mitura1/horizon_disentangled"
cd "$REPO"

if [ -f "$REPO/activate_jupiter.sh" ]; then
    source "$REPO/activate_jupiter.sh"
elif [ -f "/e/project1/numedai-scale/mitura1/venvs/activate_lymph.sh" ]; then
    source /e/project1/numedai-scale/mitura1/venvs/activate_lymph.sh
fi

# ─── CUDA / PyTorch Optimization ────────────────────────────────────────
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-72}
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# ─── Directories ────────────────────────────────────────────────────────
JOB_ID="${SLURM_JOB_ID:-manual_$(date +%s)}"
OUTPUT_DIR="/e/scratch/numedai-scale/mitura1/horizon_profiling/profile_1epoch_${JOB_ID}"
LOG_DIR="/e/scratch/numedai-scale/mitura1/horizon_logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# ─── Data Location ──────────────────────────────────────────────────────
# Primary: pre-downloaded on project storage
HDF5_DIR="/e/project1/numedai-scale/mitura1/horizon_data/hdf5"

# Fallback: fast scratch staging
SCRATCH_HDF5="/e/fscratch/numedai-scale/mitura1/horizon_hdf5_${JOB_ID}"

echo "════════════════════════════════════════════════════════════"
echo "Horizon Disentangled — 1-Epoch Profiling (JUPITER Booster)"
echo "Job ID:    $JOB_ID"
echo "Node:      $(hostname)"
echo "GPUs:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n1 || echo 'N/A')"
echo "Output:    $OUTPUT_DIR"
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "════════════════════════════════════════════════════════════"

# ─── Data Staging ────────────────────────────────────────────────────────
# If data is on project storage, stage to fast NVMe scratch for I/O perf
if [ -d "$HDF5_DIR" ] && ls "$HDF5_DIR"/*.h5 1>/dev/null 2>&1; then
    echo ">>> Found HDF5 data on project storage ($HDF5_DIR)."
    echo ">>> Staging to fast scratch ($SCRATCH_HDF5) for I/O performance..."
    mkdir -p "$SCRATCH_HDF5"
    cp -u "$HDF5_DIR"/*.h5 "$SCRATCH_HDF5/" 2>/dev/null || true
    DATA_DIR="$SCRATCH_HDF5"
    echo "✓ Staged $(du -sh "$SCRATCH_HDF5" | cut -f1) to NVMe scratch."
else
    echo ">>> No pre-downloaded HDF5 data found."
    echo ">>> Downloading from XNAT directly (this may take ~10 min on first run)..."
    mkdir -p "$SCRATCH_HDF5"
    python3 scripts/data/xnat_download.py \
        --download-dataset \
        --project temp \
        --resource-label HORIZON_HDF5 \
        --output "$SCRATCH_HDF5" \
        --credentials scripts/xnat_credentials.txt
    DATA_DIR="$SCRATCH_HDF5"
fi

# ─── GPU 0 Only: 1-Epoch Profiling Run ──────────────────────────────────
# Pin to single GPU for consistent benchmarking
export CUDA_VISIBLE_DEVICES=0

echo ""
echo ">>> Launching 1-Epoch Profiler (Phases A, B, C)..."
echo "    Data Dir:  $DATA_DIR"
echo "    Output:    $OUTPUT_DIR"
echo ""

python3 scripts/training/profile_1epoch_all_phases.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --profile-type torch \
    --phases A,B,C \
    --batch-b 16 \
    --batch-c 12 \
    --max-steps 10 \
    --num-workers 8

# ─── Summary ─────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "Profiling Complete!"
echo "════════════════════════════════════════════════════════════"
if [ -f "$OUTPUT_DIR/profile_summary.json" ]; then
    echo "Profile Summary:"
    cat "$OUTPUT_DIR/profile_summary.json"
else
    echo "WARNING: No profile_summary.json found. Check logs for errors."
fi
echo ""
echo "Artifacts:"
ls -lh "$OUTPUT_DIR/"
echo ""
echo "View traces in Perfetto: https://ui.perfetto.dev"
echo "  Upload: $OUTPUT_DIR/trace_phase_*.json.gz"
echo "════════════════════════════════════════════════════════════"

# ─── Cleanup scratch if temporary ────────────────────────────────────────
if [ -d "$SCRATCH_HDF5" ]; then
    echo "Cleaning up scratch staging $SCRATCH_HDF5..."
    rm -rf "$SCRATCH_HDF5"
fi

echo "Job finished successfully at $(date)"
