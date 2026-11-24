#!/bin/bash
#SBATCH --account=numedai-scale
#SBATCH --partition=booster
#SBATCH --job-name=horizon-high-occ
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --output=/e/scratch/numedai-scale/mitura1/horizon_logs/high_occ_%j.out
#SBATCH --error=/e/scratch/numedai-scale/mitura1/horizon_logs/high_occ_%j.err
# ═══════════════════════════════════════════════════════════════════════════
# Horizon Disentangled — High-Occupancy Scalability Profiling (JUPITER GH200)
#
# Pushes GPU occupancy to >70% by:
#   Phase A: Heavy 3D encoder (SwinUNETR-class), batch=16, 128³ volumes
#   Phase B: 756M VAE (vae_channels=256,512,1024), batch=32, grad_accum=2
#   Phase C: ODE solver batch=48, 4 integration steps
#
# Previous baseline (Job 3097317):
#   A: 0.09 GB (0.1%)  B: 38.15 GB (40%)  C: 16.03 GB (17%)
#
# Target:
#   A: 15–25 GB (~20%)  B: 70–85 GB (~80%)  C: 30–50 GB (~40%)
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ─── JUPITER Lmod ────────────────────────────────────────────────────────
module purge 2>/dev/null || true
module load Stages/2026 2>/dev/null || true
module load GCCcore/14.3.0 2>/dev/null || true
module load Python 2>/dev/null || true

# ─── Environment ─────────────────────────────────────────────────────────
REPO="/e/project1/numedai-scale/mitura1/horizon_disentangled"
cd "$REPO"

if [ -f "/e/project1/numedai-scale/mitura1/venvs/activate_lymph.sh" ]; then
    source /e/project1/numedai-scale/mitura1/venvs/activate_lymph.sh
fi

# ─── CUDA / PyTorch Optimization ────────────────────────────────────────
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-72}
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# ─── Directories ────────────────────────────────────────────────────────
JOB_ID="${SLURM_JOB_ID:-manual_$(date +%s)}"
OUTPUT_DIR="/e/scratch/numedai-scale/mitura1/horizon_profiling/high_occ_${JOB_ID}"
LOG_DIR="/e/scratch/numedai-scale/mitura1/horizon_logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# ─── Data ───────────────────────────────────────────────────────────────
HDF5_DIR="/e/project1/numedai-scale/mitura1/horizon_data/hdf5"
SCRATCH_HDF5="/e/fscratch/numedai-scale/mitura1/horizon_hdf5_${JOB_ID}"

echo "════════════════════════════════════════════════════════════"
echo "Horizon — High-Occupancy Scalability Profiling (JUPITER)"
echo "Job ID:    $JOB_ID"
echo "Node:      $(hostname)"
echo "GPUs:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n1 || echo 'N/A')"
echo "Output:    $OUTPUT_DIR"
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "════════════════════════════════════════════════════════════"

# ─── Data Staging ────────────────────────────────────────────────────────
if [ -d "$HDF5_DIR" ] && ls "$HDF5_DIR"/*.h5 1>/dev/null 2>&1; then
    echo ">>> Staging HDF5 to NVMe scratch..."
    mkdir -p "$SCRATCH_HDF5"
    cp -u "$HDF5_DIR"/*.h5 "$SCRATCH_HDF5/" 2>/dev/null || true
    DATA_DIR="$SCRATCH_HDF5"
    echo "✓ Staged $(du -sh "$SCRATCH_HDF5" | cut -f1) to NVMe."
else
    DATA_DIR="$HDF5_DIR"
fi

# ─── GPU 0: High-Occupancy Profiling ────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0

echo ""
echo ">>> Launching High-Occupancy Profiler..."
echo "    Phase A: Heavy encoder, batch=16, vol=(4,128,128,64)"
echo "    Phase B: 756M VAE, batch=32, grad_accum=2, eff_batch=64"
echo "    Phase C: ODE solver, batch=24"
echo ""

python3 scripts/training/profile_high_occupancy.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --profile-type torch \
    --phases A,B,C \
    --batch-a 16 \
    --vol-shape 4,128,128,64 \
    --batch-b 32 \
    --grad-accum-b 2 \
    --vae-scale heavy \
    --batch-c 24 \
    --max-steps 10 \
    --num-workers 8

# ─── Summary ─────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "High-Occupancy Profiling Complete!"
echo "════════════════════════════════════════════════════════════"
if [ -f "$OUTPUT_DIR/profile_summary_high_occ.json" ]; then
    echo "Profile Summary:"
    cat "$OUTPUT_DIR/profile_summary_high_occ.json"
else
    echo "WARNING: No summary file found."
fi
echo ""
echo "Artifacts:"
ls -lh "$OUTPUT_DIR/"
echo ""
echo "View traces in Perfetto: https://ui.perfetto.dev"
echo "════════════════════════════════════════════════════════════"

# ─── Cleanup scratch ────────────────────────────────────────────────────
if [ -d "$SCRATCH_HDF5" ]; then
    echo "Cleaning up scratch staging..."
    rm -rf "$SCRATCH_HDF5"
fi

echo "Job finished successfully at $(date)"
