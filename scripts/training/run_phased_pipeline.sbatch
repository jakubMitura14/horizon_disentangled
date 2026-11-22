#!/bin/bash
#SBATCH --job-name=horizon-phased
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --output=logs/phased_%j_%a.out
#SBATCH --error=logs/phased_%j_%a.err
#SBATCH --array=0-2
# ──────────────────────────────────────────────────────────────────────────────
# Horizon Disentangled — 3-Phase Staged Pipeline (H200 Cluster)
#
# SLURM array job: index 0=Phase A, 1=Phase B, 2=Phase C.
# Submit with dependency chain for sequential execution:
#
#   # Phase A
#   JOB_A=$(sbatch --array=0 scripts/training/run_phased_pipeline.sbatch | awk '{print $4}')
#   # Phase B (after A)
#   JOB_B=$(sbatch --array=1 --dependency=afterok:$JOB_A scripts/training/run_phased_pipeline.sbatch | awk '{print $4}')
#   # Phase C (after B)
#   sbatch --array=2 --dependency=afterok:$JOB_B scripts/training/run_phased_pipeline.sbatch
#
# Or run all three sequentially in a single allocation:
#   sbatch --array=0-2%1 scripts/training/run_phased_pipeline.sbatch
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ─── Environment ─────────────────────────────────────────────────────────────
module purge
module load cuda/12.4 nccl/2.22

export CONDA_PREFIX="${CONDA_PREFIX:-/DATA/conda/envs/nukl-to-ct}"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd /DATA/horizon_disentangled
PYTHON="$CONDA_PREFIX/bin/python"

# ─── NCCL / Communication tuning for H200 NVLink 4 ──────────────────────────
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=NVL
export NCCL_SOCKET_IFNAME=ib0
export NCCL_ALGO=Ring,Tree
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL

# ─── PyTorch tuning ─────────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ─── Paths ───────────────────────────────────────────────────────────────────
CONFIG="configs/phased_pipeline.json"
H5_DIR="hdf5"
CKPT_DIR="experiments/checkpoints"
LOG_DIR="logs"
mkdir -p "$H5_DIR" "$CKPT_DIR" "$LOG_DIR"

NGPUS=$SLURM_NTASKS_PER_NODE
PHASE=$SLURM_ARRAY_TASK_ID

# ─── Master address for torchrun ─────────────────────────────────────────────
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID))

echo "============================================================"
echo "Job $SLURM_JOB_ID | Phase $PHASE | $(date)"
echo "Node: $(hostname) | GPUs: $NGPUS | Conda: $CONDA_PREFIX"
echo "MASTER_ADDR=$MASTER_ADDR:$MASTER_PORT"
echo "============================================================"

# ─── GPU health check ───────────────────────────────────────────────────────
nvidia-smi --query-gpu=index,name,memory.total,memory.free,temperature.gpu \
           --format=csv,noheader

case $PHASE in
  0)
    echo ">>> Phase A: Frozen Encoder Feature Extraction → HDF5"
    srun torchrun \
      --nnodes=1 \
      --nproc_per_node=$NGPUS \
      --rdzv_backend=c10d \
      --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
      -m src.pipeline.phase_a_encode \
        --config "$CONFIG" \
        --output "$H5_DIR/encoder_features.h5" \
      2>&1 | tee "$LOG_DIR/phase_a_${SLURM_JOB_ID}.log"
    echo "Phase A complete: $(du -h $H5_DIR/encoder_features.h5)"
    ;;

  1)
    echo ">>> Phase B: VAE Level 2 Disentanglement Training"
    srun torchrun \
      --nnodes=1 \
      --nproc_per_node=$NGPUS \
      --rdzv_backend=c10d \
      --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
      -m src.pipeline.phase_b_latents \
        --config "$CONFIG" \
        --encoder-features "$H5_DIR/encoder_features.h5" \
        --output "$H5_DIR/latent_decomposition.h5" \
        --checkpoint-dir "$CKPT_DIR" \
      2>&1 | tee "$LOG_DIR/phase_b_${SLURM_JOB_ID}.log"
    echo "Phase B complete: $(du -h $H5_DIR/latent_decomposition.h5)"
    ;;

  2)
    echo ">>> Phase C: NJDE Longitudinal Dynamics Training"
    srun torchrun \
      --nnodes=1 \
      --nproc_per_node=$NGPUS \
      --rdzv_backend=c10d \
      --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
      -m src.pipeline.phase_c_dynamics \
        --config "$CONFIG" \
        --latent-features "$H5_DIR/latent_decomposition.h5" \
        --decoder-checkpoint "$CKPT_DIR/phase_b_vae.pt" \
        --output "$H5_DIR/counterfactual_results.h5" \
      2>&1 | tee "$LOG_DIR/phase_c_${SLURM_JOB_ID}.log"
    echo "Phase C complete."
    ;;

  *)
    echo "ERROR: Unknown PHASE=$PHASE (expected 0, 1, or 2)"
    exit 1
    ;;
esac

echo "============================================================"
echo "Phase $PHASE finished at $(date)"
echo "============================================================"
