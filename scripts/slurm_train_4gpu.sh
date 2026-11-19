#!/bin/bash
#SBATCH --job-name=horizon_disentangled
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --account=horizon

# ═══════════════════════════════════════════════════════════════════════
# Horizon Disentangled — 4× H200 GPU Training Script
# SLURM Batch Submission for torchrun DDP
#
# GPU Allocation (Section 5.3 of main_pipeline.md):
#   GPU 0: Glioblastoma Stages 1–3, 5 (BrainSegFounder + AutoencoderKlMaisi)
#   GPU 1: Breast Stages 1–3, 5 (MedNeXt-v2 + BreastDCEDL ViT)
#   GPU 2: Stage 4 Julia SciML + VLM (MedGemma 1.5 4B)
#   GPU 3: Data parallelism / multi-draw Monte Carlo sampling
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

# ─── Environment ──────────────────────────────────────────────────────
export CONDA_ENV="/DATA/conda/envs/nukl-to-ct"
export PATH="${CONDA_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

# CUDA unified memory pool (Section 5.3)
export PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync,garbage_collection_threshold:0.8"
export JULIA_CUDA_MEMORY_POOL="none"
export JULIA_CUDA_SOFT_MEMORY_LIMIT="60GiB"

# Julia environment
export JULIA_PROJECT="/DATA/horizon_disentangled/src"
export JULIA_NUM_THREADS=16

# NCCL settings for 4-GPU single-node
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=NVL

# Working directory
cd /DATA/horizon_disentangled
mkdir -p logs experiments/checkpoints artifacts/audit_logs

echo "═══════════════════════════════════════════════════════════"
echo "Job:        ${SLURM_JOB_ID}"
echo "Node:       ${SLURM_NODELIST}"
echo "GPUs:       ${SLURM_GPUS_ON_NODE:-4}"
echo "Date:       $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "Python:     $(python --version)"
echo "PyTorch:    $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:       $(python -c 'import torch; print(torch.version.cuda)')"
echo "═══════════════════════════════════════════════════════════"

# Parse arguments
STAGE="${1:-all}"
DOMAIN="${2:-gbm}"
EPOCHS="${3:-100}"

# ─── Execution Modes ─────────────────────────────────────────────────

if [[ "${STAGE}" == "all" ]]; then
    echo "Running full pipeline: Stage 1 → 2 → 3 → 4 → 5"
    echo "Domain: ${DOMAIN} | Epochs: ${EPOCHS}"

    # For full pipeline, run sequentially per stage
    # Stage 1: Use DDP across all 4 GPUs for data parallelism
    echo ""
    echo "──── Stage 1: Multimodal Supervisors (DDP 4-GPU) ────"
    torchrun \
        --nproc_per_node=4 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29500 \
        -m src.pipeline.cli \
        --stage 1 \
        --domain "${DOMAIN}" \
        --epochs "${EPOCHS}"

    echo ""
    echo "──── Stage 2: Hierarchical Causal VAE (DDP 4-GPU) ────"
    torchrun \
        --nproc_per_node=4 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29501 \
        -m src.pipeline.cli \
        --stage 2 \
        --domain "${DOMAIN}" \
        --epochs "${EPOCHS}"

    echo ""
    echo "──── Stage 3: Latent OOD Detection (single GPU) ────"
    python -m src.pipeline.cli \
        --stage 3 \
        --domain "${DOMAIN}" \
        --epochs $((EPOCHS / 3))

    echo ""
    echo "──── Stage 4: Biophysical PDE-NJDE (GPU 2 + Julia) ────"
    CUDA_VISIBLE_DEVICES=2 python -m src.pipeline.cli \
        --stage 4 \
        --domain "${DOMAIN}" \
        --epochs $((EPOCHS / 2))

    echo ""
    echo "──── Stage 5: Counterfactual Synthesis & Reporting ────"
    python -m src.pipeline.cli \
        --stage 5 \
        --domain "${DOMAIN}"

elif [[ "${STAGE}" == "1" ]] || [[ "${STAGE}" == "2" ]]; then
    # Stages 1 & 2 benefit from DDP across all 4 GPUs
    echo "Running Stage ${STAGE} with DDP (4-GPU)"
    torchrun \
        --nproc_per_node=4 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29500 \
        -m src.pipeline.cli \
        --stage "${STAGE}" \
        --domain "${DOMAIN}" \
        --epochs "${EPOCHS}"

elif [[ "${STAGE}" == "4" ]]; then
    # Stage 4: Pin to GPU 2 for Julia SciML cohabitation
    echo "Running Stage 4 on GPU 2 (Julia SciML engine)"
    CUDA_VISIBLE_DEVICES=2 python -m src.pipeline.cli \
        --stage 4 \
        --domain "${DOMAIN}" \
        --epochs "${EPOCHS}"

else
    # Stages 3, 5: Single GPU
    echo "Running Stage ${STAGE} (single GPU)"
    python -m src.pipeline.cli \
        --stage "${STAGE}" \
        --domain "${DOMAIN}" \
        --epochs "${EPOCHS}"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Job ${SLURM_JOB_ID} completed at $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "═══════════════════════════════════════════════════════════"
