#!/bin/bash
#SBATCH --job-name=horizon_disentangled
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=${SLURM_NODES:-1}
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=72:00:00
#SBATCH --partition=h200
#SBATCH --account=horizon
#SBATCH --constraint=h200

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

# ─── Fault tolerance ──────────────────────────────────────────────────
# Trap ERR to surface failure reason; SIGTERM for SLURM preemption handling
trap 'echo "ERROR: Job ${SLURM_JOB_ID:-local} failed at $(date -u +"%Y-%m-%dT%H:%M:%SZ"). Latest checkpoint in experiments/checkpoints/"; exit 1' ERR
trap 'echo "PREEMPTED: SLURM signal received at $(date -u). Checkpointing handled by rank-0 barrier in trainer."; exit 0' SIGTERM SIGUSR1

# ─── Environment ──────────────────────────────────────────────────────
export CONDA_ENV="/DATA/conda/envs/nukl-to-ct"
export PATH="${CONDA_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

# CUDA unified memory pool — expandable_segments avoids OOM on large allocs
export PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync,expandable_segments:True,garbage_collection_threshold:0.8"
export JULIA_CUDA_MEMORY_POOL="none"
export JULIA_CUDA_SOFT_MEMORY_LIMIT="60GiB"

# Julia environment
export JULIA_PROJECT="/DATA/horizon_disentangled/src"
export JULIA_NUM_THREADS=16

# ─── NCCL: H200 NVLink 4.0 + NDR400 InfiniBand optimised ────────────
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5           # GPU-Direct RDMA over InfiniBand
export NCCL_P2P_LEVEL=NVL             # Use NVLink for intra-node P2P
export NCCL_IB_TIMEOUT=60
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=ib0
# Allow NCCL to pick Ring vs Tree based on message size
export NCCL_ALGO=Ring,Tree
# LL128 protocol: low-latency for small messages on NVLink
export NCCL_PROTO=Simple,LL128
# Parallelism: 4 channels minimum, 16 maximum for bandwidth saturation
export NCCL_MIN_NCHANNELS=4
export NCCL_MAX_NCHANNELS=16
# Multi-node: allow cross-NIC routing for multi-rail InfiniBand
export NCCL_CROSS_NIC=1
# RoCE v2 GID for InfiniBand (required on most clusters)
export NCCL_IB_GID_INDEX=3
# Multiple QPs per connection increases InfiniBand bandwidth utilisation
export NCCL_IB_QPS_PER_CONNECTION=4
# Avoid record-streams overhead (saves ~5% on comm-compute overlap)
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
# Serialize NCCL streams with compute to maximise pipeline overlap
export CUDA_DEVICE_MAX_CONNECTIONS=1

export MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | head -n1 || echo localhost)}
export MASTER_PORT=${MASTER_PORT:-29500}

# ─── CPU threading — prevent oversubscription when using many DataLoader workers ──
# Each GPU process should use 1 OpenMP/MKL thread to avoid contention with
# DataLoader workers (which already parallelize via multiprocessing).
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
# Keep torchrun's worker coordination lean
export TORCHELASTIC_ERROR_FILE="/tmp/torch_elastic_error_${SLURM_JOB_ID:-local}.json"

# ─── PyTorch profiling (optional, off by default) ──────────────────────
# Uncomment to enable Kineto profiling on first 3 steps:
# export KINETO_LOG_LEVEL=1
# export TORCH_PROFILER_PROFILE_MEMORY=1

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
        --nnodes="${SLURM_NNODES:-1}" \
        --node_rank="${SLURM_NODEID:-0}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        -m src.pipeline.cli \
        --stage 1 \
        --domain "${DOMAIN}" \
        --epochs "${EPOCHS}"

    echo ""
    echo "──── Stage 2: Hierarchical Causal VAE (DDP 4-GPU) ────"
    torchrun \
        --nproc_per_node=4 \
        --nnodes="${SLURM_NNODES:-1}" \
        --node_rank="${SLURM_NODEID:-0}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
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
    echo "──── Stage 4: Biophysical PDE-NJDE (DDP all GPUs) ────"
    torchrun \
        --nproc_per_node=4 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29503 \
        -m src.pipeline.cli \
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
        --nnodes="${SLURM_NNODES:-1}" \
        --node_rank="${SLURM_NODEID:-0}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        -m src.pipeline.cli \
        --stage "${STAGE}" \
        --domain "${DOMAIN}" \
        --epochs "${EPOCHS}"

elif [[ "${STAGE}" == "4" ]]; then
    # Stage 4: DDP across all 4 GPUs — PDE sub-modules are individually DDP-wrapped
    # (wrap_for_cluster() in solver.py handles per-submodule DDP)
    echo "Running Stage 4 with DDP (4-GPU, via torchrun)"
    torchrun \
        --nproc_per_node=4 \
        --nnodes="${SLURM_NNODES:-1}" \
        --node_rank="${SLURM_NODEID:-0}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        -m src.pipeline.cli \
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
