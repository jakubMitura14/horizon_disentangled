#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Horizon Disentangled — Multi-Stage Cluster Pipeline on JUPITER Booster
#
# Submits a chain of SLURM jobs with dependencies for the full 5-stage
# Horizon Disentangled pipeline on JUPITER Booster (4× GH200 per node).
#
# Usage:
#   bash scripts/training/slurm_pipeline_jupiter.sh gbm 100
#   bash scripts/training/slurm_pipeline_jupiter.sh breast 200
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

DOMAIN="${1:-gbm}"
EPOCHS="${2:-100}"
PARTITION="${3:-booster}"
ACCOUNT="${4:-numedai-scale}"
NODES="${5:-1}"

REPO="/e/project1/numedai-scale/mitura1/horizon_disentangled"
LOG_DIR="/e/scratch/numedai-scale/mitura1/horizon_logs"
DATA_DIR="/e/project1/numedai-scale/mitura1/horizon_data/hdf5"
CKPT_DIR="/e/project1/numedai-scale/mitura1/horizon_data/checkpoints"

cd "$REPO"
mkdir -p "$LOG_DIR"

echo "═══════════════════════════════════════════════════════════"
echo "Horizon Disentangled — JUPITER Pipeline Submission"
echo "Domain:    ${DOMAIN}"
echo "Epochs:    ${EPOCHS}"
echo "Partition: ${PARTITION}"
echo "Account:   ${ACCOUNT}"
echo "Nodes:     ${NODES}"
echo "═══════════════════════════════════════════════════════════"

# ─── Common SLURM preamble (written as heredoc) ─────────────────────────
# Each stage runs a wrapper that:
#   1. Loads JUPITER modules
#   2. Activates the Horizon environment
#   3. Stages data to fast scratch
#   4. Runs torchrun with 4 GPUs per node

WRAPPER_ENV=$(cat << 'WRAPEOF'
module purge 2>/dev/null || true
module load Stages/2026 GCCcore/14.3.0 2>/dev/null || true
source /e/project1/numedai-scale/mitura1/horizon_disentangled/activate_jupiter.sh 2>/dev/null || \
    source /e/project1/numedai-scale/mitura1/venvs/activate_lymph.sh 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-72}
export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=NVL
WRAPEOF
)

# Stage 1: Vision Supervisor (DDP, 4 GPUs)
JOB1=$(sbatch \
    --parsable \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --job-name="hd_s1_${DOMAIN}" \
    --output="$LOG_DIR/stage1_${DOMAIN}_%j.out" \
    --error="$LOG_DIR/stage1_${DOMAIN}_%j.err" \
    --nodes="$NODES" \
    --ntasks-per-node=4 \
    --cpus-per-task=72 \
    --time=24:00:00 \
    --signal=B:SIGTERM@120 \
    --requeue \
    --wrap="${WRAPPER_ENV}
cd $REPO
torchrun --nnodes=\$SLURM_JOB_NUM_NODES --nproc_per_node=4 \
    --rdzv_backend=c10d --rdzv_endpoint=\$(hostname):29500 \
    scripts/training/train_full.py \
    --stage 1 --domain ${DOMAIN} --epochs ${EPOCHS} \
    --data-dir ${DATA_DIR} --ckpt-dir ${CKPT_DIR} \
    --precision bf16")
echo "Stage 1 submitted: Job ${JOB1}"

# Stage 2: Causal VAE (DDP, 4 GPUs)
JOB2=$(sbatch \
    --parsable \
    --dependency=afterok:${JOB1} \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --job-name="hd_s2_${DOMAIN}" \
    --output="$LOG_DIR/stage2_${DOMAIN}_%j.out" \
    --error="$LOG_DIR/stage2_${DOMAIN}_%j.err" \
    --nodes="$NODES" \
    --ntasks-per-node=4 \
    --cpus-per-task=72 \
    --time=24:00:00 \
    --signal=B:SIGTERM@120 \
    --requeue \
    --wrap="${WRAPPER_ENV}
cd $REPO
torchrun --nnodes=\$SLURM_JOB_NUM_NODES --nproc_per_node=4 \
    --rdzv_backend=c10d --rdzv_endpoint=\$(hostname):29500 \
    scripts/training/train_full.py \
    --stage 2 --domain ${DOMAIN} --epochs ${EPOCHS} \
    --data-dir ${DATA_DIR} --ckpt-dir ${CKPT_DIR} \
    --precision bf16")
echo "Stage 2 submitted: Job ${JOB2} (after ${JOB1})"

# Stage 3: OOD Filter (single GPU, lightweight)
JOB3=$(sbatch \
    --parsable \
    --dependency=afterok:${JOB2} \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --job-name="hd_s3_${DOMAIN}" \
    --output="$LOG_DIR/stage3_${DOMAIN}_%j.out" \
    --error="$LOG_DIR/stage3_${DOMAIN}_%j.err" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=72 \
    --time=04:00:00 \
    --signal=B:SIGTERM@120 \
    --requeue \
    --wrap="${WRAPPER_ENV}
cd $REPO
export CUDA_VISIBLE_DEVICES=0
python3 scripts/training/train_full.py \
    --stage 3 --domain ${DOMAIN} --epochs $((EPOCHS / 3)) \
    --data-dir ${DATA_DIR} --ckpt-dir ${CKPT_DIR} \
    --precision bf16")
echo "Stage 3 submitted: Job ${JOB3} (after ${JOB2})"

# Stage 4: PDE-NJDE Dynamics (DDP, 4 GPUs)
JOB4=$(sbatch \
    --parsable \
    --dependency=afterok:${JOB3} \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --job-name="hd_s4_${DOMAIN}" \
    --output="$LOG_DIR/stage4_${DOMAIN}_%j.out" \
    --error="$LOG_DIR/stage4_${DOMAIN}_%j.err" \
    --nodes="$NODES" \
    --ntasks-per-node=4 \
    --cpus-per-task=72 \
    --time=24:00:00 \
    --signal=B:SIGTERM@120 \
    --requeue \
    --wrap="${WRAPPER_ENV}
cd $REPO
torchrun --nnodes=\$SLURM_JOB_NUM_NODES --nproc_per_node=4 \
    --rdzv_backend=c10d --rdzv_endpoint=\$(hostname):29500 \
    scripts/training/train_full.py \
    --stage 4 --domain ${DOMAIN} --epochs $((EPOCHS / 2)) \
    --data-dir ${DATA_DIR} --ckpt-dir ${CKPT_DIR} \
    --precision bf16")
echo "Stage 4 submitted: Job ${JOB4} (after ${JOB3})"

# Stage 5: Counterfactual Synthesis & Reporting (single GPU)
JOB5=$(sbatch \
    --parsable \
    --dependency=afterok:${JOB4} \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --job-name="hd_s5_${DOMAIN}" \
    --output="$LOG_DIR/stage5_${DOMAIN}_%j.out" \
    --error="$LOG_DIR/stage5_${DOMAIN}_%j.err" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=72 \
    --time=04:00:00 \
    --signal=B:SIGTERM@120 \
    --requeue \
    --wrap="${WRAPPER_ENV}
cd $REPO
export CUDA_VISIBLE_DEVICES=0
python3 scripts/training/train_full.py \
    --stage 5 --domain ${DOMAIN} --epochs 0 \
    --data-dir ${DATA_DIR} --ckpt-dir ${CKPT_DIR} \
    --precision bf16")
echo "Stage 5 submitted: Job ${JOB5} (after ${JOB4})"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Pipeline chain: ${JOB1} → ${JOB2} → ${JOB3} → ${JOB4} → ${JOB5}"
echo "Monitor with: squeue -u \$USER"
echo "Logs at: $LOG_DIR/"
echo "═══════════════════════════════════════════════════════════"
