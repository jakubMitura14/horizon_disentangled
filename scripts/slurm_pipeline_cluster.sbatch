#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Horizon Disentangled — Multi-Node SLURM Cluster Pipeline
#
# Submits a sequence of SLURM jobs with dependencies for the full
# 5-stage pipeline across a cluster with 4× H200 GPUs per node.
#
# Usage:
#   bash scripts/slurm_pipeline_cluster.sbatch gbm 100
#   bash scripts/slurm_pipeline_cluster.sbatch breast 200
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

DOMAIN="${1:-gbm}"
EPOCHS="${2:-100}"
PARTITION="${3:-gpu}"
ACCOUNT="${4:-horizon}"
NODES_PER_JOB="${5:-1}"          # 5th argument: nodes per job (default 1, use 4 for full scale)

cd /DATA/horizon_disentangled
mkdir -p logs

echo "═══════════════════════════════════════════════════════════"
echo "Horizon Disentangled — Cluster Pipeline Submission"
echo "Domain:    ${DOMAIN}"
echo "Epochs:    ${EPOCHS}"
echo "Partition: ${PARTITION}"
echo "═══════════════════════════════════════════════════════════"

# Stage 1: DDP across 4 GPUs
JOB1=$(sbatch \
    --parsable \
    --job-name="hd_s1_${DOMAIN}" \
    --output="logs/stage1_${DOMAIN}_%j.out" \
    --error="logs/stage1_${DOMAIN}_%j.err" \
    --nodes="${NODES_PER_JOB:-1}" \
    --ntasks-per-node=4 \
    --gres=gpu:h200:4 \
    --cpus-per-task=16 \
    --mem=512G \
    --time=24:00:00 \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --wrap="bash scripts/slurm_train_4gpu.sh 1 ${DOMAIN} ${EPOCHS}")
echo "Stage 1 submitted: Job ${JOB1}"

# Stage 2: DDP across 4 GPUs (depends on Stage 1)
JOB2=$(sbatch \
    --parsable \
    --dependency=afterok:${JOB1} \
    --job-name="hd_s2_${DOMAIN}" \
    --output="logs/stage2_${DOMAIN}_%j.out" \
    --error="logs/stage2_${DOMAIN}_%j.err" \
    --nodes="${NODES_PER_JOB:-1}" \
    --ntasks-per-node=4 \
    --gres=gpu:h200:4 \
    --cpus-per-task=16 \
    --mem=512G \
    --time=24:00:00 \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --wrap="bash scripts/slurm_train_4gpu.sh 2 ${DOMAIN} ${EPOCHS}")
echo "Stage 2 submitted: Job ${JOB2} (after ${JOB1})"

# Stage 3: Single GPU, lightweight
JOB3=$(sbatch \
    --parsable \
    --dependency=afterok:${JOB2} \
    --job-name="hd_s3_${DOMAIN}" \
    --output="logs/stage3_${DOMAIN}_%j.out" \
    --error="logs/stage3_${DOMAIN}_%j.err" \
    --nodes="${NODES_PER_JOB:-1}" \
    --ntasks-per-node=1 \
    --gres=gpu:1 \
    --cpus-per-task=16 \
    --mem=128G \
    --time=4:00:00 \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --wrap="bash scripts/slurm_train_4gpu.sh 3 ${DOMAIN} $((EPOCHS / 3))")
echo "Stage 3 submitted: Job ${JOB3} (after ${JOB2})"

# Stage 4: GPU 2 (Julia SciML)
JOB4=$(sbatch \
    --parsable \
    --dependency=afterok:${JOB3} \
    --job-name="hd_s4_${DOMAIN}" \
    --output="logs/stage4_${DOMAIN}_%j.out" \
    --error="logs/stage4_${DOMAIN}_%j.err" \
    --nodes="${NODES_PER_JOB:-1}" \
    --ntasks-per-node=1 \
    --gres=gpu:1 \
    --cpus-per-task=16 \
    --mem=256G \
    --time=24:00:00 \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --wrap="bash scripts/slurm_train_4gpu.sh 4 ${DOMAIN} $((EPOCHS / 2))")
echo "Stage 4 submitted: Job ${JOB4} (after ${JOB3})"

# Stage 5: Counterfactual Synthesis & Reporting
JOB5=$(sbatch \
    --parsable \
    --dependency=afterok:${JOB4} \
    --job-name="hd_s5_${DOMAIN}" \
    --output="logs/stage5_${DOMAIN}_%j.out" \
    --error="logs/stage5_${DOMAIN}_%j.err" \
    --nodes="${NODES_PER_JOB:-1}" \
    --ntasks-per-node=1 \
    --gres=gpu:1 \
    --cpus-per-task=16 \
    --mem=256G \
    --time=4:00:00 \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --wrap="bash scripts/slurm_train_4gpu.sh 5 ${DOMAIN} 0")
echo "Stage 5 submitted: Job ${JOB5} (after ${JOB4})"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Pipeline chain: ${JOB1} → ${JOB2} → ${JOB3} → ${JOB4} → ${JOB5}"
echo "Monitor with: squeue -u \$USER"
echo "═══════════════════════════════════════════════════════════"
