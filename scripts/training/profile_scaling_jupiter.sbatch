#!/bin/bash
#SBATCH --account=numedai-scale
#SBATCH --partition=booster
#SBATCH --job-name=horizon-scaling
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=02:00:00
#SBATCH --output=/e/scratch/numedai-scale/mitura1/horizon_logs/scaling_%j.out
#SBATCH --error=/e/scratch/numedai-scale/mitura1/horizon_logs/scaling_%j.err
# ═══════════════════════════════════════════════════════════════════════════════
# Horizon Disentangled — 1-GPU vs 4-GPU Scaling Benchmark (JUPITER GH200)
#
# Runs the full scaling comparison:
#   Step 1: Single GPU baseline (all 3 phases)
#   Step 2: 4-GPU with FSDP + torch.compile + spectral Laplacian
#   Step 3: Scaling analysis (computed offline from JSON outputs)
#
# Features benchmarked:
#   Phase A: Real SwinUNETR encoder (75.5M params)
#   Phase B: 756M VAE with FSDP (FULL_SHARD) + torch.compile
#   Phase C: ODE solver with spectral cuFFT Laplacian
#
# Expected runtime: ~1.5 hours total (warmup + compile + benchmarks)
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ─── JUPITER Lmod ────────────────────────────────────────────────────────────
module purge 2>/dev/null || true
module load Stages/2026 2>/dev/null || true
module load GCCcore/14.3.0 2>/dev/null || true
module load Python 2>/dev/null || true

# ─── Environment ─────────────────────────────────────────────────────────────
REPO="/e/project1/numedai-scale/mitura1/horizon_disentangled"
cd "$REPO"

if [ -f "/e/project1/numedai-scale/mitura1/venvs/activate_lymph.sh" ]; then
    source /e/project1/numedai-scale/mitura1/venvs/activate_lymph.sh
fi

# ─── CUDA / PyTorch Optimization ─────────────────────────────────────────────
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-72}
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# NCCL for NVLink-C2C (JUPITER GH200)
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# ─── Directories ─────────────────────────────────────────────────────────────
JOB_ID="${SLURM_JOB_ID:-manual_$(date +%s)}"
OUTPUT_BASE="/e/scratch/numedai-scale/mitura1/horizon_profiling/scaling_${JOB_ID}"
LOG_DIR="/e/scratch/numedai-scale/mitura1/horizon_logs"
mkdir -p "$OUTPUT_BASE" "$LOG_DIR"

# Master node for torchrun
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

echo "════════════════════════════════════════════════════════════════"
echo "Horizon — 1-GPU vs 4-GPU Scaling Benchmark (JUPITER GH200)"
echo "Job ID:       $JOB_ID"
echo "Node:         $(hostname)"
echo "Master:       ${MASTER_ADDR}:${MASTER_PORT}"
echo "GPUs:         $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n1 || echo 'N/A')"
echo "Output:       $OUTPUT_BASE"
echo "Timestamp:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "════════════════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Single GPU Baseline
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "▶▶▶ STEP 1: Single GPU Baseline (all 3 phases)"
echo "    Phase A: SwinUNETR, batch=8"
echo "    Phase B: 756M VAE (no FSDP, no compile), batch=32"
echo "    Phase C: FD Laplacian (no spectral), batch=24"
echo ""

export CUDA_VISIBLE_DEVICES=0

python3 scripts/training/profile_scaling_benchmark.py \
    --ngpus 1 \
    --phases A,B,C \
    --batch-a 8 \
    --batch-b 32 \
    --batch-c 24 \
    --grad-accum-b 2 \
    --max-steps 10 \
    --output-dir "$OUTPUT_BASE" \
    --run-label "1gpu_baseline" \
    --run-spectral-micro

echo ""
echo "✓ Single GPU baseline complete."
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: 4-GPU with FSDP + compile + spectral
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "▶▶▶ STEP 2: 4-GPU Scaling (FSDP + compile + spectral)"
echo "    Phase A: SwinUNETR + DDP, batch=8/gpu"
echo "    Phase B: 756M VAE + FSDP + compile, batch=32/gpu"
echo "    Phase C: Spectral Laplacian + DDP, batch=24/gpu"
echo ""

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
    --batch-a 8 \
    --batch-b 32 \
    --batch-c 24 \
    --grad-accum-b 2 \
    --max-steps 10 \
    --use-fsdp \
    --use-compile \
    --use-spectral \
    --output-dir "$OUTPUT_BASE" \
    --run-label "4gpu_fsdp_compile_spectral"

echo ""
echo "✓ 4-GPU scaling benchmark complete."
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Generate comparison report
# ═══════════════════════════════════════════════════════════════════════════════
echo "▶▶▶ STEP 3: Generating comparison report..."

python3 -c "
import json, os, sys

base = '$OUTPUT_BASE'
f1 = os.path.join(base, '1gpu_baseline', 'scaling_results.json')
f4 = os.path.join(base, '4gpu_fsdp_compile_spectral', 'scaling_results.json')

if not os.path.exists(f1) or not os.path.exists(f4):
    print('ERROR: Missing result files')
    sys.exit(1)

with open(f1) as f: r1 = json.load(f)
with open(f4) as f: r4 = json.load(f)

print()
print('=' * 90)
print('SCALING COMPARISON: 1-GPU vs 4-GPU (JUPITER GH200)')
print('=' * 90)
print(f\"{'Phase':<8} {'Metric':<20} {'1-GPU':<15} {'4-GPU':<15} {'Scaling':<12} {'Efficiency':<10}\")
print('-' * 80)

for p in ['A', 'B', 'C']:
    s = r1['phases'].get(p, {})
    m = r4['phases'].get(p, {})
    if not s or not m:
        continue

    tp_key = 'throughput_traj_s' if p == 'C' else 'throughput_vol_s'
    tp_unit = 'traj/s' if p == 'C' else 'vol/s'
    tp1 = s.get(tp_key, 0)
    tp4 = m.get(tp_key, 0)
    speedup = tp4 / tp1 if tp1 > 0 else 0
    eff = speedup / 4.0 * 100

    print(f\"{p:<8} {'Throughput':<20} {tp1:.2f} {tp_unit:<4} {tp4:.2f} {tp_unit:<4} {speedup:.2f}x       {eff:.0f}%\")
    print(f\"{'':8} {'Peak VRAM/GPU':<20} {s.get('peak_vram_gb',0):.2f} GB      {m.get('peak_vram_gb',0):.2f} GB\")
    print(f\"{'':8} {'Occupancy':<20} {s.get('occupancy_pct',0):.1f}%          {m.get('occupancy_pct',0):.1f}%\")

    # Bottleneck analysis
    if eff < 90:
        if p == 'B' and m.get('use_fsdp'):
            print(f\"{'':8} {'Bottleneck':<20} FSDP all-gather/reduce-scatter communication\")
        elif p == 'C':
            print(f\"{'':8} {'Bottleneck':<20} Sequential ODE time-stepping (Amdahl's law)\")
        else:
            print(f\"{'':8} {'Bottleneck':<20} DDP gradient all-reduce overhead\")
    else:
        print(f\"{'':8} {'Status':<20} Near-linear scaling!\")
    print('-' * 80)

# Save combined report
report = {
    '1gpu': r1,
    '4gpu': r4,
    'scaling': {}
}
for p in ['A', 'B', 'C']:
    s = r1['phases'].get(p, {})
    m = r4['phases'].get(p, {})
    if s and m:
        tp_key = 'throughput_traj_s' if p == 'C' else 'throughput_vol_s'
        tp1 = s.get(tp_key, 0)
        tp4 = m.get(tp_key, 0)
        speedup = tp4 / tp1 if tp1 > 0 else 0
        report['scaling'][p] = {
            'speedup': round(speedup, 2),
            'efficiency_pct': round(speedup / 4.0 * 100, 1),
            'throughput_1gpu': tp1,
            'throughput_4gpu': tp4,
            'vram_1gpu_gb': s.get('peak_vram_gb', 0),
            'vram_4gpu_per_gpu_gb': m.get('peak_vram_gb', 0),
        }

out = os.path.join(base, 'scaling_comparison.json')
with open(out, 'w') as f:
    json.dump(report, f, indent=2)
print(f'\nFull report saved to: {out}')
print('=' * 90)
"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Scaling Benchmark Complete!"
echo "════════════════════════════════════════════════════════════════"
echo "Results: $OUTPUT_BASE/"
ls -lh "$OUTPUT_BASE/"
echo ""
echo "Job finished at $(date)"
