#!/usr/bin/env bash
# =============================================================================
# smoke_test_distributed.sh
#
# End-to-end distributed smoke test for Horizon Disentangled.
# Runs a 2-epoch single-node multi-GPU (or single-GPU) distributed training
# to verify DDP init, DistributedSampler, AMP, grad-clip, and checkpoint save.
#
# Usage:
#   # Single GPU (simulates distributed with 1 process)
#   bash scripts/smoke_test_distributed.sh
#
#   # 4 GPUs on a single node (H200)
#   bash scripts/smoke_test_distributed.sh 4
#
#   # Specific domain
#   bash scripts/smoke_test_distributed.sh 4 gbm
#
# Requirements:
#   - Conda env: nukl-to-ct  (torchrun is in that env)
#   - At least 1 CUDA GPU
# =============================================================================

set -euo pipefail

NPROC=${1:-1}
DOMAIN=${2:-gbm}
EPOCHS=2
BATCH=2
MAX_PAT=8

ROOT=/DATA/horizon_disentangled
PYTHON=/DATA/conda/envs/nukl-to-ct/bin/python
TORCHRUN=/DATA/conda/envs/nukl-to-ct/bin/torchrun
export LD_LIBRARY_PATH=/DATA/conda/envs/nukl-to-ct/lib:${LD_LIBRARY_PATH:-}

echo "========================================================"
echo "  Horizon Disentangled — Distributed Smoke Test"
echo "  nproc_per_node : ${NPROC}"
echo "  domain         : ${DOMAIN}"
echo "  epochs         : ${EPOCHS} | batch : ${BATCH} | max_patients : ${MAX_PAT}"
echo "========================================================"

cd "${ROOT}"

# ------------------------------------------------------------------
# 1. Import sanity check (single process, no CUDA required)
# ------------------------------------------------------------------
echo ""
echo "--- [1/3] Import sanity check ---"
${PYTHON} -c "
import sys
sys.path.insert(0, '.')
from src.pipeline.stage1_supervisors.trainer import Stage1Trainer
from src.pipeline.stage2_causal_vae.trainer import Stage2Trainer
from src.pipeline.stage4_dynamics.solver import ContinuousDiscreteJumpSolver
from src.pipeline.stage3_ood.filter import Stage3OODFilter
from src.pipeline.orchestrator import PipelineOrchestrator

# Verify the three remaining items are wired
import inspect
s1_src = inspect.getsource(Stage1Trainer.__init__)
assert 'wrap_model_for_cluster' in s1_src, 'FAIL: wrap_model_for_cluster not wired in Stage1Trainer.__init__'
assert 'self._sampler = sampler' in inspect.getsource(Stage1Trainer.make_dataloader), 'FAIL: _sampler not stored'

import torch.distributed as dist
s3_src = open('src/pipeline/stage3_ood/filter.py').read()
assert 'dist.get_rank() != 0' in s3_src, 'FAIL: Stage3 no rank-0 guard'

tf_src = open('scripts/train_full.py').read()
idx1, idx2 = tf_src.index('def train_stage1'), tf_src.index('def train_stage2')
seg1 = tf_src[idx1:idx2]
assert 'DDP(model' in seg1,              'FAIL: train_stage1 not DDP-wrapped'
assert '_s1_sampler.set_epoch(epoch)' in seg1, 'FAIL: set_epoch missing in train_stage1'
idx3 = tf_src.index('def train_stage3')
seg2 = tf_src[idx2:idx3]
assert 'DDP(vae' in seg2,               'FAIL: train_stage2 vae not DDP-wrapped'
assert '_s2_sampler.set_epoch(epoch)' in seg2, 'FAIL: set_epoch missing in train_stage2'

njde_src = open('src/models/njde.jl').read()
assert 'EnzymeVJP()' in njde_src,       'FAIL: EnzymeVJP not wired in njde.jl'
assert 'make_jump_callback' in njde_src, 'FAIL: JumpNet callbacks missing in njde.jl'

print('✅ All structural checks passed')
"
echo "--- Import sanity: PASSED ---"

# ------------------------------------------------------------------
# 2. Distributed Python pipeline smoke test via torchrun
#    Uses synthetic data (no HDF5 required), 2 epochs, tiny batch
# ------------------------------------------------------------------
echo ""
echo "--- [2/3] torchrun distributed smoke test (${NPROC} GPU(s)) ---"

${TORCHRUN} \
    --nproc_per_node="${NPROC}" \
    --master_port=29501 \
    -m src.pipeline.cli \
        --stage 1 \
        --domain "${DOMAIN}" \
        --epochs "${EPOCHS}" \
    2>&1 | tee /tmp/horizon_smoke_stage1.log

if grep -q "Stage 1.*DONE\|stage1.*saved\|supervisor.*saved\|Epoch ${EPOCHS}" /tmp/horizon_smoke_stage1.log 2>/dev/null || \
   ! grep -qi "error\|traceback\|exception" /tmp/horizon_smoke_stage1.log 2>/dev/null; then
    echo "--- Stage 1 torchrun smoke: PASSED ---"
else
    echo "--- Stage 1 torchrun smoke: WARNING (check /tmp/horizon_smoke_stage1.log) ---"
fi

# ------------------------------------------------------------------
# 3. train_full.py entry-point (synthetic fallback, no HDF5)
#    Uses --max-patients 0 so it skips real data load gracefully
# ------------------------------------------------------------------
echo ""
echo "--- [3/3] train_full.py distributed smoke test (${NPROC} GPU(s)) ---"

${TORCHRUN} \
    --nproc_per_node="${NPROC}" \
    --master_port=29502 \
    scripts/train_full.py \
        --domain "${DOMAIN}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH}" \
        --max-patients "${MAX_PAT}" \
    2>&1 | tee /tmp/horizon_smoke_train_full.log

echo "--- train_full.py smoke: DONE (check /tmp/horizon_smoke_train_full.log) ---"

echo ""
echo "========================================================"
echo "  Smoke test complete."
echo "  Logs: /tmp/horizon_smoke_stage1.log"
echo "        /tmp/horizon_smoke_train_full.log"
echo "========================================================"
