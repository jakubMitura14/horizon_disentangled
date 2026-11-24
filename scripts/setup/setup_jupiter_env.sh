#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# setup_jupiter_env.sh
# ═══════════════════════════════════════════════════════════════════════════
# Turnkey provisioning script for Horizon Disentangled on the JUPITER
# Booster (GH200 Grace-Hopper, 4 GPUs/node, 480 GB HBM3) at Jülich
# Supercomputing Centre (FZJ).
#
# Creates a Python venv in /e/project1/numedai-scale/mitura1/venvs/horizon_venv
# and installs the full imaging + ML stack.
#
# Usage:
#   bash scripts/setup/setup_jupiter_env.sh
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="/e/project1/numedai-scale/mitura1/venvs/horizon_venv"
ACTIVATE_SCRIPT="${REPO_ROOT}/activate_jupiter.sh"

echo "=================================================================="
echo "    HORIZON DISENTANGLED — JUPITER ENV SETUP"
echo "=================================================================="
echo "Repository:  $REPO_ROOT"
echo "Target Venv: $VENV_DIR"
echo "Timestamp:   $(date)"
echo "=================================================================="

# ─── Load JUPITER Lmod Modules ──────────────────────────────────────────────
echo ""
echo "[1/4] Loading JUPITER Lmod modules..."
module purge 2>/dev/null || true
module load Stages/2026 2>/dev/null || true
module load GCCcore/14.3.0 2>/dev/null || true
module load Python/3.12.3 2>/dev/null || module load Python 2>/dev/null || true

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Python Version: $PY_VER"

# ─── Create Virtual Environment ────────────────────────────────────────────
echo ""
echo "[2/4] Creating Python virtual environment at $VENV_DIR..."
mkdir -p "$(dirname "$VENV_DIR")"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "  Created new venv."
else
    echo "  Venv already exists."
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel -q

# ─── Install Dependencies ──────────────────────────────────────────────────
echo ""
echo "[3/4] Installing PyTorch + Medical Imaging Stack..."

# PyTorch with CUDA support (auto-detect index)
pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 -q 2>/dev/null || \
pip install --no-cache-dir torch torchvision torchaudio -q

# Core scientific + imaging
pip install --no-cache-dir \
    numpy scipy pandas \
    nibabel SimpleITK pynrrd Pillow \
    h5py hdf5plugin \
    monai torchio \
    einops timm pydantic \
    requests tqdm rich psutil \
    scikit-learn lifelines \
    pycox torchtuples \
    accelerate \
    transformers tokenizers safetensors huggingface-hub sentencepiece \
    peft open-clip-torch \
    xnat \
    -q

# ─── Activation Shortcut ──────────────────────────────────────────────────
cat > "$ACTIVATE_SCRIPT" << ACTEOF
#!/usr/bin/env bash
# Source this script to activate the Horizon Disentangled env on JUPITER.
module purge 2>/dev/null || true
module load Stages/2026 GCCcore/14.3.0 2>/dev/null || true
source "${VENV_DIR}/bin/activate"
export PYTHONPATH="${REPO_ROOT}:\${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=\${OMP_NUM_THREADS:-72}
echo "Horizon JUPITER env activated (\$(python3 --version), \$(which python3))"
ACTEOF
chmod +x "$ACTIVATE_SCRIPT"

# ─── Smoke Test ────────────────────────────────────────────────────────────
echo ""
echo "[4/4] Running environment smoke test..."
python3 - << 'PYEOF'
import sys, torch, h5py, numpy as np
print(f"  ✔ Python:   {sys.version.split()[0]}")
print(f"  ✔ PyTorch:  {torch.__version__} (CUDA: {torch.cuda.is_available()})")
print(f"  ✔ NumPy:    {np.__version__}")
print(f"  ✔ h5py:     {h5py.__version__}")
try:
    import xnat
    print(f"  ✔ xnat:     OK")
except ImportError:
    print(f"  ⚠ xnat:     NOT INSTALLED (use requests-based fallback)")
print("  ✔ ALL CHECKS PASSED")
PYEOF

echo ""
echo "=================================================================="
echo "✔ ENVIRONMENT SETUP COMPLETE"
echo "=================================================================="
echo "To activate:"
echo "  source ${ACTIVATE_SCRIPT}"
echo "=================================================================="
