#!/bin/bash
set -e

# Ensure Julia is in path
if [ -f "$HOME/.juliaup/bin/julia" ]; then
    export PATH="$HOME/.juliaup/bin:$PATH"
fi

# Configuration
export DATA_DIR="src/data_store"
export DEBUG_MODE="false" # Default to debug
export CUDA_VISIBLE_DEVICES=1

echo "============================================================"
echo "Starting End-to-End Pilot Study (Julia/Lux.jl)"
echo "DATA_DIR=$DATA_DIR"
echo "DEBUG_MODE=$DEBUG_MODE"
echo "============================================================"

# 1. Clean up logs
echo "[0/6] Cleaning up logs..."
rm -rf logs

# 2. Data Download & Preprocessing
echo "[1/6] Phase 0: Data Download & Preprocessing..."
julia --project=src src/data/download_and_preprocess.jl

# 3. Phase 1: Supervisors
echo "[2/6] Phase 1: Training Supervisor Models..."
julia --project=src src/train_supervisors.jl

# 4. Phase 2: VAE
echo "[3/6] Phase 2: Training Causal VAE..."
julia --project=src src/train_vae.jl

# 5. Phase 3: OOD
echo "[4/6] Phase 3: Training OOD Detector..."
julia --project=src src/train_ood.jl

# 6. Phase 4: NJDE
echo "[5/6] Phase 4: Training Neural Jump ODE..."
julia --project=src src/train_njde.jl

# 7. Validation
echo "[6/6] Validation..."
julia --project=src src/validate_counterfactual.jl

echo "============================================================"
echo "Pilot Study Pipeline Completed Successfully!"
echo "============================================================"
