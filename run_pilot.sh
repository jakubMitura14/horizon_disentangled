#!/bin/bash
set -e

# Ensure Julia is in path (if installed via juliaup or locally)
if [ -f "$HOME/.juliaup/bin/julia" ]; then
    export PATH="$HOME/.juliaup/bin:$PATH"
fi

echo "============================================================"
echo "Starting End-to-End Pilot Study on Minimal Synthetic Data (Julia/Lux.jl)"
echo "============================================================"

# 1. Clean up
echo "[0/5] Cleaning up..."
rm -rf src/mock_data
rm -f src/*.png

# 2. Data Generation
echo "[1/5] Phase 0: Data Generation..."
julia --project=src src/data/mock_data.jl

# 3. Phase 1: Supervisors
echo "[2/5] Phase 1: Training Supervisor Models..."
julia --project=src src/train_supervisors.jl

# 4. Phase 2: VAE
echo "[3/5] Phase 2: Training Causal VAE..."
julia --project=src src/train_vae.jl

# 5. Phase 3: OOD
echo "[4/5] Phase 3: Training OOD Detector..."
julia --project=src src/train_ood.jl

# 6. Phase 4: NJDE
echo "[5/5] Phase 4: Training Neural Jump ODE..."
julia --project=src src/train_njde.jl

# 7. Validation
echo "[6/5] Validation..."
julia --project=src src/validate_counterfactual.jl

echo "============================================================"
echo "Pilot Study Pipeline Completed Successfully!"
echo "============================================================"
