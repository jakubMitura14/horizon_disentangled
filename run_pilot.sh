#!/bin/bash
set -e

# Set PYTHONPATH to current directory to ensure imports work
export PYTHONPATH=.

echo "============================================================"
echo "Starting End-to-End Pilot Study on Minimal Synthetic Data"
echo "============================================================"

# 1. Clean up previous runs
echo "[0/5] Cleaning up previous artifacts..."
rm -rf src/mock_data
rm -f src/*.pth src/*.png

# 2. Phase 1: Data Generation & Supervisors
# This script internally generates the mock dataset if it doesn't exist
echo "[1/5] Phase 1: Training Supervisor Models (Segmentation, Ordinal, Survival)..."
python3 src/train_supervisor.py

# 3. Phase 2: Causal Disentanglement
echo "[2/5] Phase 2: Training Causal VAE (Disentanglement)..."
python3 src/train_vae.py

# 4. Phase 3: Out-of-Distribution Detection
echo "[3/5] Phase 3: Training OOD Detector..."
python3 src/test_ood.py

# 5. Phase 4: Temporal Modeling
echo "[4/5] Phase 4: Training Neural Jump ODE (NJDE)..."
python3 src/train_njde.py

# 6. Validation
echo "[5/5] Validation: Generating Counterfactual Analysis..."
python3 src/validate_counterfactual.py

echo "============================================================"
echo "Pilot Study Pipeline Completed Successfully!"
echo "Outputs generated:"
echo " - Checkpoints: src/*.pth"
echo " - Plots: src/counterfactual_plot.png"
echo "============================================================"
