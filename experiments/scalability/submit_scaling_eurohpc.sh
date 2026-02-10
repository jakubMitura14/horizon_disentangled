#!/bin/bash

# --- SLURM Job Submission Directives ---
#SBATCH --job-name=causalpca_scaling
#SBATCH -t 48:00:00
#SBATCH -p kisski-h100
#SBATCH --constraint=inet
#SBATCH -G H100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mail-user=jakub.mitura14@gmail.com
#SBATCH --mail-type=all
#SBATCH --output=/user/joanna.wybranska/u10867/.project/dir.project/horizon_disentangled/experiments/scalability/logs/scaling_job_%j.out
#SBATCH --error=/user/joanna.wybranska/u10867/.project/dir.project/horizon_disentangled/experiments/scalability/logs/scaling_job_%j.err

set -e

# --- Environment Setup ---
echo "--- Setting up the environment... ---"

# Load modules
module load apptainer
module load miniforge3
module load julia/1.11.6

# Initialize Conda
echo "Initializing Conda..."
if command -v conda &> /dev/null; then
    source $(conda info --base)/etc/profile.d/conda.sh
else
    echo "Conda command not found in PATH. Attempting to locate..."
    source /usr/local/miniforge3/etc/profile.d/conda.sh || echo "Could not source conda.sh"
fi

# Define paths
PROJECT_ROOT="/user/joanna.wybranska/u10867/.project/dir.project"
ENV_YAML_PATH="$PROJECT_ROOT/horizon_disentangled/experiments/scalability/environment.yml"
CONDA_ENV_DIR="/mnt/vast-kisski/projects/ovgu_medicine_llm/ollama_data/conda_env_causalpca"

# Create/Update Conda Environment
if [ -d "$CONDA_ENV_DIR" ]; then
    echo "--- Conda environment found at $CONDA_ENV_DIR. ---"
else
    echo "--- Creating Conda environment at $CONDA_ENV_DIR... ---"
    conda env create --prefix "$CONDA_ENV_DIR" --file "$ENV_YAML_PATH"
fi

echo "Activating Conda environment..."
conda activate "$CONDA_ENV_DIR"

# Fix libcurand symlink if missing
if [ -f "$CONDA_ENV_DIR/lib/libcurand.so.10.3.7.77" ] && [ ! -L "$CONDA_ENV_DIR/lib/libcurand.so.10" ]; then
    echo "Fixing libcurand symlink..."
    ln -sf "$CONDA_ENV_DIR/lib/libcurand.so.10.3.7.77" "$CONDA_ENV_DIR/lib/libcurand.so.10"
fi

# Export Paths
# NOTE: We keep LD_LIBRARY_PATH for Python, but Julia might conflict.
# We will check if Julia can find the GPU.
export LD_LIBRARY_PATH=$CONDA_ENV_DIR/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export JULIA_PROJECT="$PROJECT_ROOT/horizon_disentangled/experiments/scalability"
export JULIA_DEPOT_PATH="$PROJECT_ROOT/.julia_depot"

# Verify
echo "Python: $(which python)"
echo "Julia: $(which julia)"
nvidia-smi

# --- Execution ---
cd "$PROJECT_ROOT"

echo "Instantiating Julia Environment..."
# Added Pkg.add to ensure registration of required packages on compute nodes
julia --project="$JULIA_PROJECT" -e 'using Pkg; Pkg.add(["LuxCUDA", "ArgParse", "Statistics", "Random", "Printf", "ComponentArrays", "Lux", "MPI", "Optimisers", "Zygote", "NCCL"]); Pkg.instantiate()'

echo "Checking Julia CUDA status..."
julia --project="$JULIA_PROJECT" -e 'using CUDA; @show CUDA.functional(); using Lux, LuxCUDA; @show cpu_device(); @show gpu_device()' || true

# Skipping sysimage building as it has issues on this cluster and timing skip first epoch anyway.

echo "Starting Scalability Tests (Single Node Strong Scaling)..."
# Pass --slurm flag to orchestrator.
# Arguments: <MAX_GPUS>
TOTAL_GPUS=${SLURM_GPUS_ON_NODE:-4}

bash horizon_disentangled/experiments/scalability/run_scaling_tests.sh --slurm $TOTAL_GPUS

echo "Job Complete."
