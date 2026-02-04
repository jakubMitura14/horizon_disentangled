#!/bin/bash

# --- SLURM Job Submission Directives ---
#SBATCH --job-name=causalpca_scaling
#SBATCH -t 48:00:00
#SBATCH -p kisski-h100
#SBATCH --constraint=inet
#SBATCH -G H100:4
#SBATCH --mail-user=jakub.mitura14@gmail.com
#SBATCH --mail-type=all
#SBATCH --output=/user/joanna.wybranska/u10867/.project/dir.project/logs/scaling_job_%j.out
#SBATCH --error=/user/joanna.wybranska/u10867/.project/dir.project/logs/scaling_job_%j.err

set -e

# --- Environment Setup ---
echo "--- Setting up the environment... ---"

# Load modules
module load apptainer
module load miniforge3
# Assuming Julia is available as a module (common on HPC)
# If not, one might need to download it or use a container.
# We load it as requested.
module load julia

# Initialize Conda
echo "Initializing Conda..."
# Try to source conda profile; fallback to standard locations if `conda info` fails in batch mode
if command -v conda &> /dev/null; then
    source $(conda info --base)/etc/profile.d/conda.sh
else
    echo "Conda command not found in PATH. Attempting to locate..."
    # Fallback assumption based on module load
    source /usr/local/miniforge3/etc/profile.d/conda.sh || echo "Could not source conda.sh"
fi

# Define paths
PROJECT_ROOT="/user/joanna.wybranska/u10867/.project/dir.project"
ENV_YAML_PATH="$PROJECT_ROOT/experiments/scalability/environment.yml"
# Using a project-specific env path to avoid conflicts with other projects
# But using the vast storage as suggested by the example for capacity
CONDA_ENV_DIR="/mnt/vast-kisski/projects/ovgu_medicine_llm/ollama_data/conda_env_causalpca"

# Create/Update Conda Environment
if [ -d "$CONDA_ENV_DIR" ]; then
    echo "--- Conda environment found at $CONDA_ENV_DIR. ---"
    # Optional: Update it? For reproducibility, we assume it's good or manual update needed.
    # echo "Updating..."
    # conda env update --prefix "$CONDA_ENV_DIR" --file "$ENV_YAML_PATH" --prune
else
    echo "--- Creating Conda environment at $CONDA_ENV_DIR... ---"
    conda env create --prefix "$CONDA_ENV_DIR" --file "$ENV_YAML_PATH"
fi

echo "Activating Conda environment..."
conda activate "$CONDA_ENV_DIR"

# Fix libcurand symlink if missing (common issue on some clusters)
if [ -f "$CONDA_ENV_DIR/lib/libcurand.so.10.3.7.77" ] && [ ! -L "$CONDA_ENV_DIR/lib/libcurand.so.10" ]; then
    echo "Fixing libcurand symlink..."
    ln -sf "$CONDA_ENV_DIR/lib/libcurand.so.10.3.7.77" "$CONDA_ENV_DIR/lib/libcurand.so.10"
fi

# Export Paths
export LD_LIBRARY_PATH=$CONDA_ENV_DIR/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export JULIA_PROJECT="$PROJECT_ROOT/experiments/scalability"
export JULIA_DEPOT_PATH="$PROJECT_ROOT/.julia_depot" # Local depot to avoid home quota issues

# Verify
echo "Python: $(which python)"
echo "Julia: $(which julia)"
nvidia-smi

# --- Execution ---
cd "$PROJECT_ROOT"

echo "Instantiating Julia Environment..."
julia --project="$JULIA_PROJECT" -e 'using Pkg; Pkg.instantiate()'

echo "Starting Scalability Tests..."
# Pass --slurm flag to orchestrator.
# Arguments: <NODES> <GPUS_PER_NODE>
# $SLURM_NNODES is usually set, otherwise default to 1.
# $SLURM_GPUS_ON_NODE might be set, or we infer from SBATCH directive (4).

NODES=${SLURM_NNODES:-1}
GPUS_PER_NODE=4

bash experiments/scalability/run_scaling_tests.sh --slurm $NODES $GPUS_PER_NODE

echo "Job Complete."
