# Scalability Experiments for EuroHPC JUPITER

This directory contains the experimental setup for benchmarking the scalability of the CausalPCa Generative AI models (VAE and NJDE).

## Objectives
1.  **Strong Scaling**: Measure the reduction in training time per epoch as the number of GPUs increases (1 -> 4 -> 8 -> 16 -> 32).
2.  **Efficiency Analysis**: Quantify the parallel efficiency to identify communication bottlenecks (e.g., gradient synchronization overhead).
3.  **Validation**: Ensure model convergence remains stable with distributed large-batch training.

## Directory Structure
*   `src/`: Source code for Julia (Lux) and Python (PyTorch Lightning) training drivers.
*   `logs/`: Output logs from experiments.
*   `run_scaling_tests.sh`: Orchestration script for local and Slurm execution.

## 1. Localhost Simulation (Development)
You can run a simulation of distributed training on a single machine (CPU-only or GPU) to verify code correctness and MPI setup.

```bash
bash run_scaling_tests.sh
```
*   **Requirements**: OpenMPI, Julia (Lux, MPI.jl), Python (PyTorch Lightning).
*   **Output**: `results_scaling.csv` containing timing metrics for 1, 2, and 4 processes.

## 2. Execution on GPU Server (Slurm/JUPITER)
To run the actual scalability benchmarks on the JUPITER Booster Module (or similar Slurm clusters with NVIDIA GPUs), follow these steps.

### Prerequisites
*   Load necessary modules (example for JUPITER):
    ```bash
    module load Julia/1.10
    module load Python/3.11
    module load CUDA/12
    module load OpenMPI  # or ParaStationMPI
    ```
*   Ensure `Project.toml` is instantiated:
    ```bash
    julia --project=experiments/scalability -e 'using Pkg; Pkg.instantiate()'
    ```

### Example Slurm Job Script (`submit_scaling.sbatch`)
Create a submission script to request resources (e.g., 4 Nodes, 16 GPUs).

```bash
#!/bin/bash
#SBATCH --job-name=causalpca_scaling
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --partition=booster  # JUPITER Booster Partition
#SBATCH --account=<your_project_id>
#SBATCH --output=logs/slurm-%j.out

# Export Environment
export JULIA_DEPOT_PATH="/p/project/<project_id>/user/.julia"
export SRUN_CPUS_PER_TASK=16

# Run the Orchestrator in Slurm Mode
# Arguments: --slurm <NODES> <GPUS_PER_NODE>
bash run_scaling_tests.sh --slurm $SLURM_JOB_NUM_NODES 4
```

### Manual Execution via `srun`
If you are in an interactive session (`salloc`), you can run individual steps:

**Julia (Lux + MPI + CUDA):**
```bash
srun --ntasks=16 --ntasks-per-node=4 --gpus-per-node=4 \
    julia --project=experiments/scalability src/train_lux_distributed.jl
```

**Python (PyTorch Lightning + NCCL):**
```bash
srun --ntasks=16 --ntasks-per-node=4 --gpus-per-node=4 \
    python3 src/train_lightning.py \
    --accelerator gpu \
    --strategy ddp \
    --nodes 4 \
    --gpus 4
```

## Results Analysis
The scripts output logs to `logs/`. Look for "Epoch Time" and "Speedup" metrics. The `results_scaling.csv` (generated in Local mode) provides a template for data collection. In Slurm mode, parse the logs to populate the final report tables.
