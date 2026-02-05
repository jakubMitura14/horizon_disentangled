# Scalability Experiments for EuroHPC JUPITER

This directory contains the experimental setup for benchmarking the scalability of the CausalPCa Generative AI models (VAE and NJDE).

## Objectives
1.  **Strong Scaling**: Measure the reduction in training time per epoch as the number of GPUs increases (1 -> 2 -> 4) within a single node.
2.  **Efficiency Analysis**: Quantify the parallel efficiency to identify communication bottlenecks (e.g., gradient synchronization overhead).
3.  **Validation**: Ensure model convergence remains stable with distributed large-batch training.

## Architectures
To ensure the GPU workload is sufficient to measure scalability (avoiding the "small workload" bottleneck), we use **Heavy 3D ResNet** backbones in both Julia and Python benchmarks.
*   **Input**: 3D Volume `(Batch=4, Channels=1, Depth=64, Height=96, Width=96)`
*   **Model**: 3D ResNet-18 variant with `Conv3D` layers, Batch Normalization, and Residual connections.

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

### Slurm Job Script (`submit_scaling_eurohpc.sbatch`)
The provided sbatch script runs a **Single Node Strong Scaling** test, iterating through 1, 2, and 4 GPUs on a single allocated node.

```bash
sbatch experiments/scalability/submit_scaling_eurohpc.sbatch
```

### Manual Execution via `srun`
If you are in an interactive session (`salloc`), you can run individual steps using `run_scaling_tests.sh` in Slurm mode:

```bash
# Auto-detect available GPUs and run full suite
bash experiments/scalability/run_scaling_tests.sh --slurm
```

## Results Analysis
The scripts output logs to `logs/`. Look for "Epoch Time" and "Speedup" metrics. The `results_scaling.csv` (generated in Local mode) provides a template for data collection.
