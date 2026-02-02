# Scalability Experiments for EuroHPC JUPITER

This directory contains the experimental setup for benchmarking the scalability of the CausalPCa Generative AI models.

## Approaches
1.  **Julia (Lux + MPI.jl)**: Uses `MPI.jl` to simulate distributed gradient synchronization (Allreduce).
2.  **Python (PyTorch Lightning)**: Uses `pytorch_lightning` with `DDPStrategy` to simulate distributed training.

## Distributed Simulation (Localhost)
We verify the correctness of the distributed code logic by running on `localhost` using OpenMPI with oversubscription. This simulates the multi-process environment of a Supercomputer node (or multiple nodes) without requiring physical access to a cluster during development.

## Requirements
*   **Julia**: `Lux`, `MPI`, `Optimisers`, `Zygote`.
*   **Python**: `torch`, `pytorch-lightning`.
*   **System**: `openmpi-bin` (for `mpirun`), `libopenmpi-dev`.

## Running the Experiments
```bash
bash run_scaling_tests.sh
```
This script launches both Julia and Python benchmarks for 1, 2, and 4 processes and saves the timing results to `results_scaling.csv`.

## Future Porting to Slurm
To run on JUPITER, the `run_scaling_tests.sh` script will be modified to use `srun` instead of `mpiexecjl` / `python` directly, and will target the specific GPU partitions.
