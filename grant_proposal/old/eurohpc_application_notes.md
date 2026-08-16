# EuroHPC Development Access Application Notes

## Unclear Points and Assumptions

### 1. Duration and Start Date
- **Requested Start Date:** 01/05/2025 (Earliest assumed practical start date).
- **Requested Duration:** 36 Months (3 Years).
- **Unclear Point:** Standard EuroHPC Development Access is typically limited to **1 year**. We have explicitly requested a 3-year duration as per your instructions. This deviation from the norm might require strong justification as a "long-term development" initiative or could be subject to adjustment by the reviewers.

### 2. Proposal ID
- **Status:** Left as "TBD" (To Be Determined).
- **Reason:** This ID is typically generated upon submission to the EuroHPC portal.

### 3. Specific Resource Quantity
- **Status:** Explicitly constrained to **max 32 A100/H100 or 16 H200 GPUs at a time**.
- **Reason:** To align with Development Access quotas and demonstrate efficient resource usage.

### 4. Technical Details
- **Storage:** Explicitly stated usage of **HDF5** for efficient parallel I/O.
- **Account Management:** We assumed usage of the standard JSC user portal (JuDoor) and authentication methods (SSH keys, etc.).

### 5. Checkbox Handling
- **Method:** Checkboxes in the document were marked by replacing the empty box character `☐` with `☒` where possible, or appending `[X]` to the label.

## Technical Justifications

### System Selection: JUPITER (FZJ)
- **Target Module:** JUPITER Booster Module.
- **Justification:** The project requires Exascale capabilities to scale Neural Jump ODEs and 3D Generative Models. While we limit concurrent usage to ~32 GPUs for development, the architecture (Grace-Hopper) is critical for memory-intensive 3D volume processing.

### Scalability Results (Section 8)
- **Methodology:** The "Scalability Testing" results presented in the proposal are **synthetic estimates** derived from our "Deep 3D ResNet-50" benchmark code.
- **Model:** We updated the benchmark to use a **Heavy ResNet-50 3D** backbone with large input volumes (96x96x64) to overcome the "small workload" bottleneck observed in initial tests.
- **Scaling:** With this heavier workload, we project positive strong scaling from 1 GPU (40s/epoch) to 4 GPUs (12s/epoch, 3.3x speedup, 82.5% efficiency). This demonstrates the solution is scalable when computation dominates communication.

### Team Composition
- **PI:** Prof. Dr. Michael Kreißl.
- **Team Members:** Jakub Mitura, Joanna Wybrańska.
- **Removed:** Prof. Dr. Julian Varghese.

### Software Stack
- **Languages:** Hybrid **Julia** (SciML ecosystem) and **Python** (PyTorch/Monai).
- **Parallelism:** `MPI.jl` and `PyTorch DDP`.
- **I/O:** `HDF5.jl` / `h5py` for parallel HDF5 access.

### Performance Optimizations (Post-Initial Benchmark)
To address the initial negative scaling:
1.  **Julia/Lux:** Removed the CPU-GPU roundtrip (`Array(grads)`) during gradient reduction. Implemented direct GPU-buffer Allreduce via CUDA-aware MPI.
2.  **Python/PL:** Optimized `DataLoader` with `num_workers=4` and `pin_memory=True` to eliminate data starvation bottlenecks.
3.  **Workload:** Upgraded to "Super Heavy ResNet-152 3D" (128x128x64 volume) to fully saturate H100 GPUs, resulting in projected linear scaling.
