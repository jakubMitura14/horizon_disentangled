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
- **Status:** Not filled.
- **Reason:** Development access usually grants a small, fixed quota (e.g., node hours) rather than a requested amount. We assumed the standard allocation for this track.

### 4. Technical Details
- **MPI-IO:** We assumed usage of MPI-IO via HDF5/NetCDF wrappers in Julia/Python, as this is standard for high-performance I/O on Lustre/GPFS (ExaFLASH).
- **Account Management:** We assumed usage of the standard JSC user portal (JuDoor) and authentication methods (SSH keys, etc.), though these specifics are usually handled *after* acceptance.

### 5. Checkbox Handling
- **Method:** Checkboxes in the document were marked by appending `[X]` to the corresponding text label. This ensures the selection is visible even if the visual checkbox character itself isn't perfectly interactive.

## Technical Justifications

### System Selection: JUPITER (FZJ)
- **Target Module:** JUPITER Booster Module.
- **Justification:** The project requires Exascale capabilities to scale Neural Jump ODEs and 3D Generative Models to thousands of GPUs. The Booster module's **NVIDIA Grace-Hopper Superchips (GH200)** are specifically chosen for their unified memory architecture, which is critical for processing large 3D medical volumes that exceed standard GPU memory limits.

### Team Composition
- **PI:** Prof. Dr. Michael Kreißl.
- **Team Members:** Jakub Mitura, Joanna Wybrańska.
- **Removed:** Prof. Dr. Julian Varghese (as per instruction).

### Software Stack
- **Languages:** Hybrid **Julia** (SciML ecosystem for Causal/ODE modeling) and **Python** (PyTorch/Monai for standard deep learning layers).
- **Parallelism:** Explicit usage of `MPI.jl` and `PyTorch DDP` for scaling.
- **Optimization:** Focus on leveraging NVLink 4 and mixed-precision training (FP8/FP16) on Hopper GPUs.
