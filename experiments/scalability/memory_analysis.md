# Memory Analysis & Profiling Guide for EuroHPC JUPITER

## 1. Objective
The primary goal of this guide is to standardize the memory profiling methodology for the "Generative AI for Nuclear Medicine Optimization" project on the JUPITER Booster (NVIDIA Grace-Hopper GH200). We must justify the allocation of large-memory GPUs (80GB+ HBM3) by demonstrating that our **Super Heavy 3D ResNet-152** and **Neural Jump ODE** models are memory-bound or require the full capacity for efficient batch processing.

Key Metrics to Validate:
- **Peak HBM Usage**: Must approach 80GB (e.g., >70GB) to justify the hardware.
- **Memory Bandwidth Utilization**: High utilization justifies HBM3 over DDR.
- **Unified Memory Traffic**: Validate the efficiency of Grace-Hopper NVLink-C2C for host-device transfers.

## 2. Methodologies by Framework

### 2.1 Julia (Lux.jl + CUDA.jl)

Julia's `CUDA.jl` provides low-level memory management tools.

**A. Instantaneous Memory Check**
Insert these calls within the training loop (e.g., after a forward pass) to log usage:
```julia
using CUDA
# Print summary of memory usage
CUDA.memory_status()

# Programmatic access (in bytes)
free_bytes, total_bytes = CUDA.Mem.info()
used_bytes = total_bytes - free_bytes
println("GPU Memory Used: $(used_bytes / 1024^3) GB")
```

**B. Profiling Allocations**
Use `CUDA.@profile` to visualize memory events in Nsight Systems (see Section 3).
To debug allocation hotspots (e.g., unnecessary copies):
```bash
JULIA_CUDA_MEMORY_POOL=none julia --project src/train_lux_distributed.jl
```
*Note: Disabling the memory pool is slow but exposes every allocation to the profiler.*

**C. System Image Impact**
Ensure you are using the custom system image (`precompile_sysimage.jl`). Loading heavy dependencies at runtime can fragment memory. Always run with:
```bash
julia --sysimage=sysimage.so ...
```

### 2.2 Python (PyTorch Lightning)

PyTorch provides a high-level memory tracker.

**A. Peak Memory Tracking**
Add this to your `LightningModule.on_train_epoch_end`:
```python
import torch

# Current usage
current_mem = torch.cuda.memory_allocated() / 1024**3
# Peak usage since last reset
max_mem = torch.cuda.max_memory_allocated() / 1024**3

print(f"Epoch End - Current: {current_mem:.2f} GB, Peak: {max_mem:.2f} GB")
torch.cuda.reset_peak_memory_stats()
```

**B. PyTorch Profiler**
Wrap the training step to capture a memory timeline:
```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True
) as prof:
    model(input)

print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
```

## 3. System-Level Profiling (Nsight Systems)

**Nsight Systems (`nsys`)** is the definitive tool for JUPITER. It captures the interaction between the Grace CPU and Hopper GPU.

### Command for JUPITER (Slurm)
Modify your `srun` command to include the profiler. This generates a `.nsys-rep` file viewable in the Nsight GUI.

```bash
# Profile CUDA kernels, NVTX annotations, and OS runtime events
nsys profile \
  --trace=cuda,nvtx,osrt \
  --gpu-metrics-device=all \
  --cuda-memory-usage=true \
  --output=profile_memory_%p \
  python3 src/train_lightning.py
```

**What to Look For:**
1.  **Memory Throughput**: Check the "GPU Metrics" row for HBM bandwidth. It should be high (near peak) during training steps.
2.  **H2D/D2H Transfers**: On Grace-Hopper, these should be fast via NVLink-C2C. Excessive transfers indicate poor data caching or lack of `pin_memory=True`.
3.  **Fragmentation**: If "Memory Usage" is high but "Allocation" is low, the allocator is fragmented.

## 4. Continuous Monitoring Script

For long-running jobs, run a background monitoring process to log memory usage to a CSV. This is lightweight and runs alongside the job.

**`monitor_memory.sh`**:
```bash
#!/bin/bash
OUTPUT_FILE="memory_log_${SLURM_JOB_ID}.csv"
echo "timestamp, gpu_idx, utilization_gpu, memory_used_mb, memory_total_mb" > $OUTPUT_FILE

while true; do
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total \
               --format=csv,noheader >> $OUTPUT_FILE
    sleep 1
done
```

**Usage in Slurm Script**:
```bash
./monitor_memory.sh &
MONITOR_PID=$!

srun julia --project src/train_lux_distributed.jl

kill $MONITOR_PID
```
