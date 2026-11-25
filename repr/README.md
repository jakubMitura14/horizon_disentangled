# Horizon Disentangled — Reproducible SLURM Scaling & Profiling Suite (`repr/`)

Self-contained reproducibility package for benchmarking, profiling, and scaling the **Horizon Disentangled** 3D oncology pipeline on **JUPITER** (4× NVIDIA GH200 120 GB, ~95 GB usable CUDA VRAM per GPU).

## Directory Layout

```text
repr/
├── README.md                              # This file: setup, SLURM execution, profiling, results
├── requirements.txt                       # Python package requirements (PyTorch 2.6+/2.9+, MONAI, etc.)
├── data/
│   ├── README.md                          # Data specification (synthetic GPU tensors + cohort metadata CSVs)
│   ├── dataset_encoded.csv                # Encoded clinical cohort table
│   ├── dataset_summary.csv                # Cohort summary metadata
│   ├── phased_pipeline.json               # Phased pipeline configuration
│   └── reference_results/                 # Verified JSON outputs from JUPITER GH200 jobs
│       ├── scaling_1vs4gpu_3114093.json
│       ├── native_max_phase_a_3114634.json
│       ├── native_max_phase_b_3114634.json
│       ├── native_max_phase_c_3114634.json
│       ├── phase_b_dwsep_l512_b4_3116189.json
│       ├── phase_c_batch64_3116189.json
│       └── phase_c_2mm_b6_3116189.json
├── scripts/
│   ├── profile_scaling_benchmark.py       # Main PyTorch/FSDP/DDP/cuFFT benchmark runner
│   ├── profile_scaling_jupiter.sbatch     # Job 1: Apples-to-apples 1-GPU vs 4-GPU scaling
│   ├── profile_native_max_jupiter.sbatch  # Job 2: Zero-cropping (256×256×192) max-occupancy
│   └── profile_push_occupancy_v2.sbatch   # Job 3: Phase B DW-sep+ckpt (79%) & Phase C batch=64 (95.1%) / 2mm
├── src/
│   └── pipeline/
│       ├── stage1_supervisors/models.py   # Phase A: SwinUNETR / VoComni-H / VISTA3D / MedNeXt encoders
│       ├── stage2_causal_vae/             # Phase B: TwoScaleHierarchicalCausalVAE + MedVAEBackbone
│       └── stage4_dynamics/               # Phase C: Spectral cuFFT PDE + Systemic NJ-ODE + RK4 solver
└── docs/
    ├── README.md                          # Documentation index
    ├── benchmark_and_slurm.md             # Detailed docs for profile_scaling_benchmark.py & .sbatch scripts
    ├── stage1_models.md                   # Detailed docs for all classes & functions in stage1_supervisors
    ├── stage2_causal_vae.md               # Detailed docs for all classes & functions in stage2_causal_vae
    └── stage4_dynamics.md                 # Detailed docs for all 6 modules in stage4_dynamics
```

---

## 1. Environment Setup (JUPITER Supercomputer)

The SLURM scripts load JUPITER's `Stages/2026` modules and activate the Python virtual environment:

```bash
module purge
module load Stages/2026
module load GCCcore/14.3.0
module load Python
source /e/project1/numedai-scale/mitura1/venvs/activate_lymph.sh
```

If recreating the environment from scratch on another system:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Key environment variables set in all `.sbatch` scripts:
- `PYTORCH_ALLOC_CONF="expandable_segments:True"` — reduces CUDA allocator fragmentation on GH200
- `NCCL_P2P_LEVEL=NVL` — uses NVLink-C2C for intra-node GPU communication
- `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` — ensures clean error propagation across ranks

---

## 2. Running on SLURM

Submit any experiment directly from `repr/` (recommended) or from the repository root:

```bash
cd /e/project1/numedai-scale/mitura1/horizon_disentangled/repr

# 1. Apples-to-apples 1-GPU vs 4-GPU scaling comparison (97.7–99.2% efficiency)
sbatch scripts/profile_scaling_jupiter.sbatch

# 2. Zero-cropping full native 3D resolution (4, 256, 256, 192) & 4mm PDE
sbatch scripts/profile_native_max_jupiter.sbatch

# 3. High-occupancy Phase B (DW-sep + grad checkpoint) & Phase C (batch=64 & 2mm PDE)
sbatch scripts/profile_push_occupancy_v2.sbatch
```

Monitor jobs and inspect outputs:

```bash
squeue -u "$USER"
sacct -j <JOB_ID> --format=JobID,State,Elapsed,ExitCode
tail -f /e/scratch/numedai-scale/mitura1/horizon_logs/<log_name>_<JOB_ID>.out
```

---

## 3. Running Directly / Custom Profiling

### Single-GPU Quick Verification
```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/profile_scaling_benchmark.py \
    --ngpus 1 --phases A,B,C --batch-a 2 --batch-b 4 --batch-c 8 \
    --max-steps 3 --use-spectral --run-spectral-micro \
    --output-dir /tmp/horizon_test --run-label smoke_1gpu
```

### 4-GPU `torchrun` Execution
```bash
torchrun --nnodes=1 --nproc_per_node=4 scripts/profile_scaling_benchmark.py \
    --ngpus 4 --phases B \
    --latent-input-shape 16,64,64,48 --l2-channels 256 --batch-b 4 --grad-accum-b 4 \
    --max-steps 8 --use-fsdp --no-compile-b \
    --output-dir /tmp/horizon_test --run-label phase_b_native
```

### Profiling with PyTorch Profiler / Nsight Systems
To capture a CUDA trace (`nsys`) on a single phase:
```bash
nsys profile -t cuda,nvtx,osrt -o phase_b_trace \
    torchrun --nnodes=1 --nproc_per_node=4 scripts/profile_scaling_benchmark.py \
    --ngpus 4 --phases B --latent-input-shape 16,64,64,48 --l2-channels 256 \
    --batch-b 4 --max-steps 4 --use-fsdp --no-compile-b
```

---

## 4. Reference Results (JUPITER 4× GH200, 95 GB Usable VRAM)

### A. Apples-to-Apples 1-GPU vs 4-GPU Scaling (`Job 3114093`)
Both runs use identical `compile=False, use_spectral=True` settings:

| Phase | Model | 1-GPU | 4-GPU | Speedup | Efficiency | VRAM/GPU |
|-------|-------|------:|------:|--------:|-----------:|---------:|
| **A** | `SwinUNETR-48` (64.4M) | 27.54 vol/s | 107.59 vol/s | **3.91×** | **97.7%** | 22.88 GB (24.1%) |
| **B** | `CausalVAE` (756.1M) | 21.65 vol/s | 84.70 vol/s | **3.91×** | **97.8%** | 72.48 GB (76.3%) |
| **C** | `Spectral-GBM` (3.8M, 4mm) | 17.29 traj/s | 68.63 traj/s | **3.97×** | **99.2%** | 79.14 GB (83.3%) |

### B. Zero-Cropping Native 3D Max-Occupancy (`Jobs 3114634` & `3116189`)

| Phase | Configuration | Volume / Grid | Batch/GPU | Throughput | Peak VRAM | Occupancy |
|-------|---------------|---------------|:---------:|-----------:|----------:|----------:|
| **A** | `VoComni-H` (998.9M), FSDP + compile | `(4, 256, 256, 192)` | 2 | 10.79 vol/s | 50.16 GB | **52.8%** |
| **B** | `CausalVAE` `l2=256`, FSDP, no-ckpt | `(16, 64, 64, 48)` → `(4, 256, 256, 192)` | 4 | 14.22 vol/s | 66.16 GB | **69.6%** |
| **B** | `CausalVAE` `l2=512`, DW-sep + ckpt | `(16, 64, 64, 48)` → `(4, 256, 256, 192)` | 4 | 0.56 vol/s | 75.08 GB | **79.0%** |
| **C** | `Spectral-GBM`, 4mm | `(48, 56, 40)` | 56 | 68.86 traj/s | 79.14 GB | **83.3%** |
| **C** | `Spectral-GBM`, 4mm | `(48, 56, 40)` | 64 | 69.02 traj/s | 90.39 GB | **95.1%** |
| **C** | `Spectral-GBM`, 2mm ultra-fine | `(96, 112, 80)` | 6 | 8.60 traj/s | 69.85 GB | **73.5%** |

---

## 5. Detailed Documentation

See [`docs/README.md`](docs/README.md) for complete file-by-file and function-by-function documentation of every module in `repr/`.
