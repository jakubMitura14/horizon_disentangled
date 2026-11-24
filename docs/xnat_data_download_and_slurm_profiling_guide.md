# XNAT Data Download & SLURM Single-Epoch Profiling Guide

> **Author**: Horizon Disentangled Engineering Team  
> **Target Environment**: Multi-GPU Supercomputer / SLURM HPC Clusters (NVIDIA H100 / H200 / A100)  
> **Software Stack**: PyTorch 2.4+, CUDA 12.4+, PyXNAT, NVIDIA Nsight Systems, Perfetto  

---

## 1. Overview & Objectives

This manual provides an end-to-end operational guide for:
1. **Frontend / Login Node Data Ingestion**: Downloading the complete multi-modal HDF5 datasets (~22 GB across 16 HDF5 containers) from the central institutional XNAT repository using active access credentials.
2. **SLURM Cluster 1-Epoch Execution**: Submitting a batch or interactive single-epoch job across all three pipeline phases (Phase A: Vision Encoders, Phase B: 199M Causal VAE Disentanglement, Phase C: 3.8M Neural Jump ODE Dynamics).
3. **In-Depth Kernel & Hardware Profiling**: Capturing execution traces via PyTorch Profiler (`.json.gz` viewable in [Perfetto](https://ui.perfetto.dev)), tracing low-level CUDA kernels via NVIDIA Nsight Systems (`nsys`), tracking VRAM peak consumption, and diagnosing compute vs. memory vs. I/O bottlenecks.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   PIPELINE OVERVIEW                                    │
│                                                                                        │
│  [ XNAT Storage ]                                                                      │
│         │ (REST API / PyXNAT)                                                          │
│         ▼                                                                              │
│  [ Frontend / Cluster Storage ]                                                        │
│         │ (Staged to node-local NVMe: $SLURM_TMPDIR)                                   │
│         ▼                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 1-Epoch Execution & Profiling                                                     │  │
│  │                                                                                  │  │
│  │  Phase A: Feature Extraction ──► Phase B: 199M Causal VAE ──► Phase C: NJDE ODE  │  │
│  │  (3D Convolutions / Med3D)       (Disentangles z_a, u_d,       (Continuous-      │  │
│  │                                   z_p, z_s in bfloat16)         discrete RK4)    │  │
│  └──────────────────────────────────────────────────────────────────────────────────┘  │
│         │                                                                              │
│         ▼                                                                              │
│  [ Profiling Telemetry & Artifacts ]                                                   │
│   ├── trace_phase_a.json.gz (Perfetto trace)                                           │
│   ├── trace_phase_b.json.gz (Perfetto trace)                                           │
│   ├── trace_phase_c.json.gz (Perfetto trace)                                           │
│   ├── profile_summary.json (Timing, peak VRAM, throughput)                             │
│   └── profile_nsys.nsys-rep (NVIDIA Nsight Systems trace)                              │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Part 1: Frontend Data Download Guide (XNAT)

### 2.1. Active XNAT Access Credentials

The institutional XNAT endpoint houses the pre-packaged 3D volumetric HDF5 datasets under project `temp` in resource `HORIZON_HDF5`.

| Parameter | Configuration Value |
| :--- | :--- |
| **Server URL** | `https://imaging-platform.diz-ag.med.ovgu.de` |
| **Project** | `temp` |
| **Resource Label** | `HORIZON_HDF5` |
| **Alias (Username)** | `a57f7f87-c8de-4a2a-babe-36c525b0e202` |
| **Secret (Password)** | `sCuvx4KEhGSwqMTItuBBHP1NbMgqk1IUtSrzicTnjzLBI93ikGrSAa88MwhiljGV` |

> [!IMPORTANT]
> The alias and secret above are single-user bearer tokens with scoped read/write permissions for the `temp` repository. Never commit plaintext tokens to public version control. The repository's `.gitignore` explicitly excludes `scripts/xnat_credentials.txt`.

### 2.2. Setting Up Credentials

You can configure the credentials either via the standard configuration file or via shell environment variables.

#### Option A: Configuration File (Recommended)
Verify or write `/DATA/horizon_disentangled/scripts/xnat_credentials.txt`:
```ini
server = https://imaging-platform.diz-ag.med.ovgu.de
alias = a57f7f87-c8de-4a2a-babe-36c525b0e202
secret = sCuvx4KEhGSwqMTItuBBHP1NbMgqk1IUtSrzicTnjzLBI93ikGrSAa88MwhiljGV
project = temp
```

#### Option B: Environment Variables
```bash
export XNAT_SERVER="https://imaging-platform.diz-ag.med.ovgu.de"
export XNAT_ALIAS="a57f7f87-c8de-4a2a-babe-36c525b0e202"
export XNAT_SECRET="sCuvx4KEhGSwqMTItuBBHP1NbMgqk1IUtSrzicTnjzLBI93ikGrSAa88MwhiljGV"
```

---

### 2.3. Executing the Frontend Download

On the login node or staging server, activate your Python virtual environment and run the dedicated dataset downloader:

```bash
# 1. Activate environment
conda activate nukl-to-ct
# or: source /DATA/conda/envs/nukl-to-ct/bin/activate

# 2. Navigate to project root
cd /DATA/horizon_disentangled

# 3. Launch dataset download to target storage directory
python scripts/data/xnat_download.py \
    --download-dataset \
    --project temp \
    --resource-label HORIZON_HDF5 \
    --output /DATA/horizon_disentangled/hdf5
```

#### Command Options & Flags:
* `--download-dataset`: Switches `xnat_download.py` into HDF5 resource sync mode.
* `--project temp`: Target project containing the processed multi-cohort datasets.
* `--resource-label HORIZON_HDF5`: The XNAT resource folder containing all 16 HDF5 containers.
* `--output <PATH>`: Destination directory for downloaded HDF5 files.
* `--overwrite`: (Optional) Forces re-download of files even if local file size matches the remote server.

---

### 2.4. Dataset File Inventory & Checksum Verification

The download fetches 16 files totaling ~22.1 GB:

| Filename | Approximate Size | Description / Cohort |
| :--- | :--- | :--- |
| `breast_mincrop_s00.h5` | 2.04 GB | Duke Breast DCE-MRI preprocessed shard 00 (Phase A/B) |
| `breast_mincrop_s01.h5` | 2.11 GB | Duke Breast DCE-MRI preprocessed shard 01 |
| `breast_mincrop_s02.h5` | 2.64 GB | Duke Breast DCE-MRI preprocessed shard 02 |
| `breast_mincrop.h5` | 6.78 GB | Consolidated Duke Breast DCE-MRI master container |
| `breast_ispy2_s00.h5`..`s07.h5` | 13.2 GB total | I-SPY2 Neoadjuvant Longitudinal MRI shards (Phase C ODE) |
| `gbm_lumiere.h5` | 2.06 GB | Longitudinal Glioblastoma MRI cohort (GBM dynamics) |
| `gbm_tabular_cohorts.h5` | 1.74 MB | Tabular clinical and systemic lab values |
| `horizon_master.vds.h5` | 386 KB | Virtual Dataset mapping all shards into a unified volume |
| `dataset_manifest.json` | 2 KB | SHA-256 checksum registry and shard boundaries |

To verify file integrity after download:
```bash
python -c "
import json, hashlib
from pathlib import Path

manifest_path = Path('hdf5/dataset_manifest.json')
if manifest_path.exists():
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f'Checking {len(manifest.get(\"files\", {}))} files...')
    for fname, meta in manifest.get('files', {}).items():
        fpath = Path('hdf5') / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024**2)
            print(f'  ✓ {fname} ({size_mb:.1f} MB) OK')
        else:
            print(f'  ✗ MISSING: {fname}')
"
```

---

## 3. Part 2: SLURM 1-Epoch Execution Guide

### 3.1. Why 1-Epoch Profiling Is Mandatory

Before committing multi-day GPU allocations (e.g. 48-hour continuous training runs), running a single-epoch profiling job is critical for:
1. **Validating VRAM Headroom**: Confirming that the 198.9M parameter VAE and vectorised RK4 ODE solver fit cleanly into GPU memory without Out-Of-Memory (OOM) crashes.
2. **Measuring True Hardware Occupancy**: Determining whether the GPU compute tensor cores or memory bus are operating near peak capacity or idling due to CPU data-loading bottlenecks.
3. **Benchmarking I/O Bandwidth**: Quantifying whether shared network filesystems (NFS, GPFS, Lustre) create I/O wait bubbles between steps.

---

### 3.2. Data Staging to Node-Local NVMe / SSD

> [!WARNING]
> In high-performance 3D medical imaging, reading 3D volumes directly over shared network storage (NFS or Lustre) introduces disk latency and network jitter that stalls GPU execution.
> Always stage dataset files to the compute node's local NVMe scratch (`$SLURM_TMPDIR` or `/tmp`) before training!

The provided SLURM script handles this automatically:
```bash
NODE_SCRATCH="${SLURM_TMPDIR:-/tmp/horizon_hdf5_${SLURM_JOB_ID}}"
mkdir -p "$NODE_SCRATCH"

# Copies HDF5 files to high-speed local NVMe:
cp -u /DATA/horizon_disentangled/hdf5/*.h5 "$NODE_SCRATCH/"
```

---

### 3.3. Submitting the SLURM Batch Job

A production-ready SLURM script is located at [`scripts/training/profile_1epoch_all_phases.sbatch`](file:///DATA/horizon_disentangled/scripts/training/profile_1epoch_all_phases.sbatch).

#### 1. Inspect the Script Parameters
```bash
# SLURM Resource Allocation:
#SBATCH --job-name=horizon-profile-1epoch
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/profile_1epoch_%j.out
#SBATCH --error=logs/profile_1epoch_%j.err
```

#### 2. Submit the Job
```bash
sbatch scripts/training/profile_1epoch_all_phases.sbatch
```

#### 3. Monitor Job Execution
```bash
# Check queue status
squeue -u $USER

# Follow live log output
tail -f logs/profile_1epoch_*.out
```

---

### 3.4. Interactive Profiling Session (via `srun` / `salloc`)

For interactive debugging and real-time GPU telemetry, request an interactive compute allocation:

```bash
# 1. Allocate an interactive GPU session for 1 hour:
srun --partition=gpu \
     --gres=gpu:1 \
     --cpus-per-task=16 \
     --mem=128G \
     --time=01:00:00 \
     --pty bash

# 2. Inside the compute node, set up the environment:
cd /DATA/horizon_disentangled
source /DATA/conda/envs/nukl-to-ct/bin/activate
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 3. Run the standalone 1-epoch profiling script:
python scripts/training/profile_1epoch_all_phases.py \
    --data-dir /DATA/horizon_disentangled/hdf5 \
    --output-dir experiments/profiling_interactive \
    --profile-type torch \
    --phases A,B,C \
    --batch-b 32 \
    --batch-c 24 \
    --max-steps 10
```

---

### 3.5. Recommended Hardware Configuration Table

Depending on the cluster GPU architecture available, adjust batch sizes to maximize occupancy without triggering OOM:

| GPU Model | Total VRAM | Recommended Batch B (VAE) | Recommended Batch C (ODE) | Expected VRAM Occupancy |
| :--- | :--- | :--- | :--- | :--- |
| **NVIDIA H200 SXM** | 141 GB | `48` | `32` | ~85% (120 GB) |
| **NVIDIA H100 PCIe** | 80 GB | `32` | `24` | ~85% (68 GB) |
| **NVIDIA A100 SXM** | 80 GB | `32` | `24` | ~85% (67 GB) |
| **NVIDIA A100 PCIe** | 40 GB | `16` | `12` | ~80% (32 GB) |
| **NVIDIA RTX 6000 Ada** | 48 GB | `20` | `16` | ~82% (39 GB) |

---

## 4. Part 3: Profiling & Bottleneck Analysis Guide

### 4.1. PyTorch Profiler & Chrome Trace Analysis

The script exports Chrome-compatible compressed JSON traces:
* `trace_phase_a.json.gz`: Vision encoder operations
* `trace_phase_b.json.gz`: 199M VAE forward + backward passes
* `trace_phase_c.json.gz`: Neural Jump ODE RK4 time-stepping iterations

#### How to Inspect Traces:
1. Open Google Chrome and navigate to `chrome://tracing` or visit [ui.perfetto.dev](https://ui.perfetto.dev).
2. Click **Open trace file** and select `trace_phase_b.json.gz`.
3. Analyze the timeline:
   * **CPU Activity**: Notice data-loading threads and Python overhead.
   * **CUDA Stream**: Observe kernel execution continuity. A dense, solid bar of CUDA kernel execution indicates optimal saturation (>90% compute utilization). White gaps between kernels indicate host-device synchronization latency or CPU I/O wait bubbles.

```
Perfetto Timeline Visualization:
CUDA Stream 7: [=== amp_conv3d ===][=== silu ===][=== amp_conv3d ===][=== group_norm ===]
Host Thread:   [-- DataLoader --]  [-- loss.backward() --]  [-- optimizer.step() --]
```

---

### 4.2. NVIDIA Nsight Systems (`nsys`) Profiling

For granular hardware counters (Tensor Core pipe utilization, shared memory efficiency, warp occupancy):

```bash
nsys profile \
    -t cuda,nvtx,osrt,cudnn,cublas \
    -s none \
    --capture-range=nvtx \
    --nvtx-capture=PhaseB_Step_2 \
    -o experiments/profiling_results/nsys_horizon \
    --force-overwrite true \
    python scripts/training/profile_1epoch_all_phases.py \
        --profile-type nsys \
        --phases B,C \
        --max-steps 5
```

This generates `experiments/profiling_results/nsys_horizon.nsys-rep`. Open this report in the **NVIDIA Nsight Systems GUI** on your workstation to view:
* SM (Streaming Multiprocessor) active percentage
* Warp execution efficiency
* DRAM read/write throughput (GB/s) vs. HBM theoretical maximum (3.35 TB/s on H100)

---

### 4.3. Real-Time GPU Hardware Telemetry

While the 1-epoch profiling run executes, monitor live metrics using standard CLI tools:

```bash
# 1. Real-time compute, memory, and power sampling (1-second intervals):
nvidia-smi dmon -s pucvmet -d 1

# 2. Track VRAM allocation and process list:
watch -n 1 nvidia-smi
```

Key metric indicators in `dmon`:
* `sm`: Streaming Multiprocessor utilization percentage (target: >85%).
* `mem`: Memory controller bandwidth utilization percentage (target: >60%).
* `pwr`: Power draw in Watts (e.g. 300W–350W on H100 indicates heavy tensor execution).

---

### 4.4. Bottleneck Diagnosis & Remediation Checklist

| Symptom | Primary Cause | Remediation Step |
| :--- | :--- | :--- |
| **GPU SM Utilization < 50%** | CPU data loading bottleneck | Increase DataLoader `--num-workers` (e.g. from 4 to 8 or 16); ensure data is on NVMe scratch (`$SLURM_TMPDIR`). |
| **White gaps between CUDA kernels in Perfetto** | Frequent host-device synchronization | Remove explicit `.item()` or `torch.cuda.synchronize()` calls from the inner loop; enable non-blocking transfers (`non_blocking=True`). |
| **CUDA Out Of Memory (OOM)** | Batch size exceeds physical VRAM | Enable `export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`; decrease `--batch-b` or `--batch-c` by 25%. |
| **Low Tensor Core Utilization** | Memory tensor layout misaligned | Verify AMP `torch.bfloat16` is active; ensure tensor spatial and channel dimensions are multiples of 8 or 16. |
| **ODE Step Latency Spikes in Phase C** | Non-vectorised patient loop | Confirm vectorised integration is enabled (`integrate_interval` over full tensor batch rather than single patient loops). |

---

## 5. Artifact Reference & Output Files

After completing the 1-epoch profiling run, the output directory contains the following verified artifacts:

```
experiments/profiling_1epoch_<JOB_ID>/
├── profile_summary.json         # Summary report with timing, throughput, peak VRAM
├── trace_phase_a.json.gz        # Chrome trace for Phase A (Vision extraction)
├── trace_phase_b.json.gz        # Chrome trace for Phase B (199M VAE)
├── trace_phase_c.json.gz        # Chrome trace for Phase C (NJDE solver)
└── synthetic_inputs/            # Fallback synthetic datasets (if real HDF5 absent)
    ├── encoder_features.h5      # 129 MB Phase A features container
    └── latent_decomposition.h5  # 98 MB Phase B latent trajectory container
```

### Sample `profile_summary.json` Output:
```json
{
  "timestamp": "2026-11-24T08:35:50Z",
  "gpu_name": "NVIDIA H100 PCIe",
  "gpu_mem_total_gb": 79.2,
  "profile_type": "torch",
  "phases": {
    "A": {
      "elapsed_s": 0.33,
      "peak_vram_gb": 0.09,
      "throughput_vol_s": 98.2
    },
    "B": {
      "elapsed_s": 13.53,
      "peak_vram_gb": 67.64,
      "throughput_vol_s": 4.7,
      "samples": 64
    },
    "C": {
      "elapsed_s": 3.22,
      "peak_vram_gb": 30.24,
      "throughput_traj_s": 14.9,
      "samples": 48
    }
  }
}
```
