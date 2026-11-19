# XNAT Dataset Access & SLURM Cluster Staging Guide

**Server URL**: `https://imaging-platform.diz-ag.med.ovgu.de`  
**Project**: `temp`  
**Resource Label**: `HORIZON_HDF5`  
**Dataset Size**: ~22 GB (Native Uncropped Multimodal 3D Volumetric Datasets)  
**Synchronization Tool**: [scripts/xnatt.py](file:///DATA/horizon_disentangled/scripts/xnatt.py)  
**Credentials Configuration**: [scripts/xnat_credentials.txt](file:///DATA/horizon_disentangled/scripts/xnat_credentials.txt)

> [!NOTE]
> `breast_mincrop.h5` (6.7 GB) exceeded the nginx 4 GB upload limit and was split into three shards:
> `breast_mincrop_s00.h5` (360 patients), `breast_mincrop_s01.h5` (360 patients), `breast_mincrop_s02.h5` (361 patients).
> To regenerate shards locally, run: `python scripts/split_mincrop.py`

---

## 1. Dataset Overview

The dataset stored in the XNAT `temp` project under the `HORIZON_HDF5` resource provides the complete, native, uncropped 3D volumetric imaging and longitudinal clinical data required for training the 5-stage Horizon Disentangled architecture on multi-GPU SLURM clusters (e.g. 4–8× NVIDIA H200/H100 nodes).

```mermaid
flowchart TD
    subgraph XNAT_Server ["XNAT Server (imaging-platform.diz-ag.med.ovgu.de)"]
        direction TB
        subgraph Temp_Project ["Project: temp"]
            subgraph Horizon_Resource ["Resource: HORIZON_HDF5 (ID: 40120)"]
                F1["gbm_lumiere.h5 (2.1 GB)<br/>91 Glioblastoma Patients (4x256x256x192)"]
                F2["breast_mincrop_s00/01/02.h5 (7.1 GB)<br/>1,081 DUKE Breast DCE (split 3×~2.3 GB)"]
                F3["breast_ispy2_s00..s07.h5 (13 GB)<br/>985 I-SPY 2 Multicenter Cohort"]
                F4["gbm_tabular_cohorts.h5 (1.8 MB)<br/>Genomic, Survival & Clinical Covariates"]
                F5["horizon_master.vds.h5 (387 KB)<br/>HDF5 Virtual Dataset Catalog"]
                F6["dataset_manifest.json<br/>Integrity Checksums & Metadata"]
            end
        end
    end

    subgraph Cluster_Deployment ["SLURM Compute Cluster Node"]
        direction TB
        CLI["scripts/xnatt.py --download-dataset"]
        NVME["Local NVMe Burst Buffer (/tmp/horizon_hdf5)"]
        H200["8× NVIDIA H200 SXM (141 GB HBM3e)"]
    end

    Horizon_Resource -->|Fast Stream (~150 MiB/s)| CLI
    CLI --> NVME
    NVME -->|Zero-Copy cuFile / Pinned Mem| H200
```

### File Inventory & Technical Dimensions

| File Name | Size | Domain | Subjects / Visits | Key Tensor Shapes | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`gbm_lumiere.h5`** | 2.1 GB | Glioblastoma | 91 patients (longitudinal) | MRI: `(4, 256, 256, 192)`<br/>Mask: `(2, 256, 256, 192)` | Native 4-channel MRI (FLAIR, T1w, T1ce, T2w) + viable/necrotic tumor masks. |
| **`breast_mincrop_s00.h5`** | 2.1 GB | Breast DCE | 360 patients (DUKE_001–DUKE_362) | DCE: `(3, 256, 256, 28–64)`<br/>Mask: `(1, 256, 256, 28–64)` | Shard 0 of 3: 3 kinetic phases (PE2, PE6, SER) + 3D whole-breast parenchyma segmentations. |
| **`breast_mincrop_s01.h5`** | 2.2 GB | Breast DCE | 360 patients (DUKE_363–DUKE_722) | DCE: `(3, 256, 256, 28–64)`<br/>Mask: `(1, 256, 256, 28–64)` | Shard 1 of 3. Same schema as s00. Regenerate all shards with `python scripts/split_mincrop.py`. |
| **`breast_mincrop_s02.h5`** | 2.8 GB | Breast DCE | 361 patients (DUKE_723–SPY1_1239) | DCE: `(3, 256, 256, 28–64)`<br/>Mask: `(1, 256, 256, 28–64)` | Shard 2 of 3. Includes DUKE_723–SPY1 cohort, larger 512×512 patients. |
| **`breast_ispy2_s00.h5`** | 1.3 GB | Breast Trial | I-SPY 2 Shard 0 | DCE: `(3, 256, 256, 64)` | Longitudinal neoadjuvant chemotherapy trial scans (T0, T1, T2, T3). |
| **`breast_ispy2_s01.h5`** | 1.7 GB | Breast Trial | I-SPY 2 Shard 1 | DCE: `(3, 256, 256, 64)` | Multi-center ACRIN 6698 trial protocols and pathological complete response (pCR). |
| **`breast_ispy2_s02.h5`** | 1.9 GB | Breast Trial | I-SPY 2 Shard 2 | DCE: `(3, 256, 256, 64)` | Neoadjuvant kinetic washouts and functional tumor volumes (FTV). |
| **`breast_ispy2_s03.h5`** | 1.5 GB | Breast Trial | I-SPY 2 Shard 3 | DCE: `(3, 256, 256, 64)` | Multi-sequence dynamic contrast breast scans. |
| **`breast_ispy2_s04.h5`** | 1.4 GB | Breast Trial | I-SPY 2 Shard 4 | DCE: `(3, 256, 256, 64)` | Longitudinal treatment responses across targeted chemotherapy arms. |
| **`breast_ispy2_s05.h5`** | 1.6 GB | Breast Trial | I-SPY 2 Shard 5 | DCE: `(3, 256, 256, 64)` | Longitudinal visits with paired receptor subtype annotations (HR/HER2). |
| **`breast_ispy2_s06.h5`** | 1.8 GB | Breast Trial | I-SPY 2 Shard 6 | DCE: `(3, 256, 256, 64)` | Patient cohort with post-surgical residual cancer burden (RCB). |
| **`breast_ispy2_s07.h5`** | 2.4 GB | Breast Trial | I-SPY 2 Shard 7 | DCE: `(3, 256, 256, 64)` | Extended patient cohorts and multi-center validation scans. |
| **`gbm_tabular_cohorts.h5`**| 1.8 MB | Glioblastoma | All GBM Cohorts | Tabular: `(N, 256)` float32 | Age, KPS, MGMT methylation, IDH1 status, OS days, event indicators. |
| **`horizon_master.vds.h5`** | 387 KB | Multimodal | Unified Cohort | Virtual Reference Map | HDF5 Virtual Dataset (VDS) mapping all individual shards into single hierarchy. |
| **`dataset_manifest.json`** | ~4 KB | Metadata | All Files | JSON Catalog | Exact file sizes, SHA256 checksums, and remote resource URIs. |

---

## 2. Authentication & Access Credentials

Authentication uses pre-configured token credentials loaded from [scripts/xnat_credentials.txt](file:///DATA/horizon_disentangled/scripts/xnat_credentials.txt) or environment variables:

```ini
server = https://imaging-platform.diz-ag.med.ovgu.de
alias = ee21adf2-44a6-4a97-a53c-34c889273e90
secret = dEowoHHC1em1UwN70C4WG23I0mjn0Js1APpdPrCzbQJAN8p6P5ljrkuzik9k8AI0
project = temp
```

> [!NOTE]
> The credentials use XNAT Application Token authentication (`keycloak` backend). No cleartext passwords or interactive prompts are required during automated SLURM cluster execution.

---

## 3. How to Download & Stage the Dataset

### 3.1 One-Line CLI Download (Recommended for SLURM)

To download and sync the entire uncropped HDF5 dataset from XNAT into a target directory (such as local node NVMe storage `/tmp/horizon_hdf5` or shared scratch):

```bash
python scripts/xnatt.py --download-dataset \
    --project temp \
    --resource-label HORIZON_HDF5 \
    --output /tmp/horizon_hdf5
```

**Features of the Downloader**:
- **Automatic Resume & Verification**: Compares remote file sizes with local files; skips identical files and resumes interrupted transfers automatically.
- **Fast Transfer**: Achieves $> 150 \text{ MiB/s}$ throughput over HTTPS.
- **Overwrite Flag**: Pass `--overwrite` to force re-downloading corrupted files.

### 3.2 Programmatic Access in Python

You can also access or download the dataset directly in Python using `xnat-py`:

```python
import xnat
from pathlib import Path

server = "https://imaging-platform.diz-ag.med.ovgu.de"
alias = "ee21adf2-44a6-4a97-a53c-34c889273e90"
secret = "dEowoHHC1em1UwN70C4WG23I0mjn0Js1APpdPrCzbQJAN8p6P5ljrkuzik9k8AI0"

with xnat.connect(server=server, user=alias, password=secret) as session:
    project = session.projects["temp"]
    
    # Locate the HORIZON_HDF5 resource
    resource = None
    for r in project.resources.values():
        if r.label == "HORIZON_HDF5":
            resource = r
            break
            
    print(f"Connected! Resource contains {len(resource.files)} files.")
    
    # Download specific shard, e.g. gbm_lumiere.h5
    dest = Path("/tmp/horizon_hdf5/gbm_lumiere.h5")
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    if "gbm_lumiere.h5" in resource.files:
        print("Downloading gbm_lumiere.h5...")
        resource.files["gbm_lumiere.h5"].download(str(dest))
        print("Download complete!")
```

---

## 4. Reading the Data in PyTorch on SLURM Nodes

Once staged locally to `/tmp/horizon_hdf5`, load the volumes using `h5py` and `hdf5plugin` without slicing/cropping:

```python
import h5py
import hdf5plugin # Registers Bitshuffle, Blosc, and Zstd compression filters
import torch

# 1. Open Glioblastoma LUMIERE shard
with h5py.File("/tmp/horizon_hdf5/gbm_lumiere.h5", "r") as f:
    patient_group = f["patients/LUM_001/visits/v_0"]
    
    # Native uncropped 4-channel MRI: (4, 256, 256, 192) float32
    mri_np = patient_group["brain_mri"][:]
    # 2-channel tumor mask: (2, 256, 256, 192) uint8
    mask_np = patient_group["seg_tumor"][:]
    
    # Convert directly to PyTorch CUDA bfloat16 tensor
    mri_tensor = torch.from_numpy(mri_np).to(device="cuda", dtype=torch.bfloat16)
    mask_tensor = torch.from_numpy(mask_np).to(device="cuda", dtype=torch.uint8)

print(f"Loaded MRI tensor: {mri_tensor.shape}, {mri_tensor.dtype} on {mri_tensor.device}")
# Output: torch.Size([4, 256, 256, 192]), torch.bfloat16 on cuda:0
```

---

## 5. Multi-Node SLURM Job Integration

To prevent shared network filesystem (NFS/Lustre) thrashing and lock contention when scaling across multiple nodes and GPUs, stage the dataset from XNAT into the local high-speed NVMe burst buffer (`/tmp/horizon_hdf5`) on each node during job startup.

In your SLURM submission script (`scripts/slurm_pipeline_cluster.sbatch`):

```bash
#!/bin/bash
#SBATCH --job-name=horizon_cluster_train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --partition=h200

# Step 1: Stage dataset from XNAT onto local NVMe SSD of every compute node in parallel
echo "Staging uncropped HDF5 dataset from XNAT onto local NVMe (/tmp/horizon_hdf5)..."
srun --ntasks-per-node=1 python scripts/xnatt.py --download-dataset \
    --project temp \
    --resource-label HORIZON_HDF5 \
    --output /tmp/horizon_hdf5

# Step 2: Launch multi-GPU FSDP-2 training using local NVMe data
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=8 \
    scripts/train_full.py \
    --domain gbm \
    --data-dir /tmp/horizon_hdf5 \
    --batch-size 8 \
    --precision fp8
```

---

## 6. How to Upload Additional Shards to XNAT

If new patient shards or derived latent features are generated (e.g. Stage 2 compressed representations), upload them back to the `temp` project with a single command:

```bash
python scripts/xnatt.py --upload-dataset \
    --input /DATA/horizon_disentangled/hdf5 \
    --project temp \
    --resource-label HORIZON_HDF5
```

> [!WARNING]
> The XNAT server's nginx proxy enforces a **~4 GB per-file upload limit** (HTTP 413 error).  
> Files larger than ~3.5 GB must be split before uploading.  
> To split `breast_mincrop.h5` into 3 shards (`_s00`, `_s01`, `_s02`), run:
> ```bash
> python scripts/split_mincrop.py
> ```
> The script outputs `breast_mincrop_s00.h5` (2.1 GB), `_s01.h5` (2.2 GB), `_s02.h5` (2.8 GB) into `/DATA/horizon_disentangled/hdf5/`.
> These can be downloaded and merged back via HDF5 external links or by iterating patient groups across shards.

### Merging Shards Back Locally

After downloading all three `breast_mincrop_s0*.h5` shards, access all 1,081 patients transparently:

```python
import h5py

shard_files = [
    "/tmp/horizon_hdf5/breast_mincrop_s00.h5",   # DUKE_001–DUKE_362
    "/tmp/horizon_hdf5/breast_mincrop_s01.h5",   # DUKE_363–DUKE_722
    "/tmp/horizon_hdf5/breast_mincrop_s02.h5",   # DUKE_723–SPY1_1239
]

for shard_path in shard_files:
    with h5py.File(shard_path, "r") as f:
        patients = list(f["patients"].keys())
        print(f"{shard_path}: {len(patients)} patients")
        # e.g. access: f["patients/DUKE_001/visits/v_0/breast_mincrop"][:]
```
