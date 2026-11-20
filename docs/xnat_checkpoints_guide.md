# XNAT Model Checkpoints & Foundation Weights Access Guide

**Server**: `https://imaging-platform.diz-ag.med.ovgu.de`  
**Project**: `temp`  
**Resource Label**: `HORIZON_CHECKPOINTS` (ID: 40121)  
**Upload Tool**: [`scripts/upload_checkpoints.py`](../scripts/upload_checkpoints.py) + [`scripts/upload_new_models.py`](../scripts/upload_new_models.py)  
**Total Size**: **163.5 GB** across **136 XNAT files** ✅ verified 2026-11-20

> [!NOTE]
> Files larger than **3 GB** are split into binary `.part00`, `.part01`, … pieces due to the
> nginx upload size limit on the XNAT server. Reconstruct them locally with:
> ```bash
> cat filename.ext.part* > filename.ext
> ```

---

## 1. What's Stored

### 1.1 Training Checkpoints (`checkpoints/`)

Produced by running the full 5-stage pipeline on the local H100 GPU:

| Remote Name | Size | Description |
|:---|---:|:---|
| `checkpoints/stage1_breast_real.pt` | 968 MB | Stage 1 supervisor — Breast I-SPY2 (SwinUNETR + Merlin InfoNCE) |
| `checkpoints/stage1_gbm_real.pt` | 453 MB | Stage 1 supervisor — GBM LUMIERE (MedNeXt + Merlin) |
| `checkpoints/stage1_gbm_supervisor.pt` | 299 MB | Stage 1 GBM supervisor weights only |
| `checkpoints/stage2_causal_vae.pt` | 155 MB | Stage 2 Hierarchical Causal VAE (AutoencoderKL 40M) |
| `checkpoints/stage2_vae_real.pt` | 155 MB | Stage 2 VAE — real data fine-tuned |
| `checkpoints/stage3_ood_filter.pt` | 883 KB | Stage 3 OOD filter (Manifold VAE + k-NN state) |
| `checkpoints/stage4_breast_real.pt` | 15 MB | Stage 4 PDE-NJDE dynamics — Breast Fast-Slow Tofts |
| `checkpoints/stage4_gbm_dynamics.pt` | 15 MB | Stage 4 PDE-NJDE dynamics — GBM Fisher-KPP |
| `checkpoints/stage4_gbm_real.pt` | 15 MB | Stage 4 GBM dynamics — real data |

> [!NOTE]
> These checkpoints were trained on a **single H100 PCIe GPU** (small batch / local scale).
> For the full 4× H200 FSDP cluster run, these will be replaced by distributed checkpoints.
> See [Section 6](#6-h200-readiness-gaps) for details.

### 1.2 Foundation Model Weights (`models/`)

Pre-trained backbone weights used by the pipeline stages:

#### BrainSegFounder (~1.3 GB)
| Remote Name | Size | Description |
|:---|---:|:---|
| `BrainSegFounder/model_weights_ATLAS-finetune.pt` | 391 MB | SwinUNETR fine-tuned on ATLAS stroke dataset |
| `BrainSegFounder/model_weights_ATLAS-pretrain.pt` | 227 MB | SwinUNETR pre-trained on ATLAS |
| `BrainSegFounder/model_weights_BRATS-finetune.pt` | 245 MB | SwinUNETR fine-tuned on BraTS tumor |
| `BrainSegFounder/model_weights_BRATS-pretrain.pt` | 252 MB | SwinUNETR pre-trained on BraTS |
| `BrainSegFounder/model_weights_UKB-pretrain.pt` | 226 MB | SwinUNETR pre-trained on 82K UK Biobank |

#### Merlin (~1.5 GB)
| Remote Name | Size | Description |
|:---|---:|:---|
| `Merlin/merlin_clip_epoch99.pt` | 1.08 GB | Merlin CLIP encoder (I3-ResNet + LongFormer), epoch 99 |
| `Merlin/merlin_disease_prediction.pt` | 453 MB | Merlin 5-year disease prediction head |

#### VISTA3D-CTMR (~1.7 GB)
| Remote Name | Size | Description |
|:---|---:|:---|
| `VISTA3D/model.pt` | 870 MB | VISTA3D 218M param universal segmentation (PyTorch) |
| `VISTA3D/model.safetensors` | 870 MB | Same — SafeTensors format |

#### MedVAE (~2.8 GB)
| Remote Name | Size | Description |
|:---|---:|:---|
| `MedVAE/vae_4x_1c_2D.ckpt` | 220 MB | 4× compression, 1-channel 2D |
| `MedVAE/vae_4x_1c_3D.ckpt` | 640 MB | 4× compression, 1-channel **3D** (used in Stage 2) |
| `MedVAE/vae_4x_3c_2D.ckpt` | 220 MB | 4× compression, 3-channel 2D |
| `MedVAE/vae_4x_4c_2D.ckpt` | 220 MB | 4× compression, 4-channel 2D |
| `MedVAE/vae_8x_1c_2D.ckpt` | 330 MB | 8× compression, 1-channel 2D |
| `MedVAE/vae_8x_1c_3D.ckpt` | 980 MB | **8× compression, 1-channel 3D** ← primary for H200 |
| `MedVAE/vae_8x_4c_2D.ckpt` | 340 MB | 8× compression, 4-channel 2D |

#### VoCo / VoComni (~5.7 GB)
| Remote Name | Size | Parts | Description |
|:---|---:|:---:|:---|
| `VoCo/VoCo_B_SSL_head.pt` | 220 MB | — | VoCo-B SSL projection head |
| `VoCo/VoComni_H.pt` | 4.65 GB | **2** (`.part00`, `.part01`) | **VoComni-H 1.16B** ← primary backbone |
| `VoCo/VoComni_L.pt` | 1.17 GB | — | VoComni-L 290M |

#### SuPreM (~6.3 GB, 17 checkpoints)
| Remote Name | Size | Description |
|:---|---:|:---|
| `SuPreM/supervised_suprem_swinunetr_2100.pth` | 724 MB | SuPreM SwinUNETR supervised |
| `SuPreM/supervised_suprem_unet_2100.pth` | 223 MB | SuPreM UNet supervised |
| `SuPreM/supervised_clip_driven_universal_swin_unetr_2100.pth` | 724 MB | CLIP-driven universal SwinUNETR |
| `SuPreM/self_supervised_nv_swin_unetr_50000.pth` | 720 MB | NV SwinUNETR SSL 50K steps |
| `SuPreM/self_supervised_unimiss_nnunet_small_5022.pth` | 1.0 GB | UniMiSS nnUNet small |
| *(12 more variants — see `checkpoints_manifest.json`)* | | |

#### NV-Generate-MR-Brain (~4.1 GB)
| Remote Name | Size | Description |
|:---|---:|:---|
| `NV-Generate/diff_unet_3d_rflow-mr-brain_v0.pt` | 2.17 GB | NVIDIA rectified flow diffusion UNet v0 |
| `NV-Generate/diff_unet_3d_rflow-mr-brain_v1.pt` | 2.17 GB | NVIDIA rectified flow diffusion UNet v1 |

#### BrainGemma3D (~9.9 GB)
| Remote Name | Parts | Description |
|:---|:---:|:---|
| `BrainGemma3D/projector_vis_scale.pt` | — | Vision-language projector (19 MB) |
| `BrainGemma3D/vision_model.safetensors` | **2** | Vision encoder (3.5 GB) |
| `BrainGemma3D/language_model-00001-of-00002.safetensors` | **2** | LM shard 1/2 (4.96 GB) |
| `BrainGemma3D/language_model-00002-of-00002.safetensors` | **2** | LM shard 2/2 (3.64 GB) |

#### Med3DVLM-Qwen-2.5-7B (~28 GB)
| Remote Name | Parts | Description |
|:---|:---:|:---|
| `Med3DVLM/mm_projector.safetensors` | — | Multimodal projector (2.36 GB) |
| `Med3DVLM/vision_encoder.safetensors` | — | Vision encoder (35 MB) |
| `Med3DVLM/model-00001-of-00007.safetensors` | **2** | LLM shard 1 (4.97 GB) |
| `Med3DVLM/model-00002-of-00007.safetensors` | **2** | LLM shard 2 (4.78 GB) |
| `Med3DVLM/model-00003-of-00007.safetensors` | **2** | LLM shard 3 (4.93 GB) |
| `Med3DVLM/model-00004-of-00007.safetensors` | **2** | LLM shard 4 (4.93 GB) |
| `Med3DVLM/model-00005-of-00007.safetensors` | **2** | LLM shard 5 (5.00 GB) |
| `Med3DVLM/model-00006-of-00007.safetensors` | **2** | LLM shard 6 (3.92 GB) |
| `Med3DVLM/model-00007-of-00007.safetensors` | — | LLM shard 7 (2.17 GB) |

#### MAISI (~13 GB)
| Remote Name | Size | Description |
|:---|---:|:---|
| `MAISI/autoencoder.pt` | 80 MB | MAISI autoencoder |
| `MAISI/diffusion_unet.pt` | 2.17 GB | Diffusion UNet |
| `MAISI/mask_generation_diffusion_unet.pt` | 790 MB | Mask generation network |
| `MAISI/controlnet.pt` | 290 MB | ControlNet |
| `MAISI/controlnet_20datasets.pt` | 280 MB | ControlNet (20 datasets) |
| `MAISI/input_unet3d_v1.pt` | 690 MB | Input 3D UNet v1 |

### 1.3 Config / Tokenizer Files

> [!IMPORTANT]
> These small files are **required to instantiate** the VLMs and VAEs. Without them,
> `from_pretrained()` calls will fail even if all weights are present.

#### Packed Archive (easiest to download)
| Remote Name | Size | Description |
|:---|---:|:---|
| `model_configs.tar.gz` | 9.3 MB | **All** config/tokenizer/YAML/architecture files packed together ✅ uploaded |

Extract with:
```bash
mkdir -p /tmp/horizon_models
tar -xzf model_configs.tar.gz -C /tmp/horizon_models/
# Results in /tmp/horizon_models/Med3DVLM-Qwen-2.5-7B/, /tmp/horizon_models/BrainGemma3D/, etc.
```

#### Individual Config Files
Uploaded individually as `Med3DVLM/config.json`, `BrainGemma3D/tokenizer.json`, etc.:

| Remote Name | Description |
|:---|:---|
| `Med3DVLM/config.json` | Model architecture config |
| `Med3DVLM/tokenizer_config.json` | Qwen-2.5 tokenizer config |
| `Med3DVLM/vocab.json` + `merges.txt` | BPE vocabulary |
| `Med3DVLM/model.safetensors.index.json` | Shard index |
| `Med3DVLM/modeling.py` | Custom model class |
| `BrainGemma3D/model_config.json` | Full model config |
| `BrainGemma3D/braingemma3d_architecture.py` | Architecture definition |
| `BrainGemma3D/tokenizer.json` | Gemma tokenizer (32 MB) |
| `BrainGemma3D/tokenizer.model` | SentencePiece model |
| `BrainGemma3D/language_model_config.json` | LM head config |
| `MedVAE/config.json` + `medvae_*.yaml` | VAE arch configs (5 files) |
| `Merlin/config.json` | Merlin inference config |
| `VISTA3D/config.json` + `metadata.json` | VISTA3D configs |
| `NV-Generate/config.json` | NV-Generate config |
| `MAISI/configs/inference.json` | MAISI inference config |
| `MAISI/configs/label_dict.json` | Anatomy label map |

---

## 2. Authentication

Same credentials as the HDF5 dataset resource:

```ini
server = https://imaging-platform.diz-ag.med.ovgu.de
alias  = ee21adf2-44a6-4a97-a53c-34c889273e90
secret = dEowoHHC1em1UwN70C4WG23I0mjn0Js1APpdPrCzbQJAN8p6P5ljrkuzik9k8AI0
```

---

## 3. Downloading Checkpoints

### 3.1 Download All (Python)

```python
import xnat
from pathlib import Path

SERVER = "https://imaging-platform.diz-ag.med.ovgu.de"
ALIAS  = "ee21adf2-44a6-4a97-a53c-34c889273e90"
SECRET = "dEowoHHC1em1UwN70C4WG23I0mjn0Js1APpdPrCzbQJAN8p6P5ljrkuzik9k8AI0"
OUT    = Path("/tmp/horizon_checkpoints")
OUT.mkdir(parents=True, exist_ok=True)

with xnat.connect(server=SERVER, user=ALIAS, password=SECRET) as session:
    res = session.projects["temp"].resources["40121"]
    for f in res.files.values():
        dest = OUT / f.name
        if not dest.exists():
            print(f"Downloading {f.name}...")
            f.download(str(dest))

print("Done!")
```

### 3.2 Download Only Configs (fastest for quick start)

```python
import xnat
from pathlib import Path

SERVER = "https://imaging-platform.diz-ag.med.ovgu.de"
ALIAS  = "ee21adf2-44a6-4a97-a53c-34c889273e90"
SECRET = "dEowoHHC1em1UwN70C4WG23I0mjn0Js1APpdPrCzbQJAN8p6P5ljrkuzik9k8AI0"
OUT    = Path("/tmp/horizon_configs")
OUT.mkdir(parents=True, exist_ok=True)

with xnat.connect(server=SERVER, user=ALIAS, password=SECRET) as session:
    res = session.projects["temp"].resources["40121"]
    # Download the packed config archive — only 9.3 MB
    res.files["model_configs.tar.gz"].download(str(OUT / "model_configs.tar.gz"))

import subprocess
subprocess.run(["tar", "-xzf", str(OUT / "model_configs.tar.gz"), "-C", str(OUT)])
print(f"Configs extracted to {OUT}")
```

### 3.3 Reconstruct Split Files

After downloading, reassemble any `.partXX` files:

```bash
cd /tmp/horizon_checkpoints

# VoComni-H
cat VoComni_H.pt.part* > VoComni_H.pt

# BrainGemma3D language model
cat language_model-00001-of-00002.safetensors.part* > language_model-00001-of-00002.safetensors
cat language_model-00002-of-00002.safetensors.part* > language_model-00002-of-00002.safetensors
cat vision_model.safetensors.part* > vision_model.safetensors

# Med3DVLM shards (models 1–6 are split)
for i in 1 2 3 4 5 6; do
    cat model-0000${i}-of-00007.safetensors.part* > model-0000${i}-of-00007.safetensors
done

# Or reconstruct all split files at once
for base in $(ls *.part00 2>/dev/null | sed 's/\.part00$//'); do
    echo "Reconstructing $base ..."
    cat ${base}.part* > ${base} && rm ${base}.part*
done
```

### 3.4 SLURM Staging (Recommended for Cluster)

Stage checkpoints onto local NVMe at job start:

```bash
#!/bin/bash
#SBATCH --job-name=horizon_train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4            # 4× H200 per node

CKPT_DIR=/tmp/horizon_checkpoints
HDF5_DIR=/tmp/horizon_hdf5
CFG_DIR=/tmp/horizon_configs

SERVER="https://imaging-platform.diz-ag.med.ovgu.de"
ALIAS="ee21adf2-44a6-4a97-a53c-34c889273e90"
SECRET="dEowoHHC1em1UwN70C4WG23I0mjn0Js1APpdPrCzbQJAN8p6P5ljrkuzik9k8AI0"

# ── Stage HDF5 datasets ───────────────────────────────────────────────────────
srun python scripts/xnatt.py --download-dataset \
    --project temp --resource-label HORIZON_HDF5 \
    --output $HDF5_DIR &

# ── Stage checkpoints + configs ───────────────────────────────────────────────
srun python - << 'PYEOF'
import xnat, subprocess
from pathlib import Path

SERVER = "https://imaging-platform.diz-ag.med.ovgu.de"
ALIAS  = "ee21adf2-44a6-4a97-a53c-34c889273e90"
SECRET = "dEowoHHC1em1UwN70C4WG23I0mjn0Js1APpdPrCzbQJAN8p6P5ljrkuzik9k8AI0"

CKPT_DIR = Path("/tmp/horizon_checkpoints")
CFG_DIR  = Path("/tmp/horizon_configs")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
CFG_DIR.mkdir(parents=True, exist_ok=True)

with xnat.connect(server=SERVER, user=ALIAS, password=SECRET) as session:
    res = session.projects["temp"].resources["40121"]
    for f in res.files.values():
        name = f.name
        dest = CFG_DIR / name if name == "model_configs.tar.gz" else CKPT_DIR / name
        if not dest.exists():
            print(f"  ↓ {name}")
            f.download(str(dest))

# Extract configs
subprocess.run(["tar", "-xzf", str(CFG_DIR / "model_configs.tar.gz"), "-C", str(CFG_DIR)])

# Reconstruct split files
for part in sorted(CKPT_DIR.glob("*.part00")):
    base = str(part).replace(".part00", "")
    print(f"  ⚙ Reconstructing {part.name.replace('.part00', '')}...")
    subprocess.run(f"cat {base}.part* > {base}", shell=True)
    for p in CKPT_DIR.glob(f"{part.stem[:-2]}*.part*"):
        p.unlink()
print("Staging complete.")
PYEOF

wait
echo "All data staged."

# ── Launch 4× H200 distributed training ──────────────────────────────────────
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$SLURM_NODELIST:29500 \
    scripts/train_full.py \
        --data-dir  $HDF5_DIR \
        --ckpt-dir  $CKPT_DIR \
        --cfg-dir   $CFG_DIR \
        --strategy  fsdp \
        --precision bf16
```

---

## 4. Re-uploading / Updating Checkpoints

```bash
# Full env setup (required on this machine)
export LD_LIBRARY_PATH=/DATA/conda/envs/nukl-to-ct/lib:$LD_LIBRARY_PATH
PYTHON=/DATA/conda/envs/nukl-to-ct/bin/python
cd /DATA/horizon_disentangled

# Upload all (resumes automatically — skips already-uploaded files by size)
$PYTHON scripts/upload_checkpoints.py

# Upload only training checkpoints (skip 74 GB foundation models)
$PYTHON scripts/upload_checkpoints.py --only-checkpoints

# Upload only config/tokenizer files
$PYTHON scripts/upload_checkpoints.py --only-configs

# Upload only model weight files
$PYTHON scripts/upload_checkpoints.py --only-models

# Dry run (see what would be uploaded)
$PYTHON scripts/upload_checkpoints.py --dry-run
```

> [!WARNING]
> The nginx proxy enforces a **~3.5 GB per-file upload limit** (HTTP 413).
> `upload_checkpoints.py` automatically splits files exceeding this via `split -b 3072m`.
> Temporary split parts are stored in `/DATA/horizon_disentangled/.xnat_splits/` and can be
> deleted after a successful upload.

---

## 5. Manifest

A machine-readable inventory is stored as `checkpoints_manifest.json` in the XNAT resource.
Download it to inspect what's available:

```python
import xnat, json

SERVER = "https://imaging-platform.diz-ag.med.ovgu.de"
ALIAS  = "ee21adf2-44a6-4a97-a53c-34c889273e90"
SECRET = "dEowoHHC1em1UwN70C4WG23I0mjn0Js1APpdPrCzbQJAN8p6P5ljrkuzik9k8AI0"

with xnat.connect(server=SERVER, user=ALIAS, password=SECRET) as session:
    res = session.projects["temp"].resources["40121"]
    res.files["checkpoints_manifest.json"].download("/tmp/checkpoints_manifest.json")

with open("/tmp/checkpoints_manifest.json") as f:
    manifest = json.load(f)
    for entry in manifest["files"]:
        print(entry["remote_name"], "→", entry["size_bytes"]//1e6, "MB",
              f"[{entry['parts']} parts]" if entry.get("split") else "")
```

---

## 6. Additional Models ✅ Complete

All three additional model groups have been fully downloaded from HuggingFace and
uploaded to HORIZON_CHECKPOINTS.

### Hulu-Med-32B (66.4 GB) ✅

| Remote Name | Parts | Description |
|:---|:---:|:---|
| `Hulu-Med-32B/model-00001-of-00013.safetensors` | **2** | Shard 1/13 (~5.2 GB) |
| `Hulu-Med-32B/model-00002-of-00013.safetensors` | **2** | Shard 2/13 |
| `Hulu-Med-32B/model-00003-of-00013.safetensors` | **2** | Shard 3/13 |
| `Hulu-Med-32B/model-00004-of-00013.safetensors` | **2** | Shard 4/13 |
| `Hulu-Med-32B/model-00005-of-00013.safetensors` | **2** | Shard 5/13 |
| `Hulu-Med-32B/model-00006-of-00013.safetensors` | **2** | Shard 6/13 |
| `Hulu-Med-32B/model-00007-of-00013.safetensors` | **2** | Shard 7/13 |
| `Hulu-Med-32B/model-00008-of-00013.safetensors` | **2** | Shard 8/13 |
| `Hulu-Med-32B/model-00009-of-00013.safetensors` | **2** | Shard 9/13 |
| `Hulu-Med-32B/model-00010-of-00013.safetensors` | **2** | Shard 10/13 |
| `Hulu-Med-32B/model-00011-of-00013.safetensors` | **2** | Shard 11/13 |
| `Hulu-Med-32B/model-00012-of-00013.safetensors` | **2** | Shard 12/13 |
| `Hulu-Med-32B/model-00013-of-00013.safetensors` | **2** | Shard 13/13 (~4.0 GB) |
| `Hulu-Med-32B/config.json` + 17 config files | — | All tokenizer/arch/processing configs |

**HuggingFace repo**: `ZJU-AI4H/Hulu-Med-32B` (Apache 2.0)

Load with:
```python
from transformers import AutoModel, AutoProcessor
import torch

model = AutoModel.from_pretrained(
    "/path/to/Hulu-Med-32B",
    trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
)
processor = AutoProcessor.from_pretrained("/path/to/Hulu-Med-32B", trust_remote_code=True)
```

### STU-Net (15.8 GB) ✅

| Remote Name | Size | Description |
|:---|---:|:---|
| `STU-Net/small_ep4k.model` | 113 MB | STU-Net Small — 14M params |
| `STU-Net/base_ep4k.model` | 446 MB | STU-Net Base — 58M params |
| `STU-Net/large_ep4k.model` | 3.3 GB | STU-Net Large — 440M params |
| `STU-Net/huge_ep4k.model` | **11 GB** | **STU-Net Huge — 1.4B params** ← H200 primary |

**HuggingFace repo**: `ziyanhuang/STU-Net` (Apache 2.0)

> [!NOTE]
> STU-Net uses NNUNet's `.model` checkpoint format. Load with:
> ```python
> from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
> predictor = nnUNetPredictor()
> predictor.initialize_from_trained_model_folder(
>     "/path/to/STU-Net/huge_ep4k.model",
>     use_folds=(0,), checkpoint_name="checkpoint_final.pth"
> )
> ```

### VoCo / VoComni — Complete Set (10.7 GB) ✅

All VoCo variants now available:

| Remote Name | Size | Description |
|:---|---:|:---|
| `VoCo/VoCo_B_SSL_head.pt` | 211 MB | VoCo-B SSL projection head |
| `VoCo/VoCo_H_SSL_head.pt` | 3.2 GB | **VoCo-H SSL projection head** ← H200 primary |
| `VoCo/VoCo_L_SSL_head.pt` | 816 MB | VoCo-L SSL projection head |
| `VoCo/VoComni_B.pt` | 285 MB | VoComni-B backbone |
| `VoCo/VoComni_H.pt` | 4.4 GB | VoComni-H backbone (1.16B) |
| `VoCo/VoComni_L.pt` | 1.1 GB | VoComni-L backbone |
| `VoCo/VoComni_nnunet.pt` | 120 MB | VoComni NNUNet variant |

**HuggingFace repo**: `Luffy503/VoCo` (Apache 2.0)

### 🟡 Note — Training checkpoints are single-GPU H100 scale

Stages 1–4 checkpoints in `checkpoints/` were trained on a single H100 PCIe with small batch
sizes. After the first 4× H200 FSDP run, these will be superseded by distributed checkpoints.
The current checkpoints serve as **warm-start initializations** for the cluster run.

---

## 7. Re-uploading / Refreshing Models

```bash
export LD_LIBRARY_PATH=/DATA/conda/envs/nukl-to-ct/lib:$LD_LIBRARY_PATH
PYTHON=/DATA/conda/envs/nukl-to-ct/bin/python
cd /DATA/horizon_disentangled

# Original 66-file checkpoint upload (auto-resumes)
$PYTHON scripts/upload_checkpoints.py

# New models upload — Hulu-Med-32B, STU-Net, VoCo (auto-resumes)
$PYTHON scripts/upload_new_models.py --no-wait

# Dry run to see what would be uploaded
$PYTHON scripts/upload_new_models.py --dry-run
```

---

## 8. HDF5 Dataset (HORIZON_HDF5)

For reference, the companion HDF5 dataset is at:

| Resource | ID | Size | Files |
|:---|---:|---:|---:|
| `HORIZON_HDF5` | 40120 | ~23 GB | 16 |

See [`docs/xnat_data_guide.md`](xnat_data_guide.md) for HDF5 access instructions.
