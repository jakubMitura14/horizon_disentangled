# XNAT Model Checkpoints & Foundation Weights Access Guide

**Server**: `https://imaging-platform.diz-ag.med.ovgu.de`  
**Project**: `temp`  
**Resource Label**: `HORIZON_CHECKPOINTS` (ID: 40121)  
**Upload Tool**: [`scripts/upload_checkpoints.py`](../scripts/upload_checkpoints.py)  
**Total Size**: ~76.6 GB across 66 source files (split parts may be larger in count)

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

### 3.2 Reconstruct Split Files

After downloading, reassemble any `.partXX` files:

```bash
cd /tmp/horizon_checkpoints

# VoComni-H
cat VoComni_H.pt.part* > VoComni_H.pt

# BrainGemma3D language model
cat language_model-00001-of-00002.safetensors.part* > language_model-00001-of-00002.safetensors
cat language_model-00002-of-00002.safetensors.part* > language_model-00002-of-00002.safetensors

# Med3DVLM shards (models 1–6 are split)
for i in 1 2 3 4 5 6; do
    cat model-0000${i}-of-00007.safetensors.part* > model-0000${i}-of-00007.safetensors
done
```

Or use the helper script:

```bash
# Reconstruct all .partXX files in a directory
for base in $(ls *.part00 2>/dev/null | sed 's/\.part00$//'); do
    echo "Reconstructing $base ..."
    cat ${base}.part* > ${base} && rm ${base}.part*
done
```

### 3.3 SLURM Staging (Recommended for Cluster)

Stage checkpoints onto local NVMe at job start:

```bash
#!/bin/bash
#SBATCH --job-name=horizon_train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8

CKPT_DIR=/tmp/horizon_checkpoints
HDF5_DIR=/tmp/horizon_hdf5

# Stage datasets (one per node, in parallel)
srun python scripts/xnatt.py --download-dataset \
    --project temp --resource-label HORIZON_HDF5 \
    --output $HDF5_DIR &

# Stage checkpoints
srun python - << 'PYEOF'
import xnat
from pathlib import Path
OUT = Path("/tmp/horizon_checkpoints")
OUT.mkdir(parents=True, exist_ok=True)
with xnat.connect(
    server="https://imaging-platform.diz-ag.med.ovgu.de",
    user="ee21adf2-44a6-4a97-a53c-34c889273e90",
    password="dEowoHHC1em1UwN70C4WG23I0mjn0Js1APpdPrCzbQJAN8p6P5ljrkuzik9k8AI0"
) as session:
    res = session.projects["temp"].resources["40121"]
    for f in res.files.values():
        dest = OUT / f.name
        if not dest.exists():
            f.download(str(dest))
PYEOF

wait
echo "Staging complete."

# Launch training
srun torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=8 \
    scripts/train_full.py --data-dir $HDF5_DIR --ckpt-dir $CKPT_DIR
```

---

## 4. Re-uploading / Updating Checkpoints

```bash
# Upload all (resumes automatically — skips already-uploaded files)
python scripts/upload_checkpoints.py

# Upload only training checkpoints (skip 74 GB foundation models)
python scripts/upload_checkpoints.py --only-checkpoints

# Dry run (see what would be uploaded)
python scripts/upload_checkpoints.py --dry-run
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
with xnat.connect(server=SERVER, user=ALIAS, password=SECRET) as session:
    res = session.projects["temp"].resources["40121"]
    res.files["checkpoints_manifest.json"].download("/tmp/checkpoints_manifest.json")

with open("/tmp/checkpoints_manifest.json") as f:
    manifest = json.load(f)
    for entry in manifest["files"]:
        print(entry["remote_name"], "→", entry["size_bytes"]//1e6, "MB",
              f"[{entry['parts']} parts]" if entry.get("split") else "")
```
