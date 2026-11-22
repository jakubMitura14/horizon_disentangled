#!/usr/bin/env python3
"""
Upload all Horizon Disentangled model checkpoints + foundation model weights to XNAT.

Resource: temp/HORIZON_CHECKPOINTS (ID: 40121)
Server  : https://imaging-platform.diz-ag.med.ovgu.de

For files >3.5 GB (nginx 413 limit) the file is split into binary parts using
  split -b 3G file.ext → file.ext.part00, file.ext.part01, ...
and reconstructed on download with:
  cat file.ext.part* > file.ext

Usage:
  python scripts/upload_checkpoints.py [--dry-run] [--only-checkpoints]
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import xnat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
ALIAS    = "ee21adf2-44a6-4a97-a53c-34c889273e90"
SECRET   = "dEowoHHC1em1UwN70C4WG23I0mjn0Js1APpdPrCzbQJAN8p6P5ljrkuzik9k8AI0"
SERVER   = "https://imaging-platform.diz-ag.med.ovgu.de"
PROJECT  = "temp"
RESOURCE = "HORIZON_CHECKPOINTS"
RES_ID   = "40121"

BASE     = Path("/DATA/horizon_disentangled")
SPLIT_MB = 3072   # 3 GB split threshold in MiB
SPLIT_DIR = BASE / ".xnat_splits"   # temp dir for split parts

# ── File groups to upload ─────────────────────────────────────────────────────
CHECKPOINT_FILES = [
    # (local_path, remote_name)
    (BASE / "experiments/checkpoints/stage1_breast_real.pt",   "checkpoints/stage1_breast_real.pt"),
    (BASE / "experiments/checkpoints/stage1_gbm_real.pt",      "checkpoints/stage1_gbm_real.pt"),
    (BASE / "experiments/checkpoints/stage1_gbm_supervisor.pt","checkpoints/stage1_gbm_supervisor.pt"),
    (BASE / "experiments/checkpoints/stage2_causal_vae.pt",    "checkpoints/stage2_causal_vae.pt"),
    (BASE / "experiments/checkpoints/stage2_vae_real.pt",      "checkpoints/stage2_vae_real.pt"),
    (BASE / "experiments/checkpoints/stage3_ood_filter.pt",    "checkpoints/stage3_ood_filter.pt"),
    (BASE / "experiments/checkpoints/stage4_breast_real.pt",   "checkpoints/stage4_breast_real.pt"),
    (BASE / "experiments/checkpoints/stage4_gbm_dynamics.pt",  "checkpoints/stage4_gbm_dynamics.pt"),
    (BASE / "experiments/checkpoints/stage4_gbm_real.pt",      "checkpoints/stage4_gbm_real.pt"),
]

MODEL_FILES = [
    # BrainSegFounder (5 checkpoints, ~1.3 GB total)
    (BASE / "models/BrainSegFounder/model_weights_ATLAS-finetune.pt",  "BrainSegFounder/model_weights_ATLAS-finetune.pt"),
    (BASE / "models/BrainSegFounder/model_weights_ATLAS-pretrain.pt",  "BrainSegFounder/model_weights_ATLAS-pretrain.pt"),
    (BASE / "models/BrainSegFounder/model_weights_BRATS-finetune.pt",  "BrainSegFounder/model_weights_BRATS-finetune.pt"),
    (BASE / "models/BrainSegFounder/model_weights_BRATS-pretrain.pt",  "BrainSegFounder/model_weights_BRATS-pretrain.pt"),
    (BASE / "models/BrainSegFounder/model_weights_UKB-pretrain.pt",    "BrainSegFounder/model_weights_UKB-pretrain.pt"),
    # Merlin (~1.5 GB total)
    (BASE / "models/Merlin/i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt",
                                                                        "Merlin/merlin_clip_epoch99.pt"),
    (BASE / "models/Merlin/resnet_clinical_longformer_five_year_disease_prediction.pt",
                                                                        "Merlin/merlin_disease_prediction.pt"),
    # VISTA3D-CTMR (~1.7 GB)
    (BASE / "models/VISTA3D-CTMR/vista3d_pretrained_model/model.pt",           "VISTA3D/model.pt"),
    (BASE / "models/VISTA3D-CTMR/vista3d_pretrained_model/model.safetensors",  "VISTA3D/model.safetensors"),
    # MedVAE (7 checkpoints, ~2.8 GB)
    (BASE / "models/MedVAE/model_weights/vae_4x_1c_2D.ckpt", "MedVAE/vae_4x_1c_2D.ckpt"),
    (BASE / "models/MedVAE/model_weights/vae_4x_1c_3D.ckpt", "MedVAE/vae_4x_1c_3D.ckpt"),
    (BASE / "models/MedVAE/model_weights/vae_4x_3c_2D.ckpt", "MedVAE/vae_4x_3c_2D.ckpt"),
    (BASE / "models/MedVAE/model_weights/vae_4x_4c_2D.ckpt", "MedVAE/vae_4x_4c_2D.ckpt"),
    (BASE / "models/MedVAE/model_weights/vae_8x_1c_2D.ckpt", "MedVAE/vae_8x_1c_2D.ckpt"),
    (BASE / "models/MedVAE/model_weights/vae_8x_1c_3D.ckpt", "MedVAE/vae_8x_1c_3D.ckpt"),
    (BASE / "models/MedVAE/model_weights/vae_8x_4c_2D.ckpt", "MedVAE/vae_8x_4c_2D.ckpt"),
    # VoCo (needs split for VoComni_H 4.4 GB)
    (BASE / "models/VoCo/VoCo_B_SSL_head.pt", "VoCo/VoCo_B_SSL_head.pt"),
    (BASE / "models/VoCo/VoComni_H.pt",        "VoCo/VoComni_H.pt"),        # 4.4 GB → split
    (BASE / "models/VoCo/VoComni_L.pt",        "VoCo/VoComni_L.pt"),
    # SuPreM (14 checkpoints, ~6.3 GB)
    (BASE / "models/SuPreM/noisy_supervised_suprem_swinunetr_2100.pth",          "SuPreM/noisy_supervised_suprem_swinunetr_2100.pth"),
    (BASE / "models/SuPreM/repro_self_supervised_nv_swin_unetr_1000.pth",        "SuPreM/repro_self_supervised_nv_swin_unetr_1000.pth"),
    (BASE / "models/SuPreM/repro_self_supervised_nv_swin_unetr_3000.pth",        "SuPreM/repro_self_supervised_nv_swin_unetr_3000.pth"),
    (BASE / "models/SuPreM/repro_self_supervised_nv_swin_unetr_5050.pth",        "SuPreM/repro_self_supervised_nv_swin_unetr_5050.pth"),
    (BASE / "models/SuPreM/repro_self_supervised_nv_swin_unetr_7000.pth",        "SuPreM/repro_self_supervised_nv_swin_unetr_7000.pth"),
    (BASE / "models/SuPreM/self_supervised_models_genesis_unet_620.pt",          "SuPreM/self_supervised_models_genesis_unet_620.pt"),
    (BASE / "models/SuPreM/self_supervised_nv_swin_unetr_50000.pth",             "SuPreM/self_supervised_nv_swin_unetr_50000.pth"),
    (BASE / "models/SuPreM/self_supervised_nv_swin_unetr_5050.pt",               "SuPreM/self_supervised_nv_swin_unetr_5050.pt"),
    (BASE / "models/SuPreM/self_supervised_unimiss_nnunet_small_5022.pth",       "SuPreM/self_supervised_unimiss_nnunet_small_5022.pth"),
    (BASE / "models/SuPreM/self_supervised_unimiss_nnunet_tiny_5022.pth",        "SuPreM/self_supervised_unimiss_nnunet_tiny_5022.pth"),
    (BASE / "models/SuPreM/supervised_clip_driven_universal_swin_unetr_2100.pth","SuPreM/supervised_clip_driven_universal_swin_unetr_2100.pth"),
    (BASE / "models/SuPreM/supervised_clip_driven_universal_unet_2100.pth",      "SuPreM/supervised_clip_driven_universal_unet_2100.pth"),
    (BASE / "models/SuPreM/supervised_dodnet_unet_920.pth",                      "SuPreM/supervised_dodnet_unet_920.pth"),
    (BASE / "models/SuPreM/supervised_med3D_residual_unet_1623.pth",             "SuPreM/supervised_med3D_residual_unet_1623.pth"),
    (BASE / "models/SuPreM/supervised_suprem_segresnet_2100.pth",                "SuPreM/supervised_suprem_segresnet_2100.pth"),
    (BASE / "models/SuPreM/supervised_suprem_swinunetr_2100.pth",                "SuPreM/supervised_suprem_swinunetr_2100.pth"),
    (BASE / "models/SuPreM/supervised_suprem_unet_2100.pth",                     "SuPreM/supervised_suprem_unet_2100.pth"),
    # NV-Generate-MR-Brain (~4.1 GB)
    (BASE / "models/NV-Generate-MR-Brain/models/diff_unet_3d_rflow-mr-brain_v0.pt", "NV-Generate/diff_unet_3d_rflow-mr-brain_v0.pt"),
    (BASE / "models/NV-Generate-MR-Brain/models/diff_unet_3d_rflow-mr-brain_v1.pt", "NV-Generate/diff_unet_3d_rflow-mr-brain_v1.pt"),
    # BrainGemma3D (~9.9 GB — vision + projector safe, language model shards need split)
    (BASE / "models/BrainGemma3D/projector_vis_scale.pt",                                "BrainGemma3D/projector_vis_scale.pt"),
    (BASE / "models/BrainGemma3D/vision_model/model.safetensors",                        "BrainGemma3D/vision_model.safetensors"),
    (BASE / "models/BrainGemma3D/language_model/model-00001-of-00002.safetensors",       "BrainGemma3D/language_model-00001-of-00002.safetensors"),  # 4.7 GB → split
    (BASE / "models/BrainGemma3D/language_model/model-00002-of-00002.safetensors",       "BrainGemma3D/language_model-00002-of-00002.safetensors"),
    # Med3DVLM-Qwen-2.5-7B (~28 GB — all shards need split check)
    (BASE / "models/Med3DVLM-Qwen-2.5-7B/mm_projector.safetensors",         "Med3DVLM/mm_projector.safetensors"),
    (BASE / "models/Med3DVLM-Qwen-2.5-7B/vision_encoder.safetensors",       "Med3DVLM/vision_encoder.safetensors"),
    (BASE / "models/Med3DVLM-Qwen-2.5-7B/model-00001-of-00007.safetensors", "Med3DVLM/model-00001-of-00007.safetensors"),  # 4.7 GB → split
    (BASE / "models/Med3DVLM-Qwen-2.5-7B/model-00002-of-00007.safetensors", "Med3DVLM/model-00002-of-00007.safetensors"),  # 4.5 GB → split
    (BASE / "models/Med3DVLM-Qwen-2.5-7B/model-00003-of-00007.safetensors", "Med3DVLM/model-00003-of-00007.safetensors"),  # 4.6 GB → split
    (BASE / "models/Med3DVLM-Qwen-2.5-7B/model-00004-of-00007.safetensors", "Med3DVLM/model-00004-of-00007.safetensors"),  # 4.6 GB → split
    (BASE / "models/Med3DVLM-Qwen-2.5-7B/model-00005-of-00007.safetensors", "Med3DVLM/model-00005-of-00007.safetensors"),  # 4.7 GB → split
    (BASE / "models/Med3DVLM-Qwen-2.5-7B/model-00006-of-00007.safetensors", "Med3DVLM/model-00006-of-00007.safetensors"),  # 3.7 GB → split
    (BASE / "models/Med3DVLM-Qwen-2.5-7B/model-00007-of-00007.safetensors", "Med3DVLM/model-00007-of-00007.safetensors"),  # 2.1 GB — ok
    # MAISI (~13 GB)
    (BASE / "models/MAISI/maisi_ct_generative/models/autoencoder.pt",                   "MAISI/autoencoder.pt"),
    (BASE / "models/MAISI/maisi_ct_generative/models/diffusion_unet.pt",                "MAISI/diffusion_unet.pt"),
    (BASE / "models/MAISI/maisi_ct_generative/models/mask_generation_diffusion_unet.pt","MAISI/mask_generation_diffusion_unet.pt"),
    (BASE / "models/MAISI/maisi_ct_generative/models/controlnet.pt",                    "MAISI/controlnet.pt"),
    (BASE / "models/MAISI/maisi_ct_generative/models/controlnet-20datasets-e20wl100fold0bc_noi_dia_fsize_current.pt",
                                                                                          "MAISI/controlnet_20datasets.pt"),
    (BASE / "models/MAISI/maisi_ct_generative/models/input_unet3d_data-all_steps1000size512ddpm_random_current_inputx_v1.pt",
                                                                                          "MAISI/input_unet3d_v1.pt"),
]

# ── Config / tokenizer / architecture files ───────────────────────────────────
# Small files (<50 MB each) — no splitting needed.
# These are REQUIRED to instantiate the VLMs and VAEs at inference / training time.
CONFIG_FILES = [
    # Med3DVLM-Qwen-2.5-7B — model architecture + tokenizer
    (BASE / "models/Med3DVLM-Qwen-2.5-7B/config.json",                "Med3DVLM/config.json"),
    (BASE / "models/Med3DVLM-Qwen-2.5-7B/generation_config.json",     "Med3DVLM/generation_config.json"),
    (BASE / "models/Med3DVLM-Qwen-2.5-7B/tokenizer_config.json",      "Med3DVLM/tokenizer_config.json"),
    (BASE / "models/Med3DVLM-Qwen-2.5-7B/vocab.json",                 "Med3DVLM/vocab.json"),
    (BASE / "models/Med3DVLM-Qwen-2.5-7B/merges.txt",                 "Med3DVLM/merges.txt"),
    (BASE / "models/Med3DVLM-Qwen-2.5-7B/special_tokens_map.json",    "Med3DVLM/special_tokens_map.json"),
    (BASE / "models/Med3DVLM-Qwen-2.5-7B/added_tokens.json",          "Med3DVLM/added_tokens.json"),
    (BASE / "models/Med3DVLM-Qwen-2.5-7B/model.safetensors.index.json","Med3DVLM/model.safetensors.index.json"),
    (BASE / "models/Med3DVLM-Qwen-2.5-7B/modeling.py",                "Med3DVLM/modeling.py"),
    # BrainGemma3D — model architecture + language model tokenizer
    (BASE / "models/BrainGemma3D/model_config.json",                               "BrainGemma3D/model_config.json"),
    (BASE / "models/BrainGemma3D/braingemma3d_architecture.py",                    "BrainGemma3D/braingemma3d_architecture.py"),
    (BASE / "models/BrainGemma3D/braingemma3d_interpretability.py",                "BrainGemma3D/braingemma3d_interpretability.py"),
    (BASE / "models/BrainGemma3D/vision_model/config.json",                        "BrainGemma3D/vision_model_config.json"),
    (BASE / "models/BrainGemma3D/language_model/config.json",                      "BrainGemma3D/language_model_config.json"),
    (BASE / "models/BrainGemma3D/language_model/generation_config.json",           "BrainGemma3D/language_model_generation_config.json"),
    (BASE / "models/BrainGemma3D/language_model/model.safetensors.index.json",     "BrainGemma3D/language_model_index.json"),
    (BASE / "models/BrainGemma3D/language_model/tokenizer_config.json",            "BrainGemma3D/tokenizer_config.json"),
    (BASE / "models/BrainGemma3D/language_model/special_tokens_map.json",          "BrainGemma3D/special_tokens_map.json"),
    (BASE / "models/BrainGemma3D/language_model/added_tokens.json",                "BrainGemma3D/added_tokens.json"),
    (BASE / "models/BrainGemma3D/language_model/tokenizer.json",                   "BrainGemma3D/tokenizer.json"),    # 32 MB — no split needed
    (BASE / "models/BrainGemma3D/language_model/tokenizer.model",                  "BrainGemma3D/tokenizer.model"),
    # MedVAE — config + architecture yamls
    (BASE / "models/MedVAE/config.json",                               "MedVAE/config.json"),
    (BASE / "models/MedVAE/model_weights/medvae_4x1.yaml",             "MedVAE/medvae_4x1.yaml"),
    (BASE / "models/MedVAE/model_weights/medvae_4x3.yaml",             "MedVAE/medvae_4x3.yaml"),
    (BASE / "models/MedVAE/model_weights/medvae_4x4.yaml",             "MedVAE/medvae_4x4.yaml"),
    (BASE / "models/MedVAE/model_weights/medvae_8x1.yaml",             "MedVAE/medvae_8x1.yaml"),
    (BASE / "models/MedVAE/model_weights/medvae_8x4.yaml",             "MedVAE/medvae_8x4.yaml"),
    # Merlin — inference config
    (BASE / "models/Merlin/config.json",                                "Merlin/config.json"),
    # VISTA3D — model config + inference scripts
    (BASE / "models/VISTA3D-CTMR/config.json",                         "VISTA3D/config.json"),
    (BASE / "models/VISTA3D-CTMR/metadata.json",                       "VISTA3D/metadata.json"),
    # NV-Generate — config
    (BASE / "models/NV-Generate-MR-Brain/config.json",                 "NV-Generate/config.json"),
    # MAISI — inference configs (needed to run generation)
    (BASE / "models/MAISI/maisi_ct_generative/configs/inference.json",    "MAISI/configs/inference.json"),
    (BASE / "models/MAISI/maisi_ct_generative/configs/label_dict.json",   "MAISI/configs/label_dict.json"),
    (BASE / "models/MAISI/maisi_ct_generative/configs/metadata.json",     "MAISI/configs/metadata.json"),
    # Packed archive with ALL configs (convenient single-file download)
    # NOTE: Regenerate with: tar -czf /tmp/model_configs.tar.gz -C /DATA/horizon_disentangled/models .
    # Then upload manually or add path here if the archive is committed to the repo.
]


def split_file(src: Path, remote_name: str) -> list[tuple[Path, str]]:
    """Split a file into 3 GB binary parts. Returns list of (part_path, remote_name)."""
    stem = src.name
    out_dir = SPLIT_DIR / remote_name.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / (stem + ".part")
    result = subprocess.run(
        ["split", "-b", f"{SPLIT_MB}m", "--numeric-suffixes=0", "--suffix-length=2",
         str(src), str(prefix)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"split failed: {result.stderr}")
    parts = sorted(out_dir.glob(stem + ".part*"))
    log.info(f"  Split into {len(parts)} parts: {[p.name for p in parts]}")
    return [(p, remote_name + f".part{i:02d}") for i, p in enumerate(parts)]


def upload_file(res, local_path: Path, remote_name: str, existing: dict, dry_run: bool) -> dict:
    """Upload a single file, splitting if needed. Returns manifest entry."""
    sz = local_path.stat().st_size
    sz_mb = sz / 1024**2

    # Decide if split needed
    needs_split = sz_mb > SPLIT_MB

    if needs_split:
        parts = split_file(local_path, remote_name)
        entries = []
        for part_path, part_remote in parts:
            part_sz = part_path.stat().st_size
            part_name = Path(part_remote).name
            if part_name in existing and existing[part_name] == part_sz:
                log.info(f"    ↷ Skip {part_name} (already on XNAT)")
            else:
                if not dry_run:
                    log.info(f"    ↑ Uploading part {part_name} ({part_sz/1e6:.0f} MB)...")
                    t0 = time.time()
                    res.upload(str(part_path), remotepath=part_name, overwrite=True)
                    dt = time.time() - t0
                    log.info(f"      ✓ {part_name} in {dt:.1f}s ({part_sz/1e6/max(dt,0.001):.0f} MB/s)")
                else:
                    log.info(f"    [DRY] Would upload part {part_name} ({part_sz/1e6:.0f} MB)")
            entries.append({"filename": part_name, "size_bytes": part_sz, "part_of": remote_name})
        return {
            "remote_name": remote_name, "local_path": str(local_path),
            "size_bytes": sz, "split": True, "parts": len(parts),
            "reconstruct_cmd": f"cat {Path(remote_name).name}.part* > {Path(remote_name).name}"
        }
    else:
        fname = Path(remote_name).name
        if fname in existing and existing[fname] == sz:
            log.info(f"  ↷ Skip {fname} (already on XNAT, size match)")
        else:
            if not dry_run:
                log.info(f"  ↑ Uploading {fname} ({sz_mb:.0f} MB)...")
                t0 = time.time()
                res.upload(str(local_path), remotepath=fname, overwrite=True)
                dt = time.time() - t0
                log.info(f"    ✓ {fname} in {dt:.1f}s ({sz_mb/max(dt,0.001):.0f} MB/s)")
            else:
                log.info(f"  [DRY] Would upload {fname} ({sz_mb:.0f} MB)")
        return {"remote_name": remote_name, "local_path": str(local_path),
                "size_bytes": sz, "split": False}


def main():
    parser = argparse.ArgumentParser(description="Upload Horizon checkpoints to XNAT")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be uploaded without uploading")
    parser.add_argument("--only-checkpoints", action="store_true", help="Only upload experiments/checkpoints/, skip models/")
    parser.add_argument("--only-configs", action="store_true", help="Only upload config/tokenizer files (CONFIG_FILES)")
    parser.add_argument("--only-models", action="store_true", help="Only upload model weight files (MODEL_FILES)")
    args = parser.parse_args()

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    all_files = CHECKPOINT_FILES + ([] if args.only_checkpoints else MODEL_FILES + CONFIG_FILES)
    # Filter to existing local files only
    all_files = [(lp, rn) for lp, rn in all_files if lp.exists()]

    # Apply selective filters
    if args.only_configs:
        all_files = [(lp, rn) for lp, rn in zip([x[0] for x in CONFIG_FILES], [x[1] for x in CONFIG_FILES]) if lp.exists()]
    elif args.only_models:
        all_files = [(lp, rn) for lp, rn in MODEL_FILES if lp.exists()]

    total_bytes = sum(lp.stat().st_size for lp, _ in all_files)
    log.info("=" * 70)
    log.info(f"XNAT CHECKPOINT UPLOAD {'[DRY RUN] ' if args.dry_run else ''}")
    log.info(f"  Files   : {len(all_files)}")
    log.info(f"  Total   : {total_bytes/1e9:.1f} GB")
    log.info(f"  Resource: {PROJECT}/{RESOURCE} (ID {RES_ID})")
    log.info("=" * 70)

    if args.dry_run:
        for lp, rn in all_files:
            sz = lp.stat().st_size
            split = sz / 1024**2 > SPLIT_MB
            log.info(f"  {'[SPLIT] ' if split else '        '}{rn}  ({sz/1e9:.2f} GB)")
        return

    log.info("Connecting to XNAT...")
    with xnat.connect(server=SERVER, user=ALIAS, password=SECRET) as session:
        res = session.projects[PROJECT].resources[RES_ID]
        log.info("Fetching existing files (this may take a moment)...")
        existing = {Path(f.name).name: int(f.size) if f.size else 0 for f in res.files.values()}
        log.info(f"Resource currently has {len(existing)} file(s).")

        manifest = {
            "resource": RESOURCE, "resource_id": RES_ID, "project": PROJECT,
            "upload_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "files": []
        }

        for i, (local_path, remote_name) in enumerate(all_files, 1):
            log.info(f"\n[{i}/{len(all_files)}] {remote_name} ({local_path.stat().st_size/1e9:.2f} GB)")
            entry = upload_file(res, local_path, remote_name, existing, args.dry_run)
            manifest["files"].append(entry)

        # Save + upload manifest
        manifest_path = BASE / "experiments" / "checkpoints_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        log.info(f"\nUploading checkpoints_manifest.json...")
        res.upload(str(manifest_path), remotepath="checkpoints_manifest.json", overwrite=True)

    log.info("\n" + "=" * 70)
    log.info("ALL CHECKPOINTS SUCCESSFULLY SYNCED TO XNAT!")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
