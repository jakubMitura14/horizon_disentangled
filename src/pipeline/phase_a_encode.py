"""Phase A: Pretrained Encoder Feature Extraction → HDF5

Runs frozen pretrained encoders on all raw DCE-MRI volumes (each timepoint
independently) and saves dense feature representations to HDF5. This is a
one-time preprocessing step that decouples expensive 3D encoding from
downstream latent training.

DDP-aware: when launched via torchrun, shards patients across GPUs and
each rank writes to a separate temp file, then rank-0 merges.

Encoders:
  1. SwinUNETR (SuPreM/VoComni/BrainSeg): 768-D global + multi-scale spatial
  2. MedVAE-4×: per-channel VAE latent (4, D/4, H/4, W/4)

Usage:
    # Single GPU
    python -m src.pipeline.phase_a_encode --config configs/phased_pipeline.json

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=4 -m src.pipeline.phase_a_encode \
        --config configs/phased_pipeline.json
"""

import os
import json
import argparse
import logging
import torch
import torch.distributed as dist
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from src.pipeline.config import PipelineConfig
from src.pipeline.stage1_supervisors.models import SwinUNETREncoder
from src.pipeline.stage2_causal_vae.medvae_backbone import MedVAEBackbone
from src.pipeline.data.dataset import MultimodalOncologyDataset

logger = logging.getLogger("phase_a")


def setup_distributed():
    """Initialize DDP if running under torchrun/SLURM."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    elif "SLURM_PROCID" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ.get("SLURM_NPROCS", 1))
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


@torch.no_grad()
def encode_all_patients(
    config_path: str,
    output_path: str = "hdf5/encoder_features.h5",
):
    """
    Forward-pass all patient volumes through frozen encoders → save to HDF5.

    Each timepoint is encoded independently — the ODE in Phase C handles
    temporal structure.

    DDP: patients are sharded across ranks. Each rank writes a temp file,
    then rank-0 merges into the final output.
    """
    rank, world_size, local_rank = setup_distributed()
    is_main = (rank == 0)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Load config
    with open(config_path, "r") as f:
        phased_cfg = json.load(f)

    pa_cfg = phased_cfg["phase_a"]
    domain = phased_cfg["domain"]
    vis_cfg = pa_cfg["encoders"]["visual_encoder"]
    vae_cfg = pa_cfg["encoders"]["vae_encoder"]
    selected_visual = vis_cfg["selected"]
    selected_vae = vae_cfg["selected"]
    vis_opts = vis_cfg["options"][selected_visual]
    vae_opts = vae_cfg["options"][selected_vae]

    dtype = getattr(torch, pa_cfg.get("dtype", "bfloat16"))
    store_dtype = np.float16  # compact storage

    if is_main:
        logger.info(f"Phase A: Encoding with visual={selected_visual}, vae={selected_vae}")
        logger.info(f"  Domain: {domain}, dtype: {dtype}, device: {device}")

    # ─── Load frozen visual encoder ───────────────────────────────────
    visual_encoder = SwinUNETREncoder(
        in_channels=vis_opts["in_channels"],
        feature_dim=vis_opts["feature_dim"],
        feature_size=vis_opts["feature_size"],
        depths=tuple(vis_opts["depths"]),
        pretrained=selected_visual,
    ).to(device, dtype=dtype).eval()
    for p in visual_encoder.parameters():
        p.requires_grad = False

    # ─── Load frozen MedVAE encoder ───────────────────────────────────
    medvae = MedVAEBackbone(
        model_name=selected_vae,
        target_latent_channels=vae_opts["latent_channels"],
        freeze_backbone=True,
    ).to(device, dtype=dtype).eval()
    for p in medvae.parameters():
        p.requires_grad = False

    # ─── Load dataset ─────────────────────────────────────────────────
    pipeline_cfg = PipelineConfig()
    dataset = MultimodalOncologyDataset(
        vds_path=pipeline_cfg.data.master_vds_path,
        domain=domain,
        load_images=True,
    )

    # ─── Shard patients across GPUs ───────────────────────────────────
    all_indices = list(range(len(dataset)))
    my_indices = all_indices[rank::world_size]
    if is_main:
        logger.info(f"  Total patients: {len(dataset)}, this rank: {len(my_indices)}")

    # ─── Create per-rank output HDF5 ──────────────────────────────────
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if world_size > 1:
        rank_path = output_dir / f"encoder_features_rank{rank}.h5"
    else:
        rank_path = Path(output_path)

    batch_size = pa_cfg.get("batch_size", 8)
    n_encoded = 0
    n_skipped = 0

    with h5py.File(str(rank_path), "w") as out_h5:
        out_h5.attrs["encoder_visual"] = selected_visual
        out_h5.attrs["encoder_vae"] = selected_vae
        out_h5.attrs["domain"] = domain
        out_h5.attrs["rank"] = rank

        for idx in my_indices:
            sample = dataset[idx]
            uid = sample["uid"]

            for visit in sample["visits"]:
                if not visit["has_image"] or visit["image"] is None:
                    n_skipped += 1
                    continue

                img = visit["image"].unsqueeze(0).to(device, dtype=dtype)
                vk = visit["key"]
                t_days = visit.get("t_days", 0.0)

                # ── Visual encoder: global 768-D embedding ──
                h_visual = visual_encoder(img)  # (1, feature_dim)

                # ── MedVAE: per-channel latent encoding ──
                C_in = img.shape[1]
                z_channels = []
                for c in range(C_in):
                    z_ch = medvae.encode_stage_2_inputs(
                        img[:, c : c + 1].to(torch.float32)
                    )
                    z_channels.append(z_ch.half().cpu())
                z_vae = torch.cat(z_channels, dim=1).squeeze(0).numpy()

                # ── Save to HDF5 ──
                grp = out_h5.require_group(f"patients/{uid}/visits/{vk}")
                grp.create_dataset(
                    "h_visual",
                    data=h_visual.cpu().to(torch.float16).squeeze(0).numpy(),
                    compression="gzip",
                    compression_opts=pa_cfg.get("compression_level", 4),
                )
                grp.create_dataset(
                    "z_vae_per_channel",
                    data=z_vae.astype(store_dtype),
                    compression="gzip",
                    compression_opts=pa_cfg.get("compression_level", 4),
                )
                grp.attrs["t_days"] = float(t_days)
                grp.attrs["domain"] = domain
                n_encoded += 1

            # Save patient-level clinical data
            p_grp = out_h5.require_group(f"patients/{uid}")
            p_grp.attrs["pcr"] = int(sample.get("pcr", -1))
            p_grp.attrs["event"] = int(sample.get("event", 0))
            p_grp.attrs["os_weeks"] = float(sample.get("os_weeks", -1.0))
            if "clinical" in sample:
                p_grp.create_dataset(
                    "clinical",
                    data=sample["clinical"].numpy(),
                    compression="gzip",
                )

    logger.info(f"  Rank {rank}: encoded {n_encoded} volumes, skipped {n_skipped}")

    # ─── Merge rank files (rank-0 only) ───────────────────────────────
    if world_size > 1:
        dist.barrier()
        if is_main:
            logger.info("Merging rank files...")
            with h5py.File(output_path, "w") as merged:
                merged.attrs["encoder_visual"] = selected_visual
                merged.attrs["encoder_vae"] = selected_vae
                merged.attrs["domain"] = domain
                merged.attrs["world_size"] = world_size

                for r in range(world_size):
                    rp = output_dir / f"encoder_features_rank{r}.h5"
                    with h5py.File(str(rp), "r") as rank_h5:
                        for uid_key in rank_h5.get("patients", {}):
                            rank_h5.copy(
                                f"patients/{uid_key}",
                                merged.require_group("patients"),
                                name=uid_key,
                            )
                    rp.unlink()
            logger.info(f"Phase A complete → {output_path}")
        dist.barrier()
    else:
        logger.info(f"Phase A complete → {output_path}")

    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Phase A: Encoder Preprocessing")
    parser.add_argument("--config", type=str, default="configs/phased_pipeline.json")
    parser.add_argument("--output", type=str, default="hdf5/encoder_features.h5")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    encode_all_patients(args.config, args.output)


if __name__ == "__main__":
    main()
