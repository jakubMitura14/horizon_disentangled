"""Phase B: Latent Disentanglement Training on Cached Encoder Features → HDF5

Trains the Causal VAE Level 2 (ConvNeXt downsamplers + latent projection heads +
disentanglement networks) on pre-computed encoder features from Phase A, then saves
the full latent decomposition (Z_A, Z_D, Z_P, Z_S) + OOD scores to HDF5.

DDP-aware: scales to multi-GPU cluster via torchrun / SLURM.

Usage:
    # Single GPU
    python -m src.pipeline.phase_b_latents --config configs/phased_pipeline.json

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=4 -m src.pipeline.phase_b_latents \
        --config configs/phased_pipeline.json --epochs 100
"""

import os
import json
import argparse
import logging
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from src.pipeline.config import PipelineConfig
from src.pipeline.stage2_causal_vae.architecture import TwoScaleHierarchicalCausalVAE
from src.pipeline.stage2_causal_vae.disentanglement import AsymmetricCausalDisentanglement
from src.pipeline.stage2_causal_vae.uncertainty import LatentUncertaintyManager
from src.pipeline.stage3_ood.filter import Stage3OODFilter

logger = logging.getLogger("phase_b")


# ═══════════════════════════════════════════════════════════════════════
# Dataset: reads Phase A HDF5
# ═══════════════════════════════════════════════════════════════════════

class EncoderFeatureDataset(Dataset):
    """Reads pre-computed encoder features from Phase A HDF5.

    Each item is a single visit (timepoint) for a single patient, containing
    the visual embedding and VAE latent from the frozen encoders.
    The ODE in Phase C handles temporal structure — here we just train
    the disentanglement on individual volumes.
    """

    def __init__(self, encoder_h5_path: str, domain: str = "breast"):
        self.h5_path = encoder_h5_path
        self.h5_file = None
        self.domain = domain

        # Build index: (uid, visit_key) pairs
        self.index = []
        with h5py.File(encoder_h5_path, "r", swmr=True) as h5:
            for uid in h5.get("patients", {}):
                p_grp = h5[f"patients/{uid}"]
                visits = p_grp.get("visits", {})
                for vk in sorted(visits.keys()):
                    if "z_vae_per_channel" in visits[vk]:
                        clinical = p_grp["clinical"][:] if "clinical" in p_grp else np.zeros(16, dtype=np.float32)
                        self.index.append((uid, vk, clinical))

    def __len__(self):
        return len(self.index)

    def _get_h5(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r", swmr=True)
        return self.h5_file

    def __getitem__(self, idx):
        uid, vk, clinical = self.index[idx]
        h5 = self._get_h5()
        grp = h5[f"patients/{uid}/visits/{vk}"]

        z_vae = torch.from_numpy(grp["z_vae_per_channel"][:].astype(np.float32))
        h_visual = torch.from_numpy(grp["h_visual"][:].astype(np.float32))
        t_days = float(grp.attrs.get("t_days", 0.0))
        clinical_t = torch.from_numpy(clinical.astype(np.float32))

        return {
            "uid": uid,
            "visit_key": vk,
            "z_vae": z_vae,              # (C_in*4, D/4, H/4, W/4)
            "h_visual": h_visual,        # (feature_dim,)
            "clinical": clinical_t,      # (16,)
            "t_days": t_days,
        }


def _worker_init_fn(worker_id):
    """Each DataLoader worker opens its own HDF5 handle."""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        worker_info.dataset.h5_file = None
        np.random.seed((torch.initial_seed() + worker_id) % 2**32)


# ═══════════════════════════════════════════════════════════════════════
# Phase B Training
# ═══════════════════════════════════════════════════════════════════════

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


def train_phase_b(
    config_path: str,
    encoder_features_path: str = "hdf5/encoder_features.h5",
    output_path: str = "hdf5/latent_decomposition.h5",
    checkpoint_dir: str = "experiments/checkpoints",
    epochs: Optional[int] = None,
):
    """Train Causal VAE Level 2 on cached encoder features, then save latents."""
    rank, world_size, local_rank = setup_distributed()
    is_main = (rank == 0)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    with open(config_path, "r") as f:
        phased_cfg = json.load(f)
    pb_cfg = phased_cfg["phase_b"]
    cluster_cfg = phased_cfg.get("cluster", {})
    domain = phased_cfg["domain"]

    num_epochs = epochs or pb_cfg["training"]["epochs"]
    lr = pb_cfg["training"]["lr"]
    batch_size = pb_cfg["training"]["batch_size"]
    grad_accum = pb_cfg["training"]["gradient_accumulation_steps"]
    warmup_epochs = pb_cfg["training"]["warmup_epochs"]

    if is_main:
        logger.info(f"Phase B: Training VAE Level 2, epochs={num_epochs}, lr={lr}")

    # ─── Dataset ──────────────────────────────────────────────────────
    dataset = EncoderFeatureDataset(encoder_features_path, domain=domain)
    sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size) if world_size > 1 else None
    n_workers = min(16, max(1, (os.cpu_count() or 4) // max(world_size, 1)))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=(n_workers > 0),
        prefetch_factor=4 if n_workers > 0 else None,
        worker_init_fn=_worker_init_fn,
        drop_last=True,
    )

    # ─── Models ───────────────────────────────────────────────────────
    pipeline_cfg = PipelineConfig()
    vae = TwoScaleHierarchicalCausalVAE(
        in_channels=4,
        anatomy_channels=pb_cfg["vae_level2"]["disease_spatial_channels"],
        disease_channels=pb_cfg["vae_level2"]["disease_spatial_channels"],
        global_disease_dim=pb_cfg["vae_level2"]["disease_global_dim"],
        patient_dim=pb_cfg["vae_level2"]["patient_dim"],
        style_dim=pb_cfg["vae_level2"]["style_dim"],
    ).to(device)

    disentangle = AsymmetricCausalDisentanglement(
        disease_dim=pb_cfg["vae_level2"]["disease_global_dim"],
        patient_dim=pb_cfg["vae_level2"]["patient_dim"],
        style_dim=pb_cfg["vae_level2"]["style_dim"],
        anatomy_channels=pb_cfg["vae_level2"]["disease_spatial_channels"],
        beta_tc=pb_cfg["disentanglement"]["beta_tc"],
        gamma_club=pb_cfg["disentanglement"]["gamma_club"],
        alpha_dp=pb_cfg["disentanglement"]["alpha_dp"],
    ).to(device)

    # DDP wrap with H200-optimized settings
    nccl_bucket = cluster_cfg.get("nccl_bucket_cap_mb", 50)
    if dist.is_initialized():
        vae = DDP(
            vae, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False, static_graph=True,
            bucket_cap_mb=nccl_bucket, gradient_as_bucket_view=True,
        )
        disentangle = DDP(
            disentangle, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False, static_graph=True,
            bucket_cap_mb=nccl_bucket, gradient_as_bucket_view=True,
        )

    # Compile for max performance
    compile_mode = cluster_cfg.get("compile_mode", "max-autotune")
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        vae = torch.compile(vae, mode=compile_mode, dynamic=True)
        if is_main:
            logger.info(f"torch.compile applied (mode={compile_mode})")
    except Exception as e:
        if is_main:
            logger.warning(f"torch.compile skipped: {e}")

    # ─── Optimizers ───────────────────────────────────────────────────
    effective_lr = lr * max(world_size, 1) * grad_accum
    optimizer_vae = torch.optim.AdamW(vae.parameters(), lr=effective_lr, weight_decay=pb_cfg["training"]["weight_decay"])
    raw_dis = disentangle.module if dist.is_initialized() else disentangle
    optimizer_aux = torch.optim.AdamW(raw_dis.parameters(), lr=effective_lr * 0.5, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    # ─── Training Loop ────────────────────────────────────────────────
    if is_main:
        logger.info(f"  Dataset: {len(dataset)} volumes, {len(loader)} batches/epoch")

    for epoch in range(num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in loader:
            z_vae = batch["z_vae"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # Phase A already computed per-channel MedVAE latents.
                # Use encode_from_cached() to skip encode_per_channel()
                # and go straight to cross_seq_fusion → L2 → heads → decoder.
                raw_vae_ref = vae.module if dist.is_initialized() else vae
                if hasattr(raw_vae_ref, '_orig_mod'):
                    raw_vae_ref = raw_vae_ref._orig_mod
                enc = raw_vae_ref.encode_from_cached(z_vae)
                recon, mask_pred = raw_vae_ref.decode(enc["z_a"], enc["u_d"], enc["z_s"])
                # Recon loss: compare decoded output against anatomy z_a
                # (decoder outputs in_channels dims, same as z_a channels)
                z_a_target = enc["z_a"].detach()
                if recon.shape[2:] != z_a_target.shape[2:]:
                    recon = torch.nn.functional.interpolate(recon, size=z_a_target.shape[2:], mode="trilinear", align_corners=False)
                out = enc
                out["recon"] = recon
                out["mask_pred"] = mask_pred

                loss_recon = torch.nn.functional.mse_loss(out["recon"], z_a_target)
                kl_d = -0.5 * torch.mean(1 + out["logvar_d"] - out["mu_d"].pow(2) - out["logvar_d"].exp())
                kl_p = -0.5 * torch.mean(1 + out["logvar_p"] - out["mu_p"].pow(2) - out["logvar_p"].exp())
                kl_s = -0.5 * torch.mean(1 + out["logvar_s"] - out["mu_s"].pow(2) - out["logvar_s"].exp())
                loss_kl = kl_d + kl_p + kl_s

                raw_d = disentangle.module if dist.is_initialized() else disentangle
                loss_dis, loss_aux, _ = raw_d.compute_disentangle_loss(
                    out["z_d_glob"], out["z_p"], out["z_s"], out["z_a"]
                )

                kl_w = pb_cfg["disentanglement"]["kl_weight"]
                total_loss = loss_recon + kl_w * loss_kl + loss_dis

            scaler.scale(total_loss / grad_accum).backward(retain_graph=True)
            scaler.scale(loss_aux / grad_accum).backward()

            # Gradient accumulation step logic
            # (backward accumulated, now step/zero only on accum boundary)

            scaler.unscale_(optimizer_vae)
            scaler.unscale_(optimizer_aux)
            raw_vae = vae.module if dist.is_initialized() else vae
            nn.utils.clip_grad_norm_(raw_vae.parameters(), max_norm=1.0)
            nn.utils.clip_grad_norm_(raw_dis.parameters(), max_norm=1.0)
            scaler.step(optimizer_vae)
            scaler.step(optimizer_aux)
            scaler.update()
            optimizer_vae.zero_grad(set_to_none=True)
            optimizer_aux.zero_grad(set_to_none=True)

            epoch_loss += total_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        if is_main and (epoch % 10 == 0 or epoch == num_epochs - 1):
            logger.info(f"  Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f} ({elapsed:.1f}s)")

    # ─── Save checkpoint ──────────────────────────────────────────────
    if is_main:
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, "phase_b_vae.pt")
        raw_vae = vae.module if dist.is_initialized() else vae
        raw_dis = disentangle.module if dist.is_initialized() else disentangle
        torch.save({
            "vae_state": raw_vae.state_dict(),
            "disentangle_state": raw_dis.state_dict(),
            "config": pb_cfg,
        }, ckpt_path)
        logger.info(f"  Checkpoint saved → {ckpt_path}")

    # ─── Extract & save all latents ───────────────────────────────────
    if is_main:
        logger.info("Extracting latent decomposition for all volumes...")
    _extract_all_latents(vae, dataset, device, output_path, pb_cfg,
                         rank, world_size, is_main)

    if dist.is_initialized():
        dist.destroy_process_group()

    if is_main:
        logger.info(f"Phase B complete → {output_path}")


@torch.no_grad()
def _extract_all_latents(
    vae, dataset, device, output_path, pb_cfg,
    rank, world_size, is_main
):
    """Run trained VAE on all cached features and save latent decomposition."""
    raw_vae = vae.module if dist.is_initialized() else vae
    if hasattr(raw_vae, '_orig_mod'):
        raw_vae = raw_vae._orig_mod
    raw_vae.eval()

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if world_size > 1:
        rank_path = output_dir / f"latents_rank{rank}.h5"
    else:
        rank_path = Path(output_path)

    my_indices = list(range(len(dataset)))[rank::world_size]

    with h5py.File(str(rank_path), "w") as out_h5:
        for idx in my_indices:
            sample = dataset[idx]
            uid = sample["uid"]
            vk = sample["visit_key"]
            z_vae = sample["z_vae"].unsqueeze(0).to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                enc = raw_vae.encode(z_vae)

            grp = out_h5.require_group(f"patients/{uid}/visits/{vk}")
            for key in ["z_a", "u_d", "z_d_glob", "z_p", "z_s",
                        "mu_d", "logvar_d", "mu_p", "logvar_p", "mu_s", "logvar_s"]:
                val = enc[key].cpu().squeeze(0).numpy()
                store_dt = np.float16 if val.ndim > 1 else np.float32
                grp.create_dataset(key, data=val.astype(store_dt),
                                   compression="gzip", compression_opts=4)

            # Compose initial systemic state v_sys_init
            v_sys = np.zeros(160, dtype=np.float32)
            z_d = enc["z_d_glob"].cpu().squeeze(0).numpy()
            z_p = enc["z_p"].cpu().squeeze(0).numpy()
            v_sys[0:64] = z_d[:64] if len(z_d) >= 64 else np.pad(z_d, (0, 64 - len(z_d)))
            v_sys[64:96] = z_p[:32] if len(z_p) >= 32 else np.pad(z_p, (0, 32 - len(z_p)))
            clinical = sample["clinical"].numpy()
            v_sys[96:112] = clinical[:16]
            grp.create_dataset("v_sys_init", data=v_sys, compression="gzip")
            grp.attrs["t_days"] = sample["t_days"]

    # Merge rank files
    if world_size > 1:
        dist.barrier()
        if is_main:
            with h5py.File(output_path, "w") as merged:
                merged.attrs["config"] = json.dumps(pb_cfg)
                for r in range(world_size):
                    rp = output_dir / f"latents_rank{r}.h5"
                    with h5py.File(str(rp), "r") as rh5:
                        for uid_key in rh5.get("patients", {}):
                            rh5.copy(f"patients/{uid_key}", merged.require_group("patients"), name=uid_key)
                    rp.unlink()
        dist.barrier()


def main():
    parser = argparse.ArgumentParser(description="Phase B: Latent Disentanglement Training")
    parser.add_argument("--config", type=str, default="configs/phased_pipeline.json")
    parser.add_argument("--encoder-features", type=str, default="hdf5/encoder_features.h5")
    parser.add_argument("--output", type=str, default="hdf5/latent_decomposition.h5")
    parser.add_argument("--checkpoint-dir", type=str, default="experiments/checkpoints")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    train_phase_b(args.config, args.encoder_features, args.output, args.checkpoint_dir, args.epochs)


if __name__ == "__main__":
    main()
