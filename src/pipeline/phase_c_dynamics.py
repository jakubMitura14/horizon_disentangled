"""Phase C: NJDE Longitudinal Dynamics Training on Cached Latents

Trains the Neural Jump ODE (systemic drift + treatment jumps + breast PDE)
on pre-computed latent trajectories from Phase B. The VAE decoder is loaded
frozen (eval-mode, no gradient) for reconstruction plausibility loss.

Each patient's longitudinal visits are loaded as a time series of latent
states — the ODE handles all temporal structure.

DDP-aware: scales to multi-GPU cluster via torchrun / SLURM.

Usage:
    # Single GPU
    python -m src.pipeline.phase_c_dynamics --config configs/phased_pipeline.json

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=4 -m src.pipeline.phase_c_dynamics \
        --config configs/phased_pipeline.json --epochs 50
"""

import os
import json
import argparse
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.nn.utils.rnn import pad_sequence
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from src.pipeline.stage4_dynamics.solver import ContinuousDiscreteJumpSolver
from src.pipeline.stage4_dynamics.masked_loss import MaskedLeaveOneOutLoss

logger = logging.getLogger("phase_c")


# ═══════════════════════════════════════════════════════════════════════
# Dataset: reads Phase B latent trajectories
# ═══════════════════════════════════════════════════════════════════════

class LatentTrajectoryDataset(Dataset):
    """Reads pre-computed latent trajectories from Phase B HDF5.

    Each item is a full patient trajectory: a time series of latent states
    at observed timepoints. The ODE handles temporal structure — we don't
    concatenate timepoints, we return them as a sequence.

    For I-SPY 2: T0 (day 0), T1 (day 21), T2 (day 84), T3 (day 168).
    """

    def __init__(self, latent_h5_path: str, min_visits: int = 2):
        self.h5_path = latent_h5_path
        self.h5_file = None
        self.min_visits = min_visits

        # Build index: patients with ≥ min_visits timepoints
        self.patient_index = []
        with h5py.File(latent_h5_path, "r", swmr=True) as h5:
            for uid in h5.get("patients", {}):
                p_grp = h5[f"patients/{uid}"]
                visits_grp = p_grp.get("visits", {})
                visit_keys = sorted(visits_grp.keys())
                if len(visit_keys) >= min_visits:
                    self.patient_index.append((uid, visit_keys))

    def __len__(self):
        return len(self.patient_index)

    def _get_h5(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r", swmr=True)
        return self.h5_file

    def __getitem__(self, idx):
        uid, visit_keys = self.patient_index[idx]
        h5 = self._get_h5()

        u_d_series = []
        v_sys_series = []
        t_days_list = []

        for vk in visit_keys:
            grp = h5[f"patients/{uid}/visits/{vk}"]
            u_d = torch.from_numpy(grp["u_d"][:].astype(np.float32))
            v_sys = torch.from_numpy(grp["v_sys_init"][:].astype(np.float32))
            t_days = float(grp.attrs.get("t_days", 0.0))

            u_d_series.append(u_d)
            v_sys_series.append(v_sys)
            t_days_list.append(t_days)

        return {
            "uid": uid,
            "u_d": torch.stack(u_d_series),        # (T, C, D, H, W)
            "v_sys": torch.stack(v_sys_series),     # (T, 160)
            "t_days": torch.tensor(t_days_list),    # (T,)
            "n_visits": len(visit_keys),
        }


def _trajectory_collate_fn(batch: List[Dict]) -> Dict:
    """Collate variable-length trajectories with padding."""
    max_t = max(item["n_visits"] for item in batch)

    u_d_padded = []
    v_sys_padded = []
    t_days_padded = []
    masks = []

    for item in batch:
        T = item["n_visits"]
        pad_t = max_t - T

        u_d = item["u_d"]
        v_sys = item["v_sys"]
        t_days = item["t_days"]

        if pad_t > 0:
            u_d = torch.cat([u_d, u_d[-1:].expand(pad_t, -1, -1, -1, -1)], dim=0)
            v_sys = torch.cat([v_sys, v_sys[-1:].expand(pad_t, -1)], dim=0)
            t_days = torch.cat([t_days, t_days[-1:].expand(pad_t)], dim=0)

        u_d_padded.append(u_d)
        v_sys_padded.append(v_sys)
        t_days_padded.append(t_days)
        mask = torch.zeros(max_t, dtype=torch.bool)
        mask[:T] = True
        masks.append(mask)

    return {
        "u_d": torch.stack(u_d_padded),          # (B, T, C, D, H, W)
        "v_sys": torch.stack(v_sys_padded),       # (B, T, 160)
        "t_days": torch.stack(t_days_padded),     # (B, T)
        "mask": torch.stack(masks),                # (B, T)
        "uids": [item["uid"] for item in batch],
    }


def _worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        worker_info.dataset.h5_file = None
        np.random.seed((torch.initial_seed() + worker_id) % 2**32)


# ═══════════════════════════════════════════════════════════════════════
# Phase C Training
# ═══════════════════════════════════════════════════════════════════════

def setup_distributed():
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


def train_phase_c(
    config_path: str,
    latent_features_path: str = "hdf5/latent_decomposition.h5",
    decoder_checkpoint: str = "experiments/checkpoints/phase_b_vae.pt",
    output_path: str = "hdf5/counterfactual_results.h5",
    epochs: Optional[int] = None,
):
    """Train NJDE on cached latent trajectories with frozen decoder for loss."""
    rank, world_size, local_rank = setup_distributed()
    is_main = (rank == 0)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    with open(config_path, "r") as f:
        phased_cfg = json.load(f)
    pc_cfg = phased_cfg["phase_c"]
    cluster_cfg = phased_cfg.get("cluster", {})
    domain = phased_cfg["domain"]

    num_epochs = epochs or pc_cfg["training"]["epochs"]
    lr = pc_cfg["training"]["lr"]
    batch_size = pc_cfg["training"]["batch_size"]
    grad_accum = pc_cfg["training"]["gradient_accumulation_steps"]

    if is_main:
        logger.info(f"Phase C: Training NJDE, epochs={num_epochs}, lr={lr}")

    # ─── Dataset ──────────────────────────────────────────────────────
    dataset = LatentTrajectoryDataset(latent_features_path, min_visits=2)
    sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size) if world_size > 1 else None
    n_workers = min(8, max(1, (os.cpu_count() or 4) // max(world_size, 1)))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=(n_workers > 0),
        prefetch_factor=4 if n_workers > 0 else None,
        collate_fn=_trajectory_collate_fn,
        worker_init_fn=_worker_init_fn,
        drop_last=True,
    )

    # ─── Solver (trainable NJDE + PDE) ────────────────────────────────
    solver = ContinuousDiscreteJumpSolver(domain=domain).to(device)

    nccl_bucket = cluster_cfg.get("nccl_bucket_cap_mb", 50)
    if dist.is_initialized():
        solver.wrap_for_cluster()

    # Compile
    compile_mode = cluster_cfg.get("compile_mode", "max-autotune")
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        solver.spatial_pde = torch.compile(solver.spatial_pde, mode=compile_mode, dynamic=True)
        solver.systemic_ode = torch.compile(solver.systemic_ode, mode=compile_mode, dynamic=True)
        if is_main:
            logger.info(f"torch.compile applied to PDE + ODE (mode={compile_mode})")
    except Exception as e:
        if is_main:
            logger.warning(f"torch.compile skipped: {e}")

    # ─── Frozen decoder for plausibility loss ─────────────────────────
    decoder = None
    if os.path.exists(decoder_checkpoint):
        from src.pipeline.stage2_causal_vae.architecture import TwoScaleHierarchicalCausalVAE
        ckpt = torch.load(decoder_checkpoint, map_location=device, weights_only=False)
        decoder_vae = TwoScaleHierarchicalCausalVAE(
            in_channels=4,
            disease_channels=pc_cfg["njde"]["spatial_channels"],
        )
        decoder_vae.load_state_dict(ckpt["vae_state"], strict=False)
        decoder_vae = decoder_vae.to(device).eval()
        for p in decoder_vae.parameters():
            p.requires_grad = False
        decoder = decoder_vae
        if is_main:
            logger.info(f"  Frozen decoder loaded from {decoder_checkpoint}")
    else:
        if is_main:
            logger.warning(f"  Decoder checkpoint not found: {decoder_checkpoint}, skipping recon loss")

    # ─── Loss + optimizer ─────────────────────────────────────────────
    masked_loss = MaskedLeaveOneOutLoss()
    effective_lr = lr * max(world_size, 1) * grad_accum
    optimizer = torch.optim.AdamW(solver.parameters(), lr=effective_lr, weight_decay=pc_cfg["training"]["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    if is_main:
        logger.info(f"  Dataset: {len(dataset)} patient trajectories, {len(loader)} batches/epoch")

    # ─── Training Loop ────────────────────────────────────────────────
    for epoch in range(num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in loader:
            u_d = batch["u_d"].to(device)       # (B, T, C, D, H, W)
            v_sys = batch["v_sys"].to(device)   # (B, T, 160)
            t_days = batch["t_days"].to(device) # (B, T)
            mask = batch["mask"].to(device)     # (B, T)

            B, T, C, D, H, W = u_d.shape

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # For each patient, integrate from t0 to t1, t1 to t2, etc.
                # and compare predicted state at each timepoint with observed
                total_traj_loss = torch.tensor(0.0, device=device)
                n_valid = 0

                for b in range(B):
                    n_vis = mask[b].sum().int().item()
                    if n_vis < 2:
                        continue

                    u_curr = u_d[b, 0:1]            # (1, C, D, H, W)
                    v_curr = v_sys[b, 0:1]          # (1, 160)
                    phi = torch.zeros_like(u_curr[:, 0:1])
                    c_drug = torch.zeros(1, 16, device=device)

                    for t_idx in range(1, n_vis):
                        t_start = t_days[b, t_idx - 1].item()
                        t_end = t_days[b, t_idx].item()
                        if t_end <= t_start:
                            continue

                        dt_last = torch.tensor([[t_end - t_start]], device=device)

                        # Integrate one interval
                        u_pred, v_pred = solver.integrate_interval(
                            u_curr, v_curr, t_start, t_end,
                            phi, c_drug, t_start,
                        )

                        # Loss: compare predicted vs observed at this timepoint
                        u_target = u_d[b, t_idx : t_idx + 1]
                        v_target = v_sys[b, t_idx : t_idx + 1]

                        loss_u = F.mse_loss(u_pred, u_target)
                        loss_v = F.mse_loss(v_pred, v_target)
                        total_traj_loss = total_traj_loss + loss_u + 0.1 * loss_v

                        # Use predicted state as next starting point (teacher forcing off)
                        u_curr = u_pred.detach()
                        v_curr = v_pred.detach()
                        n_valid += 1

                if n_valid > 0:
                    total_traj_loss = total_traj_loss / n_valid

            scaler.scale(total_traj_loss / grad_accum).backward()

            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(solver.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss += total_traj_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        if is_main and (epoch % 5 == 0 or epoch == num_epochs - 1):
            logger.info(f"  Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.6f} ({elapsed:.1f}s)")

    # ─── Save checkpoint ──────────────────────────────────────────────
    if is_main:
        ckpt_dir = Path("experiments/checkpoints")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "solver_state": solver.state_dict(),
            "config": pc_cfg,
        }, str(ckpt_dir / "phase_c_njde.pt"))
        logger.info(f"  NJDE checkpoint saved → {ckpt_dir / 'phase_c_njde.pt'}")

    if dist.is_initialized():
        dist.destroy_process_group()

    if is_main:
        logger.info("Phase C complete.")


def main():
    parser = argparse.ArgumentParser(description="Phase C: NJDE Longitudinal Dynamics")
    parser.add_argument("--config", type=str, default="configs/phased_pipeline.json")
    parser.add_argument("--latent-features", type=str, default="hdf5/latent_decomposition.h5")
    parser.add_argument("--decoder-checkpoint", type=str, default="experiments/checkpoints/phase_b_vae.pt")
    parser.add_argument("--output", type=str, default="hdf5/counterfactual_results.h5")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    train_phase_c(args.config, args.latent_features, args.decoder_checkpoint, args.output, args.epochs)


if __name__ == "__main__":
    main()
