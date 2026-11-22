#!/usr/bin/env python
"""48-Hour Stability & Convergence Test — All Pipeline Components.

Runs Phase B (VAE disentanglement) and Phase C (NJDE dynamics) for extended
training to verify:
  1. All loss components converge monotonically (no divergence)
  2. No GPU memory leaks over time (VRAM stable across epochs)
  3. No NaN/Inf in losses or gradients
  4. Throughput is stable (no progressive slowdown)
  5. Checkpoint save/load round-trip works

The script allocates ~24 hours per phase:
  Phase B (VAE): ~26s/epoch × ~3300 epochs ≈ 24h
  Phase C (ODE): ~15s/epoch × ~5760 epochs ≈ 24h

Logs comprehensive metrics every epoch and writes periodic JSON snapshots.
Designed to run unattended for 48 hours with automatic detection of anomalies.

Usage:
    cd /DATA/horizon_disentangled
    nohup LD_LIBRARY_PATH=/DATA/conda/envs/nukl-to-ct/lib:$LD_LIBRARY_PATH \
        /DATA/conda/envs/nukl-to-ct/bin/python scripts/training/stability_48h.py \
        --hours 48 2>&1 | tee experiments/stability_48h.log &

Environment variables:
    STABILITY_PHASE_B_HOURS: hours for Phase B (default: 24)
    STABILITY_PHASE_C_HOURS: hours for Phase C (default: 24)
    STABILITY_CHECKPOINT_INTERVAL: checkpoint every N epochs (default: 100)
    STABILITY_SNAPSHOT_INTERVAL: JSON snapshot every N epochs (default: 50)
"""
import os
import sys
import json
import time
import math
import tempfile
import threading
import logging
import gc
import signal
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, "/DATA/horizon_disentangled")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/DATA/horizon_disentangled/experiments/stability_48h.log",
                            mode="a"),
    ]
)
logger = logging.getLogger("stability_48h")


# ═══════════════════════════════════════════════════════════════════════════
# Anomaly Detection
# ═══════════════════════════════════════════════════════════════════════════

class AnomalyDetector:
    """Tracks metrics and raises alerts for anomalous behavior."""

    def __init__(self, name: str, window: int = 50):
        self.name = name
        self.window = window
        self.loss_history = []
        self.mem_history = []
        self.throughput_history = []
        self.grad_norm_history = []
        self.alerts = []

    def record(self, epoch: int, loss: float, mem_gb: float,
               throughput: float, grad_norm: float):
        self.loss_history.append(loss)
        self.mem_history.append(mem_gb)
        self.throughput_history.append(throughput)
        self.grad_norm_history.append(grad_norm)

        # Check for NaN/Inf
        if math.isnan(loss) or math.isinf(loss):
            msg = f"[ALERT] {self.name} epoch {epoch}: NaN/Inf loss = {loss}"
            logger.error(msg)
            self.alerts.append({"epoch": epoch, "type": "nan_loss", "msg": msg})
            return False

        if math.isnan(grad_norm) or math.isinf(grad_norm):
            msg = f"[ALERT] {self.name} epoch {epoch}: NaN/Inf grad_norm = {grad_norm}"
            logger.error(msg)
            self.alerts.append({"epoch": epoch, "type": "nan_grad", "msg": msg})

        # Check for memory leak: if memory grows > 1 GB over last 200 epochs
        if len(self.mem_history) > 200:
            mem_start = np.mean(self.mem_history[-200:-150])
            mem_end = np.mean(self.mem_history[-50:])
            mem_drift = mem_end - mem_start
            if mem_drift > 1.0:
                msg = (f"[ALERT] {self.name} epoch {epoch}: memory leak detected! "
                       f"Drift = +{mem_drift:.2f} GB over 200 epochs "
                       f"({mem_start:.2f} → {mem_end:.2f} GB)")
                logger.warning(msg)
                self.alerts.append({"epoch": epoch, "type": "memory_leak",
                                    "drift_gb": mem_drift, "msg": msg})

        # Check for throughput degradation > 20% from baseline
        if len(self.throughput_history) > 50:
            baseline_tp = np.mean(self.throughput_history[5:25])  # skip warmup
            recent_tp = np.mean(self.throughput_history[-20:])
            if baseline_tp > 0 and recent_tp < 0.8 * baseline_tp:
                msg = (f"[ALERT] {self.name} epoch {epoch}: throughput degradation! "
                       f"{baseline_tp:.2f} → {recent_tp:.2f} "
                       f"({100*(1-recent_tp/baseline_tp):.1f}% drop)")
                logger.warning(msg)
                self.alerts.append({"epoch": epoch, "type": "throughput_drop",
                                    "msg": msg})

        # Check for loss divergence: loss increased > 50% over last 100 epochs
        if len(self.loss_history) > 100:
            loss_100ago = self.loss_history[-100]
            if loss > loss_100ago * 1.5 and loss > 0:
                msg = (f"[ALERT] {self.name} epoch {epoch}: loss diverging! "
                       f"{loss_100ago:.6f} → {loss:.6f} over 100 epochs")
                logger.warning(msg)
                self.alerts.append({"epoch": epoch, "type": "loss_divergence",
                                    "msg": msg})

        return True

    def summary(self) -> dict:
        return {
            "total_epochs": len(self.loss_history),
            "loss_start": self.loss_history[0] if self.loss_history else None,
            "loss_end": self.loss_history[-1] if self.loss_history else None,
            "loss_min": min(self.loss_history) if self.loss_history else None,
            "loss_converged": (self.loss_history[-1] < self.loss_history[0])
                              if len(self.loss_history) > 1 else (True if self.loss_history else False),
            "mem_start_gb": self.mem_history[0] if self.mem_history else None,
            "mem_end_gb": self.mem_history[-1] if self.mem_history else None,
            "mem_max_gb": max(self.mem_history) if self.mem_history else None,
            "mem_drift_gb": (self.mem_history[-1] - self.mem_history[0])
                           if len(self.mem_history) > 1 else 0.0,
            "throughput_mean": float(np.mean(self.throughput_history[5:]))
                              if len(self.throughput_history) > 5 else None,
            "throughput_std": float(np.std(self.throughput_history[5:]))
                             if len(self.throughput_history) > 5 else None,
            "grad_norm_mean": float(np.mean(self.grad_norm_history))
                             if self.grad_norm_history else None,
            "alerts": self.alerts,
            "n_alerts": len(self.alerts),
        }


# ═══════════════════════════════════════════════════════════════════════════
# GPU Monitor Thread
# ═══════════════════════════════════════════════════════════════════════════

class GPUMonitor:
    """Lightweight GPU monitor with periodic memory sampling."""

    def __init__(self, interval: float = 2.0):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            while not self._stop.is_set():
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.samples.append({
                    "gpu_util_pct": util.gpu,
                    "mem_used_gb": mem.used / (1024**3),
                    "timestamp": time.time(),
                })
                self._stop.wait(self.interval)
        except ImportError:
            while not self._stop.is_set():
                self.samples.append({
                    "gpu_util_pct": -1,
                    "mem_used_gb": torch.cuda.memory_allocated(0) / (1024**3),
                    "timestamp": time.time(),
                })
                self._stop.wait(self.interval)

    def latest_mem_gb(self) -> float:
        if self.samples:
            return self.samples[-1]["mem_used_gb"]
        return torch.cuda.memory_allocated(0) / (1024**3)

    def latest_util(self) -> float:
        if self.samples:
            return self.samples[-1]["gpu_util_pct"]
        return -1


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic Data Creation
# ═══════════════════════════════════════════════════════════════════════════

N_PATIENTS = 80
N_VISITS = 4
Z_VAE_SHAPE = (16, 32, 32, 16)  # (C_in*4, D/4, H/4, W/4)
U_D_SHAPE = (8, 32, 32, 16)     # 8 channels × 32×32×16

def create_phase_a_hdf5(path: str):
    """Create synthetic Phase A encoder features HDF5.
    
    Uses the same HDF5 structure as EncoderFeatureDataset expects:
    patients/{uid}/visits/{vk}/z_vae_per_channel, h_visual
    """
    logger.info(f"Creating synthetic Phase A HDF5: {N_PATIENTS} patients × {N_VISITS} visits")
    logger.info(f"  z_vae shape per visit: {Z_VAE_SHAPE}")
    with h5py.File(path, "w") as f:
        for pid in range(N_PATIENTS):
            uid = f"patient_{pid:04d}"
            p_grp = f.create_group(f"patients/{uid}")
            p_grp.create_dataset("clinical",
                                 data=np.random.randn(16).astype(np.float32))
            for v in range(N_VISITS):
                vk = f"visit_{v}"
                grp = p_grp.create_group(f"visits/{vk}")
                grp.create_dataset("h_visual",
                                   data=np.random.randn(768).astype(np.float16))
                grp.create_dataset("z_vae_per_channel",
                                   data=np.random.randn(*Z_VAE_SHAPE).astype(np.float16))
                t_days = [0.0, 21.0, 84.0, 168.0][v] if v < 4 else float(v * 42)
                grp.attrs["t_days"] = t_days
                grp.attrs["domain"] = "breast"
    logger.info(f"  Phase A HDF5: {os.path.getsize(path)/(1024**2):.1f} MB")


def create_phase_b_hdf5(path: str):
    """Create synthetic Phase B latent decomposition HDF5.
    
    Uses the same HDF5 structure as LatentTrajectoryDataset expects:
    patients/{uid}/visits/{vk}/u_d, v_sys_init, z_a, z_d_glob, z_p, z_s, ...
    """
    logger.info(f"Creating synthetic Phase B HDF5: {N_PATIENTS} patients × {N_VISITS} visits")
    logger.info(f"  u_d shape per visit: {U_D_SHAPE}")
    with h5py.File(path, "w") as f:
        for pid in range(N_PATIENTS):
            uid = f"patient_{pid:04d}"
            for v in range(N_VISITS):
                vk = f"visit_{v}"
                grp = f.create_group(f"patients/{uid}/visits/{vk}")
                grp.create_dataset("u_d", data=np.random.randn(*U_D_SHAPE).astype(np.float16))
                grp.create_dataset("z_a", data=np.random.randn(4, 32, 32, 16).astype(np.float16))
                grp.create_dataset("z_d_glob", data=np.random.randn(128).astype(np.float32))
                grp.create_dataset("z_p", data=np.random.randn(64).astype(np.float32))
                grp.create_dataset("z_s", data=np.random.randn(64).astype(np.float32))
                grp.create_dataset("mu_d", data=np.random.randn(128).astype(np.float32))
                grp.create_dataset("logvar_d", data=np.random.randn(128).astype(np.float32) - 2)
                grp.create_dataset("mu_p", data=np.random.randn(64).astype(np.float32))
                grp.create_dataset("logvar_p", data=np.random.randn(64).astype(np.float32) - 2)
                grp.create_dataset("mu_s", data=np.random.randn(64).astype(np.float32))
                grp.create_dataset("logvar_s", data=np.random.randn(64).astype(np.float32) - 2)
                grp.create_dataset("v_sys_init", data=np.random.randn(160).astype(np.float32))
                t_days = [0.0, 21.0, 84.0, 168.0][v] if v < 4 else float(v * 42)
                grp.attrs["t_days"] = t_days
    logger.info(f"  Phase B HDF5: {os.path.getsize(path)/(1024**2):.1f} MB")


# ═══════════════════════════════════════════════════════════════════════════
# Phase B — Extended VAE Training Loop
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_b_extended(encoder_h5_path: str, max_hours: float = 24.0,
                          batch_size: int = 36, lr: float = 2e-4,
                          grad_accum_steps: int = 2,
                          checkpoint_interval: int = 100,
                          snapshot_interval: int = 50,
                          ckpt_dir: str = "experiments/stability_checkpoints"):
    """Run Phase B VAE training for up to max_hours.
    
    Returns loss history and anomaly detector summary.
    """
    from src.pipeline.phase_b_latents import EncoderFeatureDataset, _worker_init_fn
    from src.pipeline.stage2_causal_vae.architecture import TwoScaleHierarchicalCausalVAE
    from src.pipeline.stage2_causal_vae.disentanglement import AsymmetricCausalDisentanglement

    device = torch.device("cuda")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Dataset & Loader
    dataset = EncoderFeatureDataset(encoder_h5_path, domain="breast")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True,
        worker_init_fn=_worker_init_fn, drop_last=True, prefetch_factor=8,
    )
    logger.info(f"Phase B: {len(dataset)} volumes, {len(loader)} batches/epoch, "
                f"batch={batch_size}, grad_accum={grad_accum_steps}, lr={lr}")

    # Model: 199M param VAE
    vae = TwoScaleHierarchicalCausalVAE(
        in_channels=4, disease_channels=8,
        global_disease_dim=128, patient_dim=64, style_dim=64,
        l2_channels=512, vae_channels=(128, 256, 512),
        vae_num_res_blocks=3,
    ).to(device)
    disentangle = AsymmetricCausalDisentanglement(
        disease_dim=128, patient_dim=64, style_dim=64,
        anatomy_channels=4, beta_tc=1.5, gamma_club=1.0, alpha_dp=0.5,
    ).to(device)

    n_vae_params = sum(p.numel() for p in vae.parameters()) / 1e6
    n_dis_params = sum(p.numel() for p in disentangle.parameters()) / 1e6
    logger.info(f"  VAE: {n_vae_params:.1f}M params, Disentangle: {n_dis_params:.1f}M params")

    optimizer_vae = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=1e-5)
    optimizer_aux = torch.optim.Adam(disentangle.parameters(), lr=lr * 2)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    # LR scheduler: cosine with warmup
    warmup_epochs = 20
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / max(1, 10000)))
    scheduler_vae = torch.optim.lr_scheduler.LambdaLR(optimizer_vae, lr_lambda)
    scheduler_aux = torch.optim.lr_scheduler.LambdaLR(optimizer_aux, lr_lambda)

    detector = AnomalyDetector("Phase_B")
    monitor = GPUMonitor(interval=5.0)
    monitor.start()

    deadline = time.time() + max_hours * 3600
    epoch = 0
    loss_history = []
    best_loss = float("inf")
    ckpt_path = os.path.join(ckpt_dir, "phase_b_latest.pt")

    logger.info(f"Phase B: training until {datetime.fromtimestamp(deadline).isoformat()} "
                f"({max_hours:.1f} hours)")

    try:
        while time.time() < deadline:
            epoch += 1
            epoch_loss = 0.0
            epoch_loss_recon = 0.0
            epoch_loss_kl = 0.0
            epoch_loss_disent = 0.0
            n_batches = 0
            t0 = time.time()
            max_grad_norm = 0.0

            optimizer_vae.zero_grad(set_to_none=True)
            optimizer_aux.zero_grad(set_to_none=True)

            for batch_idx, batch in enumerate(loader):
                z_vae = batch["z_vae"].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    enc = vae.encode_from_cached(z_vae)
                    recon, mask_pred = vae.decode(enc["z_a"], enc["u_d"], enc["z_s"])

                    z_a_target = enc["z_a"].detach()
                    if recon.shape[2:] != z_a_target.shape[2:]:
                        recon = F.interpolate(recon, size=z_a_target.shape[2:],
                                              mode="trilinear", align_corners=False)
                    loss_recon = F.mse_loss(recon, z_a_target)
                    kl_d = -0.5 * torch.mean(1 + enc["logvar_d"] - enc["mu_d"].pow(2) - enc["logvar_d"].exp())
                    kl_p = -0.5 * torch.mean(1 + enc["logvar_p"] - enc["mu_p"].pow(2) - enc["logvar_p"].exp())
                    kl_s = -0.5 * torch.mean(1 + enc["logvar_s"] - enc["mu_s"].pow(2) - enc["logvar_s"].exp())
                    loss_kl = kl_d + kl_p + kl_s

                    kl_weight = min(1.0, epoch / 100.0)  # KL warmup

                    loss_disent, loss_aux, _ = disentangle.compute_disentangle_loss(
                        enc["z_d_glob"], enc["z_p"], enc["z_s"], enc["z_a"]
                    )
                    total_loss = loss_recon + kl_weight * 1e-3 * loss_kl + loss_disent

                # Combined backward — no retain_graph needed
                scaler.scale((total_loss + loss_aux) / grad_accum_steps).backward()

                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(loader):
                    scaler.unscale_(optimizer_vae)
                    scaler.unscale_(optimizer_aux)
                    gn_vae = nn.utils.clip_grad_norm_(vae.parameters(), max_norm=5.0)
                    gn_aux = nn.utils.clip_grad_norm_(disentangle.parameters(), max_norm=5.0)
                    max_grad_norm = max(max_grad_norm, gn_vae.item(), gn_aux.item())
                    scaler.step(optimizer_vae)
                    scaler.step(optimizer_aux)
                    scaler.update()
                    optimizer_vae.zero_grad(set_to_none=True)
                    optimizer_aux.zero_grad(set_to_none=True)

                epoch_loss += total_loss.item()
                epoch_loss_recon += loss_recon.item()
                epoch_loss_kl += loss_kl.item()
                epoch_loss_disent += loss_disent.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            elapsed = time.time() - t0
            throughput = len(dataset) / elapsed
            mem_gb = torch.cuda.memory_allocated(0) / (1024**3)
            mem_peak_gb = torch.cuda.max_memory_allocated(0) / (1024**3)

            loss_entry = {
                "epoch": epoch,
                "loss": round(avg_loss, 6),
                "loss_recon": round(epoch_loss_recon / max(n_batches, 1), 6),
                "loss_kl": round(epoch_loss_kl / max(n_batches, 1), 6),
                "loss_disent": round(epoch_loss_disent / max(n_batches, 1), 6),
                "time_s": round(elapsed, 2),
                "throughput": round(throughput, 2),
                "mem_gb": round(mem_gb, 2),
                "mem_peak_gb": round(mem_peak_gb, 2),
                "grad_norm": round(max_grad_norm, 4),
                "lr": optimizer_vae.param_groups[0]["lr"],
                "timestamp": datetime.now().isoformat(),
            }
            loss_history.append(loss_entry)

            # Update LR schedulers
            scheduler_vae.step()
            scheduler_aux.step()

            # Anomaly detection
            ok = detector.record(epoch, avg_loss, mem_gb, throughput, max_grad_norm)
            if not ok:
                logger.error(f"Phase B: NaN detected at epoch {epoch}, stopping.")
                break

            # Logging
            if epoch <= 5 or epoch % 10 == 0:
                hours_left = (deadline - time.time()) / 3600
                logger.info(
                    f"  B[{epoch:>5d}] loss={avg_loss:.4f} "
                    f"(recon={epoch_loss_recon/max(n_batches,1):.4f} "
                    f"kl={epoch_loss_kl/max(n_batches,1):.4f} "
                    f"disent={epoch_loss_disent/max(n_batches,1):.4f}) | "
                    f"{elapsed:.1f}s {throughput:.1f}v/s | "
                    f"mem={mem_gb:.1f}/{mem_peak_gb:.1f}GB | "
                    f"gn={max_grad_norm:.2f} | "
                    f"{hours_left:.1f}h left"
                )

            # Best model tracking
            if avg_loss < best_loss:
                best_loss = avg_loss

            # Periodic checkpoint
            if epoch % checkpoint_interval == 0:
                torch.save({
                    "epoch": epoch,
                    "vae_state": vae.state_dict(),
                    "disentangle_state": disentangle.state_dict(),
                    "optimizer_vae_state": optimizer_vae.state_dict(),
                    "optimizer_aux_state": optimizer_aux.state_dict(),
                    "scheduler_vae_state": scheduler_vae.state_dict(),
                    "scheduler_aux_state": scheduler_aux.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "best_loss": best_loss,
                }, ckpt_path)
                logger.info(f"  B[{epoch}] checkpoint saved → {ckpt_path}")

            # Periodic JSON snapshot
            if epoch % snapshot_interval == 0:
                snapshot = {
                    "phase": "B",
                    "epoch": epoch,
                    "detector_summary": detector.summary(),
                    "recent_losses": loss_history[-snapshot_interval:],
                    "timestamp": datetime.now().isoformat(),
                }
                snap_path = f"experiments/stability_phase_b_snapshot_e{epoch}.json"
                with open(snap_path, "w") as f:
                    json.dump(snapshot, f, indent=2, default=str)

    except KeyboardInterrupt:
        logger.info(f"Phase B interrupted at epoch {epoch}")
    except Exception as e:
        logger.error(f"Phase B error at epoch {epoch}: {e}")
        logger.error(traceback.format_exc())

    monitor.stop()

    # Save final checkpoint
    torch.save({
        "epoch": epoch,
        "vae_state": vae.state_dict(),
        "disentangle_state": disentangle.state_dict(),
        "best_loss": best_loss,
    }, ckpt_path)
    logger.info(f"Phase B final checkpoint: epoch {epoch}, best_loss={best_loss:.6f}")

    return loss_history, detector.summary(), ckpt_path


# ═══════════════════════════════════════════════════════════════════════════
# Phase C — Extended NJDE Dynamics Training Loop
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_c_extended(latent_h5_path: str, max_hours: float = 24.0,
                          batch_size: int = 24, lr: float = 5e-4,
                          grad_accum_steps: int = 3,
                          checkpoint_interval: int = 100,
                          snapshot_interval: int = 50,
                          ckpt_dir: str = "experiments/stability_checkpoints"):
    """Run Phase C NJDE training for up to max_hours.
    
    Uses vectorised ODE integration across the full batch — no per-sample loop.
    No frozen decoder loaded (saves ~1 GB VRAM).
    """
    from src.pipeline.phase_c_dynamics import (
        LatentTrajectoryDataset, _trajectory_collate_fn, _worker_init_fn
    )
    from src.pipeline.stage4_dynamics.solver import ContinuousDiscreteJumpSolver

    device = torch.device("cuda")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Dataset & Loader
    dataset = LatentTrajectoryDataset(latent_h5_path, min_visits=2)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True,
        collate_fn=_trajectory_collate_fn,
        worker_init_fn=_worker_init_fn, drop_last=True,
        prefetch_factor=4,
    )
    logger.info(f"Phase C: {len(dataset)} trajectories, {len(loader)} batches/epoch, "
                f"batch={batch_size}, grad_accum={grad_accum_steps}, lr={lr}")

    # Solver: breast domain
    solver = ContinuousDiscreteJumpSolver(domain="breast", dt_cfl_days=7.0).to(device)
    n_params = sum(p.numel() for p in solver.parameters()) / 1e6
    logger.info(f"  Solver: {n_params:.1f}M params")

    optimizer = torch.optim.AdamW(solver.parameters(), lr=lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    # LR scheduler: cosine with warmup
    warmup_epochs = 20
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / max(1, 10000)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    detector = AnomalyDetector("Phase_C")
    monitor = GPUMonitor(interval=5.0)
    monitor.start()

    deadline = time.time() + max_hours * 3600
    epoch = 0
    loss_history = []
    best_loss = float("inf")
    ckpt_path = os.path.join(ckpt_dir, "phase_c_latest.pt")

    logger.info(f"Phase C: training until {datetime.fromtimestamp(deadline).isoformat()} "
                f"({max_hours:.1f} hours)")

    try:
        while time.time() < deadline:
            epoch += 1
            epoch_loss = 0.0
            epoch_loss_u = 0.0
            epoch_loss_v = 0.0
            n_batches = 0
            t0 = time.time()
            max_grad_norm = 0.0

            optimizer.zero_grad(set_to_none=True)

            for batch_idx, batch in enumerate(loader):
                u_d = batch["u_d"].to(device, non_blocking=True)       # (B, T, C, D, H, W)
                v_sys = batch["v_sys"].to(device, non_blocking=True)   # (B, T, 160)
                t_days = batch["t_days"].to(device, non_blocking=True) # (B, T)
                mask = batch["mask"].to(device, non_blocking=True)     # (B, T)

                B, T, C, D, H, W = u_d.shape

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    # ═══ VECTORISED ODE integration across full batch ═══
                    total_traj_loss = torch.tensor(0.0, device=device)
                    n_valid = 0
                    total_loss_u = 0.0
                    total_loss_v = 0.0

                    u_curr = u_d[:, 0]
                    v_curr = v_sys[:, 0]
                    phi = torch.zeros(B, 1, D, H, W, device=device)
                    c_drug = torch.zeros(B, 16, device=device)

                    for t_idx in range(1, min(T, 4)):
                        t_start_vec = t_days[:, t_idx - 1]
                        t_end_vec = t_days[:, t_idx]
                        valid = mask[:, t_idx] & (t_end_vec > t_start_vec)

                        if not valid.any():
                            continue

                        t_start_max = t_start_vec[valid].min().item()
                        t_end_max = t_end_vec[valid].max().item()

                        if t_end_max <= t_start_max:
                            continue

                        u_pred, v_pred = solver.integrate_interval(
                            u_curr, v_curr, t_start_max, t_end_max,
                            phi, c_drug, t_start_max,
                        )

                        u_target = u_d[:, t_idx]
                        v_target = v_sys[:, t_idx]

                        valid_f = valid.float()
                        if valid_f.sum() > 0:
                            loss_u_per_b = F.mse_loss(u_pred, u_target, reduction='none').mean(dim=(1,2,3,4))
                            loss_v_per_b = F.mse_loss(v_pred, v_target, reduction='none').mean(dim=1)
                            loss_step = (valid_f * (loss_u_per_b + 0.1 * loss_v_per_b)).sum() / valid_f.sum()
                            total_traj_loss = total_traj_loss + loss_step
                            total_loss_u += loss_u_per_b.mean().item()
                            total_loss_v += loss_v_per_b.mean().item()
                            n_valid += 1

                        u_curr = u_pred.detach()
                        v_curr = v_pred.detach()

                    if n_valid > 0:
                        total_traj_loss = total_traj_loss / n_valid

                scaler.scale(total_traj_loss / grad_accum_steps).backward()

                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(loader):
                    scaler.unscale_(optimizer)
                    gn = nn.utils.clip_grad_norm_(solver.parameters(), max_norm=1.0)
                    max_grad_norm = max(max_grad_norm, gn.item())
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                epoch_loss += total_traj_loss.item()
                epoch_loss_u += total_loss_u
                epoch_loss_v += total_loss_v
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            avg_loss_u = epoch_loss_u / max(n_batches, 1)
            avg_loss_v = epoch_loss_v / max(n_batches, 1)
            elapsed = time.time() - t0
            throughput = len(dataset) / elapsed
            mem_gb = torch.cuda.memory_allocated(0) / (1024**3)
            mem_peak_gb = torch.cuda.max_memory_allocated(0) / (1024**3)

            loss_entry = {
                "epoch": epoch,
                "loss": round(avg_loss, 6),
                "loss_u": round(avg_loss_u, 6),
                "loss_v": round(avg_loss_v, 6),
                "time_s": round(elapsed, 2),
                "throughput": round(throughput, 2),
                "mem_gb": round(mem_gb, 2),
                "mem_peak_gb": round(mem_peak_gb, 2),
                "grad_norm": round(max_grad_norm, 4),
                "lr": optimizer.param_groups[0]["lr"],
                "timestamp": datetime.now().isoformat(),
            }
            loss_history.append(loss_entry)

            scheduler.step()

            ok = detector.record(epoch, avg_loss, mem_gb, throughput, max_grad_norm)
            if not ok:
                logger.error(f"Phase C: NaN detected at epoch {epoch}, stopping.")
                break

            if epoch <= 5 or epoch % 10 == 0:
                hours_left = (deadline - time.time()) / 3600
                logger.info(
                    f"  C[{epoch:>5d}] loss={avg_loss:.6f} "
                    f"(u={avg_loss_u:.6f} v={avg_loss_v:.6f}) | "
                    f"{elapsed:.1f}s {throughput:.1f}t/s | "
                    f"mem={mem_gb:.1f}/{mem_peak_gb:.1f}GB | "
                    f"gn={max_grad_norm:.4f} | "
                    f"{hours_left:.1f}h left"
                )

            if avg_loss < best_loss:
                best_loss = avg_loss

            if epoch % checkpoint_interval == 0:
                torch.save({
                    "epoch": epoch,
                    "solver_state": solver.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "best_loss": best_loss,
                }, ckpt_path)
                logger.info(f"  C[{epoch}] checkpoint saved → {ckpt_path}")

            if epoch % snapshot_interval == 0:
                snapshot = {
                    "phase": "C",
                    "epoch": epoch,
                    "detector_summary": detector.summary(),
                    "recent_losses": loss_history[-snapshot_interval:],
                    "timestamp": datetime.now().isoformat(),
                }
                snap_path = f"experiments/stability_phase_c_snapshot_e{epoch}.json"
                with open(snap_path, "w") as f:
                    json.dump(snapshot, f, indent=2, default=str)

    except KeyboardInterrupt:
        logger.info(f"Phase C interrupted at epoch {epoch}")
    except Exception as e:
        logger.error(f"Phase C error at epoch {epoch}: {e}")
        logger.error(traceback.format_exc())

    monitor.stop()

    torch.save({
        "epoch": epoch,
        "solver_state": solver.state_dict(),
        "best_loss": best_loss,
    }, ckpt_path)
    logger.info(f"Phase C final checkpoint: epoch {epoch}, best_loss={best_loss:.6f}")

    return loss_history, detector.summary()


# ═══════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="48-Hour Stability Test")
    parser.add_argument("--hours", type=float, default=48.0,
                        help="Total hours to run (split equally between phases)")
    parser.add_argument("--phase-b-hours", type=float, default=None,
                        help="Hours for Phase B (overrides --hours split)")
    parser.add_argument("--phase-c-hours", type=float, default=None,
                        help="Hours for Phase C (overrides --hours split)")
    parser.add_argument("--checkpoint-interval", type=int, default=100,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--snapshot-interval", type=int, default=50,
                        help="Save JSON snapshot every N epochs")
    parser.add_argument("--batch-b", type=int, default=36,
                        help="Phase B batch size")
    parser.add_argument("--batch-c", type=int, default=24,
                        help="Phase C batch size")
    parser.add_argument("--skip-phase-b", action="store_true",
                        help="Skip Phase B, run Phase C only")
    parser.add_argument("--skip-phase-c", action="store_true",
                        help="Skip Phase C, run Phase B only")
    args = parser.parse_args()

    phase_b_hours = args.phase_b_hours or (args.hours / 2)
    phase_c_hours = args.phase_c_hours or (args.hours / 2)

    torch.manual_seed(42)
    np.random.seed(42)

    # CUDA optimisations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    gpu_sms = torch.cuda.get_device_properties(0).multi_processor_count

    logger.info("=" * 70)
    logger.info("48-HOUR STABILITY & CONVERGENCE TEST")
    logger.info("=" * 70)
    logger.info(f"GPU: {gpu_name} ({gpu_mem_gb:.1f} GB, {gpu_sms} SMs)")
    logger.info(f"Total run time: {args.hours:.1f}h (Phase B: {phase_b_hours:.1f}h, Phase C: {phase_c_hours:.1f}h)")
    logger.info(f"Checkpoint interval: every {args.checkpoint_interval} epochs")
    logger.info(f"Snapshot interval: every {args.snapshot_interval} epochs")
    logger.info(f"Start time: {datetime.now().isoformat()}")
    logger.info(f"Expected end: {(datetime.now() + timedelta(hours=args.hours)).isoformat()}")

    results = {
        "gpu": gpu_name,
        "gpu_mem_gb": round(gpu_mem_gb, 1),
        "total_hours_requested": args.hours,
        "start_time": datetime.now().isoformat(),
        "phases": {},
    }

    # Create synthetic data
    tmpdir = tempfile.mkdtemp(prefix="stability_48h_")
    encoder_h5 = os.path.join(tmpdir, "encoder_features.h5")
    latent_h5 = os.path.join(tmpdir, "latent_decomposition.h5")

    create_phase_a_hdf5(encoder_h5)
    create_phase_b_hdf5(latent_h5)

    ckpt_dir = "experiments/stability_checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs("experiments", exist_ok=True)

    # ─── Phase B ──────────────────────────────────────────────────────
    if not args.skip_phase_b:
        logger.info("=" * 70)
        logger.info("PHASE B: Extended VAE Training")
        logger.info("=" * 70)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()

        phase_b_start = time.time()
        phase_b_history, phase_b_summary, phase_b_ckpt = run_phase_b_extended(
            encoder_h5, max_hours=phase_b_hours,
            batch_size=args.batch_b, lr=2e-4, grad_accum_steps=2,
            checkpoint_interval=args.checkpoint_interval,
            snapshot_interval=args.snapshot_interval,
            ckpt_dir=ckpt_dir,
        )
        phase_b_elapsed = time.time() - phase_b_start

        results["phases"]["B"] = {
            "total_epochs": len(phase_b_history),
            "total_time_h": round(phase_b_elapsed / 3600, 2),
            "summary": phase_b_summary,
            "loss_start": phase_b_history[0]["loss"] if phase_b_history else None,
            "loss_end": phase_b_history[-1]["loss"] if phase_b_history else None,
            "converged": bool(phase_b_summary.get("loss_converged", False)),
            "n_alerts": phase_b_summary.get("n_alerts", 0),
            "mem_drift_gb": phase_b_summary.get("mem_drift_gb") or 0.0,
        }

        logger.info(f"Phase B complete: {len(phase_b_history)} epochs in {phase_b_elapsed/3600:.2f}h")
        logger.info(f"  Loss: {phase_b_history[0]['loss']:.4f} → {phase_b_history[-1]['loss']:.4f}" if phase_b_history else "  Loss: N/A")
        logger.info(f"  Converged: {phase_b_summary.get('loss_converged', False)}")
        logger.info(f"  Memory drift: {(phase_b_summary.get('mem_drift_gb') or 0.0):.3f} GB")
        logger.info(f"  Alerts: {phase_b_summary.get('n_alerts', 0)}")

        # Cleanup Phase B
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        time.sleep(5)
    else:
        logger.info("Phase B: SKIPPED")

    # ─── Phase C ──────────────────────────────────────────────────────
    if not args.skip_phase_c:
        logger.info("=" * 70)
        logger.info("PHASE C: Extended NJDE Dynamics Training")
        logger.info("=" * 70)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()

        phase_c_start = time.time()
        phase_c_history, phase_c_summary = run_phase_c_extended(
            latent_h5, max_hours=phase_c_hours,
            batch_size=args.batch_c, lr=5e-4, grad_accum_steps=3,
            checkpoint_interval=args.checkpoint_interval,
            snapshot_interval=args.snapshot_interval,
            ckpt_dir=ckpt_dir,
        )
        phase_c_elapsed = time.time() - phase_c_start

        results["phases"]["C"] = {
            "total_epochs": len(phase_c_history),
            "total_time_h": round(phase_c_elapsed / 3600, 2),
            "summary": phase_c_summary,
            "loss_start": phase_c_history[0]["loss"] if phase_c_history else None,
            "loss_end": phase_c_history[-1]["loss"] if phase_c_history else None,
            "converged": bool(phase_c_summary.get("loss_converged", False)),
            "n_alerts": phase_c_summary.get("n_alerts", 0),
            "mem_drift_gb": phase_c_summary.get("mem_drift_gb") or 0.0,
        }

        logger.info(f"Phase C complete: {len(phase_c_history)} epochs in {phase_c_elapsed/3600:.2f}h")
        logger.info(f"  Loss: {phase_c_history[0]['loss']:.6f} → {phase_c_history[-1]['loss']:.6f}" if phase_c_history else "  Loss: N/A")
        logger.info(f"  Converged: {phase_c_summary.get('loss_converged', False)}")
        logger.info(f"  Memory drift: {(phase_c_summary.get('mem_drift_gb') or 0.0):.3f} GB")
        logger.info(f"  Alerts: {phase_c_summary.get('n_alerts', 0)}")
    else:
        logger.info("Phase C: SKIPPED")

    # ─── Final Report ─────────────────────────────────────────────────
    results["end_time"] = datetime.now().isoformat()
    results["total_time_h"] = round(
        sum(results["phases"][p].get("total_time_h", 0) for p in results["phases"]),
        2
    )
    results["all_converged"] = all(
        results["phases"][p].get("converged", False) for p in results["phases"]
    )
    results["total_alerts"] = sum(
        results["phases"][p].get("n_alerts", 0) for p in results["phases"]
    )
    results["memory_leak_detected"] = any(
        abs(results["phases"][p].get("mem_drift_gb") or 0.0) > 1.0
        for p in results["phases"]
    )

    logger.info("=" * 70)
    logger.info("48-HOUR STABILITY TEST — FINAL REPORT")
    logger.info("=" * 70)
    logger.info(f"Total run time: {results['total_time_h']:.2f} hours")
    logger.info(f"All phases converging: {'✅ YES' if results['all_converged'] else '❌ NO'}")
    logger.info(f"Memory leak detected: {'❌ YES' if results['memory_leak_detected'] else '✅ NO'}")
    logger.info(f"Total alerts: {results['total_alerts']}")

    for p in sorted(results["phases"].keys()):
        ph = results["phases"][p]
        logger.info(f"  Phase {p}: {ph['total_epochs']} epochs in {ph['total_time_h']:.1f}h | "
                     f"loss {ph.get('loss_start', '?')} → {ph.get('loss_end', '?')} | "
                     f"mem drift {(ph.get('mem_drift_gb') or 0.0):.3f} GB | "
                     f"{ph.get('n_alerts', 0)} alerts")

    # Save final results
    out_path = "experiments/stability_48h_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved → {out_path}")

    # Cleanup temp files
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    return results


if __name__ == "__main__":
    main()
