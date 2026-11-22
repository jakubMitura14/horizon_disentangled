#!/usr/bin/env python
"""Local 3-Phase Staged Pipeline Smoke Test & GPU Benchmark.

Runs all 3 phases end-to-end with synthetic data on a single GPU,
measuring GPU utilization, memory usage, throughput, and loss convergence.

Usage:
    cd /DATA/horizon_disentangled
    LD_LIBRARY_PATH=/DATA/conda/envs/nukl-to-ct/lib:$LD_LIBRARY_PATH \
        /DATA/conda/envs/nukl-to-ct/bin/python scripts/training/local_phased_benchmark.py
"""
import os
import sys
import json
import time
import tempfile
import threading
import logging

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/DATA/horizon_disentangled")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("phased_benchmark")

# ═══════════════════════════════════════════════════════════════════════════
# GPU Monitor Thread
# ═══════════════════════════════════════════════════════════════════════════

class GPUMonitor:
    """Background thread that samples GPU utilization every second."""

    def __init__(self, device_idx: int = 0):
        self.device_idx = device_idx
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _run(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_idx)
            while not self._stop.is_set():
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.samples.append({
                    "gpu_util_pct": util.gpu,
                    "mem_used_gb": mem.used / (1024**3),
                    "mem_total_gb": mem.total / (1024**3),
                    "timestamp": time.time(),
                })
                self._stop.wait(1.0)
        except ImportError:
            # Fallback: use torch.cuda
            while not self._stop.is_set():
                mem_alloc = torch.cuda.memory_allocated(self.device_idx) / (1024**3)
                mem_res = torch.cuda.memory_reserved(self.device_idx) / (1024**3)
                self.samples.append({
                    "gpu_util_pct": -1,  # unavailable without pynvml
                    "mem_used_gb": mem_alloc,
                    "mem_reserved_gb": mem_res,
                    "timestamp": time.time(),
                })
                self._stop.wait(1.0)

    def summary(self, label: str = ""):
        if not self.samples:
            return {}
        utils = [s["gpu_util_pct"] for s in self.samples if s["gpu_util_pct"] >= 0]
        mems = [s["mem_used_gb"] for s in self.samples]
        result = {
            "label": label,
            "n_samples": len(self.samples),
            "mem_used_gb_mean": float(np.mean(mems)),
            "mem_used_gb_peak": float(np.max(mems)),
        }
        if utils:
            result["gpu_util_pct_mean"] = float(np.mean(utils))
            result["gpu_util_pct_peak"] = float(np.max(utils))
            result["gpu_util_pct_min"] = float(np.min(utils))
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic Data Generation
# ═══════════════════════════════════════════════════════════════════════════

def create_phase_a_hdf5(path: str, n_patients: int = 20, n_visits: int = 4,
                         z_vae_shape=(16, 16, 16, 8)):
    """Create synthetic Phase A encoder features (heavy dims)."""
    logger.info(f"Creating synthetic Phase A HDF5: {n_patients} patients × {n_visits} visits")
    logger.info(f"  z_vae shape per visit: {z_vae_shape} (C_in*4, D/4, H/4, W/4)")
    with h5py.File(path, "w") as h5:
        h5.attrs["encoder_visual"] = "suprem"
        h5.attrs["encoder_vae"] = "medvae_4_1_3d"
        h5.attrs["domain"] = "breast"
        for pid in range(n_patients):
            uid = f"patient_{pid:04d}"
            p_grp = h5.create_group(f"patients/{uid}")
            p_grp.attrs["pcr"] = int(np.random.randint(0, 2))
            p_grp.attrs["event"] = int(np.random.randint(0, 2))
            p_grp.attrs["os_weeks"] = float(np.random.uniform(10, 200))
            p_grp.create_dataset("clinical",
                                 data=np.random.randn(16).astype(np.float32))
            for v in range(n_visits):
                vk = f"visit_{v}"
                grp = p_grp.create_group(f"visits/{vk}")
                grp.create_dataset("h_visual",
                                   data=np.random.randn(768).astype(np.float16))
                grp.create_dataset("z_vae_per_channel",
                                   data=np.random.randn(*z_vae_shape).astype(np.float16))
                # I-SPY 2 schedule: T0=0, T1=21, T2=84, T3=168 days
                t_days = [0.0, 21.0, 84.0, 168.0][v] if v < 4 else float(v * 42)
                grp.attrs["t_days"] = t_days
                grp.attrs["domain"] = "breast"
    size_mb = os.path.getsize(path) / (1024 * 1024)
    logger.info(f"  Phase A HDF5: {size_mb:.1f} MB")


def create_phase_b_hdf5(path: str, n_patients: int = 20, n_visits: int = 4,
                         u_d_shape=(8, 8, 8, 4)):
    """Create synthetic Phase B latent decomposition."""
    logger.info(f"Creating synthetic Phase B HDF5: {n_patients} patients × {n_visits} visits")
    with h5py.File(path, "w") as h5:
        for pid in range(n_patients):
            uid = f"patient_{pid:04d}"
            for v in range(n_visits):
                vk = f"visit_{v}"
                grp = h5.create_group(f"patients/{uid}/visits/{vk}")
                grp.create_dataset("u_d", data=np.random.randn(*u_d_shape).astype(np.float16))
                grp.create_dataset("z_a", data=np.random.randn(4, 16, 16, 8).astype(np.float16))
                grp.create_dataset("z_d_glob", data=np.random.randn(64).astype(np.float32))
                grp.create_dataset("z_p", data=np.random.randn(32).astype(np.float32))
                grp.create_dataset("z_s", data=np.random.randn(32).astype(np.float32))
                grp.create_dataset("mu_d", data=np.random.randn(64).astype(np.float32))
                grp.create_dataset("logvar_d", data=np.random.randn(64).astype(np.float32) - 2)
                grp.create_dataset("mu_p", data=np.random.randn(32).astype(np.float32))
                grp.create_dataset("logvar_p", data=np.random.randn(32).astype(np.float32) - 2)
                grp.create_dataset("mu_s", data=np.random.randn(32).astype(np.float32))
                grp.create_dataset("logvar_s", data=np.random.randn(32).astype(np.float32) - 2)
                grp.create_dataset("v_sys_init", data=np.random.randn(160).astype(np.float32))
                t_days = [0.0, 21.0, 84.0, 168.0][v] if v < 4 else float(v * 42)
                grp.attrs["t_days"] = t_days
    size_mb = os.path.getsize(path) / (1024 * 1024)
    logger.info(f"  Phase B HDF5: {size_mb:.1f} MB")


# ═══════════════════════════════════════════════════════════════════════════
# Phase B — Training Loop (Local, Single GPU)
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_b_local(encoder_h5_path: str, epochs: int = 10,
                       batch_size: int = 8, lr: float = 2e-4):
    """Train Phase B VAE Level 2 on cached features, return loss history."""
    from src.pipeline.phase_b_latents import EncoderFeatureDataset, _worker_init_fn
    from src.pipeline.stage2_causal_vae.architecture import TwoScaleHierarchicalCausalVAE
    from src.pipeline.stage2_causal_vae.disentanglement import AsymmetricCausalDisentanglement
    from torch.utils.data import DataLoader

    device = torch.device("cuda")
    logger.info(f"Phase B: epochs={epochs}, batch_size={batch_size}, lr={lr}")

    dataset = EncoderFeatureDataset(encoder_h5_path, domain="breast")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
        worker_init_fn=_worker_init_fn, drop_last=True,
        prefetch_factor=4,
    )
    logger.info(f"  Dataset: {len(dataset)} volumes, {len(loader)} batches/epoch")

    # Heavy config: larger L2 channels
    vae = TwoScaleHierarchicalCausalVAE(
        in_channels=4,
        anatomy_channels=4,
        disease_channels=8,
        global_disease_dim=64,
        patient_dim=32,
        style_dim=32,
        l2_channels=256,
        backbone='monai',
    ).to(device)

    disentangle = AsymmetricCausalDisentanglement(
        disease_dim=64, patient_dim=32, style_dim=32,
        anatomy_channels=4, beta_tc=1.5, gamma_club=1.0, alpha_dp=0.5,
    ).to(device)

    n_params_vae = sum(p.numel() for p in vae.parameters()) / 1e6
    n_params_dis = sum(p.numel() for p in disentangle.parameters()) / 1e6
    logger.info(f"  VAE: {n_params_vae:.1f}M params, Disentangle: {n_params_dis:.1f}M params")

    optimizer_vae = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=1e-4)
    optimizer_aux = torch.optim.AdamW(disentangle.parameters(), lr=lr * 0.5, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in loader:
            z_vae = batch["z_vae"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                enc = vae.encode_from_cached(z_vae)
                recon, mask_pred = vae.decode(enc["z_a"], enc["u_d"], enc["z_s"])

                # Recon loss: compare decoded output against anatomy z_a
                # (decoder outputs in_channels=4, same as z_a)
                z_a_target = enc["z_a"].detach()
                if recon.shape[2:] != z_a_target.shape[2:]:
                    recon = F.interpolate(recon, size=z_a_target.shape[2:],
                                          mode="trilinear", align_corners=False)
                loss_recon = F.mse_loss(recon, z_a_target)
                kl_d = -0.5 * torch.mean(1 + enc["logvar_d"] - enc["mu_d"].pow(2) - enc["logvar_d"].exp())
                kl_p = -0.5 * torch.mean(1 + enc["logvar_p"] - enc["mu_p"].pow(2) - enc["logvar_p"].exp())
                kl_s = -0.5 * torch.mean(1 + enc["logvar_s"] - enc["mu_s"].pow(2) - enc["logvar_s"].exp())
                loss_kl = kl_d + kl_p + kl_s

                loss_dis, loss_aux, _ = disentangle.compute_disentangle_loss(
                    enc["z_d_glob"], enc["z_p"], enc["z_s"], enc["z_a"]
                )
                total_loss = loss_recon + 1e-3 * loss_kl + loss_dis

            scaler.scale(total_loss).backward(retain_graph=True)
            scaler.scale(loss_aux).backward()

            scaler.unscale_(optimizer_vae)
            scaler.unscale_(optimizer_aux)
            nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            nn.utils.clip_grad_norm_(disentangle.parameters(), max_norm=1.0)
            scaler.step(optimizer_vae)
            scaler.step(optimizer_aux)
            scaler.update()
            optimizer_vae.zero_grad(set_to_none=True)
            optimizer_aux.zero_grad(set_to_none=True)

            epoch_loss += total_loss.item()
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        throughput = len(dataset) / elapsed
        loss_history.append({"epoch": epoch + 1, "loss": avg, "time_s": elapsed,
                             "throughput_vol_per_s": throughput})
        logger.info(f"  Epoch {epoch+1}/{epochs}: loss={avg:.4f} | "
                    f"{elapsed:.1f}s | {throughput:.1f} vol/s")

    # Save checkpoint
    ckpt_path = "/tmp/phase_b_vae_benchmark.pt"
    torch.save({
        "vae_state": vae.state_dict(),
        "disentangle_state": disentangle.state_dict(),
    }, ckpt_path)
    logger.info(f"  Phase B checkpoint: {ckpt_path}")
    return loss_history, ckpt_path


# ═══════════════════════════════════════════════════════════════════════════
# Phase C — Training Loop (Local, Single GPU)
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_c_local(latent_h5_path: str, decoder_ckpt: str, epochs: int = 10,
                       batch_size: int = 4, lr: float = 1e-3):
    """Train Phase C NJDE on cached latent trajectories."""
    from src.pipeline.phase_c_dynamics import (
        LatentTrajectoryDataset, _trajectory_collate_fn, _worker_init_fn
    )
    from src.pipeline.stage4_dynamics.solver import ContinuousDiscreteJumpSolver
    from torch.utils.data import DataLoader

    device = torch.device("cuda")
    logger.info(f"Phase C: epochs={epochs}, batch_size={batch_size}, lr={lr}")

    dataset = LatentTrajectoryDataset(latent_h5_path, min_visits=2)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, persistent_workers=True,
        collate_fn=_trajectory_collate_fn,
        worker_init_fn=_worker_init_fn, drop_last=True,
    )
    logger.info(f"  Dataset: {len(dataset)} trajectories, {len(loader)} batches/epoch")

    solver = ContinuousDiscreteJumpSolver(domain="breast", dt_cfl_days=7.0).to(device)
    n_params = sum(p.numel() for p in solver.parameters()) / 1e6
    logger.info(f"  Solver: {n_params:.1f}M params")

    # Try loading frozen decoder for plausibility loss
    decoder = None
    if os.path.exists(decoder_ckpt):
        from src.pipeline.stage2_causal_vae.architecture import TwoScaleHierarchicalCausalVAE
        ckpt = torch.load(decoder_ckpt, map_location=device, weights_only=False)
        decoder = TwoScaleHierarchicalCausalVAE(in_channels=4, disease_channels=8).to(device)
        decoder.load_state_dict(ckpt["vae_state"], strict=False)
        decoder.eval()
        for p in decoder.parameters():
            p.requires_grad = False
        logger.info(f"  Frozen decoder loaded from {decoder_ckpt}")

    optimizer = torch.optim.AdamW(solver.parameters(), lr=lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    loss_history = []
    for epoch in range(epochs):
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
                total_traj_loss = torch.tensor(0.0, device=device)
                n_valid = 0

                for b in range(B):
                    n_vis = mask[b].sum().int().item()
                    if n_vis < 2:
                        continue

                    u_curr = u_d[b, 0:1]
                    v_curr = v_sys[b, 0:1]
                    phi = torch.zeros_like(u_curr[:, 0:1])
                    c_drug = torch.zeros(1, 16, device=device)

                    for t_idx in range(1, min(n_vis, 4)):  # Cap at 4 timepoints
                        t_start = t_days[b, t_idx - 1].item()
                        t_end = t_days[b, t_idx].item()
                        if t_end <= t_start:
                            continue

                        u_pred, v_pred = solver.integrate_interval(
                            u_curr, v_curr, t_start, t_end,
                            phi, c_drug, t_start,
                        )

                        u_target = u_d[b, t_idx : t_idx + 1]
                        v_target = v_sys[b, t_idx : t_idx + 1]

                        loss_u = F.mse_loss(u_pred, u_target)
                        loss_v = F.mse_loss(v_pred, v_target)
                        total_traj_loss = total_traj_loss + loss_u + 0.1 * loss_v

                        u_curr = u_pred.detach()
                        v_curr = v_pred.detach()
                        n_valid += 1

                if n_valid > 0:
                    total_traj_loss = total_traj_loss / n_valid

            scaler.scale(total_traj_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(solver.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss += total_traj_loss.item()
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        throughput = len(dataset) / elapsed
        loss_history.append({"epoch": epoch + 1, "loss": avg, "time_s": elapsed,
                             "throughput_traj_per_s": throughput})
        logger.info(f"  Epoch {epoch+1}/{epochs}: loss={avg:.6f} | "
                    f"{elapsed:.1f}s | {throughput:.1f} traj/s")

    return loss_history


# ═══════════════════════════════════════════════════════════════════════════
# Main Benchmark
# ═══════════════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info(f"GPU: {gpu_name} ({gpu_mem_gb:.1f} GB)")

    results = {
        "gpu": gpu_name,
        "gpu_mem_gb": round(gpu_mem_gb, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "phases": {},
    }

    tmpdir = tempfile.mkdtemp(prefix="phased_bench_")
    logger.info(f"Temp dir: {tmpdir}")

    # ─── Phase A: create synthetic data (simulates frozen encoder output) ──
    # We skip actual encoder forward pass (needs real images)
    # Instead create heavy synthetic features matching Phase A output format
    encoder_h5 = os.path.join(tmpdir, "encoder_features.h5")
    latent_h5 = os.path.join(tmpdir, "latent_decomposition.h5")

    N_PATIENTS = 40
    N_VISITS = 4

    # Heavy z_vae: 4 channels × 4 latent_ch = 16 channels, spatial 32×32×16
    create_phase_a_hdf5(encoder_h5, n_patients=N_PATIENTS, n_visits=N_VISITS,
                         z_vae_shape=(16, 32, 32, 16))
    create_phase_b_hdf5(latent_h5, n_patients=N_PATIENTS, n_visits=N_VISITS,
                         u_d_shape=(8, 16, 16, 8))

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # ─── Phase B: VAE Level 2 Training ────────────────────────────────────
    logger.info("=" * 70)
    logger.info("PHASE B: VAE Level 2 Disentanglement Training")
    logger.info("=" * 70)

    monitor_b = GPUMonitor()
    monitor_b.start()
    mem_before_b = torch.cuda.memory_allocated() / (1024**3)

    phase_b_history, decoder_ckpt = run_phase_b_local(
        encoder_h5, epochs=10, batch_size=16, lr=2e-4
    )

    monitor_b.stop()
    mem_peak_b = torch.cuda.max_memory_allocated() / (1024**3)
    gpu_summary_b = monitor_b.summary("Phase B")

    results["phases"]["B"] = {
        "loss_history": phase_b_history,
        "loss_start": phase_b_history[0]["loss"],
        "loss_end": phase_b_history[-1]["loss"],
        "loss_dropped": phase_b_history[-1]["loss"] < phase_b_history[0]["loss"],
        "loss_drop_pct": round(100 * (1 - phase_b_history[-1]["loss"] / max(phase_b_history[0]["loss"], 1e-8)), 1),
        "mem_peak_gb": round(mem_peak_b, 2),
        "gpu_summary": gpu_summary_b,
        "total_time_s": sum(h["time_s"] for h in phase_b_history),
    }

    logger.info(f"Phase B Summary:")
    logger.info(f"  Loss: {phase_b_history[0]['loss']:.4f} → {phase_b_history[-1]['loss']:.4f} "
                f"({'↓ CONVERGING' if results['phases']['B']['loss_dropped'] else '→ FLAT/DIVERGING'})")
    logger.info(f"  GPU Memory Peak: {mem_peak_b:.2f} GB")
    logger.info(f"  GPU Utilization: {gpu_summary_b}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # ─── Phase C: NJDE Dynamics Training ──────────────────────────────────
    logger.info("=" * 70)
    logger.info("PHASE C: NJDE Longitudinal Dynamics Training")
    logger.info("=" * 70)

    monitor_c = GPUMonitor()
    monitor_c.start()

    phase_c_history = run_phase_c_local(
        latent_h5, decoder_ckpt, epochs=10, batch_size=8, lr=5e-4
    )

    monitor_c.stop()
    mem_peak_c = torch.cuda.max_memory_allocated() / (1024**3)
    gpu_summary_c = monitor_c.summary("Phase C")

    results["phases"]["C"] = {
        "loss_history": phase_c_history,
        "loss_start": phase_c_history[0]["loss"],
        "loss_end": phase_c_history[-1]["loss"],
        "loss_dropped": phase_c_history[-1]["loss"] < phase_c_history[0]["loss"],
        "loss_drop_pct": round(100 * (1 - phase_c_history[-1]["loss"] / max(phase_c_history[0]["loss"], 1e-8)), 1),
        "mem_peak_gb": round(mem_peak_c, 2),
        "gpu_summary": gpu_summary_c,
        "total_time_s": sum(h["time_s"] for h in phase_c_history),
    }

    logger.info(f"Phase C Summary:")
    logger.info(f"  Loss: {phase_c_history[0]['loss']:.6f} → {phase_c_history[-1]['loss']:.6f} "
                f"({'↓ CONVERGING' if results['phases']['C']['loss_dropped'] else '→ FLAT/DIVERGING'})")
    logger.info(f"  GPU Memory Peak: {mem_peak_c:.2f} GB")

    # ─── Final Report ─────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("FINAL BENCHMARK RESULTS")
    logger.info("=" * 70)

    all_converging = all(
        results["phases"][p]["loss_dropped"]
        for p in ["B", "C"]
    )
    results["all_losses_converging"] = all_converging
    results["total_time_s"] = sum(
        results["phases"][p]["total_time_s"] for p in ["B", "C"]
    )

    logger.info(f"All phases converging: {'✅ YES' if all_converging else '❌ NO'}")
    for p in ["B", "C"]:
        ph = results["phases"][p]
        logger.info(f"  Phase {p}: loss {ph['loss_start']:.4f} → {ph['loss_end']:.4f} "
                    f"({ph['loss_drop_pct']:+.1f}%) | peak {ph['mem_peak_gb']:.1f} GB | "
                    f"{ph['total_time_s']:.1f}s")
    logger.info(f"Total benchmark time: {results['total_time_s']:.1f}s")

    # Save results
    out_path = "experiments/phased_benchmark_results.json"
    os.makedirs("experiments", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved → {out_path}")

    # Cleanup temp files (keep checkpoint)
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    return results


if __name__ == "__main__":
    main()
