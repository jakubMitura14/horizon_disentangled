#!/usr/bin/env python3
"""Single-Epoch Profiling Utility for Horizon Disentangled (Phases A, B, and C).

Executes exactly 1 epoch across pipeline phases under PyTorch Profiler or Nsight Systems,
capturing:
  - CPU and CUDA operator execution times & top kernels
  - Peak and reserved GPU memory allocations (VRAM)
  - Forward + backward throughput (volumes/s or trajectories/s)
  - Chrome trace export (.json.gz) viewable in chrome://tracing or Perfetto (ui.perfetto.dev)
  - JSON summary with hardware telemetry

Usage:
  # PyTorch Profiler mode:
  python scripts/training/profile_1epoch_all_phases.py --output-dir profiling_results

  # Minimal / telemetry mode (fastest):
  python scripts/training/profile_1epoch_all_phases.py --profile-type minimal

  # Nsight Systems CLI profiling:
  nsys profile -t cuda,nvtx,osrt,cudnn,cublas -s none -o profile_all_phases \
      python scripts/training/profile_1epoch_all_phases.py --profile-type nsys
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRAINING_DIR = Path(__file__).resolve().parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from local_phased_benchmark import (
    create_phase_a_hdf5,
    create_phase_b_hdf5,
)
from src.pipeline.stage2_causal_vae.architecture import TwoScaleHierarchicalCausalVAE
from src.pipeline.stage2_causal_vae.disentanglement import AsymmetricCausalDisentanglement
from src.pipeline.phase_b_latents import EncoderFeatureDataset, _worker_init_fn
from src.pipeline.stage4_dynamics.solver import ContinuousDiscreteJumpSolver
from src.pipeline.phase_c_dynamics import LatentTrajectoryDataset, _trajectory_collate_fn
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("profile_1epoch")


def parse_args():
    parser = argparse.ArgumentParser(description="Profile 1 Epoch Across All Phases")
    parser.add_argument("--data-dir", type=str, default="/DATA/horizon_disentangled/hdf5",
                        help="Path to HDF5 data directory (falls back to synthetic if absent)")
    parser.add_argument("--output-dir", type=str, default="profiling_results",
                        help="Directory to save profiling traces and reports")
    parser.add_argument("--batch-b", type=int, default=32,
                        help="Phase B batch size (default: 32)")
    parser.add_argument("--batch-c", type=int, default=24,
                        help="Phase C batch size (default: 24)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader worker processes (default: 4)")
    parser.add_argument("--profile-type", type=str, choices=["torch", "nsys", "minimal"], default="torch",
                        help="Profiling mode: 'torch' (PyTorch Profiler), 'nsys' (Nsight hooks), 'minimal' (timers only)")
    parser.add_argument("--phases", type=str, default="A,B,C",
                        help="Comma-separated list of phases to profile (default: A,B,C)")
    parser.add_argument("--max-steps", type=int, default=10,
                        help="Max steps per phase for 1-epoch profiling run (default: 10)")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Phase A: Vision Feature Extraction (3D Convolutions)
# ═══════════════════════════════════════════════════════════════════════════

def profile_phase_a(output_dir: str, n_samples: int = 32, profile_type: str = "torch"):
    logger.info("=" * 70)
    logger.info("PROFILING PHASE A: Vision Feature Extraction (1 Epoch)")
    logger.info("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Front-end 3D Conv encoder modeling multi-scale medical feature extraction
    encoder = nn.Sequential(
        nn.Conv3d(3, 32, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm3d(32),
        nn.SiLU(),
        nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm3d(64),
        nn.SiLU(),
        nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(128),
        nn.SiLU(),
    ).to(device)
    encoder.eval()

    # Synthetic 3D raw DCE-MRI volumes: (B=4, C=3, D=64, H=64, W=32)
    loader = [torch.randn(4, 3, 64, 64, 32, device=device) for _ in range(n_samples // 4)]

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    def run_steps(prof=None):
        with torch.no_grad():
            for step, x in enumerate(loader):
                if profile_type == "nsys":
                    torch.cuda.nvtx.range_push(f"PhaseA_Step_{step}")
                _ = encoder(x)
                if profile_type == "nsys":
                    torch.cuda.nvtx.range_pop()
                if prof:
                    prof.step()
                torch.cuda.synchronize()

    if profile_type == "torch":
        trace_path = os.path.join(output_dir, "trace_phase_a")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=4, repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            run_steps(prof)
        prof.export_chrome_trace(f"{trace_path}.json.gz")
        logger.info(f"Phase A Chrome trace exported -> {trace_path}.json.gz")
        print("\n--- Phase A Top CUDA Kernels by Time ---")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    else:
        run_steps()

    elapsed = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    throughput = n_samples / max(elapsed, 1e-4)

    logger.info(f"Phase A Complete: {n_samples} volumes in {elapsed:.2f}s ({throughput:.1f} vol/s) | Peak VRAM: {peak_vram:.2f} GB")
    return {"elapsed_s": elapsed, "peak_vram_gb": peak_vram, "throughput_vol_s": throughput}


# ═══════════════════════════════════════════════════════════════════════════
# Phase B: 198.9M Hierarchical Causal VAE Disentanglement
# ═══════════════════════════════════════════════════════════════════════════

def profile_phase_b(encoder_h5: str, batch_size: int = 32, num_workers: int = 4,
                    max_steps: int = 10, output_dir: str = "profiling_results",
                    profile_type: str = "torch"):
    logger.info("=" * 70)
    logger.info(f"PROFILING PHASE B: Hierarchical Causal VAE (199M Params, Batch={batch_size})")
    logger.info("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = TwoScaleHierarchicalCausalVAE(
        in_channels=4,
        anatomy_channels=4,
        disease_channels=8,
        global_disease_dim=128,
        patient_dim=64,
        style_dim=64,
        l2_channels=512,
        vae_channels=(128, 256, 512),
        vae_num_res_blocks=3,
        backbone='monai',
    ).to(device)

    disentangle = AsymmetricCausalDisentanglement(
        disease_dim=128, patient_dim=64, style_dim=64,
        anatomy_channels=4, beta_tc=1.5, gamma_club=1.0, alpha_dp=0.5,
    ).to(device)

    optimizer_vae = torch.optim.AdamW(vae.parameters(), lr=2e-4)
    optimizer_aux = torch.optim.AdamW(disentangle.parameters(), lr=2e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    dataset = EncoderFeatureDataset(encoder_h5, domain="breast")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )
    logger.info(f"Phase B Dataset: {len(dataset)} items, {len(loader)} batches total (profiling up to {max_steps} steps)")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    total_processed = 0

    def run_steps(prof=None):
        nonlocal total_processed
        for step, batch in enumerate(loader):
            if step >= max_steps:
                break
            if profile_type == "nsys":
                torch.cuda.nvtx.range_push(f"PhaseB_Step_{step}")

            z_vae = batch["z_vae"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                enc = vae.encode_from_cached(z_vae)
                recon, _ = vae.decode(enc["z_a"], enc["u_d"], enc["z_s"])
                z_a_target = enc["z_a"].detach()
                if recon.shape[2:] != z_a_target.shape[2:]:
                    recon = F.interpolate(recon, size=z_a_target.shape[2:], mode="trilinear", align_corners=False)
                loss_recon = F.mse_loss(recon, z_a_target)
                kl_d = -0.5 * torch.mean(1 + enc["logvar_d"] - enc["mu_d"].pow(2) - enc["logvar_d"].exp())
                kl_p = -0.5 * torch.mean(1 + enc["logvar_p"] - enc["mu_p"].pow(2) - enc["logvar_p"].exp())
                kl_s = -0.5 * torch.mean(1 + enc["logvar_s"] - enc["mu_s"].pow(2) - enc["logvar_s"].exp())
                loss_kl = kl_d + kl_p + kl_s
                loss_dis, loss_aux, _ = disentangle.compute_disentangle_loss(
                    enc["z_d_glob"], enc["z_p"], enc["z_s"], enc["z_a"]
                )
                total_loss = loss_recon + 1e-3 * loss_kl + loss_dis

            combined_loss = total_loss + loss_aux
            scaler.scale(combined_loss).backward()
            scaler.unscale_(optimizer_vae)
            scaler.unscale_(optimizer_aux)
            nn.utils.clip_grad_norm_((p for p in (list(vae.parameters()) + list(disentangle.parameters()))), max_norm=1.0)
            scaler.step(optimizer_vae)
            scaler.step(optimizer_aux)
            scaler.update()
            optimizer_vae.zero_grad(set_to_none=True)
            optimizer_aux.zero_grad(set_to_none=True)

            if profile_type == "nsys":
                torch.cuda.nvtx.range_pop()
            if prof:
                prof.step()
            torch.cuda.synchronize()
            total_processed += batch_size

    if profile_type == "torch":
        trace_path = os.path.join(output_dir, "trace_phase_b")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=max(1, max_steps - 2), repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            run_steps(prof)
        prof.export_chrome_trace(f"{trace_path}.json.gz")
        logger.info(f"Phase B Chrome trace exported -> {trace_path}.json.gz")
        print("\n--- Phase B Top CUDA Kernels by Time ---")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    else:
        run_steps()

    elapsed = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    throughput = total_processed / max(elapsed, 1e-4)

    logger.info(f"Phase B Complete: {total_processed} items | Peak VRAM {peak_vram:.2f} GB | Throughput {throughput:.1f} vol/s")
    return {"elapsed_s": elapsed, "peak_vram_gb": peak_vram, "throughput_vol_s": throughput, "samples": total_processed}


# ═══════════════════════════════════════════════════════════════════════════
# Phase C: Vectorised Continuous-Discrete Jump ODE Solver
# ═══════════════════════════════════════════════════════════════════════════

def profile_phase_c(latent_h5: str, batch_size: int = 24, num_workers: int = 4,
                    max_steps: int = 10, output_dir: str = "profiling_results",
                    profile_type: str = "torch"):
    logger.info("=" * 70)
    logger.info(f"PROFILING PHASE C: Vectorised Neural Jump ODE (3.8M Params, Batch={batch_size})")
    logger.info("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    solver = ContinuousDiscreteJumpSolver(domain="breast", dt_cfl_days=7.0).to(device)
    optimizer = torch.optim.AdamW(solver.parameters(), lr=5e-4, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    dataset = LatentTrajectoryDataset(latent_h5, min_visits=2)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        collate_fn=_trajectory_collate_fn,
    )
    logger.info(f"Phase C Dataset: {len(dataset)} trajectories, {len(loader)} batches total (profiling up to {max_steps} steps)")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    total_processed = 0

    def run_steps(prof=None):
        nonlocal total_processed
        for step, batch in enumerate(loader):
            if step >= max_steps:
                break
            if profile_type == "nsys":
                torch.cuda.nvtx.range_push(f"PhaseC_Step_{step}")

            u_d = batch["u_d"].to(device, non_blocking=True)       # (B, T, C, D, H, W)
            v_sys = batch["v_sys"].to(device, non_blocking=True)   # (B, T, 160)
            t_days = batch["t_days"].to(device, non_blocking=True) # (B, T)
            mask = batch["mask"].to(device, non_blocking=True)     # (B, T)
            B, T, C, D, H, W = u_d.shape

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                u_curr = u_d[:, 0]
                v_curr = v_sys[:, 0]
                phi = torch.zeros(B, 1, D, H, W, device=device)
                c_drug = torch.zeros(B, 16, device=device)
                step_loss = torch.tensor(0.0, device=device)

                for t_idx in range(1, min(T, 3)):
                    t_start_vec = t_days[:, t_idx - 1]
                    t_end_vec = t_days[:, t_idx]
                    valid = mask[:, t_idx] & (t_end_vec > t_start_vec)

                    if not valid.any():
                        continue

                    t_start = t_start_vec[valid].min().item()
                    t_end = t_end_vec[valid].max().item()
                    if t_end <= t_start:
                        continue

                    u_pred, v_pred = solver.integrate_interval(
                        u_curr, v_curr, t_start, t_end, phi, c_drug, t_start
                    )

                    u_target = u_d[:, t_idx]
                    v_target = v_sys[:, t_idx]
                    loss_u = F.mse_loss(u_pred, u_target)
                    loss_v = F.mse_loss(v_pred, v_target)
                    step_loss = step_loss + (loss_u + 0.1 * loss_v)

            scaler.scale(step_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if profile_type == "nsys":
                torch.cuda.nvtx.range_pop()
            if prof:
                prof.step()
            torch.cuda.synchronize()
            total_processed += batch_size

    if profile_type == "torch":
        trace_path = os.path.join(output_dir, "trace_phase_c")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=max(1, max_steps - 2), repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            run_steps(prof)
        prof.export_chrome_trace(f"{trace_path}.json.gz")
        logger.info(f"Phase C Chrome trace exported -> {trace_path}.json.gz")
        print("\n--- Phase C Top CUDA Kernels by Time ---")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    else:
        run_steps()

    elapsed = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    throughput = total_processed / max(elapsed, 1e-4)

    logger.info(f"Phase C Complete: {total_processed} items | Peak VRAM {peak_vram:.2f} GB | Throughput {throughput:.1f} traj/s")
    return {"elapsed_s": elapsed, "peak_vram_gb": peak_vram, "throughput_traj_s": throughput, "samples": total_processed}


# ═══════════════════════════════════════════════════════════════════════════
# Main Profiling Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0.0

    logger.info(f"Profiling System: {gpu_name} ({total_mem:.1f} GB VRAM)")
    logger.info(f"Target Phases: {args.phases} | Mode: {args.profile_type} | Output: {args.output_dir}")

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gpu_name": gpu_name,
        "gpu_mem_total_gb": total_mem,
        "profile_type": args.profile_type,
        "phases": {}
    }

    def is_valid_phase_b_h5(p):
        if not p or not os.path.exists(p):
            return False
        try:
            return len(EncoderFeatureDataset(str(p), domain="breast")) > 0
        except Exception:
            return False

    def is_valid_phase_c_h5(p):
        if not p or not os.path.exists(p):
            return False
        try:
            return len(LatentTrajectoryDataset(str(p), min_visits=2)) > 0
        except Exception:
            return False

    # Locate or generate HDF5 datasets
    data_dir = Path(args.data_dir)
    encoder_candidates = [
        data_dir / "encoder_features.h5",
        Path("/DATA/horizon_disentangled/hdf5/encoder_features.h5"),
    ]
    latent_candidates = [
        data_dir / "latent_decomposition.h5",
        Path("/DATA/horizon_disentangled/hdf5/latent_decomposition.h5"),
    ]

    encoder_h5 = next((str(p) for p in encoder_candidates if is_valid_phase_b_h5(p)), None)
    latent_h5 = next((str(p) for p in latent_candidates if is_valid_phase_c_h5(p)), None)

    # If missing or incomplete, generate synthetic HDF5 datasets in output_dir
    synth_dir = os.path.join(args.output_dir, "synthetic_inputs")
    os.makedirs(synth_dir, exist_ok=True)

    if not encoder_h5:
        synth_enc = os.path.join(synth_dir, "encoder_features.h5")
        if not is_valid_phase_b_h5(synth_enc):
            logger.info("Real encoder features HDF5 not found; generating synthetic dataset for Phase B...")
            create_phase_a_hdf5(synth_enc, n_patients=64, n_visits=4)
        encoder_h5 = synth_enc

    if not latent_h5:
        synth_lat = os.path.join(synth_dir, "latent_decomposition.h5")
        if not is_valid_phase_c_h5(synth_lat):
            logger.info("Real latent trajectory HDF5 not found; generating synthetic dataset for Phase C...")
            create_phase_b_hdf5(synth_lat, n_patients=64, n_visits=4)
        latent_h5 = synth_lat

    selected_phases = [p.strip().upper() for p in args.phases.split(",")]

    # 1. Phase A
    if "A" in selected_phases:
        results["phases"]["A"] = profile_phase_a(
            output_dir=args.output_dir, n_samples=32, profile_type=args.profile_type
        )

    # 2. Phase B
    if "B" in selected_phases:
        results["phases"]["B"] = profile_phase_b(
            encoder_h5=encoder_h5, batch_size=args.batch_b,
            num_workers=args.num_workers, max_steps=args.max_steps,
            output_dir=args.output_dir, profile_type=args.profile_type
        )

    # 3. Phase C
    if "C" in selected_phases:
        results["phases"]["C"] = profile_phase_c(
            latent_h5=latent_h5, batch_size=args.batch_c,
            num_workers=args.num_workers, max_steps=args.max_steps,
            output_dir=args.output_dir, profile_type=args.profile_type
        )

    summary_json_path = os.path.join(args.output_dir, "profile_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 70)
    logger.info(f"PROFILING RUN COMPLETE - Summary saved to {summary_json_path}")
    logger.info("=" * 70)
    for p, stats in results["phases"].items():
        logger.info(f"Phase {p}: Time {stats.get('elapsed_s', 0):.2f}s | "
                    f"Peak VRAM: {stats.get('peak_vram_gb', 'N/A'):.2f} GB | "
                    f"Throughput: {stats.get('throughput_vol_s') or stats.get('throughput_traj_s', 'N/A'):.1f}/s")


if __name__ == "__main__":
    main()
