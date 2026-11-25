#!/usr/bin/env python3
"""Scaling Benchmark: 1-GPU vs 4-GPU Comparison for Horizon Disentangled on JUPITER GH200.

Implements and benchmarks all four scaling features:
  1. Real SwinUNETR encoder (Phase A) — 75.5M params (feature_size=48)
  2. FSDP-wrapped 756M VAE + torch.compile (Phase B)
  3. Spectral cuFFT Laplacian for PDEs (Phase C)
  4. 1-GPU vs 4-GPU scaling comparison with efficiency metrics

This script supports both single-GPU and multi-GPU (DDP/FSDP) execution.
When run with torchrun --nproc_per_node=4, it auto-detects distributed mode
and wraps models appropriately.

Usage:
    # Single GPU:
    CUDA_VISIBLE_DEVICES=0 python profile_scaling_benchmark.py --ngpus 1

    # 4 GPUs via torchrun:
    torchrun --nproc_per_node=4 profile_scaling_benchmark.py --ngpus 4 --use-fsdp

    # Full benchmark (both modes in sequence via sbatch):
    sbatch profile_scaling_jupiter.sbatch
"""

import os
import sys
import gc
import time
import json
import argparse
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("scaling_benchmark")

GH200_VRAM_GB = 95.0  # usable CUDA VRAM on JUPITER GH200


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def setup_distributed():
    """Initialize distributed process group if running under torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

        logger.info(f"Distributed init: rank={rank}/{world_size}, local_rank={local_rank}, "
                     f"device=cuda:{local_rank}")
        return rank, world_size, local_rank
    else:
        logger.info("Single-GPU mode (no torchrun environment detected)")
        return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_rank():
    """Check if this is the main (rank 0) process."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def gpu_cleanup():
    """Aggressive GPU memory cleanup between phases."""
    try:
        from torch import _dynamo
        _dynamo.reset()
    except Exception:
        pass
    gc.collect()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    remaining = torch.cuda.memory_allocated() / (1024**3)
    if is_main_rank():
        logger.info(f"GPU cleanup done: {remaining:.3f} GB still allocated")


@contextmanager
def phase_timer(name: str):
    """Context manager to time and log a phase."""
    if is_main_rank():
        logger.info(f"{'═' * 60}")
        logger.info(f"Starting {name}")
        logger.info(f"{'═' * 60}")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    if is_main_rank():
        logger.info(f"{name} completed in {elapsed:.2f}s")


def get_memory_stats() -> Dict[str, float]:
    """Get current GPU memory stats in GB."""
    return {
        "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
        "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / (1024**3),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Phase A: Real SwinUNETR Encoder
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_phase_a(
    batch_size: int = 8,
    vol_shape: Tuple[int, ...] = (4, 128, 128, 64),
    feature_size: int = 48,
    n_steps: int = 10,
    use_fsdp: bool = False,
    use_compile: bool = False,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Benchmark Phase A with the real SwinUNETR / VoComni-H encoder.

    Supports both SwinUNETR-48 (64.4M) and VoComni-H (feature_size=192, ~1.03B+).
    When distributed and (use_fsdp or n_params > 400M), wraps in FSDP FULL_SHARD
    using SwinTransformerBlock auto-wrap policy.
    """
    from src.pipeline.stage1_supervisors.models import SwinUNETREncoder

    if is_main_rank():
        logger.info(f"Phase A: SwinUNETR encoder, feature_size={feature_size}, "
                     f"batch={batch_size}, vol_shape={vol_shape}")

    torch.cuda.reset_peak_memory_stats()

    # Create encoder (randomly initialized — same architecture as pretrained)
    encoder = SwinUNETREncoder(
        in_channels=vol_shape[0],
        feature_dim=768,
        feature_size=feature_size,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        use_v2=False,
        pretrained=None,
    ).to(device).to(torch.bfloat16)

    n_params = sum(p.numel() for p in encoder.parameters())
    if is_main_rank():
        logger.info(f"  Encoder params: {n_params / 1e6:.1f}M")

    # FSDP (for >400M VoComni-H) or DDP wrap if distributed
    if dist.is_initialized():
        if use_fsdp and n_params > 400_000_000:
            import functools
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                ShardingStrategy,
                MixedPrecision,
            )
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            from monai.networks.nets.swin_unetr import SwinTransformerBlock

            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.bfloat16,
            )
            wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={SwinTransformerBlock},
            )
            encoder = FSDP(
                encoder,
                auto_wrap_policy=wrap_policy,
                mixed_precision=mp_policy,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                device_id=torch.cuda.current_device(),
                use_orig_params=True,
                limit_all_gathers=True,
                forward_prefetch=True,
            )
            if is_main_rank():
                logger.info("  FSDP applied to Phase A encoder (FULL_SHARD, SwinTransformerBlock)")
        else:
            from torch.nn.parallel import DistributedDataParallel as DDP
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            encoder = DDP(encoder, device_ids=[local_rank], static_graph=True,
                          gradient_as_bucket_view=True, bucket_cap_mb=50)

    # torch.compile (mode="default" avoids 25GB CUDA Graph private pool retention & 13-min autotuning)
    if use_compile:
        try:
            encoder = torch.compile(encoder, mode="default", dynamic=False)
            if is_main_rank():
                logger.info("  torch.compile (mode=default) applied to encoder")
        except Exception as e:
            if is_main_rank():
                logger.warning(f"  torch.compile failed: {e}")

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-4)
    encoder.train()

    # Warmup
    for _ in range(2):
        x = torch.randn(batch_size, *vol_shape, device=device, dtype=torch.bfloat16)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            emb = encoder(x)
            loss = emb.mean()
        loss.backward()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Timed benchmark
    start = time.perf_counter()
    total_items = 0

    for step in range(n_steps):
        x = torch.randn(batch_size, *vol_shape, device=device, dtype=torch.bfloat16)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            emb = encoder(x)
            loss = emb.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        total_items += batch_size

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    mem = get_memory_stats()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    total_items_global = total_items * world_size

    result = {
        "phase": "A",
        "model": f"SwinUNETR-{feature_size}",
        "params_M": round(n_params / 1e6, 1),
        "batch_size": batch_size,
        "ngpus": world_size,
        "n_steps": n_steps,
        "elapsed_s": round(elapsed, 3),
        "throughput_vol_s": round(total_items_global / elapsed, 2),
        "peak_vram_gb": round(mem["peak_allocated_gb"], 2),
        "occupancy_pct": round(mem["peak_allocated_gb"] / GH200_VRAM_GB * 100, 1),
        "vol_shape": list(vol_shape),
    }

    if is_main_rank():
        logger.info(f"  Phase A Results: {json.dumps(result, indent=2)}")

    del encoder, optimizer
    gpu_cleanup()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Phase B: 756M VAE with FSDP + torch.compile
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_phase_b(
    batch_size: int = 32,
    grad_accum: int = 2,
    n_steps: int = 10,
    use_fsdp: bool = True,
    use_compile: bool = True,
    vae_channels: Tuple[int, ...] = (256, 512, 1024),
    latent_input_shape: Tuple[int, ...] = (16, 32, 32, 16),
    device: str = "cuda",
) -> Dict[str, Any]:
    """Benchmark Phase B: 756M VAE with FSDP and torch.compile.

    FSDP shards model parameters across GPUs, dramatically reducing per-GPU
    memory footprint for the 756M param model, enabling larger batch sizes.
    torch.compile(mode='max-autotune') fuses kernels for the VAE encode/decode.
    """
    from src.pipeline.stage2_causal_vae.architecture import TwoScaleHierarchicalCausalVAE

    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if is_main_rank():
        logger.info(f"Phase B: VAE channels={vae_channels}, batch={batch_size}, "
                     f"grad_accum={grad_accum}, fsdp={use_fsdp}, compile={use_compile}, "
                     f"gpus={world_size}")

    torch.cuda.reset_peak_memory_stats()

    # Create the 756M VAE
    vae = TwoScaleHierarchicalCausalVAE(
        in_channels=4,
        anatomy_channels=4,
        disease_channels=8,
        vae_channels=vae_channels,
        vae_num_res_blocks=3,
        global_disease_dim=128,
        patient_dim=64,
        style_dim=64,
        l2_channels=512,
        backbone="monai",
    ).to(device).to(torch.bfloat16)

    n_params = sum(p.numel() for p in vae.parameters())
    if is_main_rank():
        logger.info(f"  VAE params: {n_params / 1e6:.1f}M")

    # FSDP wrapping (call vae(x, cached=True) so FSDP.forward() unshards parameters!)
    if use_fsdp and dist.is_initialized():
        import functools
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
            MixedPrecision,
        )
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        )
        wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=10_000_000
        )
        vae = FSDP(
            vae,
            auto_wrap_policy=wrap_policy,
            mixed_precision=mp_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
            limit_all_gathers=True,
            forward_prefetch=True,
        )
        if is_main_rank():
            logger.info("  FSDP applied (FULL_SHARD, bf16 params, fp32 reduce, use_orig_params=True)")
    elif dist.is_initialized():
        from torch.nn.parallel import DistributedDataParallel as DDP
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        vae = DDP(vae, device_ids=[local_rank],
                  find_unused_parameters=True,
                  gradient_as_bucket_view=True, bucket_cap_mb=50)
        if is_main_rank():
            logger.info("  DDP applied")

    # torch.compile after FSDP/DDP wrapping
    if use_compile:
        try:
            from torch import _dynamo
            _dynamo.config.suppress_errors = True
            vae = torch.compile(vae, mode="default", dynamic=False, fullgraph=False)
            if is_main_rank():
                logger.info("  torch.compile (mode=default) applied to VAE")
        except Exception as e:
            if is_main_rank():
                logger.warning(f"  torch.compile failed: {e}")

    optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-4)
    vae.train()

    # Input shape: simulating cached MedVAE features from Phase A
    # Default: (B, 16, 32, 32, 16); zero-cropped 256x256x192: (B, 16, 64, 64, 48)
    input_shape = (batch_size, *latent_input_shape)

    # Warmup
    for _ in range(2):
        x = torch.randn(*input_shape, device=device, dtype=torch.bfloat16)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            enc = vae(x, cached=True)
            recon = enc["recon"]
            recon_loss = F.mse_loss(recon, torch.zeros_like(recon))
            kl_loss = -0.5 * torch.mean(1 + enc["logvar_d"] - enc["mu_d"]**2 - enc["logvar_d"].exp())
            loss = (recon_loss + 0.01 * kl_loss) / grad_accum
        loss.backward()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Timed benchmark with gradient accumulation
    start = time.perf_counter()
    total_items = 0
    optimizer.zero_grad(set_to_none=True)

    for step in range(n_steps):
        x = torch.randn(*input_shape, device=device, dtype=torch.bfloat16)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            enc = vae(x, cached=True)
            recon = enc["recon"]
            recon_loss = F.mse_loss(recon, torch.zeros_like(recon))
            kl_loss = -0.5 * torch.mean(1 + enc["logvar_d"] - enc["mu_d"]**2 - enc["logvar_d"].exp())
            loss = (recon_loss + 0.01 * kl_loss) / grad_accum

        loss.backward()

        if (step + 1) % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_items += batch_size

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    mem = get_memory_stats()
    total_items_global = total_items * world_size

    result = {
        "phase": "B",
        "model": f"CausalVAE-{'-'.join(map(str, vae_channels))}",
        "params_M": round(n_params / 1e6, 1),
        "batch_size": batch_size,
        "effective_batch": batch_size * grad_accum,
        "grad_accum": grad_accum,
        "ngpus": world_size,
        "use_fsdp": use_fsdp and dist.is_initialized(),
        "use_compile": use_compile,
        "n_steps": n_steps,
        "elapsed_s": round(elapsed, 3),
        "throughput_vol_s": round(total_items_global / elapsed, 2),
        "peak_vram_gb": round(mem["peak_allocated_gb"], 2),
        "occupancy_pct": round(mem["peak_allocated_gb"] / GH200_VRAM_GB * 100, 1),
    }

    if is_main_rank():
        logger.info(f"  Phase B Results: {json.dumps(result, indent=2)}")

    del vae, optimizer
    gpu_cleanup()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Phase C: Spectral cuFFT Laplacian PDE Solver
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_phase_c(
    batch_size: int = 24,
    n_steps: int = 10,
    use_spectral: bool = True,
    domain: str = "gbm",
    pde_shape: Optional[Tuple[int, int, int]] = None,
    dx_mm: float = 8.0,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Benchmark Phase C: ODE solver with spectral Laplacian.

    Compares finite-difference vs spectral cuFFT Laplacian for the
    biophysical PDEs. The spectral method uses torch.fft.rfftn/irfftn
    which map to NVIDIA cuFFT for high throughput.

    Note: torch.compile is NOT used for the ODE solver because
    integrate_interval has a dynamic while-loop (@torch.compiler.disable).
    """
    from src.pipeline.stage4_dynamics.solver import ContinuousDiscreteJumpSolver

    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if is_main_rank():
        logger.info(f"Phase C: domain={domain}, batch={batch_size}, "
                     f"spectral={use_spectral}, gpus={world_size}")

    torch.cuda.reset_peak_memory_stats()

    solver = ContinuousDiscreteJumpSolver(domain=domain, dt_cfl_days=7.0)
    latent_shape = pde_shape or ((24, 28, 20) if domain == "gbm" else (32, 32, 16))

    # Configure PDE with custom spatial shape, dx_mm, and optional spectral Laplacian
    if domain == "gbm":
        from src.pipeline.stage4_dynamics.gbm_biophysical import GlioblastomaBiophysicalPDE
        solver.spatial_pde = GlioblastomaBiophysicalPDE(
            latent_shape=latent_shape, dx_mm=dx_mm, use_spectral=use_spectral
        )
    else:
        from src.pipeline.stage4_dynamics.breast_fast_slow import BreastFastSlowBiophysicalPDE
        solver.spatial_pde = BreastFastSlowBiophysicalPDE(
            latent_shape=latent_shape, dx_mm=dx_mm, use_spectral=use_spectral
        )
    if use_spectral and is_main_rank():
        logger.info(f"  Spectral cuFFT Laplacian enabled (shape={latent_shape}, dx={dx_mm}mm)")

    solver = solver.to(device).to(torch.bfloat16)

    n_params = sum(p.numel() for p in solver.parameters())
    if is_main_rank():
        logger.info(f"  Solver params: {n_params / 1e6:.1f}M")

    # DDP wrap sub-modules (the solver's own wrap_for_cluster)
    if dist.is_initialized():
        solver.wrap_for_cluster()

    optimizer = torch.optim.AdamW(solver.parameters(), lr=1e-4)
    solver.train()

    D, H, W = latent_shape

    # Warmup
    for _ in range(2):
        u_d = torch.randn(batch_size, 8, D, H, W, device=device, dtype=torch.bfloat16)
        v_sys = torch.randn(batch_size, 160, device=device, dtype=torch.bfloat16)
        phi = torch.ones(batch_size, 1, D, H, W, device=device, dtype=torch.bfloat16)

        # Simple forward: integrate 1 interval
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            c_drug = torch.zeros(batch_size, 16, device=device, dtype=torch.bfloat16)
            u_out, v_out = solver.integrate_interval(
                u_d, v_sys, 0.0, 14.0, phi, c_drug, 0.0
            )
            loss = u_out.mean() + v_out.mean()
        loss.backward()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Timed benchmark
    start = time.perf_counter()
    total_traj = 0

    for step in range(n_steps):
        u_d = torch.randn(batch_size, 8, D, H, W, device=device, dtype=torch.bfloat16)
        v_sys = torch.randn(batch_size, 160, device=device, dtype=torch.bfloat16)
        phi = torch.ones(batch_size, 1, D, H, W, device=device, dtype=torch.bfloat16)
        c_drug = torch.zeros(batch_size, 16, device=device, dtype=torch.bfloat16)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            u_out, v_out = solver.integrate_interval(
                u_d, v_sys, 0.0, 14.0, phi, c_drug, 0.0
            )
            loss = u_out.mean() + v_out.mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        total_traj += batch_size

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    mem = get_memory_stats()
    total_traj_global = total_traj * world_size

    result = {
        "phase": "C",
        "model": f"{'Spectral' if use_spectral else 'FD'}-{domain.upper()}-Solver",
        "params_M": round(n_params / 1e6, 1),
        "batch_size": batch_size,
        "ngpus": world_size,
        "use_spectral": use_spectral,
        "n_steps": n_steps,
        "elapsed_s": round(elapsed, 3),
        "throughput_traj_s": round(total_traj_global / elapsed, 2),
        "peak_vram_gb": round(mem["peak_allocated_gb"], 2),
        "occupancy_pct": round(mem["peak_allocated_gb"] / GH200_VRAM_GB * 100, 1),
        "latent_shape": list(latent_shape),
    }

    if is_main_rank():
        logger.info(f"  Phase C Results: {json.dumps(result, indent=2)}")

    del solver, optimizer
    gpu_cleanup()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Spectral vs FD micro-benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_spectral_vs_fd(
    spatial_shape: Tuple[int, int, int] = (24, 28, 20),
    batch_size: int = 32,
    n_channels: int = 8,
    n_iters: int = 100,
    dx_mm: float = 8.0,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Micro-benchmark: spectral cuFFT Laplacian vs finite-difference stencil.

    Measures raw kernel throughput (no autograd) for both approaches.
    """
    from src.pipeline.stage4_dynamics.gbm_biophysical import FiniteDifference3D
    from src.pipeline.stage4_dynamics.spectral_laplacian import SpectralLaplacian3D

    if is_main_rank():
        logger.info(f"Micro-benchmark: Spectral vs FD Laplacian, shape={spatial_shape}, "
                     f"batch={batch_size}, channels={n_channels}")

    spec_lap = SpectralLaplacian3D(spatial_shape=spatial_shape, dx_mm=dx_mm).to(device)
    u = torch.randn(batch_size, n_channels, *spatial_shape, device=device, dtype=torch.bfloat16)

    # Warmup
    for _ in range(5):
        _ = spec_lap(u)
        _ = FiniteDifference3D.laplacian(u, dx_mm)

    torch.cuda.synchronize()

    # Benchmark FD
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = FiniteDifference3D.laplacian(u, dx_mm)
    torch.cuda.synchronize()
    fd_time = time.perf_counter() - t0

    # Benchmark Spectral
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = spec_lap(u)
    torch.cuda.synchronize()
    spec_time = time.perf_counter() - t0

    # Accuracy comparison (fp32 for precision)
    u_fp32 = u.float()
    spec_lap_fp32 = SpectralLaplacian3D(spatial_shape=spatial_shape, dx_mm=dx_mm).to(device)
    fd_result = FiniteDifference3D.laplacian(u_fp32, dx_mm)
    spec_result = spec_lap_fp32(u_fp32)
    # Note: they won't match exactly due to different boundary conditions
    # (FD uses replicate padding, spectral uses periodic BCs)

    result = {
        "fd_time_ms": round(fd_time / n_iters * 1000, 3),
        "spectral_time_ms": round(spec_time / n_iters * 1000, 3),
        "speedup": round(fd_time / spec_time, 2),
        "n_iters": n_iters,
        "shape": f"{batch_size}x{n_channels}x{'x'.join(map(str, spatial_shape))}",
    }

    if is_main_rank():
        logger.info(f"  FD: {result['fd_time_ms']:.3f} ms/iter")
        logger.info(f"  Spectral: {result['spectral_time_ms']:.3f} ms/iter")
        logger.info(f"  Speedup: {result['speedup']:.2f}×")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Scaling Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def compute_scaling_analysis(
    single_gpu: Dict[str, Dict[str, Any]],
    multi_gpu: Dict[str, Dict[str, Any]],
    n_gpus: int = 4,
) -> Dict[str, Any]:
    """Compute scaling efficiency between 1-GPU and N-GPU runs."""
    analysis = {}

    for phase in ["A", "B", "C"]:
        s = single_gpu.get(phase)
        m = multi_gpu.get(phase)
        if not s or not m:
            continue

        # Throughput key differs for Phase C
        tp_key = "throughput_traj_s" if phase == "C" else "throughput_vol_s"
        tp_1 = s.get(tp_key, 0)
        tp_n = m.get(tp_key, 0)

        speedup = tp_n / tp_1 if tp_1 > 0 else 0
        ideal_speedup = n_gpus
        efficiency = speedup / ideal_speedup * 100 if ideal_speedup > 0 else 0

        # Memory analysis
        mem_1 = s.get("peak_vram_gb", 0)
        mem_n = m.get("peak_vram_gb", 0)
        mem_ratio = mem_n / mem_1 if mem_1 > 0 else 0

        analysis[phase] = {
            "throughput_1gpu": tp_1,
            f"throughput_{n_gpus}gpu": tp_n,
            "speedup": round(speedup, 2),
            "ideal_speedup": ideal_speedup,
            "scaling_efficiency_pct": round(efficiency, 1),
            "peak_vram_1gpu_gb": mem_1,
            f"peak_vram_{n_gpus}gpu_per_gpu_gb": mem_n,
            "memory_ratio": round(mem_ratio, 2),
            "bottleneck": identify_bottleneck(speedup, ideal_speedup, mem_1, mem_n, phase),
        }

    return analysis


def identify_bottleneck(
    speedup: float, ideal: float, mem_1: float, mem_n: float, phase: str
) -> str:
    """Identify the likely scaling bottleneck."""
    efficiency = speedup / ideal if ideal > 0 else 0

    if efficiency >= 0.90:
        return "Near-linear scaling — no significant bottleneck"
    elif efficiency >= 0.75:
        if phase == "B":
            return "FSDP all-gather/reduce-scatter communication overhead"
        return "Moderate DDP all-reduce overhead on gradient sync"
    elif efficiency >= 0.50:
        if mem_n > 0.8 * GH200_VRAM_GB:
            return "Memory-bound: batch size limited by VRAM, cannot fully utilise added GPUs"
        return "Communication overhead dominates; consider overlapping comm/compute"
    else:
        bottlenecks = []
        if phase == "C":
            bottlenecks.append("ODE solver's sequential time-stepping limits parallelism")
        if mem_n > mem_1 * 0.9:
            bottlenecks.append("Per-GPU memory not reduced: DDP replicates full model")
        bottlenecks.append("Amdahl's law: serial fraction too high for effective scaling")
        return "; ".join(bottlenecks)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Scaling Benchmark: 1-GPU vs 4-GPU")
    parser.add_argument("--ngpus", type=int, default=1, help="Number of GPUs (1 or 4)")
    parser.add_argument("--phases", type=str, default="A,B,C", help="Phases to benchmark")
    parser.add_argument("--use-fsdp", action="store_true", help="Use FSDP for Phase B")
    parser.add_argument("--use-compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--use-spectral", action="store_true", help="Use spectral Laplacian")
    parser.add_argument("--batch-a", type=int, default=8, help="Phase A batch size")
    parser.add_argument("--batch-b", type=int, default=32, help="Phase B batch size")
    parser.add_argument("--batch-c", type=int, default=24, help="Phase C batch size")
    parser.add_argument("--grad-accum-b", type=int, default=2, help="Phase B grad accumulation")
    parser.add_argument("--max-steps", type=int, default=10, help="Steps per phase")
    parser.add_argument("--feature-size", type=int, default=48, help="SwinUNETR feature_size (192 for VoComni-H)")
    parser.add_argument("--vol-shape", type=str, default="4,128,128,64",
                        help="Phase A volume shape C,D,H,W (use 4,256,256,192 for zero-cropping)")
    parser.add_argument("--latent-input-shape", type=str, default="16,32,32,16",
                        help="Phase B latent input shape C,D,H,W (use 16,64,64,48 for zero-cropping)")
    parser.add_argument("--pde-shape", type=str, default="24,28,20",
                        help="Phase C PDE grid D,H,W (use 48,56,40 for 4mm high-res)")
    parser.add_argument("--dx-mm", type=float, default=8.0, help="Phase C grid spacing in mm (4.0 for 48,56,40)")
    parser.add_argument("--output-dir", type=str,
                        default="/e/scratch/numedai-scale/mitura1/horizon_profiling/scaling")
    parser.add_argument("--run-label", type=str, default=None,
                        help="Label for this run (default: '{ngpus}gpu')")
    parser.add_argument("--run-spectral-micro", action="store_true",
                        help="Also run spectral vs FD micro-benchmark")
    return parser.parse_args()


def main():
    args = parse_args()

    # Distributed setup
    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    if is_main_rank():
        logger.info(f"{'═' * 70}")
        logger.info(f"HORIZON DISENTANGLED — SCALING BENCHMARK")
        logger.info(f"  GPUs: {world_size}")
        logger.info(f"  FSDP: {args.use_fsdp}")
        logger.info(f"  torch.compile: {args.use_compile}")
        logger.info(f"  Spectral Laplacian: {args.use_spectral}")
        logger.info(f"  Device: {device}")
        if torch.cuda.is_available():
            logger.info(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
            logger.info(f"  VRAM: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.1f} GB")
        logger.info(f"{'═' * 70}")

    run_label = args.run_label or f"{world_size}gpu"
    output_dir = os.path.join(args.output_dir, run_label)
    if is_main_rank():
        os.makedirs(output_dir, exist_ok=True)

    results = {
        "run_label": run_label,
        "ngpus": world_size,
        "device": torch.cuda.get_device_name(local_rank) if torch.cuda.is_available() else "cpu",
        "config": {
            "use_fsdp": args.use_fsdp,
            "use_compile": args.use_compile,
            "use_spectral": args.use_spectral,
            "batch_a": args.batch_a,
            "batch_b": args.batch_b,
            "batch_c": args.batch_c,
            "grad_accum_b": args.grad_accum_b,
            "max_steps": args.max_steps,
        },
        "phases": {},
    }

    selected_phases = [p.strip().upper() for p in args.phases.split(",")]

    vol_shape = tuple(int(x) for x in args.vol_shape.split(","))
    latent_input_shape = tuple(int(x) for x in args.latent_input_shape.split(","))
    pde_shape = tuple(int(x) for x in args.pde_shape.split(","))

    # Phase A
    if "A" in selected_phases:
        with phase_timer("Phase A: SwinUNETR / VoComni-H Encoder"):
            results["phases"]["A"] = benchmark_phase_a(
                batch_size=args.batch_a,
                vol_shape=vol_shape,
                feature_size=args.feature_size,
                n_steps=args.max_steps,
                use_fsdp=args.use_fsdp,
                use_compile=args.use_compile,
                device=device,
            )
        gpu_cleanup()

    # Phase B
    if "B" in selected_phases:
        with phase_timer("Phase B: 756M Causal VAE (FSDP + compile)"):
            results["phases"]["B"] = benchmark_phase_b(
                batch_size=args.batch_b,
                grad_accum=args.grad_accum_b,
                n_steps=args.max_steps,
                use_fsdp=args.use_fsdp,
                use_compile=args.use_compile,
                latent_input_shape=latent_input_shape,
                device=device,
            )
        gpu_cleanup()

    # Phase C
    if "C" in selected_phases:
        with phase_timer("Phase C: ODE Solver (Spectral Laplacian)"):
            results["phases"]["C"] = benchmark_phase_c(
                batch_size=args.batch_c,
                n_steps=args.max_steps,
                use_spectral=args.use_spectral,
                pde_shape=pde_shape,
                dx_mm=args.dx_mm,
                device=device,
            )
        gpu_cleanup()

    # Optional: spectral vs FD micro-benchmark
    if args.run_spectral_micro and is_main_rank():
        with phase_timer("Micro-benchmark: Spectral vs FD Laplacian"):
            results["spectral_vs_fd"] = benchmark_spectral_vs_fd(device=device)

    # Save results
    if is_main_rank():
        summary_path = os.path.join(output_dir, "scaling_results.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {summary_path}")

        # Print summary table
        print(f"\n{'═' * 80}")
        print(f"SCALING BENCHMARK RESULTS — {run_label.upper()}")
        print(f"{'═' * 80}")
        print(f"{'Phase':<10} {'Model':<30} {'Params':<10} {'Throughput':<15} "
              f"{'VRAM':<12} {'Occ%':<8}")
        print(f"{'-' * 80}")
        for p in ["A", "B", "C"]:
            if p in results["phases"]:
                r = results["phases"][p]
                tp_key = "throughput_traj_s" if p == "C" else "throughput_vol_s"
                tp_unit = "traj/s" if p == "C" else "vol/s"
                print(f"{p:<10} {r.get('model', 'N/A'):<30} "
                      f"{r.get('params_M', 'N/A'):<10} "
                      f"{r.get(tp_key, 0):.2f} {tp_unit:<6} "
                      f"{r.get('peak_vram_gb', 0):.2f} GB   "
                      f"{r.get('occupancy_pct', 0):.1f}%")
        print(f"{'═' * 80}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
