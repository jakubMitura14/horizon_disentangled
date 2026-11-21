"""
Command-Line Interface for the Horizon Disentangled Pipeline.

Usage:
  # Run all stages for Glioblastoma
  python -m src.pipeline.cli --stage all --domain gbm

  # Run a specific stage
  python -m src.pipeline.cli --stage 2 --domain breast

  # Run with custom epochs
  python -m src.pipeline.cli --stage 1 --domain gbm --epochs 100

  # Multi-GPU with torchrun
  torchrun --nproc_per_node=4 -m src.pipeline.cli --stage all --domain gbm
"""
import argparse
import sys
import os
import torch
from contextlib import nullcontext

# Ensure correct libstdc++
os.environ.setdefault(
    "LD_LIBRARY_PATH",
    "/DATA/conda/envs/nukl-to-ct/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
)

# ─── Global GPU performance settings ──────────────────────────────────────────
# 'high' precision: uses TF32 on Ampere/Hopper (2× throughput vs FP32).
# H200 TensorCores: ~990 TF32 TFLOPS vs 495 FP32 TFLOPS.
torch.set_float32_matmul_precision("high")
# HDF5 file locking: disable for read-only cluster shared filesystems (NFS/Lustre)
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def main():
    parser = argparse.ArgumentParser(
        description="Horizon Disentangled — 5-Stage Causal AI Pipeline"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["1", "2", "3", "4", "5", "all"],
        help="Pipeline stage to run (1-5 or 'all')"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="gbm",
        choices=["gbm", "breast"],
        help="Clinical domain (gbm or breast)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Custom checkpoint directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device (e.g., cuda:0, cpu)"
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable bfloat16 automatic mixed precision"
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective batch = batch_size × steps × world_size)"
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default=None,
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode (max-autotune = most H200 kernel profiling; default = fastest compile)"
    )
    parser.add_argument(
        "--nccl-bucket-mb",
        type=int,
        default=50,
        help="DDP all-reduce bucket size in MB (default 50 — good for NVLink H200)"
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile (useful for debugging)"
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=None,
        help="Gradient accumulation steps (increase effective batch size without more VRAM)"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable PyTorch profiler (writes trace to artifacts/)"
    )

    args = parser.parse_args()

    # Early cluster environment setup (before any CUDA ops)
    from src.pipeline.fp8_context import setup_cluster_env
    setup_cluster_env()

    # Import after argument parsing to avoid GPU initialization delay
    from src.pipeline.config import PipelineConfig
    from src.pipeline.orchestrator import PipelineOrchestrator

    config = PipelineConfig()

    # Apply overrides
    if args.device:
        config.hardware.device = args.device
    if args.no_amp:
        config.hardware.use_bfloat16 = False
    if args.checkpoint_dir:
        config.data.checkpoints_dir = args.checkpoint_dir
    if args.batch_size:
        config.stage1.batch_size = args.batch_size
        config.stage2.batch_size = max(1, args.batch_size // 2)
        config.stage4.batch_size = max(1, args.batch_size // 4)
    # --- Cluster performance overrides ---
    config.hardware.gradient_accumulation_steps = args.grad_accum_steps
    config.hardware.nccl_bucket_cap_mb = args.nccl_bucket_mb
    if args.compile_mode is not None:
        config.hardware.compile_mode = args.compile_mode
    if args.no_compile:
        config.hardware.use_compile = False
    if args.grad_accum:
        config.hardware.gradient_accumulation_steps = args.grad_accum

    # Print configuration
    print("=" * 60)
    print("Horizon Disentangled — Causal AI Pipeline")
    print("=" * 60)
    print(f"Stage:       {args.stage}")
    print(f"Domain:      {args.domain}")
    print(f"Device:      {config.hardware.device}")
    print(f"Distributed: {config.hardware.is_distributed}")
    if config.hardware.is_distributed:
        print(f"  Rank:      {config.hardware.rank}/{config.hardware.world_size}")
    print(f"BFloat16:    {config.hardware.use_bfloat16}")
    print(f"GradAccum:   ×{config.hardware.gradient_accumulation_steps}  (effective batch = batch × accum × GPUs)")
    print(f"Compile:     {config.hardware.use_compile} (mode={getattr(config.hardware, 'compile_mode', 'default')})")
    print(f"NCCLBucket:  {getattr(config.hardware, 'nccl_bucket_cap_mb', 50)} MB")
    if torch.cuda.is_available():
        print(f"GPU:         {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM:        {mem_gb:.1f} GB")
    print("=" * 60)

    # Optional: PyTorch profiler
    profiler_ctx = nullcontext()
    if args.profile:
        from torch.profiler import profile, ProfilerActivity, schedule as prof_schedule
        trace_dir = os.path.join(config.data.artifacts_dir, "profiler_traces")
        os.makedirs(trace_dir, exist_ok=True)
        profiler_ctx = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=prof_schedule(wait=1, warmup=2, active=5, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        print(f"Profiler:    ENABLED → {trace_dir}")

    orchestrator = PipelineOrchestrator(config)

    with profiler_ctx:
        if args.stage == "all":
            epoch_kwargs = {}
            if args.epochs:
                epoch_kwargs = {
                    "stage1_epochs": args.epochs,
                    "stage2_epochs": args.epochs,
                    "stage3_epochs": max(1, args.epochs // 3),
                    "stage4_epochs": max(1, args.epochs // 2),
                }
            results = orchestrator.run_all(domain=args.domain, **epoch_kwargs)
        else:
            stage_num = int(args.stage)
            kwargs = {}
            if args.epochs:
                kwargs["num_epochs"] = args.epochs
            results = orchestrator.run_stage(stage_num, domain=args.domain, **kwargs)

    print("\n" + "=" * 60)
    print("Pipeline Complete")
    print("=" * 60)

    if isinstance(results, dict):
        for key, val in results.items():
            if isinstance(val, dict):
                print(f"\n{key}:")
                for k2, v2 in val.items():
                    if not isinstance(v2, dict):
                        print(f"  {k2}: {v2}")
            elif not isinstance(val, (list, dict)):
                print(f"{key}: {val}")


if __name__ == "__main__":
    main()
