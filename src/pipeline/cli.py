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

# Ensure correct libstdc++
os.environ.setdefault(
    "LD_LIBRARY_PATH",
    "/DATA/conda/envs/nukl-to-ct/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
)


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

    args = parser.parse_args()

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
    if torch.cuda.is_available():
        print(f"GPU:         {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM:        {mem_gb:.1f} GB")
    print("=" * 60)

    orchestrator = PipelineOrchestrator(config)

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
