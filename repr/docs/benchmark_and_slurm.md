# Benchmark Runner & SLURM Scripts (`repr/scripts/`)

---

## 1. `scripts/profile_scaling_benchmark.py`

Main PyTorch distributed benchmark script supporting 1-GPU and multi-GPU (`torchrun` DDP/FSDP) execution across Phases A, B, and C, plus a Spectral vs. Finite-Difference micro-benchmark.

### Global Constants
- `REPO_ROOT`: `Path(__file__).resolve().parent.parent` (`repr/`), prepended to `sys.path` so `src.*` resolves locally.
- `GH200_VRAM_GB = 95.0`: Usable CUDA VRAM on JUPITER GH200 (97,871 MiB total minus driver/context overhead), used to compute `occupancy_pct`.

### Functions

#### `setup_distributed() -> Tuple[int, int, int]`
- Checks `RANK` and `WORLD_SIZE` environment variables (set by `torchrun`).
- If present: calls `torch.cuda.set_device(local_rank)`, initializes `nccl` process group, and returns `(rank, world_size, local_rank)`.
- Otherwise: logs single-GPU mode and returns `(0, 1, 0)`.

#### `cleanup_distributed() -> None`
- Calls `dist.destroy_process_group()` if `dist.is_initialized()`.

#### `is_main_rank() -> bool`
- Returns `True` on rank 0 or in non-distributed mode.

#### `gpu_cleanup() -> None`
- Resets `torch._dynamo`, runs `gc.collect()`, `torch.cuda.synchronize()`, `torch.cuda.empty_cache()`, resets peak memory stats, and logs remaining allocated GB on rank 0.

#### `phase_timer(name: str)` (Context Manager)
- Synchronizes CUDA before and after yielding, logs formatted phase banner and wall-clock duration on rank 0.

#### `get_memory_stats() -> Dict[str, float]`
- Returns `allocated_gb`, `reserved_gb`, `peak_allocated_gb`, and `peak_reserved_gb` in GiB (`1024**3`).

#### `benchmark_phase_a(batch_size=8, vol_shape=(4, 128, 128, 64), feature_size=48, n_steps=10, use_fsdp=False, use_compile=False, device="cuda") -> Dict[str, Any]`
- Instantiates `SwinUNETREncoder(in_channels=vol_shape[0], feature_dim=768, feature_size=feature_size, depths=(2,2,2,2), num_heads=(3,6,12,24), use_v2=False, pretrained=None)` in `bfloat16`.
- Distributed wrapping: if `dist.is_initialized()` and `use_fsdp and n_params > 400_000_000` (e.g. VoComni-H `feature_size=192`, 998.9M params), wraps with `FSDP` (`FULL_SHARD`, `transformer_auto_wrap_policy` on `SwinTransformerBlock`, `MixedPrecision(param=bf16, reduce=fp32, buffer=bf16)`, `use_orig_params=True`, `limit_all_gathers=True`, `forward_prefetch=True`). Otherwise wraps with `DDP(static_graph=True, gradient_as_bucket_view=True, bucket_cap_mb=50)`.
- Optionally applies `torch.compile(encoder, mode="default", dynamic=False)`.
- Runs 2 warmup steps + `n_steps` timed `AdamW` forward/backward/step iterations with synthetic `bfloat16` volumes under `torch.amp.autocast("cuda", dtype=torch.bfloat16)`.
- Returns dict with `phase`, `model`, `params_M`, `batch_size`, `ngpus`, `n_steps`, `elapsed_s`, `throughput_vol_s`, `peak_vram_gb`, `occupancy_pct`, `vol_shape`.

#### `benchmark_phase_b(batch_size=32, grad_accum=2, n_steps=10, use_fsdp=True, use_compile=True, vae_channels=(256, 512, 1024), latent_input_shape=(16, 32, 32, 16), l2_channels=512, device="cuda") -> Dict[str, Any]`
- Instantiates `TwoScaleHierarchicalCausalVAE(..., l2_channels=l2_channels, backbone="monai")` in `bfloat16`.
- Distributed wrapping: if `use_fsdp and dist.is_initialized()`, wraps with `FSDP` (`FULL_SHARD`, `size_based_auto_wrap_policy(min_num_params=10_000_000)`). Else if distributed, wraps with `DDP(find_unused_parameters=True, gradient_as_bucket_view=True, bucket_cap_mb=50)`.
- Optionally applies `torch.compile(vae, mode="default", dynamic=False, fullgraph=False)` with `_dynamo.config.suppress_errors = True`.
- Runs 2 warmup steps + `n_steps` timed iterations on `x` of shape `(batch_size, *latent_input_shape)` calling `vae(x, cached=True)`, computing `mse_loss(recon, 0) + 0.01 * kl_loss` scaled by `1 / grad_accum`, stepping `AdamW` every `grad_accum` steps.
- Returns dict with `phase`, `model`, `params_M`, `batch_size`, `effective_batch`, `grad_accum`, `ngpus`, `use_fsdp`, `use_compile`, `n_steps`, `elapsed_s`, `throughput_vol_s`, `peak_vram_gb`, `occupancy_pct`.

#### `benchmark_phase_c(batch_size=24, n_steps=10, use_spectral=True, domain="gbm", pde_shape=None, dx_mm=8.0, device="cuda") -> Dict[str, Any]`
- Instantiates `ContinuousDiscreteJumpSolver(domain=domain, dt_cfl_days=7.0)` and replaces `solver.spatial_pde` with `GlioblastomaBiophysicalPDE` or `BreastFastSlowBiophysicalPDE` configured with `latent_shape=pde_shape` and `dx_mm=dx_mm`, `use_spectral=use_spectral`.
- Casts solver to `bfloat16` and calls `solver.wrap_for_cluster()` when distributed.
- Runs 2 warmup + `n_steps` timed iterations calling `solver.integrate_interval(u_d, v_sys, 0.0, 14.0, phi, c_drug, 0.0)` and backpropagating `u_out.mean() + v_out.mean()`.
- Returns dict with `phase`, `model`, `params_M`, `batch_size`, `ngpus`, `use_spectral`, `n_steps`, `elapsed_s`, `throughput_traj_s`, `peak_vram_gb`, `occupancy_pct`, `latent_shape`.

#### `benchmark_spectral_vs_fd(spatial_shape=(24, 28, 20), batch_size=32, n_channels=8, n_iters=100, dx_mm=8.0, device="cuda") -> Dict[str, Any]`
- Micro-benchmarks raw forward latency (no autograd) of `FiniteDifference3D.laplacian(u, dx_mm)` vs. `SpectralLaplacian3D(spatial_shape, dx_mm)(u)` over `n_iters` iterations on `bfloat16` tensors.
- Returns `fd_time_ms`, `spectral_time_ms`, `speedup`, `n_iters`, `shape`.

#### `compute_scaling_analysis(single_gpu, multi_gpu, n_gpus=4) -> Dict[str, Any]`
- Utility function computing `speedup`, `scaling_efficiency_pct`, `memory_ratio`, and `bottleneck` per phase from 1-GPU and N-GPU result dicts.

#### `identify_bottleneck(speedup, ideal, mem_1, mem_n, phase) -> str`
- Heuristic classifier returning a human-readable bottleneck description based on scaling efficiency thresholds (`>=0.90`, `>=0.75`, `>=0.50`, `<0.50`) and VRAM saturation.

#### `parse_args()` & `main()`
- Parses CLI arguments:
  - `--ngpus` (int), `--phases` (comma-separated `A,B,C`), `--use-fsdp`, `--use-compile`, `--no-compile-b`, `--use-spectral`
  - `--batch-a`, `--batch-b`, `--batch-c`, `--grad-accum-b`, `--max-steps`
  - `--feature-size` (48 or 192), `--vol-shape`, `--latent-input-shape`, `--l2-channels` (512 or 256), `--pde-shape`, `--dx-mm`
  - `--output-dir`, `--run-label`, `--run-spectral-micro`
- Executes selected phases in order, saves JSON to `<output_dir>/<run_label>/scaling_results.json` on rank 0, prints formatted summary table, and calls `cleanup_distributed()`.

---

## 2. SLURM Batch Scripts (`repr/scripts/*.sbatch`)

All three scripts target JUPITER (`--account=numedai-scale`, `--partition=booster`, 1 node, 4 tasks/GPUs, 72 CPUs/task), load `Stages/2026`, `GCCcore/14.3.0`, `Python`, activate `lymph_env`, set `PYTORCH_ALLOC_CONF="expandable_segments:True"` and `NCCL_P2P_LEVEL=NVL`, and auto-resolve `REPO` to `repr/`.

### `profile_scaling_jupiter.sbatch`
- **Purpose**: Apples-to-apples 1-GPU vs 4-GPU scaling comparison (`compile=False`, `use_spectral=True` for both).
- **Step 1**: Runs `CUDA_VISIBLE_DEVICES=0 python3 scripts/profile_scaling_benchmark.py` for Phases A (`batch=8`, `128×128×64`), B (`batch=32`), C (`batch=56`, 4mm `48×56×40`) + `--run-spectral-micro`.
- **Step 2**: Runs `torchrun --nproc_per_node=4` with `--use-fsdp` and identical batch/model/compile settings.
- **Step 3**: Inline Python script reads both `scaling_results.json` files, computes speedup & scaling efficiency, prints the comparison table, and writes `scaling_comparison.json`.

### `profile_native_max_jupiter.sbatch`
- **Purpose**: Zero-cropping full native 3D resolution `(4, 256, 256, 192)` benchmark.
- **Key Design**: Launches **separate `torchrun` processes** for Phases A, B, and C (distinct `master_port` and `rdzv_id`) so CUDA allocator pools cannot fragment across phases.
- **Configs**:
  - Phase A: `--feature-size 192 --vol-shape 4,256,256,192 --batch-a 2 --use-fsdp --use-compile`
  - Phase B: `--latent-input-shape 16,64,64,48 --l2-channels 256 --batch-b 4 --grad-accum-b 4 --use-fsdp --no-compile-b`
  - Phase C: `--pde-shape 48,56,40 --dx-mm 4.0 --batch-c 56 --use-spectral`

### `profile_push_occupancy_v2.sbatch`
- **Purpose**: Sweeps higher-occupancy configurations for Phase B (using DW-sep `final_conv` + decoder checkpointing) and Phase C (4mm `batch=64` + 2mm ultra-fine grid).
- **Phase B Sweep**: Tries `l2=512` at `batch=6`, then `batch=4`; falls back to `l2=256` at `batch=8`, then `batch=6` if needed.
- **Phase C Runs**: Runs 4mm `(48, 56, 40)` at `batch=64/gpu` (95.1% VRAM), then 2mm `(96, 112, 80)` at `batch=6/gpu` (falling back to `batch=4`).
