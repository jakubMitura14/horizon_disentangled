# Codebase Documentation (`repr/docs/`)

Detailed file-by-file and function-by-function technical reference for all code in the reproducible SLURM scaling and profiling package.

## Documentation Index

| Document | Covers |
|----------|--------|
| [benchmark_and_slurm.md](benchmark_and_slurm.md) | `scripts/profile_scaling_benchmark.py` and the 3 SLURM `.sbatch` job scripts |
| [stage1_models.md](stage1_models.md) | `src/pipeline/stage1_supervisors/models.py` (Phase A encoders & supervisor trunk) |
| [stage2_causal_vae.md](stage2_causal_vae.md) | `src/pipeline/stage2_causal_vae/architecture.py` & `medvae_backbone.py` (Phase B Causal VAE) |
| [stage4_dynamics.md](stage4_dynamics.md) | `src/pipeline/stage4_dynamics/` — `solver.py`, `gbm_biophysical.py`, `breast_fast_slow.py`, `spectral_laplacian.py`, `systemic_njde.py`, `masked_loss.py` (Phase C PDE-NJDE) |

## End-to-End Benchmark Data Flow

```text
Phase A (stage1_supervisors/models.py)
  Input:  x ∈ ℝ^(B × 4 × D × H × W)  [bfloat16, channels_last_3d]
  Model:  SwinUNETREncoder (SwinUNETR-48 or VoComni-H feature_size=192)
  Output: emb ∈ ℝ^(B × 768)

Phase B (stage2_causal_vae/architecture.py)
  Input:  per_ch_z ∈ ℝ^(B × 16 × D/4 × H/4 × W/4)  [simulated cached L1 latents]
  Model:  TwoScaleHierarchicalCausalVAE (FSDP FULL_SHARD, DW-sep final_conv + decoder checkpointing)
  Output: recon ∈ ℝ^(B × 4 × D × H × W), mask_pred, z_a, u_d, z_d_glob, z_p, z_s

Phase C (stage4_dynamics/solver.py + gbm_biophysical.py / breast_fast_slow.py + spectral_laplacian.py + systemic_njde.py)
  Input:  u_d ∈ ℝ^(B × 8 × D_L × H_L × W_L), v_sys ∈ ℝ^(B × 160), phi ∈ ℝ^(B × 1 × D_L × H_L × W_L)
  Model:  ContinuousDiscreteJumpSolver (RK4 + SpectralLaplacian3D/SpectralGradient3D + SystemicNeuralJumpODE)
  Output: u_out, v_out at t = 14.0 days
```
