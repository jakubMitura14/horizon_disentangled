# Data & Reference Artifacts (`repr/data/`)

## 1. Benchmark Input Data (Synthetic On-Device Tensors)

`scripts/profile_scaling_benchmark.py` intentionally generates **synthetic `bfloat16` 3D medical tensors directly on CUDA** (`torch.randn(..., device=device, dtype=torch.bfloat16)`) for all three phases:

- **Phase A**: 4-channel 3D MRI/CT volumes `(B, 4, D, H, W)` — e.g. `(B, 4, 256, 256, 192)` for zero-cropping native resolution or `(B, 4, 128, 128, 64)` for scaling tests.
- **Phase B**: Cached Level-1 MedVAE/AutoencoderKL latent tensors `(B, 16, D/4, H/4, W/4)` — e.g. `(B, 16, 64, 64, 48)` decoding to full `(B, 4, 256, 256, 192)`.
- **Phase C**: Spatial disease fields `u_d (B, 8, D_L, H_L, W_L)`, systemic state vectors `v_sys (B, 160)`, brain phase-field masks `phi (B, 1, D_L, H_L, W_L)`, and drug embeddings `c_drug (B, 16)` on 4mm `(48, 56, 40)` or 2mm `(96, 112, 80)` grids.

Generating tensors on-device isolates **GPU compute, VRAM occupancy, FSDP/DDP all-reduce, and cuFFT performance** without filesystem/HDF5 I/O variance.

## 2. Included Cohort Metadata & Configs

| File | Description |
|------|-------------|
| `dataset_encoded.csv` | Encoded longitudinal oncology cohort table (patient/visit features). |
| `dataset_summary.csv` | Summary statistics and cohort breakdown. |
| `phased_pipeline.json` | Phased training hyperparameters and hardware configuration. |

## 3. Reference Profiling Results (`reference_results/`)

Verified JSON outputs collected from JUPITER 4× GH200 jobs:

| File | SLURM Job | Contents |
|------|-----------|----------|
| `scaling_1vs4gpu_3114093.json` | `3114093` | Apples-to-apples 1-GPU vs 4-GPU comparison + Spectral vs FD micro-benchmark |
| `native_max_phase_a_3114634.json` | `3114634` | Phase A VoComni-H 998.9M at `(4, 256, 256, 192)`, batch=2/gpu |
| `native_max_phase_b_3114634.json` | `3114634` | Phase B CausalVAE `l2=256`, batch=4/gpu (69.6% occ, 14.22 vol/s) |
| `native_max_phase_c_3114634.json` | `3114634` | Phase C Spectral GBM 4mm, batch=56/gpu (83.3% occ) |
| `phase_b_dwsep_l512_b4_3116189.json` | `3116189` | Phase B `l2=512` DW-sep + grad checkpoint, batch=4/gpu (79.0% occ) |
| `phase_c_batch64_3116189.json` | `3116189` | Phase C Spectral GBM 4mm, batch=64/gpu (95.1% occ) |
| `phase_c_2mm_b6_3116189.json` | `3116189` | Phase C Spectral GBM 2mm `(96, 112, 80)`, batch=6/gpu (73.5% occ) |
