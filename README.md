# Horizon Disentangled — Causal AI for Longitudinal Oncology

Causal AI framework integrating Neural Jump ODEs (NJDEs) and Causal Disentangled VAEs for modeling cancer progression as a "Digital Twin". Supports prostate, breast (I-SPY 2), and brain (GBM/glioma) cancer cohorts.

## Architecture

5-stage pipeline:
1. **Stage 1 — Supervisors**: Segmentation (SwinUNETR), ordinal grading (DIOR), survival analysis (LogisticHazard + SurvRNC), image-text contrastive (Merlin InfoNCE)
2. **Stage 2 — Causal VAE**: Disentangles anatomy (Z_A), disease (Z_D), patient state (Z_P), scanner style (Z_S) via FactorVAE TC + vCLUB
3. **Stage 3 — OOD Filter**: Normality Manifold VAE + k-NN latent distance scoring for anomaly detection and label noise filtering
4. **Stage 4 — Dynamics**: Neural Jump ODE for continuous progression + discrete intervention modeling (Fisher-KPP, Darcy, Linear-Quadratic RT, Extended Tofts PK)
5. **Stage 5 — Synthesis**: Counterfactual generation, treatment ranking, rectified flow refinement, VLM-powered structured reporting (MedGemma / Med3DVLM), EU AI Act audit logging

## Implementation

| Component | Language | Key Libraries |
|:---|:---|:---|
| Pipeline (training + inference) | Python 3.10+ | PyTorch, MONAI, torchdiffeq, pycox |
| Scientific models (pilot) | Julia 1.9+ | Lux.jl, DifferentialEquations.jl, SciMLSensitivity.jl |
| Preprocessing | Python | h5py, hdf5plugin, SimpleITK, nibabel, pydicom |
| Cluster training | SLURM + torchrun | 4×H200 SXM DDP, FSDP-2, FP8 (Transformer Engine) |

## Quick Start

### Python Pipeline (Real Data)
```bash
# Preprocess datasets into HDF5
python src/preprocessing/pipeline.py

# Train full pipeline on cluster
sbatch scripts/training/slurm_pipeline_cluster.sbatch

# Train on single node with 4 GPUs
bash scripts/training/slurm_train_4gpu.sh

# Run specific stage
python -m src.pipeline.cli --stage 2 --domain breast --epochs 100

# Multi-GPU with torchrun
torchrun --nproc_per_node=4 -m src.pipeline.cli --stage all --domain gbm
```

### Julia Pilot (Synthetic Data)
```bash
julia --project=src -e 'using Pkg; Pkg.instantiate()'
./run_pilot.sh
```

## Project Structure
```
├── src/
│   ├── pipeline/              # Python ML pipeline (Stages 1-5)
│   │   ├── config.py          # Hardware + training configuration
│   │   ├── orchestrator.py    # Pipeline orchestration
│   │   ├── cli.py             # Command-line interface
│   │   ├── fp8_context.py     # Mixed-precision + cluster setup
│   │   ├── data/              # Dataset + transforms
│   │   ├── stage1_supervisors/# Segmentation, grading, survival
│   │   ├── stage2_causal_vae/ # Hierarchical causal VAE
│   │   ├── stage3_ood/        # OOD detection + label noise
│   │   ├── stage4_dynamics/   # NJDE + biophysical models
│   │   └── stage5_synthesis_reporting/  # Counterfactuals + VLM
│   ├── preprocessing/         # HDF5 data preparation
│   ├── classifiers/           # Standalone classifier training
│   ├── models/                # Julia model definitions
│   ├── data/                  # Julia data generation
│   └── utils/                 # Julia utilities
├── scripts/
│   ├── training/              # SLURM jobs + training scripts
│   ├── data/                  # Data download + conversion
│   ├── setup/                 # Environment + model setup
│   ├── verification/          # Model verification scripts
│   └── tests/                 # Integration tests
├── tests/                     # pytest test suite
├── docs/                      # Documentation
├── experiments/               # Experiment configs + results
└── models/                    # Pre-trained model weights (gitignored)
```

## Datasets
- **LUMIERE**: 91 GBM patients, 638 visits, 4-sequence brain MRI (CT1, T1, T2, FLAIR)
- **I-SPY 2**: 985 breast cancer patients, longitudinal DCE-MRI with kinetic maps
- **BreastDCEDL MinCrop**: 1,088 baseline tumor crops (3-timepoint DCE)
- **CFB-GBM / MU-Glioma / Burdenko-GBM**: Clinical + radiomic feature tables

## Cluster Configuration (H200)
See [docs/h200_cluster_scaling_guide.md](docs/h200_cluster_scaling_guide.md) for NCCL tuning, DDP optimization, and SLURM configuration for 4×H200 SXM nodes.

Key optimizations:
- **DDP bucket_cap_mb=50**: Matches NVLink 4.0 bandwidth (~900 GB/s) for optimal AllReduce overlap
- **gradient_as_bucket_view=True**: Avoids gradient copy, saves ~model_size MB VRAM
- **static_graph=True**: Enables NCCL comm/compute overlap pipelining
- **channels_last_3d**: ~15% cuDNN throughput improvement for 3D convolutions on H200
- **FP8 (Transformer Engine)**: 2× throughput on attention layers (3,958 vs 1,979 BF16 TFLOPS)
- **torch.compile(mode="max-autotune")**: Profiles and selects fastest CUDA kernels per op

## Documentation
- [Architecture](docs/ARCHITECTURE.md) — Model component details
- [Dataset Requirements](docs/DATASET_REQUIREMENTS.md) — Data specifications
- [Preprocessing Plan](docs/deep_research_preprocessing_clean.md) — HDF5 pipeline design
- [Cluster Guide](docs/cluster_implementation_guide.md) — Full cluster deployment
- [Data Inventory](docs/data_inventory.md) — XNAT dataset catalogue
- [H200 Scaling Guide](docs/h200_cluster_scaling_guide.md) — GPU performance tuning
- [Pipeline Workflow](docs/PIPELINE_WORKFLOW.md) — End-to-end workflow

## License

This project is part of a research grant proposal and is currently under active development.
