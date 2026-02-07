# Pipeline Workflow

This document describes the step-by-step execution of the Causal AI Framework pipeline.

## 1. Data Download & Preprocessing
**Script:** `src/data/download_and_preprocess.jl`
- Downloads datasets from Zenodo (Prostate158, PI-CAI) and TCIA (biopsy, repeatability).
- Converts DICOM files to NIfTI format using `dcm2niix`.
- Identifies T2W, ADC, and Segmentation files.
- Generates `src/data_store/clinical_data.csv` linking patient IDs to file paths.

## 2. Supervisor Training
**Script:** `src/train_supervisors.jl`
- **Goal:** Train independent models for specific tasks (Segmentation, Gleason Grading, Survival).
- **Model:** 3D U-Net (or similar).
- **Input:** T2W + ADC MRI volumes.
- **Output:** Trained supervisor models (saved checkpoints).

## 3. Causal VAE Training
**Script:** `src/train_vae.jl`
- **Goal:** Learn a disentangled latent representation (`S`, `Z_p`, `Z_s`).
- **Model:** SDNet-based VAE with spatial anatomy tensor.
- **Input:** Images + Supervisor predictions.
- **Output:** Trained VAE model.

## 4. OOD Detection Training
**Script:** `src/train_ood.jl`
- **Goal:** Detect out-of-distribution samples (e.g. artifacts, healthy controls) to gate the causal model.
- **Model:** Autoencoder on latent space.
- **Output:** OOD detector model.

## 5. Neural Jump ODE (NJDE) Training
**Script:** `src/train_njde.jl`
- **Goal:** Model disease progression and causal effects of interventions (biopsies).
- **Model:** Neural ODE with discrete jumps (events).
- **Input:** Latent trajectories.
- **Output:** Trained NJDE model.

## 6. Validation
**Script:** `src/validate_counterfactual.jl`
- **Goal:** Verify causal reasoning capabilities.
- **Method:** Generate counterfactual trajectories (e.g. "What if biopsy was not performed?").
- **Output:** Plots and metrics.

## Execution
Run the full pipeline using:
```bash
./run_pilot.sh
```
Env vars:
- `DEBUG_MODE=true`: Run on partial data for testing.
- `CUDA_VISIBLE_DEVICES=1`: Select specific GPU.
