# Pilot Study: Causal AI Framework for Prostate Cancer (Julia/SciML)

This repository contains the **Julia implementation** of the pilot study for the Causal AI Framework, utilizing Neural Jump ODEs (NJDE) and Causal Disentangled VAEs.

## Overview

The framework aims to create a "Digital Twin" of the patient's prostate to model disease progression and simulate counterfactuals (e.g., "What if we didn't biopsy?"). It leverages the scientific machine learning (SciML) ecosystem for robust differential equation modeling.

## Architecture

The system consists of four main phases, implemented using **Lux.jl** and **DifferentialEquations.jl**:

1.  **Supervisor Models** (`src/models/supervisors.jl`):
    *   **Segmentation**: U-Net to segment prostate anatomy (PZ, TZ).
    *   **Ordinal**: Classifier for Gleason/PI-RADS scores.
    *   **Survival**: DeepSurv-like model for time-to-event prediction.

2.  **Causal VAE** (`src/models/vae.jl`):
    *   **SDNet Architecture**: Disentangles the latent space into:
        *   `s` (Anatomy): Spatial tensor derived from segmentation masks.
        *   `z_p` (Pathology): Vector encoding disease state.
        *   `z_s` (Style): Vector encoding scanner artifacts.
    *   Uses convolutional encoders and decoders.

3.  **Out-of-Distribution (OOD) Detection** (`src/models/ood.jl`):
    *   A simple Autoencoder trained on the latent vectors `z_p`.
    *   Uses reconstruction error to flag anomalies.

4.  **Neural Jump ODE (NJDE)** (`src/models/njde.jl`):
    *   Models the continuous evolution of `z_p` over time using **DifferentialEquations.jl**.
    *   Explicitly models interventions (biopsies) as discrete **"Jumps"** ($z^+ = z^- + g(z, coords)$) using `PresetTimeCallback`.

## Project Structure

```
src/
├── data/
│   └── mock_data.jl       # Generates synthetic longitudinal 3D MRI data (NIfTI + CSV)
├── models/
│   ├── layers.jl          # Common Lux layers (ResNetBlock, SPADE)
│   ├── supervisors.jl     # Segmentation/Ordinal/Survival models
│   ├── vae.jl             # Causal VAE architecture
│   ├── njde.jl            # Neural Jump ODE with SciML
│   └── ood.jl             # OOD Detector
├── train_supervisors.jl   # Script to train supervisors
├── train_vae.jl           # Script to train VAE
├── train_ood.jl           # Script to train OOD
├── train_njde.jl          # Script to train NJDE
└── validate_counterfactual.jl # Script for "What-If" analysis
```

## Running the Pilot

To run the entire end-to-end pipeline on minimal synthetic data:

```bash
./run_pilot.sh
```

This script will:
1.  Generate mock data in `src/mock_data`.
2.  Train all models sequentially using Julia.
3.  Produce a counterfactual analysis plot in `src/counterfactual_plot.png`.

## Requirements

*   Julia >= 1.9
*   `Lux.jl`
*   `DifferentialEquations.jl`
*   `SciMLSensitivity.jl`
*   `NIfTI.jl`
*   `CSV.jl`, `DataFrames.jl`
