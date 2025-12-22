# Pilot Study: Causal AI Framework for Prostate Cancer

This repository contains the implementation of the pilot study for the Causal AI Framework, utilizing Neural Jump ODEs (NJDE) and Causal Disentangled VAEs.

## Overview

The framework aims to create a "Digital Twin" of the patient's prostate to model disease progression and simulate counterfactuals (e.g., "What if we didn't biopsy?").

## Architecture

The system consists of four main phases, implemented using **PyTorch Lightning** for modularity and scalability:

1.  **Supervisor Models** (`src/models/lightning_supervisors.py`):
    *   **Segmentation**: 3D U-Net to segment prostate anatomy (PZ, TZ).
    *   **Ordinal**: Classifier for Gleason/PI-RADS scores using ordinal loss.
    *   **Survival**: DeepSurv-like model for time-to-event prediction.

2.  **Causal VAE** (`src/models/lightning_vae.py`):
    *   **SDNet Architecture**: Disentangles the latent space into:
        *   `s` (Anatomy): Spatial tensor derived from segmentation masks.
        *   `z_p` (Pathology): Vector encoding disease state.
        *   `z_s` (Style): Vector encoding scanner artifacts.
    *   Uses **SPADE** blocks for anatomy-guided generation.
    *   Includes an adversarial **Discriminator** to ensure `z_p` contains no anatomical info.

3.  **Out-of-Distribution (OOD) Detection** (`src/models/lightning_ood.py`):
    *   A simple VAE trained on the latent vectors `z_p`.
    *   Uses **Reconstruction Error** + **KNN Distance** in latent space to flag anomalies (e.g., mislabeled data).

4.  **Neural Jump ODE (NJDE)** (`src/models/lightning_njde.py`):
    *   Models the continuous evolution of `z_p` over time using an ODE solver (`torchdyn`).
    *   Explicitly models interventions (biopsies) as discrete **"Jumps"** ($z^+ = z^- + g(z, coords)$).

## Project Structure

```
src/
├── data/
│   └── mock_data.py       # Generates synthetic longitudinal 3D MRI data
├── models/
│   ├── lightning_supervisors.py # PL Modules for Phase 1
│   ├── lightning_vae.py         # PL Module for Phase 2
│   ├── lightning_ood.py         # PL Module for Phase 3
│   ├── lightning_njde.py        # PL Module for Phase 4
│   ├── causal_vae.py            # Core PyTorch VAE architecture
│   ├── njde.py                  # Core PyTorch NJDE architecture
│   ├── ood_detector.py          # Core PyTorch OOD architecture
│   └── ...
├── train_supervisor.py    # Script to train supervisors
├── train_vae.py           # Script to train VAE
├── test_ood.py            # Script to train/test OOD
├── train_njde.py          # Script to train NJDE
├── validate_counterfactual.py # Script for "What-If" analysis
└── verify_full_pipeline.py    # Legacy verification script
```

## Running the Pilot

To run the entire end-to-end pipeline on minimal synthetic data:

```bash
./run_pilot.sh
```

This script will:
1.  Generate mock data in `src/mock_data`.
2.  Train all models sequentially.
3.  Produce a counterfactual analysis plot in `src/counterfactual_plot.png`.

## Requirements

*   `torch >= 2.0`
*   `pytorch-lightning`
*   `monai`
*   `torchdyn`
*   `torchdiffeq`
*   `simpleitk`
*   `pycox`
