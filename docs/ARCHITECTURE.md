# System Architecture

This document details the architectural design of the Causal AI Framework for Prostate Cancer.

## 1. High-Level Overview

The system is designed as a "Digital Twin" pipeline that takes multimodal clinical data (MRI, demographics, biopsies) and learns a causal model of disease progression. It consists of four distinct phases:

1.  **Supervision**: Grounding the model in clinical truth (Segmentation, Grading).
2.  **Disentanglement**: Separating causal factors (Disease vs. Anatomy vs. Scanner).
3.  **Dynamics**: Modeling temporal evolution with interventions.
4.  **Validation**: Simulating counterfactuals.

## 2. Model Components

### 2.1 Supervisor Models (`src/models/supervisors.jl`)

These models are trained first to provide "silver standard" labels and feature extractors.

*   **SegmentationSupervisor**: A 3D U-Net that predicts pixel-wise class probabilities (Background, Peripheral Zone, Transition Zone, Tumor).
*   **OrdinalSupervisor**: A 3D CNN regressor/classifier optimized with ordinal loss to predict Gleason Grade Groups (0-5).
*   **SurvivalSupervisor**: A DeepSurv-like architecture that outputs a risk score from imaging features, trained with Cox Partial Likelihood.

### 2.2 Causal Disentangled VAE (`src/models/vae.jl`)

Based on the SDNet architecture, this VAE enforces a structured latent space:

*   **Encoders**:
    *   `E_anatomy(mask) -> s`: Encodes the segmentation mask into a spatial tensor.
    *   `E_pathology(image) -> z_p`: Encodes disease texture into a vector.
    *   `E_style(image) -> z_s`: Encodes scanner artifacts.
*   **Decoder**:
    *   `D(s, z_p, z_s) -> image`: Reconstructs the image. Ideally uses SPADE normalization to inject `s` at multiple scales.

### 2.3 Out-of-Distribution Detector (`src/models/ood.jl`)

A lightweight Autoencoder trained on the latent vectors `z_p`. High reconstruction error indicates a sample that deviates from the learned manifold of "plausible disease states", flagging it for review.

### 2.4 Neural Jump ODE (`src/models/njde.jl`)

The core temporal engine.

*   **State**: The disease latent vector `z_p(t)`.
*   **Dynamics**: `dz/dt = f(z, t, c)` modeled by a Neural Network (MLP).
*   **Jumps**: At intervention times $t_k$ (biopsy), the state updates instantaneously: `z(t_k+) = z(t_k-) + g(z, intervention_params)`.
*   **Solver**: `DifferentialEquations.jl` (Tsit5) with `SciMLSensitivity` for adjoint gradient backpropagation.

## 3. Training Workflow

1.  **Data Gen**: Synthetic NIfTI volumes are generated with known "ground truth" tumor blobs and scanner biases.
2.  **Phase 1**: Train Supervisors to convergence.
3.  **Phase 2**: Train VAE. The Anatomy encoder is frozen or guided by the Supervisor's output.
4.  **Phase 3**: Train OOD on the resulting `z_p` vectors.
5.  **Phase 4**: Extract `z_p` sequences from longitudinal data and train the NJDE to predict future states.

## 4. Implementation Details (Julia)

*   **Lux.jl**: Used for all neural network definitions. Explicit parameter handling (`ps`, `st`) fits well with differential equation solvers.
*   **Zygote.jl**: Handles automatic differentiation.
*   **TensorBoardLogger.jl**: Used for metrics visualization.
