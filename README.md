# Causal AI Framework for Prostate Cancer Pilot Study

This repository hosts the **Julia/SciML implementation** of a pilot study for a Causal AI Framework designed to model prostate cancer progression. The framework integrates **Neural Jump Ordinary Differential Equations (NJDEs)** and **Causal Disentangled Variational Autoencoders (CD-VAEs)** to create a "Digital Twin" of the patient, enabling counterfactual simulations for personalized treatment planning.

## Key Features

*   **Causal Disentanglement (CD-VAE):** Separates anatomical features from disease pathology and scanner-specific artifacts using **Lux.jl**.
*   **Temporal Modeling (NJDE):** Models continuous disease evolution and discrete interventions (biopsies) using **DifferentialEquations.jl** and **SciMLSensitivity.jl**.
*   **Supervisor Models:** Includes U-Net for segmentation, ordinal classifiers for Gleason grading, and DeepSurv-like models for survival analysis.
*   **Out-of-Distribution Detection:** VAE-based anomaly detection to flag inconsistencies or label noise.
*   **Logging:** Integration with **TensorBoard** for tracking training metrics.

## Prerequisites

*   **Julia**: Version 1.9 or higher.
*   **Bash**: For running the orchestration script.

## Quick Start

The entire pilot study pipeline, from synthetic data generation to counterfactual validation, can be executed using the provided shell script.

### 1. Instantiate the Environment

Before running the pipeline, ensure the Julia environment is instantiated.

```bash
julia --project=src -e 'using Pkg; Pkg.instantiate()'
```

### 2. Run the Full Pipeline

Execute the `run_pilot.sh` script to run all phases sequentially:

```bash
./run_pilot.sh
```

This script performs the following steps:
1.  **Data Generation**: Creates synthetic longitudinal 3D MRI data (NIfTI) and clinical records in `src/mock_data`.
2.  **Supervisor Training**: Trains segmentation, ordinal, and survival models. Logs to `logs/supervisors`.
3.  **VAE Training**: Trains the Causal VAE. Logs to `logs/vae`.
4.  **OOD Training**: Trains the Out-of-Distribution detector. Logs to `logs/ood`.
5.  **NJDE Training**: Trains the Neural Jump ODE. Logs to `logs/njde`.
6.  **Validation**: Performs a counterfactual analysis ("Natural History" vs. "Biopsy Intervention") and generates a plot at `src/counterfactual_plot.png`.

## Documentation

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for a detailed technical breakdown of the model architectures and design choices.

## Project Structure

*   `src/`: Contains all source code and Julia project configuration (`Project.toml`).
    *   `data/`: Data generation logic.
    *   `models/`: Lux.jl model definitions (VAE, NJDE, Supervisors, OOD).
    *   `train_*.jl`: Training scripts for each component with TensorBoard logging.
    *   `validate_counterfactual.jl`: Validation script.
*   `run_pilot.sh`: Master script to run the end-to-end pipeline.
*   `logs/`: TensorBoard logs generated during training.

## License

This project is part of a research grant proposal and is currently under active development.
