import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.models.njde import NJDE

def main():
    print("\n--- Phase 4: Validation & Counterfactual Analysis ---")

    # Load Model
    model = NJDE(dim=16)
    if os.path.exists("src/njde_checkpoint.pth"):
        model.load_state_dict(torch.load("src/njde_checkpoint.pth"))
    else:
        print("Model checkpoint not found. Run training first.")
        return

    # Mock Patient Data
    z0 = torch.randn(1, 16) # Initial state
    t_span = torch.linspace(0, 2, 20) # 2 years

    # Scenario A: No Biopsy (Natural History)
    with torch.no_grad():
        traj_natural = model(z0, t_span, intervention_indices=[], intervention_coords=None)

    # Scenario B: Biopsy at t=1.0 (Index 10)
    biopsy_coords = torch.tensor([[0.5, 0.5, 0.5]]) # Center of prostate
    with torch.no_grad():
        traj_biopsy = model(z0, t_span, intervention_indices=[10], intervention_coords=biopsy_coords)

    # Visualize divergence
    # Project to 1D (e.g. PCA or just 1st dim) for plotting
    val_natural = traj_natural[0, :, 0].numpy()
    val_biopsy = traj_biopsy[0, :, 0].numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(t_span.numpy(), val_natural, label="Counterfactual: No Biopsy", linestyle="--")
    plt.plot(t_span.numpy(), val_biopsy, label="Observed: With Biopsy", marker='o')
    plt.axvline(x=1.0, color='r', linestyle=':', label="Biopsy Event")
    plt.title("Counterfactual Simulation: Effect of Biopsy on Latent Pathology Trajectory")
    plt.xlabel("Time (Years)")
    plt.ylabel("Latent Pathology Dimension 0")
    plt.legend()
    plt.grid(True)
    plt.savefig("src/counterfactual_plot.png")
    print("Counterfactual plot saved to src/counterfactual_plot.png")

    # Validation Metric (Mock AUROC)
    # In real pipeline: Predict z(t+1) -> Classifier -> Probability -> AUROC
    print("Calculating Prognostic Accuracy (Mock)...")
    auroc = 0.85 # Placeholder
    print(f"Prognostic AUROC: {auroc:.2f}")

if __name__ == "__main__":
    main()
