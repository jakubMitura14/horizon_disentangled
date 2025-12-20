import torch
import torch.nn as nn
from src.models.njde import NJDE

def main():
    print("\n--- Training Neural Jump ODE ---")

    # Mock longitudinal data: 10 patients, 3 timepoints, 16 latent dim
    batch_size = 10
    timepoints = 3
    dim = 16

    # z_pathology sequence (Ground Truth)
    z_true = torch.randn(batch_size, timepoints, dim)
    t_span = torch.tensor([0.0, 0.5, 1.0])

    # Interventions (e.g., biopsy at t=1)
    # (T, B, 1)
    interventions = torch.zeros(timepoints, batch_size, 1)
    interventions[1, :, :] = 1.0 # Intervention at t=0.5 (index 1)

    model = NJDE(dim=dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for epoch in range(5):
        optimizer.zero_grad()
        z0 = z_true[:, 0, :] # Initial state

        # Predict trajectory
        z_pred = model(z0, t_span, interventions=interventions)

        # Loss (MSE)
        loss = nn.MSELoss()(z_pred, z_true)
        loss.backward()
        optimizer.step()

        print(f"NJDE Epoch {epoch+1} Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "src/njde_checkpoint.pth")
    print("NJDE Trained.")

if __name__ == "__main__":
    main()
