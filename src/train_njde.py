import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from src.models.njde import NJDE

def main():
    print("\n--- Training Neural Jump ODE (Phase 3) ---")

    data_dir = "src/mock_data"
    # Ensure data exists
    if not os.path.exists(os.path.join(data_dir, "clinical_data.csv")):
        from src.data.mock_data import generate_longitudinal_dataset
        generate_longitudinal_dataset(data_dir)

    df = pd.read_csv(os.path.join(data_dir, "clinical_data.csv"))

    # Organize data by patient
    patient_ids = df['patient_id'].unique()
    patients = []
    for pid in patient_ids:
        rows = df[df['patient_id'] == pid].sort_values('time_months')
        # Extract features
        # In real pipeline, we would load images and run VAE encoder to get z_p.
        # Here, we mock z_p for speed, but use the structure.

        # Mock z_p sequence
        timesteps = len(rows)
        z_p = torch.randn(timesteps, 16) # (T, D)
        times = torch.tensor(rows['time_months'].values).float() / 12.0 # Year units

        # Biopsy info
        # intervention_indices (list of indices where biopsy happened)
        # intervention_coords (tensor of coords)
        int_indices = []
        int_coords = []

        for i, (idx, row) in enumerate(rows.iterrows()):
            if row['biopsy_performed'] == 1:
                int_indices.append(i) # Jump at this timestep
                # Parse coords "x,y,z"
                try:
                    c = [float(x) for x in row['biopsy_coords'].split(',')]
                    # Normalize coords approx (0-96 -> 0-1)
                    c = [val / 96.0 for val in c]
                    int_coords.append(c)
                except:
                    int_coords.append([0.5, 0.5, 0.5]) # Default

        if len(int_coords) > 0:
            int_coords_tensor = torch.tensor(int_coords).float() # (NumJumps, 3)
        else:
            int_coords_tensor = None

        patients.append({
            "z": z_p,
            "t": times,
            "int_idx": int_indices,
            "int_coords": int_coords_tensor
        })

    model = NJDE(dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Training Loop
    # We process one patient at a time (or pad for batching, here batch=1 for simplicity)
    for epoch in range(5):
        total_loss = 0
        for p in patients:
            if len(p['t']) < 2: continue # Need at least 2 points

            z_true = p['z'] # (T, D)
            t_span = p['t'] # (T)
            z0 = z_true[0].unsqueeze(0) # (1, D)

            # Prepare intervention tensor for current patient
            # The model expects intervention_coords of shape (1, 3) if batch=1
            # But NJDE forward logic needs update to handle dynamic list of jumps
            # Simplified: We only support 1 jump per sequence in this mock loop if present

            curr_coords = None
            curr_indices = []

            if len(p['int_idx']) > 0:
                # Take first biopsy for demo
                curr_indices = p['int_idx']
                # Pad/Stack coords.
                # Model expects (B, 3) for the specific jump.
                # Since we loop inside NJDE, we pass the full list?
                # Actually, my NJDE implementation takes `intervention_coords` as ONE tensor used for jumps?
                # "curr_z = self.jump(curr_z, intervention_coords)" -> implies intervention_coords is static for the jump call?
                # Wait, if there are multiple jumps, we need multiple coords.
                # Let's assume for mock training we just use the first jump coords if any.
                curr_coords = p['int_coords'][0].unsqueeze(0) # (1, 3)
                curr_indices = [p['int_idx'][0]]

            optimizer.zero_grad()

            # Predict
            z_pred = model(z0, t_span, intervention_indices=curr_indices, intervention_coords=curr_coords)

            # Loss
            loss = nn.MSELoss()(z_pred.squeeze(0), z_true)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"NJDE Epoch {epoch+1} Loss: {total_loss/len(patients):.4f}")

    torch.save(model.state_dict(), "src/njde_checkpoint.pth")
    print("NJDE Trained.")

if __name__ == "__main__":
    main()
