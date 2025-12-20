import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from monai.data import CacheDataset

from src.preprocessing.pipeline import get_preprocessing_pipeline
from src.models.causal_vae import CausalVAE
from src.models.vae_loss import vae_loss

def main():
    data_dir = "src/mock_data"
    df = pd.read_csv(os.path.join(data_dir, "clinical_data.csv"))

    data_dicts = []
    for idx, row in df.iterrows():
        data_dicts.append({
            "t2w": row['t2w_path'],
            "adc": row['adc_path'],
            "seg": row['seg_path'],
        })

    # Reduce size for CPU verification speed
    transforms = get_preprocessing_pipeline(roi_size=(48, 48, 16))
    dataset = CacheDataset(data=data_dicts, transform=transforms, cache_rate=0.0)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init VAE
    model = CausalVAE(latent_dim=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("\n--- Training Causal VAE ---")
    for epoch in range(1):
        model.train()
        total_loss = 0
        for batch in dataloader:
            img = torch.cat([batch["t2w"], batch["adc"]], dim=1).to(device)
            mask = batch["seg"].to(device)

            optimizer.zero_grad()
            recon, mu_p, log_p, mu_s, log_s, s = model(img, mask)

            # Recon shape check: if decoder output doesn't match img, interpolate
            if recon.shape != img.shape:
                recon = torch.nn.functional.interpolate(recon, size=img.shape[2:])

            loss = vae_loss(recon, img, mu_p, log_p, mu_s, log_s, s, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"VAE Epoch {epoch+1} Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "src/vae_checkpoint.pth")
    print("VAE Trained.")

if __name__ == "__main__":
    main()
