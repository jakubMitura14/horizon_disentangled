import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from monai.data import CacheDataset
from monai.transforms import Resize

from src.preprocessing.pipeline import get_preprocessing_pipeline
from src.models.causal_vae import CausalVAE
from src.models.vae_loss import vae_loss

def main():
    data_dir = "src/mock_data"
    # Ensure data exists
    if not os.path.exists(os.path.join(data_dir, "clinical_data.csv")):
        from src.data.mock_data import generate_longitudinal_dataset
        generate_longitudinal_dataset(data_dir)

    df = pd.read_csv(os.path.join(data_dir, "clinical_data.csv"))

    data_dicts = []
    for idx, row in df.iterrows():
        data_dicts.append({
            "t2w": row['t2w_path'],
            "adc": row['adc_path'],
            "seg": row['seg_path'],
        })

    # Resize to match VAE expectations (48x48x16 used in model for simplicity, but mock data is 96x96x32)
    # The model's SPADE blocks and interpolations assume 48x48x16 output.
    # Actually, let's adjust transform to resize to 48x48x16 for speed.
    transforms = get_preprocessing_pipeline(roi_size=(48, 48, 16))

    dataset = CacheDataset(data=data_dicts, transform=transforms, cache_rate=0.0)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Init VAE
    model = CausalVAE(latent_dim=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Discriminator Optimizer (for adversarial part)
    # We train discriminator to PREDICT mask from z_p.
    # The VAE tries to MAXIMIZE this loss (fool discriminator).
    disc_opt = torch.optim.Adam(model.discriminator.parameters(), lr=1e-3)

    print("\n--- Training Causal VAE (Phase 2) ---")
    for epoch in range(2): # Short epoch for demo
        model.train()
        total_loss = 0
        for batch in dataloader:
            img = torch.cat([batch["t2w"], batch["adc"]], dim=1).to(device)
            mask = batch["seg"].to(device)

            # --- 1. Train Discriminator ---
            disc_opt.zero_grad()
            # Forward VAE (detach z_p so VAE doesn't update)
            with torch.no_grad():
                mu_p, log_p = model.enc_p(img)
                z_p = model.reparameterize(mu_p, log_p)

            pred_mask = model.discriminator(z_p)
            # Resize target mask to discriminator output size (e.g. 24x24x8 from logic)
            # Discr output is 24x24x8. Input mask is 48x48x16.
            target_mask_small = torch.nn.functional.interpolate(mask, size=pred_mask.shape[2:])
            disc_loss = torch.nn.MSELoss()(pred_mask, target_mask_small)
            disc_loss.backward()
            disc_opt.step()


            # --- 2. Train VAE ---
            optimizer.zero_grad()
            recon, mu_p, log_p, mu_s, log_s, s, pred_mask_adv = model(img, mask)

            # Recon shape check
            if recon.shape != img.shape:
                recon = torch.nn.functional.interpolate(recon, size=img.shape[2:])

            # Main Loss
            # Adv loss: we want pred_mask_adv to be BAD (high MSE with real mask)?
            # Or usually maximize entropy. Here: minimize -MSE (maximize MSE).
            # Simplified: Use vae_loss helper

            # Note: vae_loss helper in src/models/vae_loss.py might need update for adversarial term
            # For now, let's manually compute here or update vae_loss.py.
            # I will define loss here.

            recon_loss = torch.nn.L1Loss()(recon, img)
            kl_p = -0.5 * torch.sum(1 + log_p - mu_p.pow(2) - log_p.exp())
            kl_s = -0.5 * torch.sum(1 + log_s - mu_s.pow(2) - log_s.exp())

            # Adv Loss: We want Discriminator to FAIL to predict mask.
            # So we minimize ||Pred - Random||? Or maximize ||Pred - Mask||?
            # Standard approach: Generator minimizes log(1-D(G(z))).
            # Here: Minimize MSE(Pred, Uniform) or just Maximize MSE(Pred, Mask).
            # Let's maximize MSE(Pred, Mask) with a cap.
            target_mask_small = torch.nn.functional.interpolate(mask, size=pred_mask_adv.shape[2:])
            adv_loss = -torch.nn.MSELoss()(pred_mask_adv, target_mask_small)

            loss = recon_loss + 0.1 * (kl_p + kl_s) + 0.01 * adv_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"VAE Epoch {epoch+1} Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "src/vae_checkpoint.pth")
    print("VAE Trained.")

if __name__ == "__main__":
    main()
