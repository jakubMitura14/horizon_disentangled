import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam
from src.models.causal_vae import CausalVAE

class CausalVAEModule(pl.LightningModule):
    """
    LightningModule for Causal VAE (Disentanglement).
    """
    def __init__(self, latent_dim=16, anatomy_dim=4, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = CausalVAE(latent_dim, anatomy_dim)
        # We manually manage optimizers for adversarial training
        self.automatic_optimization = False

    def forward(self, img, mask):
        return self.model(img, mask)

    def training_step(self, batch, batch_idx):
        img = torch.cat([batch["t2w"], batch["adc"]], dim=1)
        mask = batch["seg"]

        opt_vae, opt_disc = self.optimizers()

        # --- 1. Train Discriminator ---
        # Predict mask from z_p
        with torch.no_grad():
            mu_p, log_p = self.model.enc_p(img)
            z_p = self.model.reparameterize(mu_p, log_p)

        pred_mask = self.model.discriminator(z_p)
        # Resize target mask to discriminator output size (e.g. 24x24x8 from logic)
        # Discr output is 24x24x8. Input mask is 48x48x16.
        target_mask_small = torch.nn.functional.interpolate(mask, size=pred_mask.shape[2:])

        disc_loss = nn.MSELoss()(pred_mask, target_mask_small)

        opt_disc.zero_grad()
        self.manual_backward(disc_loss)
        opt_disc.step()
        self.log("train_disc_loss", disc_loss, prog_bar=True)

        # --- 2. Train VAE ---
        recon, mu_p, log_p, mu_s, log_s, s, pred_mask_adv = self(img, mask)

        # Recon shape check
        if recon.shape != img.shape:
            recon = torch.nn.functional.interpolate(recon, size=img.shape[2:])

        recon_loss = nn.L1Loss()(recon, img)
        kl_p = -0.5 * torch.sum(1 + log_p - mu_p.pow(2) - log_p.exp())
        kl_s = -0.5 * torch.sum(1 + log_s - mu_s.pow(2) - log_s.exp())

        # Adv Loss: maximize MSE(Pred, Mask) -> minimize -MSE
        # We want z_p to be uninformative about mask
        target_mask_small_adv = torch.nn.functional.interpolate(mask, size=pred_mask_adv.shape[2:])
        adv_loss = -nn.MSELoss()(pred_mask_adv, target_mask_small_adv)

        vae_loss = recon_loss + 0.1 * (kl_p + kl_s) + 0.01 * adv_loss

        opt_vae.zero_grad()
        self.manual_backward(vae_loss)
        opt_vae.step()
        self.log("train_vae_loss", vae_loss, prog_bar=True)

    def configure_optimizers(self):
        opt_vae = Adam(self.model.parameters(), lr=self.hparams.lr)
        opt_disc = Adam(self.model.discriminator.parameters(), lr=self.hparams.lr)
        return [opt_vae, opt_disc], []
