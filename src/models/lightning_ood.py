import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam
from src.models.ood_detector import OODDetector

class OODModule(pl.LightningModule):
    """
    LightningModule for OOD Detector.
    """
    def __init__(self, input_dim=16, latent_dim=4, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = OODDetector(input_dim, latent_dim)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # batch is simply the latent vectors z (B, D)
        # When using TensorDataset with DataLoader, batch is a list [x]
        if isinstance(batch, list):
            x = batch[0]
        else:
            x = batch

        recon, mu, logvar = self(x)

        recon_loss = nn.MSELoss()(recon, x)
        kld_loss = 0.01 * torch.mean(mu**2 + logvar.exp() - 1 - logvar)
        loss = recon_loss + kld_loss

        self.log("train_ood_loss", loss, prog_bar=True)
        return loss

    def on_train_end(self):
        # Fit KNN on validation or training data
        # For pilot, we fit on the last batch or we need to access datamodule
        # Simplified: We rely on manual call to fit_knn in script if needed
        pass

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)
