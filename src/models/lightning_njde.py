import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam
from src.models.njde import NJDE

class NJDEModule(pl.LightningModule):
    """
    LightningModule for Neural Jump ODE.
    """
    def __init__(self, dim=16, lr=1e-2):
        super().__init__()
        self.save_hyperparameters()
        self.model = NJDE(dim)

    def forward(self, z0, t_span, intervention_indices=None, intervention_coords=None):
        return self.model(z0, t_span, intervention_indices, intervention_coords)

    def training_step(self, batch, batch_idx):
        # Batch: { "z": (B, T, D), "t": (B, T), "int_idx": list of lists, "int_coords": (B, N_jumps, 3) }
        # Simplified: We process one patient at a time (Batch size = 1)

        z_true = batch["z"][0] # (T, D)
        t_span = batch["t"][0] # (T)

        z0 = z_true[0].unsqueeze(0) # (1, D)

        # Extract interventions
        int_coords = batch["int_coords"] # Tensor

        # Logic to parse batch inputs to model args
        curr_coords = None
        curr_indices = []

        if int_coords.shape[1] > 0: # Check num jumps dim
            curr_coords = int_coords[0][0].unsqueeze(0) # First jump of first patient -> (1, 3)
            curr_indices = [1] # Simplified assumption for pilot refactor

        z_pred = self(z0, t_span, intervention_indices=curr_indices, intervention_coords=curr_coords)

        loss = nn.MSELoss()(z_pred.squeeze(0), z_true)
        self.log("train_njde_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)
