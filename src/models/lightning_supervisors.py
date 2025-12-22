import pytorch_lightning as pl
import torch
import torch.nn as nn
from monai.losses import DiceCELoss
from torch.optim import Adam
from src.models.supervisor import SegmentationSupervisor
from src.models.ordinal import OrdinalSupervisor, ordinal_loss
from src.models.survival import SurvivalSupervisor, cox_loss

class SegmentationModule(pl.LightningModule):
    """
    LightningModule for the Segmentation Supervisor.
    """
    def __init__(self, in_channels=2, out_channels=2, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = SegmentationSupervisor(in_channels, out_channels)
        self.loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs = torch.cat([batch["t2w"], batch["adc"]], dim=1)
        labels = batch["seg"]
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        self.log("train_seg_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)


class OrdinalModule(pl.LightningModule):
    """
    LightningModule for the Ordinal Supervisor.
    """
    def __init__(self, in_channels=2, num_classes=5, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = OrdinalSupervisor(in_channels, num_classes)
        # Mapping Gleason: 0->0, 6->1, 7->2, 8->3, 9->4
        self.gleason_map = {0: 0, 6: 1, 7: 2, 8: 3, 9: 4}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs = torch.cat([batch["t2w"], batch["adc"]], dim=1)
        raw_g = batch["gleason"].tolist()
        targets = torch.tensor([self.gleason_map.get(g, 0) for g in raw_g]).to(self.device)

        logits = self(inputs)
        loss = ordinal_loss(logits, targets)
        self.log("train_ordinal_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)


class SurvivalModule(pl.LightningModule):
    """
    LightningModule for the Survival Supervisor.
    """
    def __init__(self, in_channels=2, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = SurvivalSupervisor(in_channels)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs = torch.cat([batch["t2w"], batch["adc"]], dim=1)
        times = batch["time"].float()
        events = batch["event"].float()

        risk_scores = self(inputs)
        # Flatten risk
        loss = cox_loss(risk_scores.view(-1), times, events)
        self.log("train_survival_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)
