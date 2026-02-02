import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

# Mock Dataset
class Mock3DDataset(Dataset):
    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 3D Volume: [C, D, H, W]
        return torch.randn(1, 16, 48, 48)

# Simple 3D CNN
class Simple3DNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        # self.fc = nn.Linear(32 * 16 * 48 * 48, 10) # Removed unused layer to fix DDP error

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return x.mean() # Scalar output

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = output # Dummy loss
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--gpus", type=int, default=0) # 0 for CPU
    parser.add_argument("--accelerator", type=str, default="cpu")
    parser.add_argument("--strategy", type=str, default="ddp")
    parser.add_argument("--num_processes", type=int, default=1)
    args = parser.parse_args()

    print(f"Starting PyTorch Lightning Training with {args.num_processes} processes (Strategy: {args.strategy})...")

    dataset = Mock3DDataset()
    dataloader = DataLoader(dataset, batch_size=4)

    model = Simple3DNet()

    # Configure Trainer
    # Use find_unused_parameters=True if needed, but better to remove unused params.
    # I removed self.fc, so hopefully it works now.
    # But to be safe, let's explicitly enable it if the error persists or for robustness in this mock.

    strategy = args.strategy
    if strategy == "ddp":
        strategy = DDPStrategy(find_unused_parameters=True)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        strategy=strategy,
        devices=args.num_processes if args.accelerator == "cpu" else args.gpus,
        enable_progress_bar=False,
        logger=False
    )

    start_time = time.time()
    trainer.fit(model, dataloader)
    end_time = time.time()

    if trainer.global_rank == 0:
        print(f"Training finished in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
