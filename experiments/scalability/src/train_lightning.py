import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

# Mock Dataset (Heavier)
class Mock3DDataset(Dataset):
    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 3D Volume: [C, D, H, W] -> 96x96x64
        return torch.randn(1, 64, 96, 96)

class ResNetBlockBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        mid_channels = out_channels
        final_channels = out_channels * self.expansion

        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_channels)

        self.conv2 = nn.Conv3d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_channels)

        self.conv3 = nn.Conv3d(mid_channels, final_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(final_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != final_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, final_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(final_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)

class HeavyResNet3D_Deep(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # ResNet-50 3D structure: [3, 4, 6, 3] blocks
        self.in_channels = 64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * 4, 10) # Expansion=4

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(ResNetBlockBottleneck(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * 4
        for _ in range(1, blocks):
            layers.append(ResNetBlockBottleneck(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = output.mean()
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="cpu")
    parser.add_argument("--strategy", type=str, default="ddp")
    parser.add_argument("--num_processes", type=int, default=1)
    args = parser.parse_args()

    num_nodes = args.nodes
    if "SLURM_NNODES" in os.environ:
        num_nodes = int(os.environ["SLURM_NNODES"])

    dataset = Mock3DDataset()
    dataloader = DataLoader(dataset, batch_size=4)

    model = HeavyResNet3D_Deep()

    strategy = args.strategy
    if strategy == "ddp":
        if args.accelerator == "gpu":
            strategy = DDPStrategy(find_unused_parameters=False, process_group_backend="nccl")
        else:
            strategy = DDPStrategy(find_unused_parameters=False, process_group_backend="gloo")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        strategy=strategy,
        devices=args.num_processes if args.accelerator == "cpu" else args.gpus,
        num_nodes=num_nodes,
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
