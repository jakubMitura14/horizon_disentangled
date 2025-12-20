import torch
import torch.nn as nn
from monai.networks.nets import UNet

class SegmentationSupervisor(nn.Module):
    """
    Module 1: The Segmentation Supervisor (Anatomical Grounding)
    Architecture: 3D U-Net
    Input: T2W + ADC (2 channels)
    Output: 3-channel segmentation probability map (Background, PZ, TZ)

    Note: For the pilot mock data, we simplify to 2 output classes (Background, Prostate).
    """
    def __init__(self, in_channels=2, out_channels=2):
        super(SegmentationSupervisor, self).__init__()
        self.unet = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2,
            norm="INSTANCE",
            act="LEAKYRELU",
        )

    def forward(self, x):
        return self.unet(x)
