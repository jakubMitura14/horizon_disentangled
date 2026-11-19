import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class AnatomicalSpatialWarper(nn.Module):
    """
    Warps baseline anatomy Z_A(t_0) using 3D displacement field Phi(x, t)
    derived from integrated Darcy velocity v_c to render mass effect,
    ventricular compression, and midline shift.
    """
    def __init__(self, mode: str = "bilinear", padding_mode: str = "border"):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode

    def forward(
        self,
        z_a: torch.Tensor,                # (B, C, D, H, W)
        displacement_field: torch.Tensor  # (B, 3, D, H, W) [dz, dy, dx] in voxels
    ) -> torch.Tensor:
        B, C, D, H, W = z_a.shape
        device = z_a.device

        # Create normalized identity grid in [-1, 1]
        z_coords = torch.linspace(-1.0, 1.0, D, device=device)
        y_coords = torch.linspace(-1.0, 1.0, H, device=device)
        x_coords = torch.linspace(-1.0, 1.0, W, device=device)

        grid_z, grid_y, grid_x = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
        # Shape: (B, D, H, W, 3) in order [x, y, z]
        base_grid = torch.stack([grid_x, grid_y, grid_z], dim=-1).unsqueeze(0).expand(B, -1, -1, -1, -1)

        # Scale displacement field to [-1, 1] coordinate range
        # displacement_field is (B, 3, D, H, W) [dz, dy, dx]
        scale_x = 2.0 / max(W - 1, 1)
        scale_y = 2.0 / max(H - 1, 1)
        scale_z = 2.0 / max(D - 1, 1)

        disp_x = displacement_field[:, 2] * scale_x
        disp_y = displacement_field[:, 1] * scale_y
        disp_z = displacement_field[:, 0] * scale_z

        scaled_disp = torch.stack([disp_x, disp_y, disp_z], dim=-1) # (B, D, H, W, 3)
        sampling_grid = base_grid + scaled_disp

        # 3D grid sample
        warped_za = F.grid_sample(
            z_a,
            sampling_grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=True
        )
        return warped_za
