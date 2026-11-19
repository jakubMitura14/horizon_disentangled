import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

class FlowVelocityNet(nn.Module):
    """Predicts velocity field v_theta(z_tau, tau, c) for latent flow matching."""
    def __init__(self, in_channels: int = 4, cond_dim: int = 64, hidden_channels: int = 32):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.time_emb = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.cond_proj = nn.Linear(cond_dim, hidden_channels)
        
        self.res1 = nn.Sequential(
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU()
        )
        self.res2 = nn.Sequential(
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU()
        )
        self.conv_out = nn.Conv3d(hidden_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor, tau: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # z: (B, C, D, H, W), tau: (B, 1), cond: (B, cond_dim)
        h = self.conv_in(z)
        t_emb = self.time_emb(tau).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        c_emb = self.cond_proj(cond).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        h = h + t_emb + c_emb
        h = self.res1(h)
        h = self.res2(h)
        return self.conv_out(h)

class LatentRectifiedFlowRefiner(nn.Module):
    """
    3D Latent Rectified Flow Refiner:
    Refines coarse VAE decodings along straight-line flow paths,
    integrating 15-25 Euler flow steps to restore sharp anatomical gradients.
    """
    def __init__(
        self,
        channels: int = 4,
        cond_dim: int = 64,
        num_steps: int = 20,
        tau_0: float = 0.35
    ):
        super().__init__()
        self.num_steps = num_steps
        self.tau_0 = tau_0
        self.velocity_net = FlowVelocityNet(in_channels=channels, cond_dim=cond_dim)

    def refine_latent(
        self,
        z_coarse: torch.Tensor, # (B, C, D, H, W)
        condition: torch.Tensor  # (B, cond_dim) e.g. z_d_glob
    ) -> torch.Tensor:
        """
        Integrates straight-line flow from tau_0 to 1.0 using Euler steps:
        z_tau0 = (1 - tau_0) * z_coarse + tau_0 * eps
        z_{tau + dtau} = z_tau + dtau * v_theta(z_tau, tau, cond)
        """
        B = z_coarse.size(0)
        device = z_coarse.device

        eps = torch.randn_like(z_coarse)
        z_curr = (1.0 - self.tau_0) * z_coarse + self.tau_0 * eps

        tau_start = self.tau_0
        tau_end = 1.0
        dtau = (tau_end - tau_start) / self.num_steps

        for step in range(self.num_steps):
            t_val = tau_start + step * dtau
            t_tensor = torch.full((B, 1), t_val, device=device)
            v_pred = self.velocity_net(z_curr, t_tensor, condition)
            z_curr = z_curr + dtau * v_pred

        return z_curr
