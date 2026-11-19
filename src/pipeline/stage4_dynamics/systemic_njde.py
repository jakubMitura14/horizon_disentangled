import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class SystemicNeuralJumpODE(nn.Module):
    """
    Systemic Neural Jump ODE (NJ-ODE) for canonical state vector v_sys ∈ ℝ^160.
    
    Conditioned on spatial mean of U_D and time-since-last-observation (t - t_last).
    
    Upgraded architecture with:
      - 4-layer ResNet drift network with skip connections (was 3-layer MLP)
      - Deeper observation update network
      - Wider hidden dimension (512, was 256)
    
    Slots in v_sys:
      [0:64]    z_D^glob
      [64:96]   z_P
      [96:112]  genomic / receptor markers g_i
      [112:128] scalar clinical volumes & biomarkers b(t)
      [128:144] systemic therapy state c_rx(t)
      [144:160] epistemic & aleatoric variance σ²
    """
    def __init__(self, systemic_dim: int = 160, spatial_channels: int = 8, hidden_dim: int = 512):
        super().__init__()
        self.systemic_dim = systemic_dim
        self.spatial_channels = spatial_channels

        # Input: v_sys (160) + pooled U_D (8) + (t - t_last) (1) = 169
        in_dim = systemic_dim + spatial_channels + 1

        # 4-layer ResNet drift network with skip connections
        self.drift_in = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.drift_res1 = ResMLPBlock(hidden_dim)
        self.drift_res2 = ResMLPBlock(hidden_dim)
        self.drift_res3 = ResMLPBlock(hidden_dim)
        self.drift_out = nn.Linear(hidden_dim, systemic_dim)
        # Initialize output layer near zero for stable training
        nn.init.zeros_(self.drift_out.weight)
        nn.init.zeros_(self.drift_out.bias)

        # Observation update network for NJ-ODE discrete observation shock
        self.obs_update_net = nn.Sequential(
            nn.Linear(systemic_dim + 16, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, systemic_dim),
        )

    def forward_drift(
        self,
        v_sys: torch.Tensor,              # (B, 160)
        u_d_pooled: torch.Tensor,         # (B, 8)
        dt_last: torch.Tensor             # (B, 1)
    ) -> torch.Tensor:
        """Evaluates dv_sys / dt using ResNet drift network."""
        x = torch.cat([v_sys, u_d_pooled, dt_last], dim=-1)
        h = self.drift_in(x)
        h = self.drift_res1(h)
        h = self.drift_res2(h)
        h = self.drift_res3(h)
        return self.drift_out(h)

    def apply_observation_update(
        self,
        v_sys: torch.Tensor,
        obs_features: torch.Tensor
    ) -> torch.Tensor:
        """NJ-ODE jump update at observation timestamp."""
        x = torch.cat([v_sys, obs_features], dim=-1)
        delta_v = self.obs_update_net(x)
        return v_sys + delta_v


class ResMLPBlock(nn.Module):
    """Residual MLP block with LayerNorm and GELU."""
    def __init__(self, dim: int, mult: float = 2.0, dropout: float = 0.1):
        super().__init__()
        inner = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
            nn.LayerNorm(dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)
