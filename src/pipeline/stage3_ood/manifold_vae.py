import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class ResBlockMLP(nn.Module):
    """Residual MLP block with LayerNorm and GELU."""
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.act1(self.norm1(self.fc1(x)))
        out = self.norm2(self.fc2(out))
        return self.act2(out + res)

class NormalityManifoldVAE(nn.Module):
    """
    4-layer residual MLP Normality Manifold VAE (V_OOD, bottleneck dim 32).
    Trained on clean purified latent vectors:
    zeta_i = [mu_D^glob (64), mu_P (32), log_var_D (64), q_stage1 (5)] = 165-D
    """
    def __init__(self, in_dim: int = 165, hidden_dim: int = 128, bottleneck_dim: int = 32):
        super().__init__()
        self.in_dim = in_dim
        self.bottleneck_dim = bottleneck_dim

        # Encoder
        self.enc_in = nn.Linear(in_dim, hidden_dim)
        self.enc_res1 = ResBlockMLP(hidden_dim)
        self.enc_res2 = ResBlockMLP(hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, bottleneck_dim)
        self.fc_logvar = nn.Linear(hidden_dim, bottleneck_dim)

        # Decoder
        self.dec_in = nn.Linear(bottleneck_dim, hidden_dim)
        self.dec_res1 = ResBlockMLP(hidden_dim)
        self.dec_res2 = ResBlockMLP(hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, in_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, zeta: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encode
        h = F.gelu(self.enc_in(zeta))
        h = self.enc_res1(h)
        h = self.enc_res2(h)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -10.0, 10.0)
        z = self.reparameterize(mu, logvar)

        # Decode
        h_dec = F.gelu(self.dec_in(z))
        h_dec = self.dec_res1(h_dec)
        h_dec = self.dec_res2(h_dec)
        zeta_hat = self.dec_out(h_dec)

        # Loss components
        recon_loss = F.mse_loss(zeta_hat, zeta, reduction='none').sum(dim=-1) # (B,)
        kl_loss = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1) # (B,)

        return {
            "zeta_hat": zeta_hat,
            "latent_z": z,
            "mu": mu,
            "logvar": logvar,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss
        }
