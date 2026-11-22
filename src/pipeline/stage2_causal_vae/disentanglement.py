"""Stage 2 — Causal Disentanglement Losses.

Enforces statistical independence between latent factors using:
- FactorVAE Total Correlation (TC) discriminator: separates biological
  latents [Z_D, Z_P] from scanner style Z_S
- Variational CLUB (vCLUB): upper-bounds mutual information I(Z_D; Z_S),
  I(Z_A; Z_S), and I(Z_D; Z_P) for asymmetric causal structure

Key classes:
    FactorVAEDiscriminator: Binary classifier for joint vs. factorized distributions.
    VariationalCLUB: Contrastive log-ratio upper bound on mutual information.
    AsymmetricCausalDisentanglement: Composite loss combining TC + vCLUB penalties.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

class FactorVAEDiscriminator(nn.Module):
    """
    FactorVAE Total Correlation Discriminator D_omega([z_D_glob, z_P], z_S).
    Classifies whether latent components come from the joint distribution q(z)
    or the factorized product of marginals q(z_D, z_P) * q(z_S).
    """
    def __init__(self, bio_dim: int = 64 + 32, style_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        in_dim = bio_dim + style_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 2) # 2-class logits: [is_joint, is_factorized]
        )

    def forward(self, z_bio: torch.Tensor, z_style: torch.Tensor) -> torch.Tensor:
        # z_bio: (B, 96), z_style: (B, 32)
        x = torch.cat([z_bio, z_style], dim=-1)
        return self.net(x)

    def compute_tc_loss(self, z_bio: torch.Tensor, z_style: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the Total Correlation penalty for the VAE encoder and the cross-entropy
        loss for training the discriminator.
        """
        B = z_bio.size(0)
        if B < 2:
            zero = torch.tensor(0.0, device=z_bio.device, requires_grad=True)
            return zero, zero

        # Joint pair
        d_joint = self.forward(z_bio, z_style) # (B, 2)
        
        # Factorized pair: permute z_style across batch
        perm_idx = torch.randperm(B, device=z_bio.device)
        z_style_perm = z_style[perm_idx]
        d_perm = self.forward(z_bio, z_style_perm) # (B, 2)

        # Discriminator objective: classify joint as 0, permuted as 1
        labels_joint = torch.zeros(B, dtype=torch.long, device=z_bio.device)
        labels_perm = torch.ones(B, dtype=torch.long, device=z_bio.device)
        d_loss = 0.5 * (F.cross_entropy(d_joint, labels_joint) + F.cross_entropy(d_perm, labels_perm))

        # VAE Total Correlation penalty: log(D_0 / D_1) = logit_0 - logit_1
        tc_loss = torch.mean(d_joint[:, 0] - d_joint[:, 1])
        return tc_loss, d_loss

class VariationalCLUB(nn.Module):
    """
    Variational Contrastive Log-ratio Upper Bound (vCLUB):
    Estimates and minimizes mutual information I(X; Y) between continuous random vectors.
    """
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, y_dim)
        )
        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, y_dim)
        )

    def get_mu_logvar(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.p_mu(x)
        logvar = torch.clamp(self.p_logvar(x), -10.0, 10.0)
        return mu, logvar

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes sample vCLUB mutual information upper bound."""
        # x: (N, x_dim), y: (N, y_dim)
        N = x.size(0)
        if N < 2:
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        mu, logvar = self.get_mu_logvar(x)
        
        # log q(y_i | x_i)
        # -(y_i - mu_i)^2 / (2 * exp(logvar_i)) - 0.5 * logvar_i
        positive = -0.5 * (((y - mu) ** 2) * torch.exp(-logvar) + logvar).sum(dim=-1).mean()

        # log q(y_j | x_i) for all i, j
        # y: (1, N, y_dim), mu: (N, 1, y_dim), logvar: (N, 1, y_dim)
        y_exp = y.unsqueeze(0)
        mu_exp = mu.unsqueeze(1)
        logvar_exp = logvar.unsqueeze(1)
        negative = -0.5 * (((y_exp - mu_exp) ** 2) * torch.exp(-logvar_exp) + logvar_exp).sum(dim=-1).mean()

        return positive - negative

    def compute_nll(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood to train the variational distribution q(y | x)."""
        mu, logvar = self.get_mu_logvar(x)
        nll = 0.5 * (((y - mu) ** 2) * torch.exp(-logvar) + logvar).sum(dim=-1).mean()
        return nll

class AsymmetricCausalDisentanglement(nn.Module):
    """
    Composite Asymmetric Causal Disentanglement Module:
    Enforces independence of Style Z_S from all biological latents (Z_D, Z_A, Z_P),
    and reduces spurious mutual information between Disease Z_D and Patient State Z_P.
    """
    def __init__(
        self,
        disease_dim: int = 64,
        patient_dim: int = 32,
        style_dim: int = 32,
        anatomy_channels: int = 4,
        beta_tc: float = 1.5,
        gamma_club: float = 1.0,
        alpha_dp: float = 0.5
    ):
        super().__init__()
        self.beta_tc = beta_tc
        self.gamma_club = gamma_club
        self.alpha_dp = alpha_dp

        self.tc_disc = FactorVAEDiscriminator(bio_dim=disease_dim + patient_dim, style_dim=style_dim)
        self.club_ds = VariationalCLUB(x_dim=disease_dim, y_dim=style_dim)
        self.club_as = VariationalCLUB(x_dim=anatomy_channels, y_dim=style_dim)
        self.club_dp = VariationalCLUB(x_dim=disease_dim, y_dim=patient_dim)

    def compute_disentangle_loss(
        self,
        z_d_glob: torch.Tensor, # (B, 64)
        z_p: torch.Tensor,      # (B, 32)
        z_s: torch.Tensor,      # (B, 32)
        z_a: torch.Tensor       # (B, 4, D, H, W)
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        z_bio = torch.cat([z_d_glob, z_p], dim=-1)
        
        # 1. Total Correlation
        tc_loss, d_loss = self.tc_disc.compute_tc_loss(z_bio, z_s)

        # 2. vCLUB penalties
        z_a_pooled = z_a.mean(dim=(-3, -2, -1)) # pool spatial anatomy to (B, 4)
        
        mi_ds = self.club_ds(z_d_glob, z_s)
        mi_as = self.club_as(z_a_pooled, z_s)
        mi_dp = self.club_dp(z_d_glob, z_p)

        # Variational network NLL loss
        club_train_loss = (
            self.club_ds.compute_nll(z_d_glob.detach(), z_s.detach()) +
            self.club_as.compute_nll(z_a_pooled.detach(), z_s.detach()) +
            self.club_dp.compute_nll(z_d_glob.detach(), z_p.detach())
        )

        total_disentangle = (
            self.beta_tc * tc_loss +
            self.gamma_club * (mi_ds + mi_as + self.alpha_dp * mi_dp)
        )

        metrics = {
            "loss_tc": tc_loss.item(),
            "loss_disc": d_loss.item(),
            "mi_ds": mi_ds.item(),
            "mi_as": mi_as.item(),
            "mi_dp": mi_dp.item(),
            "loss_disentangle": total_disentangle.item()
        }
        return total_disentangle, d_loss + club_train_loss, metrics
