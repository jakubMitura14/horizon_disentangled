"""Stage 4 — Glioblastoma Biophysical PDE Engine.

Implements the spatial PDE system on the coarse latent grid Omega_L (24x28x20):
- Anisotropic Fisher-KPP diffusion-proliferation for viable tumor
- Screened Darcy pressure + mechanical advection for mass effect
- Necrotic damage accumulation and hypoxia-driven clearance
- Neural closure f_psi for residual morphological dynamics (channels 4-7)
- Linear-Quadratic (LQ) radiation shock with TMZ chemosensitization
- Surgical resection shock (cavity masking)

Key classes:
    GlioblastomaBiophysicalPDE: Full GBM PDE engine with jump operators.
    FiniteDifference3D: 3D Laplacian and gradient stencils on Cartesian grid.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

class FiniteDifference3D:
    """3D finite-difference gradient and Laplacian stencils on Cartesian latent grid."""
    @staticmethod
    def laplacian(u: torch.Tensor, dx: float = 8.0) -> torch.Tensor:
        # u shape: (B, C, D, H, W)
        u_pad = F.pad(u, (1, 1, 1, 1, 1, 1), mode='replicate')
        lap = (
            u_pad[:, :, 2:, 1:-1, 1:-1] + u_pad[:, :, :-2, 1:-1, 1:-1] +
            u_pad[:, :, 1:-1, 2:, 1:-1] + u_pad[:, :, 1:-1, :-2, 1:-1] +
            u_pad[:, :, 1:-1, 1:-1, 2:] + u_pad[:, :, 1:-1, 1:-1, :-2] -
            6.0 * u
        ) / (dx ** 2)
        return lap

    @staticmethod
    def gradient(u: torch.Tensor, dx: float = 8.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u_pad = F.pad(u, (1, 1, 1, 1, 1, 1), mode='replicate')
        grad_z = (u_pad[:, :, 2:, 1:-1, 1:-1] - u_pad[:, :, :-2, 1:-1, 1:-1]) / (2.0 * dx)
        grad_y = (u_pad[:, :, 1:-1, 2:, 1:-1] - u_pad[:, :, 1:-1, :-2, 1:-1]) / (2.0 * dx)
        grad_x = (u_pad[:, :, 1:-1, 1:-1, 2:] - u_pad[:, :, 1:-1, 1:-1, :-2]) / (2.0 * dx)
        return grad_z, grad_y, grad_x

class GlioblastomaBiophysicalPDE(nn.Module):
    """
    Stage 4 Glioblastoma Biophysical PDE Engine on Coarse Latent Grid Omega_L (24 x 28 x 20).
    Physical Channels:
      0: viable tumor u_v
      1: necrotic damaged tumor u_d
      2: vasogenic edema u_e
      3: tissue pressure p
      4-7: residual morphological dynamics u_res
    Coupled to Darcy mechanical advection, anisotropic Fisher-KPP diffusion,
    and Linear-Quadratic (LQ) radiation and surgical jumps.
    """
    def __init__(
        self,
        latent_shape: Tuple[int, int, int] = (24, 28, 20),
        dx_mm: float = 8.0,
        diff_w_default: float = 0.05, # mm^2/day
        kappa_darcy: float = 0.02,
        mu_screening: float = 0.1,
        gamma_pressure: float = 0.03,
        k_clear: float = 0.05,
        use_spectral: bool = False
    ):
        super().__init__()
        self.latent_shape = latent_shape
        self.dx = dx_mm
        self.diff_w = diff_w_default
        self.kappa = kappa_darcy
        self.mu_s = mu_screening
        self.gamma_p = gamma_pressure
        self.k_clear = k_clear
        self.use_spectral = use_spectral

        # Spectral Laplacian (cuFFT) — optional replacement for FD stencils
        if use_spectral:
            from src.pipeline.stage4_dynamics.spectral_laplacian import SpectralLaplacian3D, SpectralGradient3D
            self.spectral_lap = SpectralLaplacian3D(spatial_shape=latent_shape, dx_mm=dx_mm)
            self.spectral_grad = SpectralGradient3D(spatial_shape=latent_shape, dx_mm=dx_mm)

        # Neural closure f_psi for residual dynamics — deeper 3-layer ResNet
        self.neural_closure = nn.Sequential(
            nn.Conv3d(8 + 16, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, 64),
            nn.GELU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, 64),
            nn.GELU(),
            nn.Conv3d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv3d(32, 8, kernel_size=3, padding=1),
        )

        # Radiosensitivity LQ parameter MLP: maps v_sys -> [alpha, beta, gamma_TMZ]
        self.lq_param_net = nn.Sequential(
            nn.Linear(160, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3)  # [alpha > 0, beta > 0, gamma_TMZ > 0]
        )

        # Residual RT jump network g_theta_RT — deeper
        self.g_rt = nn.Sequential(
            nn.Conv3d(8 + 4, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv3d(32, 16, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(16, 4, kernel_size=1),
        )

    @torch.compiler.disable  # dynamic loop count prevents successful tracing
    def solve_screened_darcy_pressure(self, total_tumor: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Solves: (-kappa * Laplacian + mu_s) p = gamma_p * (u_v + u_d) via iterative Richardson relaxation.
        Returns pressure p and Darcy velocity v_c = -kappa * grad(p).
        """
        # total_tumor: (B, 1, D, H, W)
        rhs = self.gamma_p * total_tumor
        p = rhs / (self.mu_s + 1e-4) # Initial guess

        # Pre-compute denominator once (saves 3 redundant multiplications)
        denom = self.mu_s + 6.0 * self.kappa / (self.dx ** 2)

        # 3 Jacobi/Richardson smoothing iterations
        for _ in range(3):
            lap_p = FiniteDifference3D.laplacian(p, self.dx)
            p = (rhs + self.kappa * lap_p) / denom

        # Velocity v_c = -kappa * grad(p)
        gz, gy, gx = FiniteDifference3D.gradient(p, self.dx)
        v_z = -self.kappa * gz
        v_y = -self.kappa * gy
        v_x = -self.kappa * gx
        return p, (v_z, v_y, v_x)

    def forward_continuous_deriv(
        self,
        u_d: torch.Tensor,              # (B, 8, D, H, W)
        v_sys: torch.Tensor,            # (B, 160)
        phi_mask: torch.Tensor,         # (B, 1, D, H, W) brain phase field in [0, 1]
        dt_last: torch.Tensor           # (B,) time since last observation
    ) -> torch.Tensor:
        """Evaluates dU_D / dt on coarse latent grid."""
        # H200: channels_last_3d improves cuDNN throughput ~15% for 3D conv ops
        if u_d.dim() == 5:
            u_d = u_d.to(memory_format=torch.channels_last_3d)
            phi_mask = phi_mask.to(memory_format=torch.channels_last_3d)
        B, C, D, H, W = u_d.shape
        u_v = torch.clamp(u_d[:, 0:1], 0.0, 1.0)
        u_d_dam = torch.clamp(u_d[:, 1:2], 0.0, 1.0)
        u_e = torch.relu(u_d[:, 2:3])

        phi_tau = phi_mask + 1e-4

        # 1. Darcy Pressure & Advection
        total_tumor = u_v + u_d_dam
        p, (v_z, v_y, v_x) = self.solve_screened_darcy_pressure(total_tumor)

        # Advection term: div(phi * u_v * v_c)
        flux_z = phi_mask * u_v * v_z
        flux_y = phi_mask * u_v * v_y
        flux_x = phi_mask * u_v * v_x
        if self.use_spectral:
            gz_fz, _, _ = self.spectral_grad(flux_z)
            _, gy_fy, _ = self.spectral_grad(flux_y)
            _, _, gx_fx = self.spectral_grad(flux_x)
        else:
            gz_fz, _, _ = FiniteDifference3D.gradient(flux_z, self.dx)
            _, gy_fy, _ = FiniteDifference3D.gradient(flux_y, self.dx)
            _, _, gx_fx = FiniteDifference3D.gradient(flux_x, self.dx)
        div_advect_v = gz_fz + gy_fy + gx_fx

        # 2. Anisotropic Infiltration Diffusion: D_w * div(phi_tau * grad(u_v))
        if self.use_spectral:
            lap_v = self.spectral_lap(u_v)
        else:
            lap_v = FiniteDifference3D.laplacian(u_v, self.dx)
        diff_v = self.diff_w * phi_tau * lap_v

        # 3. Logistic Proliferation: rho * u_v * (1 - u_v - u_d)
        prolif_v = 0.03 * phi_mask * u_v * (1.0 - u_v - u_d_dam)

        # 4. Viable Tumor derivative du_v / dt
        du_v = diff_v - div_advect_v + prolif_v

        # 5. Necrotic damage & clearance du_d / dt
        # Hypoxia shedding + clear rate
        du_d = 0.01 * phi_mask * u_v * (u_v + u_d_dam) - self.k_clear * phi_mask * u_d_dam

        # 6. Edema du_e / dt
        if self.use_spectral:
            lap_e = self.spectral_lap(u_e)
        else:
            lap_e = FiniteDifference3D.laplacian(u_e, self.dx)
        du_e = 0.02 * phi_tau * lap_e + 0.02 * phi_mask * (u_v + 0.5 * u_d_dam) - 0.03 * phi_mask * u_e

        # 7. Pressure update dp / dt
        dp = p - u_d[:, 3:4]

        # 8. Neural Closure for residual channels 4-7 + physical corrections
        # Broadcast systemic state snippet (16-D) across spatial grid
        sys_snippet = v_sys[:, :16].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, D, H, W)
        neural_in = torch.cat([u_d, sys_snippet], dim=1)
        f_psi = self.neural_closure(neural_in)

        # Assemble full 8-channel derivative
        phys_deriv = torch.cat([du_v, du_d, du_e, dp, torch.zeros_like(u_d[:, 4:])], dim=1)
        return phys_deriv + 0.1 * f_psi

    def apply_resection_jump(
        self,
        u_d: torch.Tensor,
        cavity_mask: torch.Tensor
    ) -> torch.Tensor:
        """Applies instantaneous surgical resection shock."""
        # cavity_mask: 1 where tissue was resected, 0 elsewhere
        # Avoid in-place ops for autograd compatibility
        ch0 = (1.0 - cavity_mask) * u_d[:, 0:1]  # Remove viable tumor
        ch1 = (1.0 - cavity_mask) * u_d[:, 1:2]  # Remove necrotic tumor
        return torch.cat([ch0, ch1, u_d[:, 2:]], dim=1)

    def apply_radiotherapy_lq_jump(
        self,
        u_d: torch.Tensor,
        v_sys: torch.Tensor,
        dose_gy: float,
        tmz_dose: float = 1.0
    ) -> torch.Tensor:
        """
        Applies Linear-Quadratic (LQ) radiation shock:
        S_k = exp(-alpha * d - beta * d^2 - gamma_TMZ * c_TMZ)
        u_v(t^+) = S_k * u_v(t^-)
        u_d(t^+) = u_d(t^-) + (1 - S_k) * u_v(t^-)
        """
        lq_raw = F.softplus(self.lq_param_net(v_sys)) # [alpha, beta, gamma] > 0
        alpha = lq_raw[:, 0:1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = lq_raw[:, 1:2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        gamma = lq_raw[:, 2:3].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        surviving_fraction = torch.exp(-alpha * dose_gy - beta * (dose_gy ** 2) - gamma * tmz_dose)
        surviving_fraction = torch.clamp(surviving_fraction, 1e-4, 1.0)

        u_v_old = u_d[:, 0:1]
        u_d_dam_old = u_d[:, 1:2]

        u_v_new = surviving_fraction * u_v_old
        u_d_dam_new = u_d_dam_old + (1.0 - surviving_fraction) * u_v_old

        # Non-LQ residual update on channels 4-7
        rt_info = torch.tensor([dose_gy, tmz_dose, 0.0, 0.0], device=u_d.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(u_d.size(0), -1, u_d.size(2), u_d.size(3), u_d.size(4))
        g_in = torch.cat([u_d, rt_info], dim=1)
        ch4to7 = u_d[:, 4:8] + 0.1 * self.g_rt(g_in)

        # Reassemble without in-place ops for autograd compatibility
        return torch.cat([u_v_new, u_d_dam_new, u_d[:, 2:4], ch4to7], dim=1)
