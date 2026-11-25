"""Stage 4 — Breast Cancer Fast-Slow Biophysical PDE.

Dual-timescale biophysical model for breast cancer treatment response:
- Slow timeline (days): tumor cellularity evolution, vascular remodeling,
  and neoadjuvant chemotherapy shocks over a 168-day treatment course
- Fast timeline (minutes): intra-scan gadolinium washout modeled via
  closed-form Extended Tofts pharmacokinetics under bi-exponential Weinmann AIF

Key classes:
    BreastFastSlowBiophysicalPDE: Coarse-grid PDE engine for breast cancer.
    ExtendedToftsObservationOperator: Analytically differentiable PK forward model.

Example:
    >>> pde = BreastFastSlowBiophysicalPDE()
    >>> du_dt = pde.forward_continuous_deriv(u_d, v_sys, c_drug)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

class ExtendedToftsObservationOperator(nn.Module):
    """
    Closed-form analytically differentiable Extended Tofts observation operator
    under bi-exponential Weinmann AIF:
      C_p(tau) = D_bolus * sum_{r=1}^2 a_r * exp(-m_r * tau)
    Evaluates contrast concentration C_t(tau) and physical Signal Enhancement Ratio (SER).
    """
    def __init__(
        self,
        bolus_dose: float = 0.1,
        a_weinmann: Tuple[float, float] = (3.99, 4.78),
        m_weinmann: Tuple[float, float] = (0.144, 0.0111),
        tau_early_min: float = 2.25, # 135 s (I-SPY 2 early PE2 peak)
        tau_late_min: float = 7.5    # 450 s (I-SPY 2 late PE6 peak)
    ):
        super().__init__()
        self.d_bolus = bolus_dose
        self.a1, self.a2 = a_weinmann
        self.m1, self.m2 = m_weinmann
        self.tau_early = tau_early_min
        self.tau_late = tau_late_min

    def aif(self, tau: float) -> float:
        return self.d_bolus * (self.a1 * torch.exp(torch.tensor(-self.m1 * tau)) + self.a2 * torch.exp(torch.tensor(-self.m2 * tau)))

    def evaluate_contrast_concentration(
        self,
        k_trans: torch.Tensor, # (B, 1, D, H, W) in min^-1
        v_e: torch.Tensor,     # (B, 1, D, H, W)
        v_p: torch.Tensor,     # (B, 1, D, H, W)
        tau_min: float
    ) -> torch.Tensor:
        """Closed-form Extended Tofts convolution."""
        eps = 1e-5
        v_e_safe = torch.clamp(v_e, 0.05, 0.95)
        k_ep = k_trans / (v_e_safe + eps) # min^-1

        cp_tau = float(self.aif(tau_min))

        # Sum over r=1, 2: a_r * (exp(-m_r * tau) - exp(-k_ep * tau)) / (k_ep - m_r)
        term1 = self.a1 * (torch.exp(torch.tensor(-self.m1 * tau_min)) - torch.exp(-k_ep * tau_min)) / (k_ep - self.m1 + eps)
        term2 = self.a2 * (torch.exp(torch.tensor(-self.m2 * tau_min)) - torch.exp(-k_ep * tau_min)) / (k_ep - self.m2 + eps)
        
        ce_tau = k_trans * self.d_bolus * (term1 + term2)
        c_t = v_p * cp_tau + ce_tau
        return torch.clamp(c_t, min=0.0)

    def predict_ser(
        self,
        k_trans: torch.Tensor,
        v_e: torch.Tensor,
        v_p: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predicts early percent enhancement (PE_early), late (PE_late),
        and physical Signal Enhancement Ratio SER = PE_early / PE_late in [0.0, 3.0].
        """
        c_early = self.evaluate_contrast_concentration(k_trans, v_e, v_p, self.tau_early)
        c_late = self.evaluate_contrast_concentration(k_trans, v_e, v_p, self.tau_late)

        # In physics-normalized space, relative PE is proportional to contrast concentration
        pe_early_pred = c_early
        pe_late_pred = c_late
        
        # Physical SER
        ser_pred = torch.clamp(pe_early_pred / (pe_late_pred + 1e-4), 0.0, 3.0)
        return pe_early_pred, pe_late_pred, ser_pred

class BreastFastSlowBiophysicalPDE(nn.Module):
    """
    Stage 4 Breast Cancer Fast-Slow Biophysical PDE Engine on Coarse Latent Grid (32 x 32 x 16).
    Slow treatment response timeline t in [0, 168] days:
      0: tumor cellularity u_c in [0, 1] (inversely coupled to ADC)
      1: volume transfer constant K_trans > 0
      2: extravascular-extracellular fraction v_e in (0.05, 0.95)
      3: plasma fraction v_p in (0, 0.3)
      4-7: residual dynamics u_res
    Fast intra-scan gadolinium washout tau in [0, 8] min via Extended Tofts PK.
    """
    def __init__(
        self,
        latent_shape: Tuple[int, int, int] = (32, 32, 16),
        dx_mm: float = 8.0,
        diff_b_default: float = 0.02,
        use_spectral: bool = False
    ):
        super().__init__()
        self.latent_shape = latent_shape
        self.dx = dx_mm
        self.diff_b = diff_b_default
        self.use_spectral = use_spectral

        # Spectral Laplacian (cuFFT) — optional replacement for FD stencils
        if use_spectral:
            from src.pipeline.stage4_dynamics.spectral_laplacian import SpectralLaplacian3D
            self.spectral_lap = SpectralLaplacian3D(spatial_shape=latent_shape, dx_mm=dx_mm)

        self.tofts_op = ExtendedToftsObservationOperator()

        # Neural reaction closure — deeper 3-layer with GroupNorm
        self.reaction_net = nn.Sequential(
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

        # Neoadjuvant chemotherapy jump shock network g_theta_NACT — deeper
        self.g_nact = nn.Sequential(
            nn.Conv3d(8 + 16, 64, kernel_size=1, bias=False),
            nn.GroupNorm(16, 64),
            nn.GELU(),
            nn.Conv3d(64, 32, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(32, 8, kernel_size=1),
        )

    def forward_continuous_deriv(
        self,
        u_d: torch.Tensor,   # (B, 8, D, H, W)
        v_sys: torch.Tensor, # (B, 160)
        c_drug: torch.Tensor # (B, 16) drug embedding
    ) -> torch.Tensor:
        """Evaluates slow treatment timeline derivative dU_D / dt."""
        B, C, D, H, W = u_d.shape
        u_c = torch.clamp(u_d[:, 0:1], 0.0, 1.0)
        k_trans = F.softplus(u_d[:, 1:2])
        v_e = torch.sigmoid(u_d[:, 2:3]) * 0.9 + 0.05
        v_p = torch.sigmoid(u_d[:, 3:4]) * 0.3

        # Diffusion term
        if self.use_spectral:
            lap_uc = self.spectral_lap(u_c)
        else:
            from src.pipeline.stage4_dynamics.gbm_biophysical import FiniteDifference3D
            lap_uc = FiniteDifference3D.laplacian(u_c, self.dx)
        diff_term = self.diff_b * lap_uc

        # Proliferation: logistic expansion
        prolif_uc = 0.02 * u_c * (1.0 - u_c)
        du_c = diff_term + prolif_uc

        # Vascular remodeling
        dk_trans = -0.01 * k_trans
        dv_e = 0.005 * (0.4 - v_e)
        dv_p = 0.001 * (0.05 - v_p)

        phys_deriv = torch.cat([du_c, dk_trans, dv_e, dv_p, torch.zeros_like(u_d[:, 4:])], dim=1)

        # Neural closure
        sys_snippet = v_sys[:, :16].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, D, H, W)
        neural_in = torch.cat([u_d, sys_snippet], dim=1)
        f_psi = self.reaction_net(neural_in)

        return phys_deriv + 0.1 * f_psi

    def apply_chemotherapy_jump(
        self,
        u_d: torch.Tensor,
        v_sys: torch.Tensor,
        drug_embedding: torch.Tensor # (B, 16)
    ) -> torch.Tensor:
        """Applies discrete neoadjuvant chemotherapy shock (cytotoxic and anti-angiogenic)."""
        B, C, D, H, W = u_d.shape
        drug_expanded = drug_embedding.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, D, H, W)
        shock_in = torch.cat([u_d, drug_expanded], dim=1)
        delta_u = self.g_nact(shock_in)
        
        u_d_new = u_d + delta_u
        # Clamp cellularity and microvascular parameters to valid physiological ranges
        # Use torch.cat instead of in-place slice assignment to avoid autograd issues
        ch0 = torch.clamp(u_d_new[:, 0:1], 0.0, 1.0)
        ch1 = F.softplus(u_d_new[:, 1:2])
        ch2 = torch.clamp(u_d_new[:, 2:3], 0.05, 0.95)
        ch3 = torch.clamp(u_d_new[:, 3:4], 0.0, 0.3)
        ch_rest = u_d_new[:, 4:]
        return torch.cat([ch0, ch1, ch2, ch3, ch_rest], dim=1)
