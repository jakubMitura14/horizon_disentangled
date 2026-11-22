"""
Stage 2: Two-Scale Hierarchical Causal VAE

Upgraded architecture using MONAI AutoencoderKL as the generative backbone
(per main_pipeline.md Section 2.2), with deeper Level 2 ConvNeXt downsamplers,
richer global latent heads, and proper AdaGN decoder conditioning.

Architecture:
  Level 1 (Fine Spatial): AutoencoderKL encoder (4×4×4 downsample) → Z_A ∈ ℝ^(4×D/4×H/4×W/4)
  Level 2 (Coarse):       ConvNeXt downsamplers (2×2×2) → U_D ∈ ℝ^(8×D_L×H_L×W_L)
                           Pooled → z_D^glob ∈ ℝ^64, Z_P ∈ ℝ^32, Z_S ∈ ℝ^32
  Decoder: AutoencoderKL decoder with AdaGN style modulation from Z_S

Backbone options:
  - 'monai' (default): MONAI AutoencoderKL (~42M params, trained from scratch)
  - 'medvae_4_1_3d': Stanford MedVAE 4× (~160M pretrained, MIT license)
  - 'medvae_8_1_3d': Stanford MedVAE 8× (~245M pretrained, MIT license)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from monai.networks.nets import AutoencoderKL
import logging

logger = logging.getLogger(__name__)


class AdaGroupNorm3d(nn.Module):
    """Adaptive Group Normalization conditioned on style vector Z_S."""
    def __init__(self, num_groups: int, num_channels: int, style_dim: int):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.fc = nn.Sequential(
            nn.Linear(style_dim, num_channels * 2),
            nn.SiLU(),
            nn.Linear(num_channels * 2, num_channels * 2),
        )

    def forward(self, x: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:
        normalized = self.gn(x)
        style = self.fc(z_s).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = torch.chunk(style, 2, dim=1)
        return normalized * (1.0 + gamma) + beta


class ConvNeXtDownBlock(nn.Module):
    """ConvNeXt-style 3D downsampling block with depthwise convolutions."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.down = nn.Conv3d(in_channels, out_channels, kernel_size=stride, stride=stride, bias=False)
        self.norm = nn.GroupNorm(min(32, out_channels), out_channels)
        self.dwconv = nn.Conv3d(out_channels, out_channels, kernel_size=7, padding=3, groups=out_channels, bias=True)
        self.act = nn.GELU()
        self.pwconv1 = nn.Conv3d(out_channels, out_channels * 4, kernel_size=1)
        self.pwconv2 = nn.Conv3d(out_channels * 4, out_channels, kernel_size=1)
        self.layer_scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1) * 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.down(x))
        residual = x
        x = self.dwconv(x)
        x = self.act(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return residual + self.layer_scale * x


class LatentProjectionHead(nn.Module):
    """
    Deep projection head for global latent variables.
    2-layer MLP with LayerNorm, producing mu and logvar.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(hidden_dim, out_dim)
        self.logvar_head = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        return self.mu_head(h), self.logvar_head(h)


class TwoScaleHierarchicalCausalVAE(nn.Module):
    """
    Two-Scale Hierarchical Causal VAE with AutoencoderKL backbone:
    
    Level 1 (Fine Spatial Anatomy Z_A):
      - AutoencoderKL encoder: 4× spatial downsample → Z_A ∈ ℝ^(4, D/4, H/4, W/4)
      - Supervised to reconstruct non-lesion tissue
    
    Level 2 (Coarse Spatial Disease + Global Latents):
      - ConvNeXt downsamplers: 2× from L1 → U_D ∈ ℝ^(8, D/8, H/8, W/8)
      - Global Disease z_D^glob ∈ ℝ^64 (mu_D, σ²_D)
      - Patient State Z_P ∈ ℝ^32 (mu_P, σ²_P) 
      - Style/Confounders Z_S ∈ ℝ^32 (mu_S, σ²_S) → injected via AdaGN
    
    Decoder:
      - Fuse Z_A + U_D (upsampled) with AdaGN(Z_S) conditioning
      - AutoencoderKL decoder for final reconstruction
      - Multi-compartment mask prediction head
    
    Total: ~42M params
    """
    def __init__(
        self,
        in_channels: int = 4,
        anatomy_channels: int = 4,
        disease_channels: int = 8,
        global_disease_dim: int = 64,
        patient_dim: int = 32,
        style_dim: int = 32,
        # AutoencoderKL config (MAISI-compatible)
        vae_channels: Tuple[int, ...] = (64, 128, 256),
        vae_latent_channels: int = 4,
        vae_num_res_blocks: int = 2,
        # Level 2 ConvNeXt config
        l2_channels: int = 256,
        # Backbone selection: 'monai' | 'medvae_4_1_3d' | 'medvae_8_1_3d'
        backbone: str = 'monai',
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.anatomy_channels = anatomy_channels
        self.disease_channels = disease_channels
        self.global_disease_dim = global_disease_dim
        self.patient_dim = patient_dim
        self.style_dim = style_dim
        self.vae_latent_channels = vae_latent_channels
        self.backbone_type = backbone

        # ============ Level 1: VAE Encoder Backbone ============
        if backbone.startswith('medvae'):
            # Use pretrained MedVAE (MIT license, 160M–245M params)
            from .medvae_backbone import MedVAEBackbone
            self.vae_backbone = MedVAEBackbone(
                model_name=backbone,
                target_latent_channels=vae_latent_channels,
                freeze_backbone=freeze_backbone,
            )
            logger.info(f"Stage 2 using MedVAE backbone: {backbone}")
        else:
            # Default: MONAI AutoencoderKL (~42M params, trained from scratch)
            self.vae_backbone = AutoencoderKL(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                channels=vae_channels,
                latent_channels=vae_latent_channels,
                num_res_blocks=vae_num_res_blocks,
                attention_levels=(False, False, True),
                with_encoder_nonlocal_attn=True,
                with_decoder_nonlocal_attn=True,
                norm_num_groups=32,
                use_flash_attention=True,
            )
            logger.info("Stage 2 using MONAI AutoencoderKL backbone")
        
        # Cross-sequence fusion after per-channel encoding
        # in_channels * vae_latent_channels → anatomy_channels
        self.cross_seq_fusion = nn.Sequential(
            nn.Conv3d(in_channels * vae_latent_channels, l2_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, l2_channels // 2),
            nn.GELU(),
            nn.Conv3d(l2_channels // 2, l2_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, l2_channels // 2),
            nn.GELU(),
        )
        
        # Anatomy projection (Z_A)
        self.to_za = nn.Conv3d(l2_channels // 2, anatomy_channels, kernel_size=1)
        
        # ============ Level 2: ConvNeXt Downsamplers ============
        # Downsample from L1 resolution (D/4) to coarse grid (D/8)
        self.l2_down = nn.Sequential(
            ConvNeXtDownBlock(l2_channels // 2, l2_channels, stride=2),
            ConvNeXtDownBlock(l2_channels, l2_channels, stride=1),  # Same-res refinement
        )
        
        # Disease field projection (U_D)
        self.to_ud = nn.Conv3d(l2_channels, disease_channels, kernel_size=1)
        
        # Global pooling for vector latents
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Deep projection heads for global latents
        self.disease_head = LatentProjectionHead(l2_channels, 256, global_disease_dim)
        self.patient_head = LatentProjectionHead(l2_channels, 128, patient_dim)
        self.style_head = LatentProjectionHead(l2_channels, 128, style_dim)

        # ============ Decoder ============
        # Upsample U_D to L1 resolution
        self.ud_upsample = nn.Sequential(
            nn.ConvTranspose3d(disease_channels, l2_channels // 2, kernel_size=2, stride=2),
            nn.GroupNorm(32, l2_channels // 2),
            nn.GELU(),
        )
        
        # Fuse Z_A + upsampled U_D with AdaGN
        fuse_in = anatomy_channels + l2_channels // 2
        self.fuse_conv = nn.Sequential(
            nn.Conv3d(fuse_in, l2_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, l2_channels // 2),
            nn.GELU(),
            nn.Conv3d(l2_channels // 2, l2_channels // 2, kernel_size=3, padding=1, bias=False),
        )
        self.adagn1 = AdaGroupNorm3d(32, l2_channels // 2, style_dim)
        
        # Decoder upsampling stages (mirror AutoencoderKL spatial upsampling)
        self.dec_up1 = nn.Sequential(
            nn.ConvTranspose3d(l2_channels // 2, l2_channels // 4, kernel_size=2, stride=2),
            nn.GroupNorm(min(32, l2_channels // 4), l2_channels // 4),
            nn.GELU(),
            nn.Conv3d(l2_channels // 4, l2_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(32, l2_channels // 4), l2_channels // 4),
            nn.GELU(),
        )
        self.adagn2 = AdaGroupNorm3d(min(32, l2_channels // 4), l2_channels // 4, style_dim)
        
        self.dec_up2 = nn.Sequential(
            nn.ConvTranspose3d(l2_channels // 4, l2_channels // 8, kernel_size=2, stride=2),
            nn.GroupNorm(min(16, l2_channels // 8), l2_channels // 8),
            nn.GELU(),
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv3d(l2_channels // 8, l2_channels // 8, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(l2_channels // 8, in_channels, kernel_size=1),
        )

        # Mask prediction head (multi-compartment)
        self.mask_head = nn.Sequential(
            nn.ConvTranspose3d(disease_channels, 32, kernel_size=2, stride=2),
            nn.GroupNorm(16, 32),
            nn.GELU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose3d(16, 2, kernel_size=2, stride=2),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_per_channel(self, x: torch.Tensor) -> torch.Tensor:
        """
        Weight-shared per-channel VAE encoding.
        x: (B, C_in, D, H, W) → latents: (B, C_in * latent_ch, D/4, H/4, W/4)
        """
        B, C_in, D, H, W = x.shape
        per_ch_latents = []
        
        for c in range(C_in):
            x_ch = x[:, c:c+1]  # (B, 1, D, H, W)
            # Use only encoder part of AutoencoderKL
            z_ch = self.vae_backbone.encode_stage_2_inputs(x_ch)  # (B, latent_ch, D/4, H/4, W/4)
            per_ch_latents.append(z_ch)
        
        # Concatenate all channels
        return torch.cat(per_ch_latents, dim=1)  # (B, C_in * latent_ch, D/4, H/4, W/4)

    def encode(self, x: torch.Tensor) -> Dict[str, Any]:
        """Full encoding pipeline: L1 anatomy + L2 disease + global latents."""
        # Level 1: Per-channel AutoencoderKL encoding
        per_ch_z = self.encode_per_channel(x)  # (B, C_in*4, D/4, H/4, W/4)
        return self.encode_from_cached(per_ch_z)

    def encode_from_cached(self, per_ch_z: torch.Tensor) -> Dict[str, Any]:
        """Encode from pre-computed per-channel VAE latents (Phase B uses this).

        This skips encode_per_channel() and starts from cross_seq_fusion,
        allowing Phase B to operate on cached MedVAE features from Phase A HDF5.

        Args:
            per_ch_z: (B, C_in*latent_ch, D/4, H/4, W/4) cached encoder output.

        Returns:
            Dict with z_a, u_d, z_d_glob, z_p, z_s and mu/logvar for each.
        """
        # Cross-sequence fusion
        feat_l1 = self.cross_seq_fusion(per_ch_z)  # (B, l2_ch//2, D/4, H/4, W/4)
        z_a = self.to_za(feat_l1)  # (B, 4, D/4, H/4, W/4)

        # Level 2: Downsample to coarse grid
        feat_l2 = self.l2_down(feat_l1)  # (B, l2_ch, D/8, H/8, W/8)
        u_d = self.to_ud(feat_l2)  # (B, 8, D/8, H/8, W/8)

        # Global latent heads
        pooled = self.pool(feat_l2).flatten(1)
        mu_d, logvar_d = self.disease_head(pooled)
        mu_p, logvar_p = self.patient_head(pooled)
        mu_s, logvar_s = self.style_head(pooled)

        z_d_glob = self.reparameterize(mu_d, logvar_d)
        z_p = self.reparameterize(mu_p, logvar_p)
        z_s = self.reparameterize(mu_s, logvar_s)

        return {
            "z_a": z_a,
            "u_d": u_d,
            "z_d_glob": z_d_glob,
            "z_p": z_p,
            "z_s": z_s,
            "mu_d": mu_d, "logvar_d": logvar_d,
            "mu_p": mu_p, "logvar_p": logvar_p,
            "mu_s": mu_s, "logvar_s": logvar_s,
        }

    def decode(self, z_a: torch.Tensor, u_d: torch.Tensor, z_s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode from latents to reconstruction + mask prediction."""
        # Upsample U_D to match Z_A spatial shape
        u_d_up = self.ud_upsample(u_d)
        
        if u_d_up.shape[2:] != z_a.shape[2:]:
            u_d_up = F.interpolate(u_d_up, size=z_a.shape[2:], mode="trilinear", align_corners=False)

        # Fuse Z_A + U_D with AdaGN conditioning on Z_S
        fused = torch.cat([z_a, u_d_up], dim=1)
        h = self.fuse_conv(fused)
        h = F.gelu(self.adagn1(h, z_s))

        # Progressive upsampling
        h = self.dec_up1(h)
        h = F.gelu(self.adagn2(h, z_s))
        h = self.dec_up2(h)
        recon = self.final_conv(h)
        
        # Mask prediction from U_D
        mask_pred = self.mask_head(u_d)
        return recon, mask_pred

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        enc = self.encode(x)
        recon, mask_pred = self.decode(enc["z_a"], enc["u_d"], enc["z_s"])
        
        if recon.shape[2:] != x.shape[2:]:
            recon = F.interpolate(recon, size=x.shape[2:], mode="trilinear", align_corners=False)
        if mask_pred.shape[2:] != x.shape[2:]:
            mask_pred = F.interpolate(mask_pred, size=x.shape[2:], mode="trilinear", align_corners=False)

        enc["recon"] = recon
        enc["mask_pred"] = mask_pred
        return enc
