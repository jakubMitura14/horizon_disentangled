"""
MedVAE Backbone Integration for Stage 2

Wraps Stanford MedVAE (MIT license) as a drop-in replacement for MONAI AutoencoderKL
in the TwoScaleHierarchicalCausalVAE. Supports:
  - medvae_4_1_3d: 4× spatial downsampling, 1ch latent, ~160M params
  - medvae_8_1_3d: 8× spatial downsampling, 1ch latent, ~245M params

The wrapper exposes `encode_stage_2_inputs()` and `decode()` matching the interface
expected by TwoScaleHierarchicalCausalVAE.encode_per_channel().

Usage:
    from .medvae_backbone import MedVAEBackbone
    backbone = MedVAEBackbone(model_name='medvae_4_1_3d', freeze=True)
    # Use backbone.encode_stage_2_inputs(x_1ch) → (B, latent_ch, D/4, H/4, W/4)

License: MIT (MedVAE) — compatible with commercial spin-off (WP6/9).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MedVAEBackbone(nn.Module):
    """
    Wraps MedVAE's AutoencoderKL_3D as a backbone for the Hierarchical Causal VAE.
    
    Uses the low-level AutoencoderKL_3D model directly (not the MVAE sliding-window
    wrapper) for efficient training. The model's encode returns a 
    DiagonalGaussianDistribution; we sample from it.
    
    MedVAE model variants:
      - medvae_4_1_3d: 4× spatial downsampling per dim (64× total), 1 latent channel
        Input (1, D, H, W) → Latent (embed_dim, D/4, H/4, W/4) 
      - medvae_8_1_3d: 8× spatial downsampling per dim (512× total), 1 latent channel
        Input (1, D, H, W) → Latent (embed_dim, D/8, H/8, W/8)
    """
    
    def __init__(
        self,
        model_name: str = 'medvae_4_1_3d',
        target_latent_channels: int = 4,
        freeze_backbone: bool = True,
        local_ckpt_path: Optional[str] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.target_latent_channels = target_latent_channels
        self.freeze_backbone = freeze_backbone
        
        # Determine compression factor from model name
        if '8' in model_name:
            self.spatial_downsample = 8
        else:
            self.spatial_downsample = 4
        
        # Load MedVAE model using the factory (bypassing MVAE wrapper)
        try:
            from medvae.utils.factory import (
                create_model, build_model, download_model_weights,
                FILE_DICT_ASSOCIATIONS
            )
            
            if model_name not in FILE_DICT_ASSOCIATIONS:
                raise ValueError(
                    f"Model '{model_name}' not found. "
                    f"Available: {list(FILE_DICT_ASSOCIATIONS.keys())}"
                )
            
            logger.info(f"Loading MedVAE model: {model_name}")
            
            # Build model directly (not through MVAE which needs modality for transforms)
            if local_ckpt_path is not None:
                config_fpath = download_model_weights(
                    FILE_DICT_ASSOCIATIONS[model_name]["config"]
                )
                self.vae = build_model(
                    model_name, config_fpath, local_ckpt_path
                )
            else:
                self.vae = create_model(
                    model_name, training=False, state_dict=True
                )
            
            # Determine embed_dim from the model
            self._native_latent_channels = self.vae.embed_dim
            logger.info(
                f"MedVAE loaded: embed_dim={self._native_latent_channels}, "
                f"spatial_downsample={self.spatial_downsample}×"
            )
            
        except ImportError:
            raise ImportError(
                "MedVAE not installed. Install with: pip install medvae"
            )
        
        # Projection layer: match MedVAE latent dim to expected latent_channels
        if self._native_latent_channels != target_latent_channels:
            self.latent_proj = nn.Sequential(
                nn.Conv3d(self._native_latent_channels, target_latent_channels * 2,
                         kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(min(16, target_latent_channels * 2), target_latent_channels * 2),
                nn.GELU(),
                nn.Conv3d(target_latent_channels * 2, target_latent_channels,
                         kernel_size=1, bias=True),
            )
            self.decode_proj = nn.Sequential(
                nn.Conv3d(target_latent_channels, self._native_latent_channels * 2,
                         kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(min(16, self._native_latent_channels * 2),
                           self._native_latent_channels * 2),
                nn.GELU(),
                nn.Conv3d(self._native_latent_channels * 2, self._native_latent_channels,
                         kernel_size=1, bias=True),
            )
        else:
            self.latent_proj = nn.Identity()
            self.decode_proj = nn.Identity()
        
        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()
        
        param_count = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"MedVAEBackbone: {param_count:,} total, {trainable:,} trainable, "
            f"latent {self._native_latent_channels}→{target_latent_channels}"
        )
    
    def _freeze_backbone(self):
        """Freeze all MedVAE parameters (only projection heads train)."""
        for param in self.vae.parameters():
            param.requires_grad = False
        logger.info("MedVAE backbone frozen — only projection layers are trainable")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning."""
        for param in self.vae.parameters():
            param.requires_grad = True
        logger.info("MedVAE backbone unfrozen for fine-tuning")
    
    def encode_stage_2_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode single-channel input using MedVAE, matching AutoencoderKL interface.
        
        Args:
            x: (B, 1, D, H, W) single-channel 3D volume
            
        Returns:
            z: (B, target_latent_channels, D/ds, H/ds, W/ds) latent representation
        """
        if self.freeze_backbone:
            with torch.no_grad():
                posterior = self.vae.encode(x)
                z = posterior.sample()
        else:
            posterior = self.vae.encode(x)
            z = posterior.sample()
        
        # Project to target latent channels
        z = self.latent_proj(z)
        return z
    
    def decode_stage_2_outputs(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode from latent space using MedVAE.
        
        Args:
            z: (B, target_latent_channels, D', H', W') latent
            
        Returns:
            x_recon: (B, 1, D, H, W) reconstructed volume
        """
        z_native = self.decode_proj(z)
        
        if self.freeze_backbone:
            with torch.no_grad():
                x_recon = self.vae.decode(z_native)
        else:
            x_recon = self.vae.decode(z_native)
        
        return x_recon
    
    def forward(self, x: torch.Tensor, decode: bool = False):
        """Forward pass through encoder (and optionally decoder)."""
        z = self.encode_stage_2_inputs(x)
        if decode:
            return self.decode_stage_2_outputs(z)
        return z
    
    def get_param_groups(self) -> dict:
        """Return parameter groups for differential learning rates."""
        backbone_params = list(self.vae.parameters())
        proj_params = []
        if isinstance(self.latent_proj, nn.Sequential):
            proj_params += list(self.latent_proj.parameters())
            proj_params += list(self.decode_proj.parameters())
        return {
            'backbone': backbone_params,
            'projection': proj_params,
        }
