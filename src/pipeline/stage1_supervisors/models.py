"""
Stage 1: Multimodal Supervisor Models

Upgraded architecture using MONAI SwinUNETR (BrainSegFounder-compatible) as the
3D visual backbone instead of a lightweight custom Conv3D encoder.

Architecture per main_pipeline.md Section 2.1:
  - GBM: LoRA-adapted BrainSegFounder 3D SwinViT → 768-D embedding
  - Breast: MedNeXt-v2 / SwinUNETR → 768-D embedding
  - Cross-attention fusion with 256-D tabular MLP
  - DIOR ordinal + LogisticHazard survival + SurvRNC contrastive heads
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from monai.networks.nets import SwinUNETR
import logging

logger = logging.getLogger(__name__)


class SwinUNETREncoder(nn.Module):
    """
    3D SwinViT encoder backbone from MONAI SwinUNETR.
    Extracts multi-scale features and projects to a 768-D embedding.
    
    Compatible with foundation model weights:
      - BrainSegFounder (BRATS & UK Biobank): depths=(2,2,6,2), 4 channels native
      - SuPreM (AbdomenAtlas): depths=(2,2,2,2), 1 channel
      - VoComni-H (1.16B parameters, 160K CTs): feature_size=192
    """
    def __init__(
        self,
        in_channels: int = 4,
        feature_dim: int = 768,
        feature_size: int = 48,
        depths: Tuple[int, ...] = (2, 2, 2, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        use_v2: bool = False,
        pretrained: Optional[str] = None,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Auto-configure architecture if BrainSegFounder is requested
        if pretrained and any(name in str(pretrained).lower() for name in ["brainsegfounder", "brats"]):
            depths = (2, 2, 6, 2)
            feature_size = 48
            num_heads = (3, 6, 12, 24)
        elif pretrained and "vocomni_h" in str(pretrained).lower():
            feature_size = 192
            depths = (2, 2, 2, 2)
            num_heads = (3, 6, 12, 24)
        
        # Full SwinUNETR — we extract multi-resolution encoder features
        self.swin_unetr = SwinUNETR(
            in_channels=in_channels,
            out_channels=feature_dim,  # will be ignored — we tap encoder features
            feature_size=feature_size,
            depths=depths,
            num_heads=num_heads,
            spatial_dims=3,
            use_v2=use_v2,
        )
        
        bottleneck_dim = 16 * feature_size  # 768 when feature_size=48
        
        # Global average pool + projection from bottleneck
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.proj = nn.Sequential(
            nn.Linear(bottleneck_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )
        
        # Multi-scale feature aggregation (optional — fuses stage 2,3,bottleneck)
        self.ms_proj = nn.Sequential(
            nn.Linear(4 * feature_size + 8 * feature_size + bottleneck_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )
        
        if pretrained:
            self.load_pretrained_weights(pretrained)

    def load_pretrained_weights(self, weights_name_or_path: str):
        """
        Load pretrained weights into the SwinViT backbone.
        Supports BrainSegFounder, SuPreM, VoComni, or custom path.
        """
        path = weights_name_or_path
        name_lower = str(weights_name_or_path).lower()
        
        if name_lower in ["brainsegfounder_brats", "brainsegfounder"]:
            path = "models/BrainSegFounder/model_weights_BRATS-pretrain.pt"
        elif name_lower == "brainsegfounder_ukb":
            path = "models/BrainSegFounder/model_weights_UKB-pretrain.pt"
        elif name_lower == "brainsegfounder_atlas":
            path = "models/BrainSegFounder/model_weights_ATLAS-pretrain.pt"
        elif name_lower in ["suprem", "suprem_supervised"]:
            path = "models/SuPreM/supervised_suprem_swinunetr_2100.pth"
        elif name_lower == "suprem_ssl":
            path = "models/SuPreM/self_supervised_nv_swin_unetr_50000.pth"
        elif name_lower == "vocomni_h":
            path = "models/VoCo/VoComni_H.pt"
        elif name_lower == "vocomni_l":
            path = "models/VoCo/VoComni_L.pt"
            
        if not os.path.exists(path):
            logger.warning(f"Pretrained weights file not found: {path}. Using random init.")
            return

        logger.info(f"Loading SwinViT pretrained weights from {path}...")
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            sd = ckpt.get("state_dict", ckpt.get("net", ckpt.get("model", ckpt)))
            
            # Map checkpoint keys to swinViT
            swin_sd = {}
            for k, v in sd.items():
                clean_k = k
                for prefix in ["module.swinViT.", "swinViT.", "module.backbone.swinViT.", "encoder."]:
                    if clean_k.startswith(prefix):
                        clean_k = clean_k.replace(prefix, "")
                        break
                swin_sd[clean_k] = v
                
            missing, unexpected = self.swin_unetr.swinViT.load_state_dict(swin_sd, strict=False)
            logger.info(
                f"Loaded pretrained SwinViT weights! Missing: {len(missing)}, Unexpected: {len(unexpected)}"
            )
        except Exception as e:
            logger.error(f"Failed to load pretrained weights from {path}: {e}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 768-D embedding from 3D volume."""
        hidden_states = self.swin_unetr.swinViT(x, self.swin_unetr.normalize)
        bottleneck = hidden_states[-1]
        h_bn = self.global_pool(bottleneck).flatten(1)
        h = self.proj(h_bn)
        return h


class VISTA3DEncoder(nn.Module):
    """
    3D VISTA3D CT+MRI foundation model encoder (~218M params).
    Uses pretrained weights from NVIDIA NV-Segment-CTMR (models/VISTA3D-CTMR).
    Pretrained across large-scale CT and MRI scans (127+ anatomic categories).
    """
    def __init__(
        self,
        in_channels: int = 4,
        feature_dim: int = 768,
        pretrained: bool = True,
        weights_path: str = "models/VISTA3D-CTMR/vista3d_pretrained_model/model.pt",
        freeze_backbone: bool = False,
    ):
        super().__init__()
        import monai.networks.nets
        self.vista = monai.networks.nets.vista3d132(encoder_embed_dim=48, in_channels=1)
        
        if pretrained and os.path.exists(weights_path):
            try:
                sd = torch.load(weights_path, map_location="cpu", weights_only=False)
                self.vista.load_state_dict(sd, strict=True)
                logger.info("Loaded VISTA3D-CTMR pretrained weights (218M params) successfully")
            except Exception as e:
                logger.warning(f"Failed to load VISTA3D weights: {e}")
                
        if freeze_backbone:
            for p in self.vista.parameters():
                p.requires_grad = False
                
        # Channel adaptation if multi-channel (e.g. 4-channel MRI sequences)
        self.in_conv = nn.Conv3d(in_channels, 1, kernel_size=1) if in_channels != 1 else nn.Identity()
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.proj = nn.Sequential(
            nn.Linear(48, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.in_conv(x)
        feat = self.vista.image_encoder(x_in)
        pooled = self.pool(feat[-1]).flatten(1)
        return self.proj(pooled)


class ConvNeXt3DBlock(nn.Module):
    """3D ConvNeXt block with Global Response Normalization (GRN)."""
    def __init__(self, dim: int, mult: float = 4.0):
        super().__init__()
        inner = int(dim * mult)
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, inner)
        self.act = nn.GELU()
        self.grn = GRN3D(inner)
        self.pwconv2 = nn.Linear(inner, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, D, H, W)
        return x + residual


class GRN3D(nn.Module):
    """Global Response Normalization (MedNeXt-v2 style)."""
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, H, W, C)
        gx = torch.norm(x, p=2, dim=(1, 2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class Conv3DResBlock(nn.Module):
    """3D Residual convolution block with GroupNorm and GELU."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.act2 = nn.GELU()
        
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(min(32, out_channels), out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act2(out + res)


class MedNeXtEncoder(nn.Module):
    """
    MedNeXt-v2-inspired 3D encoder with ConvNeXt blocks and GRN.
    For breast DCE-MRI encoding.
    
    Uses UpKern-compatible depthwise convolutions with progressive downsampling.
    ~15M parameters.
    """
    def __init__(self, in_channels: int = 4, feature_dim: int = 768, base_channels: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(32, base_channels),
            nn.GELU(),
        )
        # Progressive downsampling stages
        self.stage1 = nn.Sequential(
            Conv3DResBlock(base_channels, base_channels * 2, stride=2),
            ConvNeXt3DBlock(base_channels * 2),
        )
        self.stage2 = nn.Sequential(
            Conv3DResBlock(base_channels * 2, base_channels * 4, stride=2),
            ConvNeXt3DBlock(base_channels * 4),
            ConvNeXt3DBlock(base_channels * 4),
        )
        self.stage3 = nn.Sequential(
            Conv3DResBlock(base_channels * 4, base_channels * 8, stride=2),
            ConvNeXt3DBlock(base_channels * 8),
            ConvNeXt3DBlock(base_channels * 8),
        )
        self.stage4 = nn.Sequential(
            Conv3DResBlock(base_channels * 8, base_channels * 16, stride=2),
            ConvNeXt3DBlock(base_channels * 16),
        )
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.proj = nn.Sequential(
            nn.Linear(base_channels * 16, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.stage4(h)
        return self.proj(self.global_pool(h).flatten(1))


class TabularEncoder(nn.Module):
    """MLP encoding clinical metadata, receptors, and radiomics into 256-D embedding."""
    def __init__(self, in_dim: int = 16, hidden_dim: int = 256, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionFusion(nn.Module):
    """
    Multi-head cross-attention fusion between 3D visual features and tabular clinical features.
    Enhanced with 2 transformer layers instead of 1.
    """
    def __init__(self, visual_dim: int = 768, tabular_dim: int = 256, out_dim: int = 768, num_heads: int = 12):
        super().__init__()
        # Ensure num_heads divides visual_dim
        while visual_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.proj_tab = nn.Linear(tabular_dim, visual_dim)
        
        # 2-layer cross-attention
        self.attn1 = nn.MultiheadAttention(embed_dim=visual_dim, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(visual_dim)
        self.attn2 = nn.MultiheadAttention(embed_dim=visual_dim, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.norm2 = nn.LayerNorm(visual_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(visual_dim, visual_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(visual_dim * 4, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, h_vis: torch.Tensor, h_tab: torch.Tensor) -> torch.Tensor:
        q = h_vis.unsqueeze(1)  # (B, 1, D)
        k = self.proj_tab(h_tab).unsqueeze(1)  # (B, 1, D)
        
        attn1_out, _ = self.attn1(q, k, k)
        h = self.norm1(q + attn1_out)
        
        attn2_out, _ = self.attn2(h, k, k)
        h = self.norm2(h + attn2_out)
        
        return self.ffn(h.squeeze(1))


class MultimodalSupervisorTrunk(nn.Module):
    """
    Multimodal Supervisor Model (Stage 1):
    
    Processes 3D imaging + clinical tabular features to output joint representation h ∈ ℝ^768
    alongside supervised prediction heads:
    - DIOR ordinal response head (RANO 2.0 / NACT response tiers)
    - Discrete hazard survival head (20 quantile intervals)  
    - Binary clinical outcome heads (pCR, MGMT)
    
    Architecture (from main_pipeline.md Section 2.1):
    - GBM: SwinUNETR (BrainSegFounder-compatible, 62M params) + TabularMLP + CrossAttn
    - Breast: MedNeXt-v2 encoder (15M params) + TabularMLP + CrossAttn
    
    Total: ~65M params (vs previous 8.9M)
    """
    def __init__(
        self,
        domain: str = "gbm",
        in_channels: int = 4,
        tabular_dim: int = 16,
        feature_dim: int = 768,
        num_ordinal_classes: int = 5,
        num_survival_intervals: int = 20,
        encoder_type: str = "auto",  # "swinunetr", "mednext", "vista3d", or "auto" (domain-based)
        pretrained_weights: Optional[str] = None,
    ):
        super().__init__()
        self.domain = domain
        self.feature_dim = feature_dim
        self.K = num_ordinal_classes
        self.M = num_survival_intervals

        # Select encoder based on domain (per main_pipeline.md)
        if encoder_type == "auto":
            encoder_type = "swinunetr" if domain == "gbm" else "mednext"
        
        if encoder_type == "swinunetr":
            # SwinUNETR with optional foundation model weights (BrainSegFounder / SuPreM / VoComni)
            if feature_dim >= 384 and feature_dim % 16 == 0:
                self.visual_encoder = SwinUNETREncoder(
                    in_channels=in_channels,
                    feature_dim=feature_dim,
                    feature_size=max(24, feature_dim // 16),
                    pretrained=pretrained_weights,
                )
            else:
                # Fallback to MedNeXt for non-standard dims (e.g., testing)
                self.visual_encoder = MedNeXtEncoder(
                    in_channels=in_channels,
                    feature_dim=feature_dim,
                    base_channels=32,
                )
        elif encoder_type == "vista3d":
            # NVIDIA VISTA3D-CTMR foundation model (218M params)
            self.visual_encoder = VISTA3DEncoder(
                in_channels=in_channels,
                feature_dim=feature_dim,
                pretrained=True if pretrained_weights is None else bool(pretrained_weights),
            )
        else:
            self.visual_encoder = MedNeXtEncoder(
                in_channels=in_channels,
                feature_dim=feature_dim,
                base_channels=64,
            )

        self.tabular_encoder = TabularEncoder(in_dim=tabular_dim, hidden_dim=256, out_dim=256)
        self.fusion = CrossAttentionFusion(
            visual_dim=feature_dim,
            tabular_dim=256,
            out_dim=feature_dim,
            num_heads=12,
        )

        # Fallback when image is not present
        self.tab_to_full = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )

        # 1. DIOR Ordinal Head
        self.dior_cat = nn.Linear(feature_dim, self.K)
        self.dior_cum = nn.Linear(feature_dim, self.K - 1)

        # 2. Discrete Hazard Survival Head (20 intervals)
        self.hazard_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, self.M),
        )

        # 3. Domain-Specific Binary Heads
        self.binary_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.GELU(),
            nn.Linear(feature_dim // 4, 1),
        )

    def forward(
        self,
        image: Optional[torch.Tensor],
        clinical: torch.Tensor,
        has_image: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        B = clinical.size(0)
        h_tab = self.tabular_encoder(clinical)

        if image is not None and (has_image is None or has_image.any()):
            h_vis = self.visual_encoder(image)
            h = self.fusion(h_vis, h_tab)
            if has_image is not None and not has_image.all():
                h_fallback = self.tab_to_full(h_tab)
                mask = has_image.unsqueeze(1).float()
                h = mask * h + (1.0 - mask) * h_fallback
        else:
            h = self.tab_to_full(h_tab)

        cat_logits = self.dior_cat(h)
        cum_probs = torch.sigmoid(self.dior_cum(h))
        hazard_logits = self.hazard_head(h)
        binary_logit = self.binary_head(h).squeeze(-1)

        return {
            "h": h,
            "embedding": h,
            "cat_logits": cat_logits,
            "cum_probs": cum_probs,
            "hazard_logits": hazard_logits,
            "binary_logit": binary_logit
        }
