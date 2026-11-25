# Stage 1 Models (`repr/src/pipeline/stage1_supervisors/models.py`)

Implements 3D visual encoders, clinical tabular MLP, cross-attention fusion, the multimodal supervisor trunk, and cluster wrapping helpers.

---

## Classes

### `SwinUNETREncoder(nn.Module)`
3D SwinViT encoder backbone wrapping MONAI's `SwinUNETR`.
- **`__init__(self, in_channels=4, feature_dim=768, feature_size=48, depths=(2,2,2,2), num_heads=(3,6,12,24), use_v2=False, pretrained=None)`**:
  - Auto-configures `depths=(2,2,6,2), feature_size=48` for BrainSegFounder or `feature_size=192` for `vocomni_h` (~998.9M–1.16B params).
  - Builds `SwinUNETR(..., spatial_dims=3, use_checkpoint=True)` for 40% activation memory savings.
  - Adds `global_pool` (`AdaptiveAvgPool3d(1)`), `proj` (2-layer MLP with `LayerNorm` and `GELU`), and `ms_proj` (multi-scale projection).
  - Calls `load_pretrained_weights(pretrained)` if provided.
- **`load_pretrained_weights(self, weights_name_or_path: str)`**:
  - Maps named presets (`brainsegfounder_*`, `suprem*`, `vocomni_h/l`) or file paths, strips DDP prefixes (`module.swinViT.`, etc.), and loads into `self.swin_unetr.swinViT` with `strict=False`. Logs warning if file is absent (benchmark uses random init).
- **`forward(self, x: torch.Tensor) -> torch.Tensor`**:
  - Converts 5D input `(B, C, D, H, W)` to `torch.channels_last_3d` for ~15% H200/GH200 cuDNN speedup, extracts `swinViT` bottleneck `hidden_states[-1]`, pools, and projects to `(B, feature_dim)`.

### `VISTA3DEncoder(nn.Module)`
Wrapper for MONAI's `vista3d132` CT+MRI foundation encoder (~218M params).
- **`__init__(self, in_channels=4, feature_dim=768, pretrained=True, weights_path=..., freeze_backbone=False)`**: Loads `vista3d132`, adapts multi-channel input via `1×1×1` `Conv3d`, and projects `48`-D encoder features to `feature_dim`.
- **`forward(self, x: torch.Tensor) -> torch.Tensor`**: Runs `in_conv` → `vista.image_encoder` → global average pool → `proj`.

### `GRN3D(nn.Module)`
3D Global Response Normalization (MedNeXt-v2 style).
- **`__init__(self, dim: int)`**: Initializes learnable `gamma` and `beta` parameters of shape `(1, 1, 1, 1, dim)`.
- **`forward(self, x: torch.Tensor) -> torch.Tensor`**: Computes L2 norm across `(D, H, W)`, normalizes across channels, and returns `gamma * (x * nx) + beta + x`.

### `ConvNeXt3DBlock(nn.Module)`
3D ConvNeXt block with `7×7×7` depthwise convolution, `LayerNorm`, `4×` inverted bottleneck linear layers, `GELU`, `GRN3D`, and residual skip connection.
- **`__init__(self, dim: int, mult: float = 4.0)`**
- **`forward(self, x: torch.Tensor) -> torch.Tensor`**

### `Conv3DResBlock(nn.Module)`
Standard 3D residual block with two `3×3×3` convolutions, `GroupNorm`, `GELU`, and optional `1×1×1` strided shortcut.
- **`__init__(self, in_channels: int, out_channels: int, stride: int = 1)`**
- **`forward(self, x: torch.Tensor) -> torch.Tensor`**

### `MedNeXtEncoder(nn.Module)`
MedNeXt-v2-inspired 3D encoder (~15M params) for breast DCE-MRI.
- **`__init__(self, in_channels=4, feature_dim=768, base_channels=64)`**: `5×5×5` stride-2 stem + 4 progressive downsampling stages (`Conv3DResBlock` + `ConvNeXt3DBlock`) + `AdaptiveAvgPool3d(1)` + `proj`.
- **`forward(self, x: torch.Tensor) -> torch.Tensor`**

### `TabularEncoder(nn.Module)`
3-layer MLP (`Linear` → `LayerNorm` → `GELU` → `Dropout(0.1)`) encoding clinical metadata into a `256`-D embedding.
- **`__init__(self, in_dim=16, hidden_dim=256, out_dim=256)`**
- **`forward(self, x: torch.Tensor) -> torch.Tensor`**

### `CrossAttentionFusion(nn.Module)`
2-layer `MultiheadAttention` + `LayerNorm` + FFN fusing 3D visual features (`h_vis`) with projected tabular features (`h_tab`).
- **`__init__(self, visual_dim=768, tabular_dim=256, out_dim=768, num_heads=12)`**
- **`forward(self, h_vis: torch.Tensor, h_tab: torch.Tensor) -> torch.Tensor`**

### `MultimodalSupervisorTrunk(nn.Module)`
Full Stage 1 multimodal model combining a visual encoder (`swinunetr`, `vista3d`, or `mednext`), `TabularEncoder`, `CrossAttentionFusion`, missing-image fallback (`tab_to_full`), DIOR ordinal heads (`dior_cat`, `dior_cum`), discrete survival hazard head (`hazard_head`), and binary outcome head (`binary_head`).
- **`__init__(...)`**
- **`forward(self, image, clinical, has_image=None) -> Dict[str, torch.Tensor]`**: Returns dict with keys `h`, `embedding`, `cat_logits`, `cum_probs`, `hazard_logits`, `binary_logit`.

---

## Functions

### `wrap_model_for_cluster(model: nn.Module, config) -> nn.Module`
Helper that wraps `model` in `FSDP` (`FULL_SHARD`, `size_based_auto_wrap_policy` >= 10M params, `MixedPrecision` bf16/fp32) when `n_params > 400_000_000` (or `DDP` otherwise) and applies `torch.compile`.
