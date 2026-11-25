# Stage 2 Causal VAE (`repr/src/pipeline/stage2_causal_vae/`)

---

## 1. `architecture.py`

Implements the Two-Scale Hierarchical Causal VAE separating anatomy (`Z_A`), spatial disease (`U_D`), global disease (`z_D^glob`), patient state (`Z_P`), and scanner style (`Z_S`).

### Classes

#### `AdaGroupNorm3d(nn.Module)`
Adaptive Group Normalization conditioned on style vector `z_s`.
- **`__init__(self, num_groups: int, num_channels: int, style_dim: int)`**: Uses affine-free `GroupNorm` and a 2-layer `SiLU` MLP producing `(gamma, beta)` scale/shift tensors.
- **`forward(self, x: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor`**: Applies `normalized * (1 + gamma) + beta`.

#### `ConvNeXtDownBlock(nn.Module)`
ConvNeXt-style 3D downsampling block for Level-2 coarse encoding.
- **`__init__(self, in_channels: int, out_channels: int, stride: int = 2)`**: Strided `Conv3d` downsampler + `GroupNorm` + `7×7×7` depthwise `Conv3d` + `GELU` + `4×` pointwise `1×1×1` expansion/projection + learnable `layer_scale`.
- **`forward(self, x: torch.Tensor) -> torch.Tensor`**: Applies downsampler and residual ConvNeXt block.

#### `LatentProjectionHead(nn.Module)`
2-layer MLP projecting pooled Level-2 features into Gaussian `(mu, logvar)` parameters.
- **`__init__(self, in_dim: int, hidden_dim: int, out_dim: int)`**
- **`forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`**: Clamps `logvar` to `[-10.0, 10.0]` for numerical stability.

#### `TwoScaleHierarchicalCausalVAE(nn.Module)`
Main hierarchical VAE (~746.7M–756M params with `vae_channels=(256, 512, 1024)`).
- **`__init__(self, in_channels=4, anatomy_channels=4, disease_channels=8, global_disease_dim=64, patient_dim=32, style_dim=32, vae_channels=(64,128,256), vae_latent_channels=4, vae_num_res_blocks=2, l2_channels=256, backbone='monai', freeze_backbone=False)`**:
  - **Level 1**: `MedVAEBackbone` or MONAI `AutoencoderKL` (`use_flash_attention=True`).
  - **Fusion & Level 2**: `cross_seq_fusion` (`3×3×3` convs), `to_za` (`1×1×1`), `l2_down` (2× `ConvNeXtDownBlock`), `to_ud`, `AdaptiveAvgPool3d`, and `disease_head` / `patient_head` / `style_head`.
  - **Decoder**: `ud_upsample` (`ConvTranspose3d`), `fuse_conv`, `adagn1`, `dec_up1`, `adagn2`, `dec_up2`, `mask_head`, and **`final_conv`**.
  - **Key Memory Optimization in `final_conv`**: Uses a depthwise-separable `3×3×3` (`groups=final_ch`) + `1×1×1` pointwise convolution instead of a dense `3×3×3` conv. At full `(256, 256, 192)` resolution, this reduces the `F.conv3d` `im2col` workspace by **32–64×** (from `40.50 GiB` at `l2_channels=512` down to `~0.63 GiB`).
- **`reparameterize(self, mu, logvar) -> torch.Tensor`**: Standard Gaussian reparameterization trick `mu + eps * exp(0.5 * logvar)`.
- **`encode_per_channel(self, x: torch.Tensor) -> torch.Tensor`**: Encodes each input channel independently through `vae_backbone.encode_stage_2_inputs()` and concatenates along channel dim.
- **`encode(self, x: torch.Tensor) -> Dict[str, Any]`**: Runs `encode_per_channel` then `encode_from_cached`.
- **`encode_from_cached(self, per_ch_z: torch.Tensor) -> Dict[str, Any]`**: Phase B entry point (`cached=True`). Computes `z_a`, `u_d`, `z_d_glob`, `z_p`, `z_s`, and their `mu`/`logvar` dicts.
- **`_decode_stage1(self, z_a, u_d, z_s) -> torch.Tensor`**: Upsamples `u_d` (trilinear fallback if shapes differ), concatenates with `z_a`, runs `fuse_conv` and `adagn1`.
- **`_decode_stage2(self, h, z_s) -> torch.Tensor`**: Runs `dec_up1` and `adagn2`.
- **`_decode_stage3(self, h) -> torch.Tensor`**: Runs `dec_up2` and `final_conv`.
- **`decode(self, z_a, u_d, z_s) -> Tuple[torch.Tensor, torch.Tensor]`**: Executes `_decode_stage1/2/3` with `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` when `self.training and torch.is_grad_enabled()`, plus `mask_head(u_d)`. Returns `(recon, mask_pred)`.
- **`forward(self, x: torch.Tensor, cached: bool = False) -> Dict[str, Any]`**: Dispatches to `encode_from_cached` (`cached=True`) or `encode` (`cached=False`), decodes, interpolates outputs to input shape if needed, and returns the full latent + reconstruction dict.

---

## 2. `medvae_backbone.py`

Optional Stanford MedVAE (`medvae_4_1_3d` / `medvae_8_1_3d`) wrapper matching the `AutoencoderKL` interface.

### `MedVAEBackbone(nn.Module)`
- **`__init__(self, model_name='medvae_4_1_3d', target_latent_channels=4, freeze_backbone=True, local_ckpt_path=None)`**: Loads MedVAE via `medvae.utils.factory`, builds `latent_proj` and `decode_proj` `Conv3d` adapters if native `embed_dim != target_latent_channels`, and optionally freezes backbone weights.
- **`_freeze_backbone(self)`** / **`unfreeze_backbone(self)`**: Toggles `requires_grad` on `self.vae.parameters()`.
- **`encode_stage_2_inputs(self, x: torch.Tensor) -> torch.Tensor`**: Encodes `(B, 1, D, H, W)` via `self.vae.encode(x).sample()` (under `torch.no_grad()` if frozen) and applies `latent_proj`.
- **`decode_stage_2_outputs(self, z: torch.Tensor) -> torch.Tensor`**: Applies `decode_proj` and `self.vae.decode(...)`.
- **`forward(self, x: torch.Tensor, decode: bool = False)`**: Encodes `x`, optionally decodes.
- **`get_param_groups(self) -> dict`**: Returns `{'backbone': ..., 'projection': ...}` for differential learning rates.
