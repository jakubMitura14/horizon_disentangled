"""Data Transforms — Physics-Based MRI Normalization and Spatial Standardization.

Implements domain-specific preprocessing transforms for brain and breast MRI:
- Brain: percentile clipping + z-score normalization per sequence
- Breast: physics-normalized 4-channel tensor (S0, PE_early, PE_late, SER)
- Spatial: center-crop or symmetric zero-pad to target 3D shape

All transforms operate on NumPy arrays and return float32 outputs suitable
for direct conversion to PyTorch tensors.

Key functions:
    normalize_brain_sequence: Percentile-clip and z-score a single brain MRI sequence.
    build_multisequence_brain_mri: Normalize and stack 4-sequence brain volume.
    build_physics_normalized_breast_dce: Construct physics-normalized breast DCE tensor.
    standardize_spatial_shape: Center-crop or zero-pad 3D/4D arrays.
"""
import numpy as np
import torch
from typing import Tuple, Optional, Union

def normalize_brain_sequence(arr: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Percentile clipping (0.5 to 99.5) and z-score normalization on non-zero brain mask."""
    arr = arr.astype(np.float32)
    if mask is None:
        mask = (arr > 0)
    
    if np.any(mask):
        p05, p995 = np.percentile(arr[mask], (0.5, 99.5))
        arr = np.clip(arr, p05, p995)
        mean_val = np.mean(arr[mask])
        std_val = np.std(arr[mask])
        if std_val > 1e-6:
            arr[mask] = (arr[mask] - mean_val) / std_val
        else:
            arr[mask] = arr[mask] - mean_val
    arr[~mask] = 0.0
    return arr

def standardize_spatial_shape(arr: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """Center crops or symmetric zero-pads 3D/4D spatial dimensions to match target_shape."""
    # arr shape is (C, D, H, W)
    C = arr.shape[0]
    curr_shape = arr.shape[1:]
    
    out = np.zeros((C, *target_shape), dtype=arr.dtype)
    
    # Calculate slice ranges for each dimension
    src_slices = []
    dst_slices = []
    for cur, tgt in zip(curr_shape, target_shape):
        if cur >= tgt:
            # Crop center
            start = (cur - tgt) // 2
            src_slices.append(slice(start, start + tgt))
            dst_slices.append(slice(0, tgt))
        else:
            # Pad center
            start = (tgt - cur) // 2
            src_slices.append(slice(0, cur))
            dst_slices.append(slice(start, start + cur))
            
    out[:, dst_slices[0], dst_slices[1], dst_slices[2]] = arr[:, src_slices[0], src_slices[1], src_slices[2]]
    return out

def build_physics_normalized_breast_dce(
    s0: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    ser: Optional[np.ndarray] = None,
    target_shape: Tuple[int, int, int] = (256, 256, 64),
    eps: float = 1e-4
) -> np.ndarray:
    """
    Constructs the physics-normalized 4-channel breast tensor (4, 256, 256, 64):
    (0) Z-scored pre-contrast S0,
    (1) Early percent enhancement PE_early = (S1 - S0) / max(S0, eps),
    (2) Late percent enhancement PE_late = (S2 - S0) / max(S0, eps),
    (3) Physical SER (0.0 to 3.0).
    """
    s0 = s0.astype(np.float32)
    s1 = s1.astype(np.float32)
    s2 = s2.astype(np.float32)

    # Valid FOV mask
    fov_mask = (s0 > 0) | (s1 > 0) | (s2 > 0)
    
    # 0: Z-scored S0
    s0_norm = np.zeros_like(s0)
    if np.any(fov_mask):
        p05, p995 = np.percentile(s0[fov_mask], (0.5, 99.5))
        clipped = np.clip(s0, p05, p995)
        mean_v = np.mean(clipped[fov_mask])
        std_v = np.std(clipped[fov_mask])
        if std_v > 1e-6:
            s0_norm[fov_mask] = (clipped[fov_mask] - mean_v) / std_v

    # 1: PE_early & 2: PE_late
    denom = np.maximum(s0, eps)
    pe_early = np.where(fov_mask, (s1 - s0) / denom, 0.0)
    pe_late = np.where(fov_mask, (s2 - s0) / denom, 0.0)

    # 3: Physical SER
    if ser is not None:
        ser_arr = ser.astype(np.float32)
    else:
        # Calculate SER from PE: PE_early / PE_late with physical clipping
        denom_pe = np.maximum(pe_late, eps)
        ser_arr = np.where(fov_mask, pe_early / denom_pe, 0.0)
    ser_arr = np.clip(ser_arr, 0.0, 3.0)

    # Stack into 4 channels
    stack = np.stack([s0_norm, pe_early, pe_late, ser_arr], axis=0)
    return standardize_spatial_shape(stack, target_shape)

def build_multisequence_brain_mri(
    vol_4seq: np.ndarray,
    target_shape: Tuple[int, int, int] = (192, 224, 160)
) -> np.ndarray:
    """
    Standardizes 4-sequence brain volume (CT1, T1, T2, FLAIR) to target spatial grid.
    vol_4seq shape: (4, D, H, W)
    """
    C = vol_4seq.shape[0]
    norm_channels = []
    for c in range(C):
        norm_channels.append(normalize_brain_sequence(vol_4seq[c]))
    stacked = np.stack(norm_channels, axis=0)
    return standardize_spatial_shape(stacked, target_shape)
