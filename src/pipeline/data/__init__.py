"""Data — Multimodal Oncology Dataset and Transforms.

HDF5-backed data pipeline for brain MRI (LUMIERE) and breast DCE-MRI
(I-SPY 2, BreastDCEDL) cohorts with physics-normalized transforms.

Key classes:
    MultimodalOncologyDataset: PyTorch Dataset reading horizon_master.vds.h5.
    worker_init_fn: Per-worker HDF5 handle + RNG seeding for DataLoader.

Key functions:
    normalize_brain_sequence: Percentile clip + z-score for brain MRI.
    build_multisequence_brain_mri: 4-sequence (CT1/T1/T2/FLAIR) standardization.
    build_physics_normalized_breast_dce: Physics-normalized 4-channel breast tensor.
    standardize_spatial_shape: Center-crop or zero-pad to target 3D shape.
"""
from src.pipeline.data.dataset import MultimodalOncologyDataset, worker_init_fn
from src.pipeline.data.transforms import (
    normalize_brain_sequence,
    build_multisequence_brain_mri,
    build_physics_normalized_breast_dce,
    standardize_spatial_shape
)

__all__ = [
    "MultimodalOncologyDataset",
    "worker_init_fn",
    "normalize_brain_sequence",
    "build_multisequence_brain_mri",
    "build_physics_normalized_breast_dce",
    "standardize_spatial_shape"
]
