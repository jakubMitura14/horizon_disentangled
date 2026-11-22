"""Stage 2 — Hierarchical Causal VAE.

Two-scale disentangled variational autoencoder separating anatomy (Z_A),
disease (Z_D), patient state (Z_P), and scanner style (Z_S) via
FactorVAE Total Correlation penalty and vCLUB mutual information bounds.

Key classes:
    TwoScaleHierarchicalCausalVAE: AutoencoderKL-based hierarchical VAE (~42M params).
    AsymmetricCausalDisentanglement: TC + vCLUB disentanglement module.
    LatentUncertaintyManager: MC posterior sampling and HDF5 export.
    Stage2Trainer: DDP-aware trainer with dual optimizer (VAE + auxiliary).
    MedVAEBackbone: Optional Stanford MedVAE backbone (MIT, 160-245M params).

Example:
    >>> from src.pipeline.stage2_causal_vae import Stage2Trainer
    >>> trainer = Stage2Trainer(config)
"""
from src.pipeline.stage2_causal_vae.architecture import TwoScaleHierarchicalCausalVAE
from src.pipeline.stage2_causal_vae.disentanglement import AsymmetricCausalDisentanglement, FactorVAEDiscriminator, VariationalCLUB
from src.pipeline.stage2_causal_vae.uncertainty import LatentUncertaintyManager
from src.pipeline.stage2_causal_vae.trainer import Stage2Trainer
from src.pipeline.stage2_causal_vae.medvae_backbone import MedVAEBackbone

__all__ = [
    "TwoScaleHierarchicalCausalVAE",
    "AsymmetricCausalDisentanglement",
    "FactorVAEDiscriminator",
    "VariationalCLUB",
    "LatentUncertaintyManager",
    "Stage2Trainer",
    "MedVAEBackbone",
]
