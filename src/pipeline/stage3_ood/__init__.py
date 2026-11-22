"""Stage 3 — Latent Out-of-Distribution Detection.

VAE-based anomaly detection on disentangled latent vectors, combining
reconstruction error with k-NN latent distance scoring to flag OOD visits
and filter label noise before Stage 4 dynamics training.

Key classes:
    NormalityManifoldVAE: 4-layer residual MLP VAE on 165-D latent vectors.
    OODScorer: Composite reconstruction + k-NN scorer with percentile threshold.
    Stage3OODFilter: End-to-end OOD detection and patient tagging pipeline.

Example:
    >>> from src.pipeline.stage3_ood import Stage3OODFilter
    >>> ood = Stage3OODFilter(config)
    >>> ood.train_manifold(clean_latents, epochs=20)
"""
from src.pipeline.stage3_ood.manifold_vae import NormalityManifoldVAE
from src.pipeline.stage3_ood.scoring import OODScorer
from src.pipeline.stage3_ood.filter import Stage3OODFilter

__all__ = [
    "NormalityManifoldVAE",
    "OODScorer",
    "Stage3OODFilter"
]
