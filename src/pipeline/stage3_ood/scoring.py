"""Stage 3 — OOD Scoring and Thresholding.

Computes the composite OOD score S_OOD combining z-scored VAE reconstruction
error with z-scored k-NN (k=15) latent distance against a clean reference bank.
Threshold is set at the 95th percentile of the calibration set.

Key classes:
    OODScorer: Calibration + scoring engine for latent OOD detection.

Example:
    >>> scorer = OODScorer(manifold_vae, k_neighbors=15)
    >>> scorer.calibrate(clean_latents, device)
    >>> scores, flags = scorer.score(test_latents)
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
from src.pipeline.stage3_ood.manifold_vae import NormalityManifoldVAE

class OODScorer:
    """
    Computes Out-of-Distribution (OOD) score:
    S_OOD(zeta_i) = (||zeta_i - zeta_hat_i||^2 - mu_rec) / sigma_rec + (||zeta_i - zeta_(k)||_2 - mu_knn) / sigma_knn
    where zeta_(k) is the distance to the k-th (k=15) nearest clean neighbor.
    """
    def __init__(
        self,
        manifold_vae: NormalityManifoldVAE,
        k_neighbors: int = 15,
        percentile_threshold: float = 95.0
    ):
        self.manifold_vae = manifold_vae
        self.k = k_neighbors
        self.percentile = percentile_threshold

        self.mu_rec = 0.0
        self.sigma_rec = 1.0
        self.mu_knn = 0.0
        self.sigma_knn = 1.0
        self.threshold = 0.0
        self.ref_bank: Optional[torch.Tensor] = None
        self.is_calibrated = False

    def calibrate(self, clean_latents: torch.Tensor, device: torch.device):
        """Calibrates normalization parameters and 95th-percentile threshold on clean reference visits."""
        self.manifold_vae.eval()
        clean_latents = clean_latents.to(device)
        self.ref_bank = clean_latents.clone()

        with torch.no_grad():
            out = self.manifold_vae(clean_latents)
            recon_errors = out["recon_loss"].cpu().numpy()

            # Compute k-NN distances against reference bank
            dist_matrix = torch.cdist(clean_latents, clean_latents, p=2) # (N, N)
            # Sort distances per row, take k-th neighbor
            k_val = min(self.k, clean_latents.size(0) - 1)
            knn_dists, _ = torch.topk(dist_matrix, k=k_val + 1, largest=False, dim=-1)
            knn_errors = knn_dists[:, k_val].cpu().numpy()

        self.mu_rec = float(np.mean(recon_errors))
        self.sigma_rec = float(np.std(recon_errors)) if np.std(recon_errors) > 1e-6 else 1.0

        self.mu_knn = float(np.mean(knn_errors))
        self.sigma_knn = float(np.std(knn_errors)) if np.std(knn_errors) > 1e-6 else 1.0

        # Standardized scores on calibration set
        scores = (recon_errors - self.mu_rec) / self.sigma_rec + (knn_errors - self.mu_knn) / self.sigma_knn
        self.threshold = float(np.percentile(scores, self.percentile))
        self.is_calibrated = True

        print(f"OOD Scorer Calibrated: mu_rec={self.mu_rec:.4f}, sigma_rec={self.sigma_rec:.4f}, "
              f"mu_knn={self.mu_knn:.4f}, sigma_knn={self.sigma_knn:.4f}, threshold_95={self.threshold:.4f}")

    def score(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates S_OOD scores and binary flags (ood_flag = 1 for score > threshold).
        Returns: (scores, ood_flags)
        """
        assert self.is_calibrated, "OODScorer must be calibrated before scoring."
        self.manifold_vae.eval()
        device = latents.device

        with torch.no_grad():
            out = self.manifold_vae(latents)
            recon_err = out["recon_loss"] # (B,)

            # Distance to k-th neighbor in reference bank
            ref = self.ref_bank.to(device)
            dist_mat = torch.cdist(latents, ref, p=2) # (B, N_ref)
            k_val = min(self.k, ref.size(0))
            knn_dists, _ = torch.topk(dist_mat, k=k_val, largest=False, dim=-1)
            knn_err = knn_dists[:, k_val - 1] # (B,)

            # Standardized score
            score = (recon_err - self.mu_rec) / self.sigma_rec + (knn_err - self.mu_knn) / self.sigma_knn
            ood_flag = (score > self.threshold).long()

        return score, ood_flag
