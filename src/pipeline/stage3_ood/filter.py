import os
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from src.pipeline.stage3_ood.manifold_vae import NormalityManifoldVAE
from src.pipeline.stage3_ood.scoring import OODScorer
from src.pipeline.config import PipelineConfig

class Stage3OODFilter:
    """Stage 3 Latent Manifold OOD Detection and Data Filtering Pipeline."""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = torch.device(config.hardware.device)

        self.manifold_vae = NormalityManifoldVAE(
            in_dim=config.stage3.latent_vec_dim,
            hidden_dim=128,
            bottleneck_dim=config.stage3.bottleneck_dim
        ).to(self.device)

        self.scorer = OODScorer(
            manifold_vae=self.manifold_vae,
            k_neighbors=config.stage3.k_neighbors,
            percentile_threshold=config.stage3.ood_percentile_threshold
        )

        self.optimizer = torch.optim.AdamW(self.manifold_vae.parameters(), lr=config.stage3.lr, weight_decay=1e-4)

    def train_manifold(self, clean_latents: torch.Tensor, epochs: int = 20, batch_size: int = 32):
        """Trains the Normality Manifold VAE on clean latent vectors."""
        self.manifold_vae.train()
        clean_latents = clean_latents.to(self.device)
        N = clean_latents.size(0)

        for epoch in range(epochs):
            perm = torch.randperm(N, device=self.device)
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, N, batch_size):
                idx = perm[i:i + batch_size]
                batch_zeta = clean_latents[idx]

                self.optimizer.zero_grad()
                out = self.manifold_vae(batch_zeta)
                loss = (out["recon_loss"] + 0.1 * out["kl_loss"]).mean()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

        print(f"Stage 3 Normality Manifold VAE trained ({epochs} epochs, final loss: {epoch_loss/max(num_batches, 1):.4f})")
        # Calibrate scorer on the reference bank
        self.scorer.calibrate(clean_latents, self.device)

    def filter_and_tag_patients(self, patient_latents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Tags patient visits with ood_flag in [0, 1]."""
        tagged_records = []
        for p in patient_latents:
            # Construct zeta vector: [mu_D, mu_P, logvar_D, q_stage1]
            mu_d = torch.from_numpy(p["z_d_glob"]).float() if isinstance(p["z_d_glob"], np.ndarray) else p["z_d_glob"].float()
            mu_p = torch.from_numpy(p["z_p"]).float() if isinstance(p["z_p"], np.ndarray) else p["z_p"].float()
            
            # Pad or construct full 165-D vector
            zeta_parts = [mu_d, mu_p]
            if "sigma2" in p:
                sig = torch.from_numpy(p["sigma2"][:64]).float() if isinstance(p["sigma2"], np.ndarray) else p["sigma2"][:64].float()
                zeta_parts.append(torch.log(torch.clamp(sig, min=1e-6)))
            else:
                zeta_parts.append(torch.zeros(64))

            if "q_stage1" in p:
                q = torch.from_numpy(p["q_stage1"]).float() if isinstance(p["q_stage1"], np.ndarray) else p["q_stage1"].float()
                zeta_parts.append(q)
            else:
                zeta_parts.append(torch.zeros(5))

            zeta = torch.cat(zeta_parts, dim=-1).unsqueeze(0).to(self.device)
            score, ood_flag = self.scorer.score(zeta)

            p_copy = dict(p)
            p_copy["ood_score"] = float(score.item())
            p_copy["ood_flag"] = int(ood_flag.item())
            tagged_records.append(p_copy)

        num_flagged = sum(r["ood_flag"] for r in tagged_records)
        print(f"Stage 3 OOD Filter: {num_flagged}/{len(tagged_records)} visits flagged as anomalous.")
        return tagged_records

    def save_checkpoint(self, path: str):
        """Save checkpoint — rank-0 only to avoid write races on shared filesystems."""
        import torch.distributed as dist
        # Non-rank-0 processes wait at the barrier; rank-0 performs the write
        if dist.is_initialized() and dist.get_rank() != 0:
            dist.barrier()
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "manifold_vae_state": self.manifold_vae.state_dict(),
            "mu_rec": self.scorer.mu_rec,
            "sigma_rec": self.scorer.sigma_rec,
            "mu_knn": self.scorer.mu_knn,
            "sigma_knn": self.scorer.sigma_knn,
            "threshold": self.scorer.threshold,
            "ref_bank": self.scorer.ref_bank.cpu() if self.scorer.ref_bank is not None else None
        }, path)
        print(f"Stage 3 OOD checkpoint saved to {path}")
        # Signal all other ranks that the write is complete
        if dist.is_initialized():
            dist.barrier()
