import os
import h5py
import hdf5plugin
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple

class LatentUncertaintyManager:
    """
    Manages epistemic and aleatoric uncertainty propagation and exports
    disentangled latent representations to Zstd-compressed HDF5 store.
    """
    def __init__(self, num_mc_samples: int = 16):
        self.num_mc_samples = num_mc_samples

    def compute_posterior_stats(
        self,
        mu_d: torch.Tensor,
        logvar_d: torch.Tensor,
        mu_p: torch.Tensor,
        logvar_p: torch.Tensor
    ) -> Dict[str, Any]:
        """Calculates posterior mean, variance, and generates Monte Carlo draws."""
        var_d = torch.exp(logvar_d)
        var_p = torch.exp(logvar_p)
        sigma2 = torch.cat([var_d, var_p], dim=-1) # (B, 64 + 32 = 96)

        # Generate M Monte Carlo draws
        B = mu_d.size(0)
        mc_d = []
        mc_p = []
        for _ in range(self.num_mc_samples):
            eps_d = torch.randn_like(mu_d)
            eps_p = torch.randn_like(mu_p)
            mc_d.append(mu_d + eps_d * torch.sqrt(var_d))
            mc_p.append(mu_p + eps_p * torch.sqrt(var_p))

        # Shape: (M, B, dim)
        mc_d = torch.stack(mc_d, dim=0)
        mc_p = torch.stack(mc_p, dim=0)

        return {
            "sigma2": sigma2,
            "mc_draws_d": mc_d,
            "mc_draws_p": mc_p
        }

    @staticmethod
    def export_latents_to_hdf5(
        latent_store_path: str,
        patient_records: List[Dict[str, Any]]
    ):
        """
        Exports patient latents into Zstandard-compressed HDF5 store
        (Filter ID 32015, Zstd clevel 5) for fast Stage 4 ingestion.
        """
        os.makedirs(os.path.dirname(latent_store_path), exist_ok=True)
        compress_kwargs = hdf5plugin.Zstd(clevel=5)

        with h5py.File(latent_store_path, 'a') as f:
            p_root = f.require_group("patients")

            for rec in patient_records:
                uid = rec["uid"]
                p_grp = p_root.require_group(uid)

                # Static / Baseline latents
                if "z_a" in rec:
                    if "z_a" in p_grp: del p_grp["z_a"]
                    p_grp.create_dataset("z_a", data=rec["z_a"], **compress_kwargs)

                if "u_d" in rec:
                    if "u_d" in p_grp: del p_grp["u_d"]
                    p_grp.create_dataset("u_d", data=rec["u_d"], **compress_kwargs)

                if "z_d_glob" in rec:
                    if "z_d_glob" in p_grp: del p_grp["z_d_glob"]
                    p_grp.create_dataset("z_d_glob", data=rec["z_d_glob"])

                if "z_p" in rec:
                    if "z_p" in p_grp: del p_grp["z_p"]
                    p_grp.create_dataset("z_p", data=rec["z_p"])

                if "z_s" in rec:
                    if "z_s" in p_grp: del p_grp["z_s"]
                    p_grp.create_dataset("z_s", data=rec["z_s"])

                if "sigma2" in rec:
                    if "sigma2" in p_grp: del p_grp["sigma2"]
                    p_grp.create_dataset("sigma2", data=rec["sigma2"])

                if "ood_flag" in rec:
                    p_grp.attrs["ood_flag"] = int(rec["ood_flag"])

                p_grp.attrs["cohort"] = str(rec.get("cohort", ""))
                p_grp.attrs["domain"] = str(rec.get("domain", ""))

        print(f"Exported {len(patient_records)} patient latents to {latent_store_path}")
