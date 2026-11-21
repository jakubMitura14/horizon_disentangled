import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, Any, Optional
from src.pipeline.stage2_causal_vae.architecture import TwoScaleHierarchicalCausalVAE
from src.pipeline.stage2_causal_vae.disentanglement import AsymmetricCausalDisentanglement
from src.pipeline.stage2_causal_vae.uncertainty import LatentUncertaintyManager
from src.pipeline.config import PipelineConfig

class Stage2Trainer:
    """Trainer for Stage 2 Hierarchical Causal VAE Disentanglement."""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = torch.device(config.hardware.device)

        if config.hardware.is_distributed and not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.is_main = (self.rank == 0)

        self.vae = TwoScaleHierarchicalCausalVAE(
            in_channels=4,
            anatomy_channels=config.stage2.anatomy_channels,
            disease_channels=config.stage2.disease_channels,
            global_disease_dim=config.stage2.global_disease_dim,
            patient_dim=config.stage2.patient_state_dim,
            style_dim=config.stage2.style_dim
        ).to(self.device)

        # Gap 3 — gradient checkpointing for VAE encoder
        if hasattr(self.vae, 'encoder'):
            for name, module in self.vae.named_modules():
                if hasattr(module, 'gradient_checkpointing'):
                    module.gradient_checkpointing = True

        self.disentangle = AsymmetricCausalDisentanglement(
            disease_dim=config.stage2.global_disease_dim,
            patient_dim=config.stage2.patient_state_dim,
            style_dim=config.stage2.style_dim,
            anatomy_channels=config.stage2.anatomy_channels,
            beta_tc=config.stage2.beta_tc,
            gamma_club=config.stage2.gamma_club,
            alpha_dp=config.stage2.alpha_dp
        ).to(self.device)

        # Gap 1 — DDP wrap
        # static_graph=True: VAE/Disentangle forward is fully deterministic (no unused params)
        # Enables comm/compute overlap that torch.compile expects; bucket_cap_mb=50 for NVLink
        if dist.is_initialized():
            self.vae = DDP(
                self.vae,
                device_ids=[config.hardware.local_rank],
                output_device=config.hardware.local_rank,
                find_unused_parameters=False,
                static_graph=True,
                bucket_cap_mb=getattr(config.hardware, 'nccl_bucket_cap_mb', 50),
                gradient_as_bucket_view=True,
            )
            self.disentangle = DDP(
                self.disentangle,
                device_ids=[config.hardware.local_rank],
                output_device=config.hardware.local_rank,
                find_unused_parameters=False,
                static_graph=True,
                bucket_cap_mb=getattr(config.hardware, 'nccl_bucket_cap_mb', 50),
                gradient_as_bucket_view=True,
            )

        self.uq_manager = LatentUncertaintyManager(num_mc_samples=config.stage2.mc_posterior_samples)

        _lr = config.hardware.lr_scaled(config.stage2.lr) if config.hardware.is_distributed else config.stage2.lr
        self.optimizer_vae = torch.optim.AdamW(self.vae.parameters(), lr=_lr, weight_decay=1e-4)
        self.optimizer_aux = torch.optim.AdamW(self.disentangle.parameters(), lr=_lr, weight_decay=1e-4)
        self.use_amp = config.hardware.use_bfloat16 and ("cuda" in str(self.device))
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        # Gradient accumulation support (matches Stage1Trainer interface)
        self.grad_accum_steps = max(1, config.hardware.gradient_accumulation_steps)
        self._accum_step = 0

    def train_step(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        import contextlib
        self.vae.train()
        self.disentangle.train()

        # H2D copy + channels_last_3d in one pass for 3D volumes
        x = x.to(device=self.device, memory_format=torch.channels_last_3d) if x.dim() == 5 else x.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        # Gradient accumulation: zero_grad only at the start of each cycle
        if self._accum_step == 0:
            self.optimizer_vae.zero_grad(set_to_none=True)
            self.optimizer_aux.zero_grad(set_to_none=True)

        # Suppress DDP all-reduce on intermediate accumulation steps
        _is_ddp = dist.is_initialized()
        _is_last = (self._accum_step + 1) % self.grad_accum_steps == 0
        _vae_ctx = (self.vae.no_sync() if (_is_ddp and not _is_last) else contextlib.nullcontext())
        _dis_ctx = (self.disentangle.no_sync() if (_is_ddp and not _is_last) else contextlib.nullcontext())

        with _vae_ctx, _dis_ctx:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.use_amp):
                out = self.vae(x)

            # 1. Reconstruction Loss
            loss_recon = F.l1_loss(out["recon"], x)

            # 2. Anatomy Loss on healthy parenchyma (1 - mask)
            if mask is not None:
                # Binary healthy mask
                healthy_weight = torch.clamp(1.0 - mask.mean(dim=1, keepdim=True).float(), 0.0, 1.0)
                loss_anatomy = torch.mean(healthy_weight * torch.abs(out["recon"] - x))
                
                # Disease mask loss
                loss_disease = F.binary_cross_entropy_with_logits(out["mask_pred"], mask.float())
            else:
                loss_anatomy = torch.tensor(0.0, device=self.device)
                loss_disease = torch.tensor(0.0, device=self.device)

            # 3. Standard Gaussian KL Divergence
            kl_d = -0.5 * torch.mean(1.0 + out["logvar_d"] - out["mu_d"].pow(2) - out["logvar_d"].exp())
            kl_p = -0.5 * torch.mean(1.0 + out["logvar_p"] - out["mu_p"].pow(2) - out["logvar_p"].exp())
            kl_s = -0.5 * torch.mean(1.0 + out["logvar_s"] - out["mu_s"].pow(2) - out["logvar_s"].exp())
            loss_kl = kl_d + kl_p + kl_s

            # 4. Asymmetric Causal Disentanglement Loss
            loss_disentangle, loss_aux, metrics_dis = self.disentangle.compute_disentangle_loss(
                out["z_d_glob"], out["z_p"], out["z_s"], out["z_a"]
            )

            total_vae_loss = (
                loss_recon +
                self.config.stage2.lambda_anatomy * loss_anatomy +
                self.config.stage2.lambda_disease * loss_disease +
                self.config.stage2.kl_weight * loss_kl +
                loss_disentangle
            )

        # Backward passes — scale by 1/grad_accum_steps for correct effective gradient
        # Both backward calls are inside _vae_ctx/_dis_ctx (no_sync on intermediate steps)
        self.scaler.scale(total_vae_loss / self.grad_accum_steps).backward(retain_graph=True)
        self.scaler.scale(loss_aux / self.grad_accum_steps).backward()

        self._accum_step += 1
        if self._accum_step % self.grad_accum_steps == 0:
            self._accum_step = 0
            # Unscale before clip so norms are in true gradient space
            self.scaler.unscale_(self.optimizer_vae)
            self.scaler.unscale_(self.optimizer_aux)
            raw_vae = self.vae.module if dist.is_initialized() else self.vae
            raw_dis = self.disentangle.module if dist.is_initialized() else self.disentangle
            nn.utils.clip_grad_norm_(raw_vae.parameters(), max_norm=1.0)
            nn.utils.clip_grad_norm_(raw_dis.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer_vae)
            self.scaler.step(self.optimizer_aux)
            self.scaler.update()

        return {
            "loss_total": total_vae_loss.item(),
            "loss_recon": loss_recon.item(),
            "loss_anatomy": loss_anatomy.item(),
            "loss_disease": loss_disease.item(),
            "loss_kl": loss_kl.item(),
            "loss_disentangle": loss_disentangle.item(),
            "loss_aux": loss_aux.item()
        }

    def save_checkpoint(self, path: str):
        """Only rank-0 writes to disk to prevent file corruption races."""
        if dist.is_initialized() and dist.get_rank() != 0:
            dist.barrier()
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        raw_vae = self.vae.module if dist.is_initialized() else self.vae
        raw_dis = self.disentangle.module if dist.is_initialized() else self.disentangle
        torch.save({
            "vae_state": raw_vae.state_dict(),
            "disentangle_state": raw_dis.state_dict(),
            "config": self.config.stage2
        }, path)
        print(f"Stage 2 Causal VAE saved to {path}")
        if dist.is_initialized():
            dist.barrier()

    def make_dataloader(self, dataset, batch_size: int, shuffle: bool = True):
        """Gap 7 — cluster-optimised DataLoader with DistributedSampler."""
        if dist.is_initialized():
            sampler = DistributedSampler(dataset, shuffle=shuffle,
                                         rank=self.rank, num_replicas=self.world_size)
            shuffle = False
        else:
            sampler = None
        # Use auto-computed workers from HardwareConfig (accounts for world_size)
        n_workers = getattr(self.config.hardware, 'dataloader_num_workers', None) or \
                    min(16, (os.cpu_count() or 4) // max(self.world_size, 1))
        pfactor = getattr(self.config.hardware, 'prefetch_factor', 4)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=n_workers,
            pin_memory=True,
            persistent_workers=(n_workers > 0),
            prefetch_factor=pfactor if n_workers > 0 else None,
            drop_last=True,
        ), sampler
