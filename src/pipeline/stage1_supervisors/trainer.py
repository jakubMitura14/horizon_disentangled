"""Stage 1 — Trainer for Multimodal Supervisors.

DDP-aware training loop with:
- Gradient accumulation with correct no_sync() suppression
- GradScaler for mixed-precision (BF16) training
- Linear warmup → cosine annealing LR schedule
- Merlin-style contrastive learning (ClinicalTextEncoder + InfoNCE)
- Rank-0-only checkpoint saving with dist.barrier() synchronization

Key classes:
    Stage1Trainer: Full training loop for MultimodalSupervisorTrunk.

Example:
    >>> from src.pipeline.stage1_supervisors.trainer import Stage1Trainer
    >>> trainer = Stage1Trainer(config, domain='gbm')
    >>> metrics = trainer.train_step(batch)
"""
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, Any, Optional
from src.pipeline.stage1_supervisors.models import MultimodalSupervisorTrunk
from src.pipeline.stage1_supervisors.losses import (
    DifferentialOrdinalLoss, LogisticHazardLoss, SurvRNCLoss,
    MerlinInfoNCELoss, ClinicalTextEncoder, MerlinMultiTaskLoss
)
from src.pipeline.data.dataset import MultimodalOncologyDataset, worker_init_fn
from src.pipeline.config import PipelineConfig

class Stage1Trainer:
    """
    Trainer for Stage 1 Multimodal Supervisors.
    
    Now includes Merlin-style contrastive learning:
    - ClinicalTextEncoder creates text embeddings from structured clinical data
    - InfoNCE loss aligns visual embeddings with clinical text embeddings
    - Multi-label BCE phenotype prediction (Merlin's L_BCE)
    """
    def __init__(self, config: PipelineConfig, domain: str = "gbm"):
        self.config = config
        self.domain = domain
        self.device = torch.device(config.hardware.device)

        # Initialise process group if running under torchrun / SLURM
        if config.hardware.is_distributed and not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.is_main = (self.rank == 0)

        self.model = MultimodalSupervisorTrunk(
            domain=domain,
            in_channels=4,
            tabular_dim=16,
            feature_dim=config.stage1.feature_dim,
            num_ordinal_classes=5,
            num_survival_intervals=config.stage1.num_survival_intervals
        ).to(self.device)

        # Gap 3 — gradient checkpointing on the SwinUNETR backbone
        if hasattr(self.model, 'visual_encoder') and hasattr(self.model.visual_encoder, 'swin_unetr'):
            self.model.visual_encoder.swin_unetr.swinViT.use_checkpoint = True

        # Gap 1 — DDP wrap (after gradient checkpointing)
        # bucket_cap_mb=50 (NVLink 4.0 bandwidth ~900 GB/s, larger buckets reduce kernel launches)
        # gradient_as_bucket_view=True avoids gradient copy → saves ~model_size MB of VRAM
        # find_unused_parameters=True required: Stage1 has conditional image path
        if dist.is_initialized():
            self.model = DDP(
                self.model,
                device_ids=[config.hardware.local_rank],
                output_device=config.hardware.local_rank,
                find_unused_parameters=True,
                bucket_cap_mb=getattr(config.hardware, 'nccl_bucket_cap_mb', 50),
                gradient_as_bucket_view=True,
            )

        # Gap 2/4 — FSDP-2 for >400M-param visual backbone + torch.compile
        # Only applied when explicitly requested (use_fsdp=True in HardwareConfig)
        # or when the model has >400M params in a distributed setting.
        from src.pipeline.stage1_supervisors.models import wrap_model_for_cluster
        self.model = wrap_model_for_cluster(self.model, config)

        self.dior_loss_fn = DifferentialOrdinalLoss(
            num_classes=5,
            lambda_ord=config.stage1.lambda_ord,
            lambda_adj=config.stage1.lambda_adj,
            lambda_diff=config.stage1.lambda_diff,
            adjacent_margin=config.stage1.adjacent_margin,
            diff_alpha=config.stage1.diff_alpha
        ).to(self.device)

        self.survival_loss_fn = LogisticHazardLoss(num_intervals=config.stage1.num_survival_intervals).to(self.device)
        self.survrnc_loss_fn = SurvRNCLoss(tau=config.stage1.survrnc_tau).to(self.device)
        
        # === Merlin-style Contrastive Learning ===
        self.use_contrastive = getattr(config.stage1, 'use_contrastive', True)
        
        if self.use_contrastive:
            self.text_encoder = ClinicalTextEncoder(
                num_clinical_tokens=32,
                token_dim=64,
                hidden_dim=256,
                out_dim=256,
                num_heads=4,
                mode="structured",
            ).to(self.device)
            
            self.merlin_loss_fn = MerlinMultiTaskLoss(
                visual_dim=config.stage1.feature_dim,
                text_dim=256,
                num_phenotype_labels=32,
                lambda_infonce=getattr(config.stage1, 'lambda_infonce', 0.5),
                lambda_bce=getattr(config.stage1, 'lambda_bce', 0.5),
                proj_dim=256,
            ).to(self.device)
            
            # Merge all parameters for optimizer
            all_params = list(self.model.parameters()) + \
                         list(self.text_encoder.parameters()) + \
                         list(self.merlin_loss_fn.parameters())
        else:
            self.text_encoder = None
            self.merlin_loss_fn = None
            all_params = list(self.model.parameters())

        _lr = config.hardware.lr_scaled(config.stage1.lr) if config.hardware.is_distributed else config.stage1.lr
        self.optimizer = torch.optim.AdamW(all_params, lr=_lr, weight_decay=1e-4)
        self.use_amp = config.hardware.use_bfloat16 and ("cuda" in str(self.device))
        # Gap 7 — GradScaler for AMP + store sampler factory
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        # Gradient accumulation: effective batch = batch_size * grad_accum_steps * world_size
        self.grad_accum_steps = max(1, config.hardware.gradient_accumulation_steps)
        self._accum_step = 0
        # CosineAnnealingLR with linear warmup — step once per epoch from the train loop
        _warmup_epochs = max(1, getattr(config.stage1, 'warmup_epochs', 5))
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1e-3, end_factor=1.0, total_iters=_warmup_epochs
        )
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, getattr(config.stage1, 'num_epochs', 100) - _warmup_epochs),
            eta_min=_lr * 0.01,
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[self.warmup_scheduler, self.cosine_scheduler],
            milestones=[_warmup_epochs],
        )

    def train_step(self, batch: Dict[str, Any], is_last_accum_step: bool = True) -> Dict[str, float]:
        """Single training micro-step with optional gradient accumulation.

        When grad_accum_steps > 1, call train_step() grad_accum_steps times before the
        optimizer updates. Only the last call (is_last_accum_step=True) steps the optimizer.
        DDP gradient sync is correctly suppressed on non-last steps via no_sync() to avoid
        redundant all-reduces on intermediate accumulation steps.
        """
        import contextlib
        self.model.train()
        if self.text_encoder is not None:
            self.text_encoder.train()

        # Zero gradients only at the start of an accumulation window
        if self._accum_step == 0:
            self.optimizer.zero_grad(set_to_none=True)

        # Suppress DDP all-reduce on intermediate accumulation steps
        _is_ddp = dist.is_initialized() and hasattr(self.model, 'no_sync')
        _sync_ctx = (
            self.model.no_sync()
            if (_is_ddp and not is_last_accum_step)
            else contextlib.nullcontext()
        )

        clinical = batch['clinical'].to(self.device)
        has_img = batch.get('has_baseline_img', torch.zeros(clinical.size(0), dtype=torch.bool)).to(self.device)
        img = batch.get('baseline_img')
        if img is not None:
            # H2D copy + channels_last_3d in one pass — avoids a redundant reformat kernel
            img = img.to(device=self.device, memory_format=torch.channels_last_3d)

        # Synthetic/ground truth targets
        B = clinical.size(0)
        # Ordinal targets (default to 0..4 or clinical)
        targets_ord = torch.clamp((clinical[:, 1].long() % 5), 0, 4).to(self.device)
        # Survival intervals (0..19) & events (0 or 1)
        os_weeks = batch.get('os_weeks', torch.ones(B) * 50.0).to(self.device)
        interval_idx = torch.clamp((os_weeks / 10.0).long(), 0, self.config.stage1.num_survival_intervals - 1)
        events = torch.where(batch.get('event', torch.zeros(B)).to(self.device) == 1, 1, 0)

        with _sync_ctx:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.use_amp):
                out = self.model(img, clinical, has_image=has_img)

                # 1. DIOR Loss
                loss_dior, metrics_dior = self.dior_loss_fn(
                    out['cat_logits'], out['cum_probs'], out['embedding'], targets_ord
                )

                # 2. Survival Losses
                loss_surv_lh = self.survival_loss_fn(out['hazard_logits'], interval_idx, events)
                loss_survrnc = self.survrnc_loss_fn(out['embedding'], os_weeks, events)

                # 3. Binary Outcome Loss (if available)
                pcr = batch.get('pcr', -torch.ones(B)).to(self.device)
                mask_bin = (pcr >= 0)
                if mask_bin.any():
                    loss_bin = nn.functional.binary_cross_entropy_with_logits(
                        out['binary_logit'][mask_bin], pcr[mask_bin].float()
                    )
                else:
                    loss_bin = torch.tensor(0.0, device=self.device)

                # 4. Merlin-style Contrastive Loss
                loss_merlin = torch.tensor(0.0, device=self.device)
                metrics_merlin = {}

                if self.use_contrastive and self.text_encoder is not None and img is not None:
                    text_emb = self.text_encoder(clinical)
                    phenotype_targets = (clinical > 0).float()
                    if phenotype_targets.size(1) < 32:
                        phenotype_targets = torch.nn.functional.pad(
                            phenotype_targets, (0, 32 - phenotype_targets.size(1)), value=0.0
                        )
                    elif phenotype_targets.size(1) > 32:
                        phenotype_targets = phenotype_targets[:, :32]
                    phenotype_mask = torch.ones_like(phenotype_targets)
                    loss_merlin, metrics_merlin = self.merlin_loss_fn(
                        out['embedding'], text_emb,
                        phenotype_targets=phenotype_targets,
                        phenotype_mask=phenotype_mask,
                    )

                total_loss = (loss_dior + loss_surv_lh +
                             self.config.stage1.lambda_survrnc * loss_survrnc +
                             0.5 * loss_bin + loss_merlin)

                # Scale by 1/grad_accum_steps so effective gradient is correct regardless of accum
                total_loss = total_loss / self.grad_accum_steps

        self.scaler.scale(total_loss).backward()
        self._accum_step += 1

        if is_last_accum_step:
            # Must unscale before clip_grad_norm_ so norms are in true gradient space
            self.scaler.unscale_(self.optimizer)
            raw_model = self.model.module if dist.is_initialized() else self.model
            nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
            if self.text_encoder is not None:
                nn.utils.clip_grad_norm_(self.text_encoder.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self._accum_step = 0

        result = {
            "loss_total": total_loss.item(),
            "loss_dior": loss_dior.item(),
            "loss_surv_lh": loss_surv_lh.item(),
            "loss_survrnc": loss_survrnc.item(),
            "loss_bin": loss_bin.item() if isinstance(loss_bin, torch.Tensor) else loss_bin,
            "loss_merlin": loss_merlin.item() if isinstance(loss_merlin, torch.Tensor) else loss_merlin,
        }
        result.update(metrics_merlin)
        return result

    def save_checkpoint(self, path: str):
        """Only rank-0 writes to disk to prevent file corruption races."""
        if dist.is_initialized() and dist.get_rank() != 0:
            dist.barrier()   # non-rank-0 waits until rank-0 finishes writing
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        raw_model = self.model.module if dist.is_initialized() else self.model
        save_dict = {
            "model_state": raw_model.state_dict(),
            "domain": self.domain,
            "config": self.config.stage1
        }
        if self.text_encoder is not None:
            save_dict["text_encoder_state"] = self.text_encoder.state_dict()
        if self.merlin_loss_fn is not None:
            save_dict["merlin_loss_state"] = self.merlin_loss_fn.state_dict()
        torch.save(save_dict, path)
        print(f"Stage 1 {self.domain} supervisor saved to {path}")
        if dist.is_initialized():
            dist.barrier()   # rank-0 signals all others that write is complete

    def make_dataloader(self, dataset, batch_size: int, shuffle: bool = True):
        """Gap 7 — cluster-optimised DataLoader with DistributedSampler."""
        if dist.is_initialized():
            sampler = DistributedSampler(dataset, shuffle=shuffle,
                                         rank=self.rank, num_replicas=self.world_size)
            shuffle = False
        else:
            sampler = None
        # Store sampler on self so orchestrator can call set_epoch each epoch
        self._sampler = sampler
        # Use auto-computed workers from HardwareConfig (accounts for world_size)
        n_workers = getattr(self.config.hardware, 'dataloader_num_workers', None) or \
                    min(16, (os.cpu_count() or 4) // max(self.world_size, 1))
        pfactor = getattr(config.hardware, 'prefetch_factor', 4)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=n_workers,
            pin_memory=True,
            persistent_workers=(n_workers > 0),
            prefetch_factor=pfactor if n_workers > 0 else None,
            worker_init_fn=worker_init_fn,
            drop_last=True,
        ), sampler
