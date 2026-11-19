import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

        self.model = MultimodalSupervisorTrunk(
            domain=domain,
            in_channels=4,
            tabular_dim=16,
            feature_dim=config.stage1.feature_dim,
            num_ordinal_classes=5,
            num_survival_intervals=config.stage1.num_survival_intervals
        ).to(self.device)

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

        self.optimizer = torch.optim.AdamW(all_params, lr=config.stage1.lr, weight_decay=1e-4)
        self.use_amp = config.hardware.use_bfloat16 and ("cuda" in str(self.device))

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        self.model.train()
        if self.text_encoder is not None:
            self.text_encoder.train()
        self.optimizer.zero_grad()

        clinical = batch['clinical'].to(self.device)
        has_img = batch.get('has_baseline_img', torch.zeros(clinical.size(0), dtype=torch.bool)).to(self.device)
        img = batch.get('baseline_img')
        if img is not None:
            img = img.to(self.device)

        # Synthetic/ground truth targets
        B = clinical.size(0)
        # Ordinal targets (default to 0..4 or clinical)
        targets_ord = torch.clamp((clinical[:, 1].long() % 5), 0, 4).to(self.device)
        # Survival intervals (0..19) & events (0 or 1)
        os_weeks = batch.get('os_weeks', torch.ones(B) * 50.0).to(self.device)
        interval_idx = torch.clamp((os_weeks / 10.0).long(), 0, self.config.stage1.num_survival_intervals - 1)
        events = torch.where(batch.get('event', torch.zeros(B)).to(self.device) == 1, 1, 0)

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

            # 4. Merlin-style Contrastive Loss (NEW)
            loss_merlin = torch.tensor(0.0, device=self.device)
            metrics_merlin = {}
            
            if self.use_contrastive and self.text_encoder is not None and img is not None:
                # Create text embeddings from clinical features
                text_emb = self.text_encoder(clinical)
                
                # Create phenotype targets from clinical features
                # Binary phenotype labels derived from clinical data
                # (in production: map to PheWAS/ICD codes from clinical notes)
                phenotype_targets = (clinical > 0).float()
                # Pad/truncate to 32 phenotype labels
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

        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        if self.text_encoder is not None:
            nn.utils.clip_grad_norm_(self.text_encoder.parameters(), max_norm=1.0)
        self.optimizer.step()

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
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            "model_state": self.model.state_dict(),
            "domain": self.domain,
            "config": self.config.stage1
        }
        if self.text_encoder is not None:
            save_dict["text_encoder_state"] = self.text_encoder.state_dict()
        if self.merlin_loss_fn is not None:
            save_dict["merlin_loss_state"] = self.merlin_loss_fn.state_dict()
        torch.save(save_dict, path)
        print(f"Stage 1 {self.domain} supervisor saved to {path}")
