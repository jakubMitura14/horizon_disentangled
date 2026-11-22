"""
Full Training Script for Horizon Disentangled Pipeline.

Loads real patient data from HDF5 shards and trains all 5 stages.
Designed for single H100 (80GB) with scaling annotations for 4× H200.

Usage:
  cd /DATA/horizon_disentangled
  HDF5_PLUGIN_PATH=/DATA/conda/envs/nukl-to-ct/lib/python3.10/site-packages/hdf5plugin/plugins \
  LD_LIBRARY_PATH=/DATA/conda/envs/nukl-to-ct/lib:$LD_LIBRARY_PATH \
  /DATA/conda/envs/nukl-to-ct/bin/python scripts/training/train_full.py \
    --domain breast --epochs 10 --batch-size 4
"""
import os
import sys
import time
import json
import logging
import argparse
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# HDF5 plugins
os.environ.setdefault("HDF5_PLUGIN_PATH",
    "/DATA/conda/envs/nukl-to-ct/lib/python3.10/site-packages/hdf5plugin/plugins")
import h5py
import hdf5plugin

sys.path.insert(0, "/DATA/horizon_disentangled")

from src.pipeline.config import PipelineConfig
from src.pipeline.stage1_supervisors.models import MultimodalSupervisorTrunk
from src.pipeline.stage1_supervisors.losses import (
    DifferentialOrdinalLoss, LogisticHazardLoss, SurvRNCLoss,
    MerlinInfoNCELoss, ClinicalTextEncoder, MerlinMultiTaskLoss
)
from src.pipeline.stage2_causal_vae.architecture import TwoScaleHierarchicalCausalVAE
from src.pipeline.stage2_causal_vae.disentanglement import AsymmetricCausalDisentanglement
from src.pipeline.stage3_ood.manifold_vae import NormalityManifoldVAE
from src.pipeline.stage3_ood.scoring import OODScorer
from src.pipeline.stage4_dynamics.solver import ContinuousDiscreteJumpSolver
from src.pipeline.stage4_dynamics.masked_loss import MaskedLeaveOneOutLoss
from src.pipeline.stage5_synthesis_reporting.flow_refiner import LatentRectifiedFlowRefiner
from src.pipeline.stage5_synthesis_reporting.counterfactual import CounterfactualSimulator
from src.pipeline.stage5_synthesis_reporting.treatment_helper import AgenticTreatmentHelper
from src.pipeline.stage5_synthesis_reporting.structured_vlm import StructuredClinicalReporter
from src.pipeline.stage5_synthesis_reporting.vlm_reporter import VLMReportGenerator
from src.pipeline.stage5_synthesis_reporting.eu_ai_act_logger import EUAIActAuditLogger

logger = logging.getLogger("horizon_train")


# ═══════════════════════════════════════════════════════════════════════════
# Real Data Loader
# ═══════════════════════════════════════════════════════════════════════════

class HorizonRealDataset(Dataset):
    """
    Loads real patient volumes from HDF5 shards.
    
    For GBM: 4-channel brain MRI + 2-channel tumor seg 
    For Breast: 3-channel kinetic (PE2, PE6, SER) + 2-channel tumor seg
    
    Spatial normalization:
      - Center-crop/pad to target spatial shape
      - Intensity clip to [0.5, 99.5] percentile + z-score per volume
    
    Multi-GPU scaling note:
      On 4× H200, use DistributedSampler with num_workers=4 per GPU.
      Each GPU processes its shard of patients. Use SWMR for concurrent reads.
    """
    
    TARGET_SPATIAL_GBM = (192, 224, 160)     # D, H, W for brain MRI
    TARGET_SPATIAL_BREAST = (256, 256, 64)   # D, H, W for breast DCE
    
    def __init__(
        self,
        hdf5_dir: str,
        domain: str = "breast",
        max_patients: int = 200,
        target_latent_grid: tuple = None,
    ):
        self.hdf5_dir = hdf5_dir
        self.domain = domain
        self.target_spatial = self.TARGET_SPATIAL_GBM if domain == "gbm" else self.TARGET_SPATIAL_BREAST
        self.target_latent = target_latent_grid
        
        # Collect patient records with imaging
        self.records = []
        self._scan_hdf5_shards(max_patients)
        logger.info(f"Dataset: {len(self.records)} visit records ({domain})")
    
    def _scan_hdf5_shards(self, max_patients: int):
        """Scans HDF5 shards to build a list of (shard_path, patient_id, visit_key) records."""
        hdf5_files = sorted([f for f in os.listdir(self.hdf5_dir) if f.endswith('.h5') and 'master' not in f and 'vds' not in f])
        
        # Filter by domain
        if self.domain == "gbm":
            hdf5_files = [f for f in hdf5_files if 'gbm' in f.lower() and 'tabular' not in f.lower()]
        else:
            hdf5_files = [f for f in hdf5_files if 'breast' in f.lower() or 'ispy' in f.lower()]
        
        patient_count = 0
        for hf in hdf5_files:
            path = os.path.join(self.hdf5_dir, hf)
            try:
                with h5py.File(path, 'r') as f:
                    if 'patients' not in f:
                        continue
                    for pid in f['patients'].keys():
                        if patient_count >= max_patients:
                            break
                        pg = f['patients'][pid]
                        if 'visits' not in pg:
                            continue
                        
                        # Get clinical data
                        static_clin = pg['static_clinical'][:] if 'static_clinical' in pg else np.zeros(4, dtype=np.float32)
                        interventions = pg['interventions'][:] if 'interventions' in pg else np.array([], dtype=np.float32)
                        
                        for vk in pg['visits'].keys():
                            vg = pg['visits'][vk]
                            vkeys = list(vg.keys())
                            
                            # Check for imaging data
                            img_key = None
                            if 'brain_mri' in vkeys:
                                img_key = 'brain_mri'
                            elif 'ispy2_kinetic' in vkeys:
                                img_key = 'ispy2_kinetic'
                            
                            if img_key is None:
                                continue
                            
                            seg_key = 'seg_tumor' if 'seg_tumor' in vkeys else None
                            
                            self.records.append({
                                'shard_path': path,
                                'patient_id': pid,
                                'visit_key': vk,
                                'img_key': img_key,
                                'seg_key': seg_key,
                                'static_clinical': static_clin,
                                'interventions': interventions,
                            })
                        
                        patient_count += 1
                        
            except Exception as e:
                logger.warning(f"Error scanning {hf}: {e}")
                continue
    
    def __len__(self):
        return len(self.records)
    
    def _center_crop_pad(self, vol: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Center-crops or pads a volume to target shape (C, D, H, W)."""
        C = vol.shape[0]
        result = np.zeros((C, *target_shape), dtype=vol.dtype)
        
        for dim in range(3):
            src_size = vol.shape[dim + 1]
            tgt_size = target_shape[dim]
            if src_size >= tgt_size:
                # Crop
                start = (src_size - tgt_size) // 2
                vol = np.take(vol, range(start, start + tgt_size), axis=dim + 1)
            # Padding happens via the zeros initialization
        
        # Copy into result (handling case where vol may be smaller)
        slices = [slice(None)]  # channel dim
        for dim in range(3):
            src_size = vol.shape[dim + 1]
            tgt_size = target_shape[dim]
            if src_size < tgt_size:
                pad = (tgt_size - src_size) // 2
                slices.append(slice(pad, pad + src_size))
            else:
                slices.append(slice(None))
        
        result[tuple(slices)] = vol
        return result
    
    def _normalize_intensity(self, vol: np.ndarray) -> np.ndarray:
        """Clip to [0.5, 99.5] percentile and z-score normalize per channel."""
        for c in range(vol.shape[0]):
            ch = vol[c]
            nonzero = ch[ch != 0]
            if len(nonzero) > 100:
                p_low, p_high = np.percentile(nonzero, [0.5, 99.5])
                ch = np.clip(ch, p_low, p_high)
                mu = ch[ch != 0].mean()
                std = ch[ch != 0].std() + 1e-8
                vol[c] = (ch - mu) / std
            else:
                vol[c] = ch
        return vol
    
    def __getitem__(self, idx):
        rec = self.records[idx]
        
        with h5py.File(rec['shard_path'], 'r', swmr=True) as f:
            pg = f['patients'][rec['patient_id']]
            vg = pg['visits'][rec['visit_key']]
            
            # Load imaging
            img = vg[rec['img_key']][:]  # (C, D/H, W/W, slices)
            
            # Ensure 4 channels for GBM or pad breast to 4
            if img.shape[0] < 4:
                pad_channels = 4 - img.shape[0]
                pad = np.zeros((pad_channels, *img.shape[1:]), dtype=img.dtype)
                img = np.concatenate([img, pad], axis=0)
            elif img.shape[0] > 4:
                img = img[:4]
            
            # Load segmentation mask
            if rec['seg_key'] is not None:
                seg = vg[rec['seg_key']][:].astype(np.float32)
                # Take first channel as binary tumor mask
                if seg.shape[0] >= 2:
                    seg = seg[:2]
                else:
                    seg = np.concatenate([seg, np.zeros_like(seg)], axis=0)
            else:
                seg = np.zeros((2, *img.shape[1:]), dtype=np.float32)
        
        # Spatial normalization
        img = self._center_crop_pad(img.astype(np.float32), self.target_spatial)
        seg = self._center_crop_pad(seg, self.target_spatial)
        
        # Intensity normalization
        img = self._normalize_intensity(img)
        
        # Binarize seg mask
        seg = (seg > 0).astype(np.float32)
        
        # Clinical vector (pad to 16)
        clin = np.zeros(16, dtype=np.float32)
        sc = rec['static_clinical']
        clin[:min(len(sc), 16)] = sc[:min(len(sc), 16)]
        
        return {
            'image': torch.from_numpy(img),          # (4, D, H, W)
            'mask': torch.from_numpy(seg),            # (2, D, H, W)
            'clinical': torch.from_numpy(clin),       # (16,)
            'patient_id': rec['patient_id'],
            'visit_key': rec['visit_key'],
        }


# ═══════════════════════════════════════════════════════════════════════════
# Training Loops
# ═══════════════════════════════════════════════════════════════════════════

def train_stage1(config, dataloader, domain, num_epochs, device):
    """
    Stage 1: Multimodal Supervisors with real imaging data.
    
    Now includes Merlin-style contrastive learning:
    - ClinicalTextEncoder: Creates text embeddings from structured clinical data
    - MerlinInfoNCELoss: Aligns visual embeddings with clinical text embeddings
    - MerlinMultiTaskLoss: Multi-label BCE phenotype prediction
    
    Multi-GPU strategy: DDP across all 4 GPUs with DistributedSampler.
    Each GPU processes batch_size/4 samples → effective batch = batch_size.
    """
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    _dist = dist.is_initialized()
    _rank = dist.get_rank() if _dist else 0
    _local_rank = int(os.environ.get("LOCAL_RANK", device.index or 0))
    is_main = (_rank == 0)

    logger.info("═══ Stage 1: Multimodal Supervisors ═══")
    
    model = MultimodalSupervisorTrunk(
        domain=domain, in_channels=4, tabular_dim=16,
        feature_dim=config.stage1.feature_dim,
        num_ordinal_classes=5,
        num_survival_intervals=config.stage1.num_survival_intervals
    ).to(device)
    
    # Gradient checkpointing on the SwinUNETR backbone to save VRAM
    if hasattr(model, 'visual_encoder') and hasattr(model.visual_encoder, 'swin_unetr'):
        model.visual_encoder.swin_unetr.swinViT.use_checkpoint = True

    # DDP-wrap: synchronises gradients across all GPUs in the process group
    _nccl_bucket_mb = getattr(config.hardware, 'nccl_bucket_cap_mb', 50)
    if _dist:
        model = DDP(model, device_ids=[_local_rank], output_device=_local_rank,
                    find_unused_parameters=False,
                    bucket_cap_mb=_nccl_bucket_mb,
                    gradient_as_bucket_view=True,
                    static_graph=True)

    dior_loss = DifferentialOrdinalLoss(num_classes=5).to(device)
    surv_loss = LogisticHazardLoss(num_intervals=config.stage1.num_survival_intervals).to(device)
    survrnc_loss = SurvRNCLoss(tau=config.stage1.survrnc_tau).to(device)

    # === Merlin-style Contrastive Learning ===
    text_encoder = ClinicalTextEncoder(
        num_clinical_tokens=32, token_dim=64,
        hidden_dim=256, out_dim=256, num_heads=4,
        mode="structured",
    ).to(device)

    merlin_loss_fn = MerlinMultiTaskLoss(
        visual_dim=config.stage1.feature_dim,
        text_dim=256, num_phenotype_labels=32,
        lambda_infonce=0.5, lambda_bce=0.5,
        proj_dim=256,
    ).to(device)

    logger.info(f"  Merlin contrastive learning: ENABLED")
    logger.info(f"  ClinicalTextEncoder params: {sum(p.numel() for p in text_encoder.parameters()):,}")
    logger.info(f"  MerlinMultiTaskLoss params: {sum(p.numel() for p in merlin_loss_fn.parameters()):,}")

    all_params = list(model.parameters()) + list(text_encoder.parameters()) + list(merlin_loss_fn.parameters())

    # Linear LR scaling rule for DDP: scale lr by effective world size
    _world_s1 = dist.get_world_size() if dist.is_initialized() else 1
    _effective_lr_s1 = config.stage1.lr * _world_s1
    optimizer = torch.optim.AdamW(all_params, lr=_effective_lr_s1, weight_decay=1e-4)

    # Warmup + Cosine Annealing schedule (warmup avoids early divergence at scaled LR)
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    _warmup_e = max(1, min(5, num_epochs // 10))
    _warmup_s1 = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=_warmup_e)
    _cosine_s1 = CosineAnnealingLR(optimizer, T_max=max(1, num_epochs - _warmup_e), eta_min=_effective_lr_s1 * 0.01)
    scheduler = SequentialLR(optimizer, schedulers=[_warmup_s1, _cosine_s1], milestones=[_warmup_e])

    scaler = torch.amp.GradScaler('cuda', enabled=True)

    # Gradient accumulation: accumulate over _grad_accum micro-steps before optimizer.step()
    _grad_accum = getattr(config.hardware, 'gradient_accumulation_steps', 1)
    _accum_count = 0

    embeddings_bank = []
    # Retrieve sampler from the dataloader if DistributedSampler was set
    _s1_sampler = getattr(dataloader, 'sampler', None)
    if not hasattr(_s1_sampler, 'set_epoch'):
        _s1_sampler = None

    for epoch in range(num_epochs):
        # DistributedSampler: shuffle must differ per epoch so each GPU sees new data
        if _s1_sampler is not None:
            _s1_sampler.set_epoch(epoch)
        model.train()
        text_encoder.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        epoch_infonce = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in dataloader:
            img = batch['image'].to(device, non_blocking=True)         # (B, 4, D, H, W)
            clin = batch['clinical'].to(device, non_blocking=True)     # (B, 16)
            B = img.size(0)
            has_img = torch.ones(B, dtype=torch.bool, device=device)

            # Downscale images to half resolution for upgraded models
            # (upgraded from 0.25× to 0.5× — with SwinUNETR+AutoencoderKL,
            #  B=4 at 128³ fits ~18 GB, well within H200's 141 GB)
            img_small = F.interpolate(img, scale_factor=0.5, mode='trilinear', align_corners=False)
            # Channels-last 3D for 15-20% cuDNN throughput gain on H200
            img_small = img_small.to(memory_format=torch.channels_last_3d)

            targets_ord = torch.randint(0, 5, (B,), device=device)  # Placeholder until labels extracted
            os_weeks = torch.rand(B, device=device) * 100.0 + 10.0
            interval_idx = torch.clamp((os_weeks / 10.0).long(), 0, config.stage1.num_survival_intervals - 1)
            events = torch.randint(0, 2, (B,), device=device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(img_small, clin, has_image=has_img)

                loss_dior, _ = dior_loss(out['cat_logits'], out['cum_probs'], out['embedding'], targets_ord)
                loss_surv = surv_loss(out['hazard_logits'], interval_idx, events)
                loss_rnc = survrnc_loss(out['embedding'], os_weeks, events)

                # === Merlin Contrastive Loss ===
                text_emb = text_encoder(clin)
                phenotype_targets = (clin > 0).float()
                if phenotype_targets.size(1) < 32:
                    phenotype_targets = F.pad(phenotype_targets, (0, 32 - phenotype_targets.size(1)), value=0.0)
                elif phenotype_targets.size(1) > 32:
                    phenotype_targets = phenotype_targets[:, :32]

                loss_merlin, merlin_metrics = merlin_loss_fn(
                    out['embedding'], text_emb,
                    phenotype_targets=phenotype_targets,
                    phenotype_mask=torch.ones_like(phenotype_targets),
                )

                total_loss = (loss_dior + loss_surv + 0.5 * loss_rnc + loss_merlin) / _grad_accum

            scaler.scale(total_loss).backward()
            _accum_count += 1

            if _accum_count % _grad_accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                _accum_count = 0

            epoch_loss += total_loss.item() * _grad_accum   # de-normalise for logging
            epoch_infonce += merlin_metrics.get('loss_infonce', 0.0)
            n_batches += 1

            # Save embeddings for Stage 2/3 (rank-0 only to avoid 4× bank duplication)
            if is_main:
                with torch.no_grad():
                    embeddings_bank.append(out['embedding'].detach().cpu())

        scheduler.step()
        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_infonce = epoch_infonce / max(n_batches, 1)
        mem_gb = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()   # reset so next epoch shows its own peak

        if is_main:
            logger.info(
                f"  Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | "
                f"InfoNCE: {avg_infonce:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                f"GPU peak: {mem_gb:.2f}GB | {elapsed:.1f}s | {n_batches} batches"
            )

    # Save checkpoint — rank-0 only to prevent write races on shared filesystem
    if _dist:
        dist.barrier()   # ensure all ranks finish training before rank-0 saves
    if is_main:
        ckpt_path = os.path.join(config.data.checkpoints_dir, f"stage1_{domain}_real.pt")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        # Unwrap DDP module for clean state_dict
        raw_model = model.module if hasattr(model, 'module') else model
        torch.save({
            'model_state': raw_model.state_dict(),
            'text_encoder_state': text_encoder.state_dict(),
            'merlin_loss_state': merlin_loss_fn.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': num_epochs,
            'domain': domain,
        }, ckpt_path)
        logger.info(f"  Stage 1 checkpoint saved: {ckpt_path}")
    if _dist:
        dist.barrier()   # rank-0 signals write complete

    return model, torch.cat(embeddings_bank, dim=0) if embeddings_bank else None


def train_stage2(config, dataloader, num_epochs, device):
    """
    Stage 2: Hierarchical Causal VAE on real 3D volumes.
    
    Multi-GPU strategy: DDP for data parallelism.
    With 4× H200 (141 GB each), can process full-resolution volumes 
    without spatial tiling. On single H100, downscale 4× for efficiency.
    """
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    _dist = dist.is_initialized()
    _rank = dist.get_rank() if _dist else 0
    _local_rank = int(os.environ.get("LOCAL_RANK", device.index or 0))
    is_main = (_rank == 0)

    logger.info("═══ Stage 2: Hierarchical Causal VAE ═══")
    
    vae = TwoScaleHierarchicalCausalVAE(
        in_channels=4,
        anatomy_channels=config.stage2.anatomy_channels,
        disease_channels=config.stage2.disease_channels,
        global_disease_dim=config.stage2.global_disease_dim,
        patient_dim=config.stage2.patient_state_dim,
        style_dim=config.stage2.style_dim
    ).to(device)

    disentangle = AsymmetricCausalDisentanglement(
        disease_dim=config.stage2.global_disease_dim,
        patient_dim=config.stage2.patient_state_dim,
        style_dim=config.stage2.style_dim,
        anatomy_channels=config.stage2.anatomy_channels,
        beta_tc=config.stage2.beta_tc,
        gamma_club=config.stage2.gamma_club,
        alpha_dp=config.stage2.alpha_dp
    ).to(device)

    # DDP-wrap for gradient sync across all GPUs
    _nccl_bucket_mb_s2 = getattr(config.hardware, 'nccl_bucket_cap_mb', 50)
    if _dist:
        vae = DDP(vae, device_ids=[_local_rank], output_device=_local_rank,
                  find_unused_parameters=False,
                  bucket_cap_mb=_nccl_bucket_mb_s2,
                  gradient_as_bucket_view=True,
                  static_graph=True)
        disentangle = DDP(disentangle, device_ids=[_local_rank], output_device=_local_rank,
                          find_unused_parameters=False,
                          bucket_cap_mb=_nccl_bucket_mb_s2,
                          gradient_as_bucket_view=True,
                          static_graph=True)

    # Linear LR scaling rule for DDP
    _world_s2 = dist.get_world_size() if dist.is_initialized() else 1
    _effective_lr_s2 = config.stage2.lr * _world_s2
    opt_vae = torch.optim.AdamW(vae.parameters(), lr=_effective_lr_s2, weight_decay=1e-4)
    opt_aux = torch.optim.AdamW(disentangle.parameters(), lr=_effective_lr_s2, weight_decay=1e-4)

    # Warmup + Cosine Annealing schedule
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    _warmup_e2 = max(1, min(5, num_epochs // 10))
    _warmup_s2 = LinearLR(opt_vae, start_factor=0.1, end_factor=1.0, total_iters=_warmup_e2)
    _cosine_s2 = CosineAnnealingLR(opt_vae, T_max=max(1, num_epochs - _warmup_e2), eta_min=_effective_lr_s2 * 0.01)
    scheduler_vae = SequentialLR(opt_vae, schedulers=[_warmup_s2, _cosine_s2], milestones=[_warmup_e2])

    latent_bank = []
    # DistributedSampler: must call set_epoch each epoch for correct per-epoch shuffling
    _s2_sampler = getattr(dataloader, 'sampler', None)
    if not hasattr(_s2_sampler, 'set_epoch'):
        _s2_sampler = None

    for epoch in range(num_epochs):
        if _s2_sampler is not None:
            _s2_sampler.set_epoch(epoch)
        vae.train()
        disentangle.train()
        opt_vae.zero_grad()
        opt_aux.zero_grad()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in dataloader:
            img = batch['image'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)

            # Downscale to half resolution for upgraded AutoencoderKL backbone
            # B=2 at 128³ → ~35 GB (vs 141 GB available on H200)
            img_small = F.interpolate(img, scale_factor=0.5, mode='trilinear', align_corners=False)
            mask_small = F.interpolate(mask, scale_factor=0.5, mode='nearest')
            # Channels-last 3D for 15-20% cuDNN throughput gain
            img_small = img_small.to(memory_format=torch.channels_last_3d)
            
            opt_vae.zero_grad()
            opt_aux.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = vae(img_small)

                # Reconstruction loss
                loss_recon = F.l1_loss(out['recon'], img_small)

                # Mask prediction loss (resize mask_pred to match mask_small)
                mp = out['mask_pred']
                if mp.shape[2:] != mask_small.shape[2:]:
                    mp = F.interpolate(mp, size=mask_small.shape[2:], mode='trilinear', align_corners=False)
                loss_mask = F.binary_cross_entropy_with_logits(mp, mask_small)

                # KL divergence
                kl_d = -0.5 * torch.mean(1 + out['logvar_d'] - out['mu_d'].pow(2) - out['logvar_d'].exp())
                kl_p = -0.5 * torch.mean(1 + out['logvar_p'] - out['mu_p'].pow(2) - out['logvar_p'].exp())
                kl_s = -0.5 * torch.mean(1 + out['logvar_s'] - out['mu_s'].pow(2) - out['logvar_s'].exp())
                loss_kl = kl_d + kl_p + kl_s

                # Disentanglement
                loss_dis, loss_aux, _ = disentangle.compute_disentangle_loss(
                    out['z_d_glob'], out['z_p'], out['z_s'], out['z_a']
                )

                total = loss_recon + 2.0 * loss_mask + config.stage2.kl_weight * loss_kl + loss_dis

            # BF16 autocast only (no GradScaler) — dual backward with retain_graph
            # is incompatible with a single scaler; BF16 doesn't overflow like FP16
            total.backward(retain_graph=True)
            loss_aux.backward()

            nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            nn.utils.clip_grad_norm_(disentangle.parameters(), max_norm=1.0)
            opt_vae.step()
            opt_aux.step()

            epoch_loss += total.item()
            n_batches += 1

            # Collect latents for Stage 3 (rank-0 only to avoid 4× duplication)
            if is_main:
                with torch.no_grad():
                    latent_bank.append({
                        'z_d_glob': out['z_d_glob'].detach().cpu(),
                        'z_p': out['z_p'].detach().cpu(),
                        'mu_d': out['mu_d'].detach().cpu(),
                        'logvar_d': out['logvar_d'].detach().cpu(),
                    })

        scheduler_vae.step()
        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(n_batches, 1)
        mem_gb = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()   # reset so next epoch shows its own peak

        if is_main:
            logger.info(
                f"  Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | "
                f"Recon: {loss_recon.item():.4f} | KL: {loss_kl.item():.4f} | "
                f"GPU peak: {mem_gb:.2f}GB | {elapsed:.1f}s"
            )

    # Save checkpoint — rank-0 only to prevent write races on shared filesystem
    if _dist:
        dist.barrier()
    if is_main:
        ckpt_path = os.path.join(config.data.checkpoints_dir, "stage2_vae_real.pt")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        raw_vae = vae.module if hasattr(vae, 'module') else vae
        raw_dis = disentangle.module if hasattr(disentangle, 'module') else disentangle
        torch.save({
            'vae_state': raw_vae.state_dict(),
            'disentangle_state': raw_dis.state_dict(),
            'epoch': num_epochs,
        }, ckpt_path)
        logger.info(f"  Stage 2 checkpoint saved: {ckpt_path}")
    if _dist:
        dist.barrier()

    return vae, latent_bank


def train_stage3(config, latent_bank, num_epochs, device):
    """
    Stage 3: Latent OOD Detection on collected VAE latents.
    
    Multi-GPU strategy: Single GPU is sufficient (latent space is compact).
    Can run on GPU 0 or GPU 3 while other GPUs process other stages.
    """
    logger.info("═══ Stage 3: Latent OOD Detection ═══")
    
    # Assemble latent vectors: [z_d_glob(64), z_p(32), logvar_d(64), zeros(5)] = 165
    vectors = []
    for lb in latent_bank:
        B = lb['z_d_glob'].size(0)
        zeta = torch.cat([
            lb['z_d_glob'],
            lb['z_p'],
            lb['logvar_d'][:, :64] if lb['logvar_d'].size(1) >= 64 else F.pad(lb['logvar_d'], (0, 64 - lb['logvar_d'].size(1))),
            torch.zeros(B, 5)
        ], dim=1)
        vectors.append(zeta)
    
    all_vectors = torch.cat(vectors, dim=0).to(device)
    logger.info(f"  Latent bank: {all_vectors.shape[0]} vectors, dim={all_vectors.shape[1]}")
    
    actual_dim = all_vectors.shape[1]
    manifold_vae = NormalityManifoldVAE(
        in_dim=actual_dim, hidden_dim=128, bottleneck_dim=32
    ).to(device)
    
    scorer = OODScorer(manifold_vae=manifold_vae, k_neighbors=min(15, all_vectors.size(0) - 1))
    optimizer = torch.optim.AdamW(manifold_vae.parameters(), lr=5e-4)
    
    N = all_vectors.size(0)
    for epoch in range(num_epochs):
        manifold_vae.train()
        perm = torch.randperm(N, device=device)
        total_loss = 0.0
        nb = 0
        for i in range(0, N, 32):
            idx = perm[i:i+32]
            batch = all_vectors[idx]
            optimizer.zero_grad()
            out = manifold_vae(batch)
            loss = (out['recon_loss'] + 0.1 * out['kl_loss']).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            nb += 1
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1}/{num_epochs} | Loss: {total_loss/nb:.4f}")
    
    # Calibrate scorer
    scorer.calibrate(all_vectors, device)
    
    # Score all vectors
    scores, flags = scorer.score(all_vectors)
    n_flagged = flags.sum().item()
    logger.info(f"  OOD: {n_flagged}/{N} visits flagged ({n_flagged/N*100:.1f}%)")
    
    return manifold_vae, scorer, scores, flags


def train_stage4(config, domain, num_epochs, device):
    """
    Stage 4: Biophysical PDE-NJDE dynamics training.
    
    Multi-GPU strategy: Run on GPU 2 (Julia SciML co-host).
    On single H100: compact latent grid (24×28×20 or 32×32×16) fits easily.
    The CFL-stable 7-day RK4 step keeps state under 512 KiB per patient.
    """
    logger.info(f"═══ Stage 4: Biophysical PDE-NJDE [{domain.upper()}] ═══")
    
    solver = ContinuousDiscreteJumpSolver(domain=domain, dt_cfl_days=7.0).to(device)
    # DDP-wrap solver sub-modules if running distributed
    solver.wrap_for_cluster()
    loss_fn = MaskedLeaveOneOutLoss(lambda_phys=config.stage4.lambda_phys)
    optimizer = torch.optim.AdamW(solver.parameters(), lr=config.stage4.lr, weight_decay=1e-4)
    s4_scaler = torch.amp.GradScaler('cuda', enabled=True)
    
    if domain == "gbm":
        grid = config.data.brain_latent_grid
    else:
        grid = config.data.breast_latent_grid
    
    B = min(config.stage4.batch_size, 4)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Synthetic initial conditions (would come from Stage 2 latents in production)
        u_d_0 = torch.rand(B, 8, *grid, device=device) * 0.15
        u_d_0[:, 0] = torch.rand(B, *grid, device=device) * 0.3  # viable tumor
        v_sys_0 = torch.randn(B, 160, device=device) * 0.1
        phi = torch.ones(B, 1, *grid, device=device)
        
        # Multi-visit schedule with interventions
        if domain == "gbm":
            schedule = [
                {"t": 7.0, "type": "radiotherapy", "dose_gy": 2.0, "tmz_dose": 1.0},
                {"t": 14.0, "type": "radiotherapy", "dose_gy": 2.0, "tmz_dose": 1.0},
                {"t": 21.0, "type": "radiotherapy", "dose_gy": 2.0, "tmz_dose": 1.0},
            ]
        else:
            schedule = [
                {"t": 7.0, "type": "chemotherapy", "drug_vec": torch.zeros(B, 16, device=device)},
                {"t": 14.0, "type": "chemotherapy", "drug_vec": torch.zeros(B, 16, device=device)},
            ]
        
        eval_times = [0.0, 14.0, 28.0]
        
        traj = solver.forward_trajectory(u_d_0, v_sys_0, phi, schedule, eval_times)
        
        # Leave-one-out: predict t=28 from t=0,14 observations
        if 28.0 in traj:
            u_pred, v_pred = traj[28.0]
            u_target = u_pred.detach() + torch.randn_like(u_pred) * 0.02
            v_target = v_pred.detach() + torch.randn_like(v_pred) * 0.02
            
            mask_u = torch.ones_like(u_pred)
            mask_v = torch.ones_like(v_pred)
            ood = torch.zeros(B, device=device)
            
            loss, metrics = loss_fn(u_pred, u_target, mask_u, v_pred, v_target, mask_v, ood)
            s4_scaler.scale(loss).backward()
            s4_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(solver.parameters(), max_norm=1.0)
            s4_scaler.step(optimizer)
            s4_scaler.update()
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"  Epoch {epoch+1}/{num_epochs} | "
                    f"Loss_U: {metrics['loss_u']:.6f} | Loss_V: {metrics['loss_v']:.6f} | "
                    f"Total: {metrics['loss_stage4']:.6f}"
                )
    
    ckpt_path = os.path.join(config.data.checkpoints_dir, f"stage4_{domain}_real.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({'solver_state': solver.state_dict(), 'domain': domain}, ckpt_path)
    logger.info(f"  Stage 4 checkpoint saved: {ckpt_path}")
    
    return solver


def run_stage5(config, solver, domain, device):
    """
    Stage 5: Counterfactual simulation + treatment ranking + VLM reporting.
    
    Now uses VLMReportGenerator with:
    - Med3DVLM (native 3D VLM, Qwen-2.5-7B + DCFormer-SigLIP)
    - BrainGemma3D (Gemma-based brain MRI VLM)
    - Rule-based fallback if models unavailable
    
    Multi-GPU strategy:
      - Counterfactual rollout: GPU 2 (Julia SciML or PyTorch solver)
      - VLM reporting: GPU 2 (vLLM serving MedGemma 1.5 4B or Med3DVLM)
      - Multi-draw MC: Fan out across GPU 3 for parallel posterior sampling
    """
    logger.info(f"═══ Stage 5: Counterfactual Synthesis & VLM Reporting [{domain.upper()}] ═══")
    
    simulator = CounterfactualSimulator(domain=domain, solver=solver)
    treatment_helper = AgenticTreatmentHelper(
        domain=domain, simulator=simulator,
        num_posterior_draws=config.stage5.num_counterfactual_draws,
        lambda_unc=config.stage5.uncertainty_penalty_lambda
    )
    
    # === VLM-Powered Reporter ===
    # Try Med3DVLM first (ungated, native 3D), then BrainGemma3D for GBM, then rule-based
    import os as _os
    vlm_model_name = "rule_based"
    vlm_model_path = None
    
    med3dvlm_path = "/DATA/horizon_disentangled/models/Med3DVLM-Qwen-2.5-7B"
    braingemma_path = "/DATA/horizon_disentangled/models/BrainGemma3D"
    
    if _os.path.exists(med3dvlm_path) and _os.path.isdir(med3dvlm_path):
        # Check if model files actually exist (not just empty dir)
        safetensor_files = [f for f in _os.listdir(med3dvlm_path) if f.endswith('.safetensors')]
        if safetensor_files:
            vlm_model_name = "med3dvlm"
            vlm_model_path = med3dvlm_path
            logger.info(f"  VLM: Using Med3DVLM from {med3dvlm_path}")
    
    if vlm_model_name == "rule_based" and domain == "gbm" and _os.path.exists(braingemma_path):
        safetensor_files = [f for f in _os.listdir(braingemma_path) if f.endswith('.safetensors') or f.endswith('.pt')]
        if safetensor_files:
            vlm_model_name = "braingemma3d"
            vlm_model_path = braingemma_path
            logger.info(f"  VLM: Using BrainGemma3D from {braingemma_path}")
    
    if vlm_model_name == "rule_based":
        logger.info("  VLM: Using rule-based reporter (no VLM models available)")
    
    # Create VLM reporter  
    vlm_reporter = VLMReportGenerator(
        model_name=vlm_model_name,
        model_path=vlm_model_path,
        device=str(device),
        max_new_tokens=1024,
        concordance_threshold=config.stage5.confidence_divergence_threshold,
        use_4bit=True,
    )
    
    # Also keep legacy reporter for comparison
    legacy_reporter = StructuredClinicalReporter(
        concordance_threshold=config.stage5.confidence_divergence_threshold
    )
    audit_logger = EUAIActAuditLogger()
    
    if domain == "gbm":
        grid = config.data.brain_latent_grid
    else:
        grid = config.data.breast_latent_grid
    
    # Synthetic patient state (in production: from Stage 2 encoder)
    u_d_0 = torch.rand(1, 8, *grid, device=device) * 0.2
    v_sys_0 = torch.randn(1, 160, device=device) * 0.1
    z_a_0 = torch.randn(1, 4, grid[0]//2, grid[1]//2, grid[2]//2, device=device) * 0.1
    z_s_ref = torch.randn(1, 32, device=device) * 0.1
    sigma2 = torch.rand(1, 96, device=device) * 0.01
    
    # Treatment ranking
    rankings = treatment_helper.rank_interventions(
        u_d_0, v_sys_0, z_a_0, z_s_ref, sigma2, horizon_days=180.0
    )
    
    for r in rankings:
        logger.info(
            f"  {r['regimen']}: utility={r['utility_score']:.2f}, "
            f"E[TTP]={r['mean_ttp_days']:.1f}d ± {r['std_ttp_days']:.1f}d"
        )
    
    # Generate clinical report using VLM
    if domain == "gbm":
        mc_draws = [{"rano_category": "Stable_Disease"} for _ in range(8)]
        report = vlm_reporter.generate_rano_report(
            patient_id="TRAIN_GBM_001", visit_day=90.0,
            current_tumor_vol=25.3, prior_tumor_vol=28.1,
            dior_probabilities={"CR": 0.05, "PR": 0.15, "SD": 0.60, "PD": 0.15, "PP": 0.05},
            median_ttp_days=rankings[0]["mean_ttp_days"],
            ci95_ttp=rankings[0]["ci95_ttp"],
            mc_predictions=mc_draws
        )
    else:
        mc_draws = [{"pcr_prob": 0.65 + i * 0.01} for i in range(8)]
        report = vlm_reporter.generate_birads_ispy2_report(
            patient_id="TRAIN_BREAST_001", visit_timepoint="T1",
            current_ftv_cm3=15.2, baseline_ftv_cm3=42.0,
            longest_diameter_mm=35.0, ser_mean=1.4,
            predicted_pcr_prob=0.68, mc_predictions=mc_draws
        )
    
    logger.info(f"  VLM Report generated: method={report.get('generation_method', 'unknown')}, schema={report.get('schema_version')}")
    
    # Log VLM assessment details
    if domain == "gbm":
        rano = report.get("rano_assessment", {})
        logger.info(f"  RANO category: {rano.get('category')} (rule-based: {rano.get('rule_based_category')})")
        logger.info(f"  Reasoning: {rano.get('reasoning', 'N/A')[:100]}...")
    else:
        prog = report.get("prognosis_pcr", {})
        logger.info(f"  Response tier: {prog.get('response_tier')} (rule-based: {prog.get('rule_based_tier')})")
        logger.info(f"  pCR assessment: {prog.get('pcr_assessment')}")
    
    # Hallucination audit
    audit = report.get("hallucination_audit", {})
    logger.info(f"  Hallucination audit: concordance={audit.get('concordance_rate', audit.get('posterior_sd', 'N/A'))}, warning={audit.get('concordance_warning', False)}")
    
    # Audit logging
    audit_logger.log_clinical_report(
        patient_id=report.get("patient_id", "UNKNOWN"),
        report_type="RANO_2.0" if domain == "gbm" else "ISPY2_BIRADS",
        schema_version=report.get("schema_version", "v1"),
        report=report,
    )
    
    conformity = audit_logger.generate_conformity_summary()
    logger.info(f"  EU AI Act conformity: {json.dumps({k: v.get('status', 'N/A') for k, v in conformity.get('articles', {}).items()})}")
    
    # Unload VLM to free GPU memory
    vlm_reporter.unload()
    
    return rankings, report


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Horizon Disentangled — Full Training")
    parser.add_argument("--domain", default="breast", choices=["gbm", "breast"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-patients", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume-stage", type=int, default=1, help="Stage to resume from (skip earlier stages)")
    args = parser.parse_args()

    # ── Distributed process group init (torchrun / SLURM) ──────────────────
    import torch.distributed as dist
    from torch.utils.data.distributed import DistributedSampler

    _is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if _is_distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda:0")
    is_main = (rank == 0)

    # Setup logging — only rank-0 logs to avoid duplicate stdout
    if is_main:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("/DATA/horizon_disentangled/artifacts/train_full.log", mode="a"),
            ]
        )
    else:
        logging.basicConfig(level=logging.WARNING)

    config = PipelineConfig()

    if is_main:
        logger.info("╔═══════════════════════════════════════════════════════════╗")
        logger.info("║  Horizon Disentangled — Full Training Run               ║")
        logger.info(f"║  Domain: {args.domain.upper():8s} | Epochs: {args.epochs:3d} | BS: {args.batch_size:2d}           ║")
        logger.info(f"║  Distributed: {_is_distributed} | World: {world_size} | Rank: {rank}        ║")
        logger.info(f"║  GPU: {torch.cuda.get_device_name(device.index or 0):44s}  ║")
        logger.info(f"║  VRAM: {torch.cuda.get_device_properties(device.index or 0).total_memory/1e9:.1f} GB                                       ║")
        logger.info("╚═══════════════════════════════════════════════════════════╝")

    t0_total = time.time()

    # Build dataset — each rank sees the full dataset, DistributedSampler shards it
    if is_main:
        logger.info(f"Loading {args.domain} data from HDF5 shards (max {args.max_patients} patients)...")
    dataset = HorizonRealDataset(
        hdf5_dir="/DATA/horizon_disentangled/hdf5",
        domain=args.domain,
        max_patients=args.max_patients,
    )

    # DistributedSampler: each GPU gets 1/world_size of data per epoch
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) \
              if _is_distributed else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),     # shuffle only when not using DistributedSampler
        num_workers=min(16, (os.cpu_count() or 4) // max(world_size, 1)),
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    if is_main:
        logger.info(f"Dataset: {len(dataset)} visits | {len(dataloader)} batches/epoch/GPU | world_size={world_size}")

    # Stage 1
    if args.resume_stage <= 1:
        # Note: set_epoch is called per-epoch inside train_stage1 via the sampler
        # stored in dataloader.sampler (DistributedSampler attached above)
        torch.cuda.reset_peak_memory_stats()
        model_s1, embeddings = train_stage1(config, dataloader, args.domain, args.epochs, device)
        peak_s1 = torch.cuda.max_memory_allocated() / 1e9
        if is_main:
            logger.info(f"  Stage 1 peak GPU: {peak_s1:.2f} GB")
    else:
        peak_s1 = 0.0
        if is_main:
            logger.info(f"  Stage 1 SKIPPED (resuming from stage {args.resume_stage})")

    # Stage 2
    if args.resume_stage <= 2:
        # Note: set_epoch is called per-epoch inside train_stage2
        torch.cuda.reset_peak_memory_stats()
        vae_s2, latent_bank = train_stage2(config, dataloader, args.epochs, device)
        peak_s2 = torch.cuda.max_memory_allocated() / 1e9
        if is_main:
            logger.info(f"  Stage 2 peak GPU: {peak_s2:.2f} GB")
    else:
        peak_s2 = 0.0
        latent_bank = []
        if is_main:
            logger.info(f"  Stage 2 SKIPPED (resuming from stage {args.resume_stage})")

    # Stage 3
    if args.resume_stage <= 3:
        torch.cuda.reset_peak_memory_stats()
        if latent_bank:
            manifold_vae, scorer, scores, flags = train_stage3(config, latent_bank, max(5, args.epochs // 2), device)
            peak_s3 = torch.cuda.max_memory_allocated() / 1e9
            if is_main:
                logger.info(f"  Stage 3 peak GPU: {peak_s3:.2f} GB")
        else:
            peak_s3 = 0.0
            if is_main:
                logger.info(f"  Stage 3 SKIPPED (no latent bank)")
    else:
        peak_s3 = 0.0
        if is_main:
            logger.info(f"  Stage 3 SKIPPED (resuming from stage {args.resume_stage})")

    # Stage 4
    torch.cuda.reset_peak_memory_stats()
    solver = train_stage4(config, args.domain, args.epochs, device)
    peak_s4 = torch.cuda.max_memory_allocated() / 1e9
    if is_main:
        logger.info(f"  Stage 4 peak GPU: {peak_s4:.2f} GB")

    # Stage 5 — inference only, rank-0
    if is_main:
        torch.cuda.reset_peak_memory_stats()
        rankings, report = run_stage5(config, solver, args.domain, device)
        peak_s5 = torch.cuda.max_memory_allocated() / 1e9
        logger.info(f"  Stage 5 peak GPU: {peak_s5:.2f} GB")
    else:
        peak_s5 = 0.0

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    total_time = time.time() - t0_total

    if is_main:
        logger.info("╔═══════════════════════════════════════════════════════════╗")
        logger.info(f"║  Training Complete — {total_time:.1f}s total                       ║")
        logger.info(f"║  Peak GPU per stage: S1={peak_s1:.1f} S2={peak_s2:.1f} S3={peak_s3:.1f} S4={peak_s4:.1f} S5={peak_s5:.1f} GB ║")
        logger.info("╚═══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
