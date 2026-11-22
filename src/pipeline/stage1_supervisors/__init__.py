"""Stage 1 — Multimodal Supervisors.

Segmentation (SwinUNETR/VISTA3D), ordinal grading (DIOR), survival analysis
(LogisticHazard + SurvRNC), and Merlin-style image-text contrastive learning.

Key classes:
    MultimodalSupervisorTrunk: Joint 3D imaging + clinical model (~65M params).
    Stage1Trainer: DDP-aware trainer with GradScaler and gradient accumulation.
    DifferentialOrdinalLoss: DIOR ordinal learning loss.
    SurvRNCLoss: Survival Rank-N-Contrast regularizer (MICCAI 2024).
    MerlinMultiTaskLoss: InfoNCE contrastive + BCE phenotype loss (Nature 2026).

Example:
    >>> from src.pipeline.stage1_supervisors import Stage1Trainer
    >>> trainer = Stage1Trainer(config, domain='gbm')
"""
from src.pipeline.stage1_supervisors.models import (
    MultimodalSupervisorTrunk, SwinUNETREncoder, VISTA3DEncoder
)
from src.pipeline.stage1_supervisors.losses import (
    DifferentialOrdinalLoss, LogisticHazardLoss, SurvRNCLoss,
    MerlinInfoNCELoss, ClinicalTextEncoder, MerlinMultiTaskLoss
)
from src.pipeline.stage1_supervisors.trainer import Stage1Trainer

__all__ = [
    "MultimodalSupervisorTrunk",
    "SwinUNETREncoder",
    "VISTA3DEncoder",
    "DifferentialOrdinalLoss",
    "LogisticHazardLoss",
    "SurvRNCLoss",
    "MerlinInfoNCELoss",
    "ClinicalTextEncoder",
    "MerlinMultiTaskLoss",
    "Stage1Trainer"
]
