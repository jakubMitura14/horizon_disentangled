"""Stage 4 — Masked Leave-One-Out Trajectory Loss.

Implements the masked trajectory supervision loss (Eq. 16) for training the
continuous-discrete jump solver. Evaluates MSE only over valid observed
spatial slots (M_U) and systemic slots (M_v), ignoring unobserved modalities
and down-weighting OOD-flagged visits. Includes optional biophysical
regularization (L_SER for breast, L_LQ for GBM).

Key classes:
    MaskedLeaveOneOutLoss: Masked MSE + biophysical regularization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional

class MaskedLeaveOneOutLoss(nn.Module):
    """
    Masked Leave-One-Out Trajectory Loss (Eq. 16 in docs/main_pipeline.md):
    Evaluates MSE exclusively over valid observed spatial slots (M_U) and systemic slots (M_v),
    ignoring unobserved modalities and masking OOD-flagged visits.
    Adds biophysical regularization L_SER/LQ.
    """
    def __init__(self, lambda_phys: float = 0.2, eps: float = 1e-4):
        super().__init__()
        self.lambda_phys = lambda_phys
        self.eps = eps

    def forward(
        self,
        u_d_pred: torch.Tensor,       # (B, 8, D, H, W) predicted coarse disease
        u_d_gt: torch.Tensor,         # (B, 8, D, H, W) ground truth
        mask_u: torch.Tensor,         # (B, 1, 1, 1, 1) or (B, 8, D, H, W) binary observed mask
        v_sys_pred: torch.Tensor,     # (B, 160)
        v_sys_gt: torch.Tensor,       # (B, 160)
        mask_v: torch.Tensor,         # (B, 160) binary slot mask
        ood_flags: Optional[torch.Tensor] = None, # (B,) 1=OOD, 0=clean
        phys_regularizer: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        B = u_d_pred.size(0)
        device = u_d_pred.device

        # If visit is flagged as OOD, zero out its supervision weight
        clean_weights = torch.ones(B, 1, device=device)
        if ood_flags is not None:
            clean_weights = (1.0 - ood_flags.float()).unsqueeze(1)

        # 1. Spatial Disease Loss
        diff_u = (u_d_pred - u_d_gt) ** 2
        masked_diff_u = clean_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * mask_u * diff_u
        loss_u = masked_diff_u.sum() / (mask_u.sum() * clean_weights.sum() + self.eps)

        # 2. Systemic Vector Loss
        diff_v = (v_sys_pred - v_sys_gt) ** 2
        masked_diff_v = clean_weights * mask_v * diff_v
        loss_v = masked_diff_v.sum() / (mask_v.sum() * clean_weights.sum() + self.eps)

        # 3. Biophysical Loss
        if phys_regularizer is not None:
            loss_phys = phys_regularizer
        else:
            loss_phys = torch.tensor(0.0, device=device)

        total_loss = loss_u + loss_v + self.lambda_phys * loss_phys
        metrics = {
            "loss_u": loss_u.item(),
            "loss_v": loss_v.item(),
            "loss_phys": loss_phys.item(),
            "loss_stage4": total_loss.item()
        }
        return total_loss, metrics
