"""Stage 1 — Loss Functions for Multimodal Supervisors.

Implements the multi-objective training losses for Stage 1:
- DIOR: Differential Ordinal Learning (CE + cumulative ordinal + adjacency + pairwise)
- LogisticHazard: Discrete hazard survival NLL for right-censored data
- SurvRNC: Survival Rank-N-Contrast contrastive regularizer (MICCAI 2024)
- MerlinInfoNCE: Symmetric image-text InfoNCE contrastive loss (Nature 2026)
- MerlinMultiTaskLoss: Combined InfoNCE + BCE phenotype prediction

Key classes:
    DifferentialOrdinalLoss: 4-component DIOR loss with pairwise alignment.
    LogisticHazardLoss: Discrete-time survival negative log-likelihood.
    SurvRNCLoss: Censoring-aware contrastive survival ranking.
    MerlinInfoNCELoss: Symmetric InfoNCE with learnable temperature.
    ClinicalTextEncoder: Attention-pooled clinical text encoder.
    MerlinMultiTaskLoss: Combined contrastive + phenotype multi-task loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict
import math

class DifferentialOrdinalLoss(nn.Module):
    """
    Differential Ordinal Learning (DIOR) Loss:
    Combines:
    1. Standard categorical cross-entropy
    2. Importance-weighted cumulative binary ordinal cross-entropy
    3. Adjacent-rank monotonicity boundary penalty
    4. Pairwise differential feature-rank alignment
    """
    def __init__(
        self,
        num_classes: int,
        lambda_ord: float = 1.0,
        lambda_adj: float = 0.5,
        lambda_diff: float = 0.1,
        adjacent_margin: float = 0.05,
        diff_alpha: float = 0.5,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.K = num_classes
        self.lambda_ord = lambda_ord
        self.lambda_adj = lambda_adj
        self.lambda_diff = lambda_diff
        self.adjacent_margin = adjacent_margin
        self.diff_alpha = diff_alpha
        self.register_buffer("class_weights", class_weights if class_weights is not None else torch.ones(num_classes))

    def forward(
        self,
        cat_logits: torch.Tensor,       # (B, K)
        cum_probs: torch.Tensor,        # (B, K-1) sigmoid probabilities
        embeddings: torch.Tensor,       # (B, D)
        targets: torch.Tensor           # (B,) integer rank in [0, K-1]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        B = targets.size(0)
        device = targets.device

        # 1. Categorical Cross Entropy
        loss_ce = F.cross_entropy(cat_logits, targets, weight=self.class_weights)

        # 2. Cumulative Ordinal Loss: y_k = I(c > k - 1) for k in 1..K-1
        k_indices = torch.arange(self.K - 1, device=device).unsqueeze(0) # (1, K-1)
        y_cum = (targets.unsqueeze(1) > k_indices).float()              # (B, K-1)
        
        # Binary cross entropy with numerical stability
        eps = 1e-7
        cum_probs_clamped = torch.clamp(cum_probs, eps, 1.0 - eps)
        loss_ord = -torch.mean(y_cum * torch.log(cum_probs_clamped) + (1.0 - y_cum) * torch.log(1.0 - cum_probs_clamped))

        # 3. Adjacent Boundary Penalty: max(0, p_{k+1} - p_k + delta)
        # Because p_k = P(c > k-1) should be monotonically non-increasing: p_{k+1} <= p_k
        if self.K > 2:
            diff_adj = cum_probs[:, 1:] - cum_probs[:, :-1] + self.adjacent_margin
            loss_adj = torch.mean(torch.relu(diff_adj))
        else:
            loss_adj = torch.tensor(0.0, device=device)

        # 4. Pairwise Differential Feature-Rank Alignment
        # (||h_i - h_j||_2 - alpha * |c_i - c_j|)^2
        if B > 1 and self.lambda_diff > 0:
            h_dist = torch.cdist(embeddings, embeddings, p=2) # (B, B)
            c_dist = torch.abs(targets.unsqueeze(0).float() - targets.unsqueeze(1).float()) # (B, B)
            loss_diff = torch.mean((h_dist - self.diff_alpha * c_dist) ** 2)
        else:
            loss_diff = torch.tensor(0.0, device=device)

        total_loss = loss_ce + self.lambda_ord * loss_ord + self.lambda_adj * loss_adj + self.lambda_diff * loss_diff
        metrics = {
            "loss_ce": loss_ce.item(),
            "loss_ord": loss_ord.item(),
            "loss_adj": loss_adj.item(),
            "loss_diff": loss_diff.item(),
            "loss_dior": total_loss.item()
        }
        return total_loss, metrics

class LogisticHazardLoss(nn.Module):
    """
    Discrete Hazard Survival Negative Log-Likelihood for right-censored time-to-event data.
    Quantizes follow-up into M intervals.
    """
    def __init__(self, num_intervals: int = 20, eps: float = 1e-7):
        super().__init__()
        self.num_intervals = num_intervals
        self.eps = eps

    def forward(
        self,
        hazard_logits: torch.Tensor, # (B, M)
        interval_idx: torch.Tensor,  # (B,) int in [0, M-1]
        events: torch.Tensor         # (B,) 1=event occurred, 0=censored
    ) -> torch.Tensor:
        hazards = torch.sigmoid(hazard_logits) # h_m in (0, 1)
        hazards = torch.clamp(hazards, self.eps, 1.0 - self.eps)
        surv_parts = torch.log(1.0 - hazards)

        B = hazard_logits.size(0)
        loss = 0.0

        for i in range(B):
            k = interval_idx[i].item()
            e = events[i].item()
            
            # Sum log(1 - h_m) for m < k
            log_surv_prior = surv_parts[i, :k].sum() if k > 0 else 0.0
            
            if e == 1:
                # Event occurred at interval k: log(h_k) + sum_{m<k} log(1 - h_m)
                nll = -(torch.log(hazards[i, k]) + log_surv_prior)
            else:
                # Censored during interval k: sum_{m<=k} log(1 - h_m)
                nll = -(log_surv_prior + surv_parts[i, k])
            loss = loss + nll

        return loss / max(B, 1)

class SurvRNCLoss(nn.Module):
    """
    Survival Rank-N-Contrast (SurvRNC) Regularizer (MICCAI 2024).
    Enforces continuous monotonic ordering of patient latent embeddings by prognosis,
    strictly accounting for right-censoring admissibility.
    """
    def __init__(self, tau: float = 0.2, eps: float = 1e-7):
        super().__init__()
        self.tau = tau
        self.eps = eps

    def forward(
        self,
        embeddings: torch.Tensor, # (B, D)
        times: torch.Tensor,      # (B,) follow-up time
        events: torch.Tensor      # (B,) 1=uncensored, 0=censored
    ) -> torch.Tensor:
        B = embeddings.size(0)
        if B < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
        sim_matrix = torch.matmul(embeddings_norm, embeddings_norm.t()) / self.tau # (B, B)

        loss = 0.0
        num_valid_anchors = 0

        for i in range(B):
            t_i = times[i].item()
            e_i = events[i].item()

            # Find positive admissible pairs P(i)
            # Uncensored anchors (e_i=1): can pair with anyone with T_j > T_i, or uncensored with T_j <= T_i
            # Censored anchors (e_i=0): can strictly pair only with T_j > T_i
            pos_indices = []
            for j in range(B):
                if i == j:
                    continue
                t_j = times[j].item()
                e_j = events[j].item()

                if e_i == 1:
                    if t_j >= t_i or e_j == 1:
                        pos_indices.append(j)
                else:
                    if t_j > t_i:
                        pos_indices.append(j)

            if not pos_indices:
                continue

            num_valid_anchors += 1
            pos_loss = 0.0

            for j in pos_indices:
                delta_ij = abs(times[i].item() - times[j].item())
                # Admissible comparison set S_{i,j} = {k : |T_i - T_k| >= delta_ij}
                neg_mask = torch.tensor(
                    [abs(times[i].item() - times[k].item()) >= delta_ij for k in range(B)],
                    device=embeddings.device,
                    dtype=torch.bool
                )
                neg_mask[i] = False # exclude anchor

                denom = torch.exp(sim_matrix[i, neg_mask]).sum()
                num = torch.exp(sim_matrix[i, j])
                
                pos_loss = pos_loss - torch.log((num / (denom + self.eps)) + self.eps)

            loss = loss + (pos_loss / len(pos_indices))

        return loss / max(num_valid_anchors, 1)



class MerlinInfoNCELoss(nn.Module):
    """
    Merlin-style Image-Text Contrastive Loss (InfoNCE):
    
    Aligns 3D medical image embeddings with clinical text embeddings via
    symmetric InfoNCE contrastive learning, adapted from:
      Merlin (Nature 2026): L_InfoNCE + λ·L_BCE multi-task recipe
    
    Instead of Merlin's CT-specific I3D ResNet weights, we use our existing
    SwinUNETR/MedNeXt visual encoder with a clinical text encoder (PubMedBERT
    or lightweight MLP on tokenized clinical text).
    
    Key design choices:
    - Symmetric InfoNCE (image→text + text→image) for bidirectional alignment
    - Learnable temperature parameter (log-scaled for stability)
    - Projection heads map both modalities to shared 256-D contrastive space
    - Hard negative mining via in-batch negatives
    """
    def __init__(
        self,
        visual_dim: int = 768,
        text_dim: int = 256,
        proj_dim: int = 256,
        init_temperature: float = 0.07,
        learnable_temperature: bool = True,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        
        # Projection heads (following CLIP/Merlin design)
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )
        
        # Learnable temperature (log-parameterized for stability)
        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature)))
        else:
            self.register_buffer('log_temperature', torch.tensor(math.log(1.0 / init_temperature)))
    
    def forward(
        self,
        visual_emb: torch.Tensor,   # (B, visual_dim)
        text_emb: torch.Tensor,     # (B, text_dim) 
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute symmetric InfoNCE loss.
        
        Each image i should match text i (positive pair).
        All other (i, j) pairs where i != j are negatives.
        """
        B = visual_emb.size(0)
        if B < 2:
            zero = torch.tensor(0.0, device=visual_emb.device, requires_grad=True)
            return zero, {"loss_infonce": 0.0, "accuracy_i2t": 0.0, "accuracy_t2i": 0.0}
        
        # Project to shared contrastive space
        z_v = F.normalize(self.visual_proj(visual_emb), p=2, dim=-1)  # (B, proj_dim)
        z_t = F.normalize(self.text_proj(text_emb), p=2, dim=-1)      # (B, proj_dim)
        
        # Temperature-scaled similarity matrix
        temperature = torch.exp(self.log_temperature).clamp(max=100.0)
        logits = (z_v @ z_t.t()) * temperature  # (B, B)
        
        # Labels: diagonal = positive pairs
        labels = torch.arange(B, device=visual_emb.device)
        
        # Symmetric InfoNCE: image→text + text→image
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        loss = (loss_i2t + loss_t2i) / 2.0
        
        # Accuracy metrics for monitoring
        with torch.no_grad():
            acc_i2t = (logits.argmax(dim=1) == labels).float().mean().item()
            acc_t2i = (logits.argmax(dim=0) == labels).float().mean().item()
        
        metrics = {
            "loss_infonce": loss.item(),
            "accuracy_i2t": acc_i2t,
            "accuracy_t2i": acc_t2i,
            "temperature": (1.0 / temperature).item(),
        }
        return loss, metrics


class ClinicalTextEncoder(nn.Module):
    """
    Lightweight clinical text encoder for Merlin-style contrastive learning.
    
    Encodes structured clinical metadata (RANO reports, clinical notes,
    receptor status, treatment history) into a 256-D text embedding.
    
    Two modes:
    1. 'structured': MLP on tokenized clinical features (default, no pretrained LM needed)
    2. 'biomedclip': Uses frozen BiomedCLIP/PubMedBERT text encoder (requires download)
    
    For structured mode, we create a pseudo-text representation from clinical
    variables: each variable gets a learned embedding, then aggregated via
    attention pooling — mimicking how Merlin's Clinical Longformer processes
    structured ICD/phenotype codes.
    """
    def __init__(
        self,
        num_clinical_tokens: int = 32,   # Number of clinical feature slots
        token_dim: int = 64,              # Per-token embedding dim
        hidden_dim: int = 256,
        out_dim: int = 256,
        num_heads: int = 4,
        mode: str = "structured",         # "structured" or "biomedclip"
    ):
        super().__init__()
        self.mode = mode
        self.out_dim = out_dim
        
        if mode == "structured":
            # Learned embeddings for each clinical feature slot
            self.token_embeddings = nn.Parameter(torch.randn(1, num_clinical_tokens, token_dim) * 0.02)
            
            # Value projection: maps scalar clinical values to token_dim
            self.value_proj = nn.Linear(1, token_dim)
            
            # Positional encoding for feature ordering
            self.pos_enc = nn.Parameter(torch.randn(1, num_clinical_tokens, token_dim) * 0.02)
            
            # Self-attention aggregation (mimics Longformer's global attention)
            self.self_attn = nn.TransformerEncoderLayer(
                d_model=token_dim, nhead=num_heads,
                dim_feedforward=hidden_dim, dropout=0.1,
                batch_first=True, activation='gelu',
            )
            
            # Attention pooling to single vector
            self.pool_query = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)
            self.pool_attn = nn.MultiheadAttention(
                embed_dim=token_dim, num_heads=num_heads,
                batch_first=True, dropout=0.1,
            )
            
            # Output projection
            self.out_proj = nn.Sequential(
                nn.Linear(token_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
                nn.LayerNorm(out_dim),
            )
        else:
            # BiomedCLIP mode — will be initialized externally
            self.out_proj = nn.Sequential(
                nn.Linear(512, hidden_dim),  # BiomedCLIP text dim = 512
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
                nn.LayerNorm(out_dim),
            )
    
    def forward(self, clinical_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clinical_values: (B, num_features) clinical feature values
            
        Returns:
            text_emb: (B, out_dim) clinical text embedding
        """
        if self.mode == "structured":
            B = clinical_values.size(0)
            num_features = clinical_values.size(1)
            num_tokens = self.token_embeddings.size(1)
            
            # Pad or truncate features to match token count
            if num_features < num_tokens:
                clinical_values = F.pad(clinical_values, (0, num_tokens - num_features), value=0.0)
            elif num_features > num_tokens:
                clinical_values = clinical_values[:, :num_tokens]
            
            # Create token representations: type embedding + value projection + position
            val_proj = self.value_proj(clinical_values.unsqueeze(-1))  # (B, T, token_dim)
            tokens = self.token_embeddings.expand(B, -1, -1) + val_proj + self.pos_enc.expand(B, -1, -1)
            
            # Self-attention over clinical tokens
            tokens = self.self_attn(tokens)  # (B, T, token_dim)
            
            # Attention pooling → single vector
            query = self.pool_query.expand(B, -1, -1)  # (B, 1, token_dim)
            pooled, _ = self.pool_attn(query, tokens, tokens)  # (B, 1, token_dim)
            pooled = pooled.squeeze(1)  # (B, token_dim)
            
            return self.out_proj(pooled)
        else:
            # BiomedCLIP mode: clinical_values expected to be pre-encoded (B, 512)
            return self.out_proj(clinical_values)


class MerlinMultiTaskLoss(nn.Module):
    """
    Complete Merlin-style Multi-Task Loss combining:
    1. InfoNCE contrastive (image-text alignment)
    2. BCE multi-label phenotype prediction (PheWAS-style)
    3. Existing DIOR + Survival losses
    
    L_total = L_InfoNCE + λ_bce * L_BCE + L_DIOR + L_survival + λ_survrnc * L_SurvRNC
    
    Following Merlin (Nature 2026), the contrastive loss is weighted equally
    with the phenotype prediction loss for balanced representation learning.
    """
    def __init__(
        self,
        visual_dim: int = 768,
        text_dim: int = 256,
        num_phenotype_labels: int = 32,  # Binary clinical phenotype targets
        lambda_infonce: float = 1.0,
        lambda_bce: float = 1.0,
        proj_dim: int = 256,
    ):
        super().__init__()
        self.lambda_infonce = lambda_infonce
        self.lambda_bce = lambda_bce
        
        # InfoNCE contrastive loss
        self.infonce = MerlinInfoNCELoss(
            visual_dim=visual_dim,
            text_dim=text_dim,
            proj_dim=proj_dim,
        )
        
        # Multi-label phenotype prediction head (Merlin's L_BCE)
        self.phenotype_head = nn.Sequential(
            nn.Linear(visual_dim, visual_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(visual_dim // 2, num_phenotype_labels),
        )
    
    def forward(
        self,
        visual_emb: torch.Tensor,     # (B, visual_dim) 
        text_emb: torch.Tensor,       # (B, text_dim)
        phenotype_targets: Optional[torch.Tensor] = None,  # (B, num_labels) binary
        phenotype_mask: Optional[torch.Tensor] = None,     # (B, num_labels) valid entries
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Merlin-style multi-task contrastive + phenotype loss.
        """
        # 1. InfoNCE contrastive loss
        loss_infonce, metrics_infonce = self.infonce(visual_emb, text_emb)
        
        # 2. Multi-label BCE phenotype loss
        phenotype_logits = self.phenotype_head(visual_emb)
        
        if phenotype_targets is not None:
            if phenotype_mask is not None:
                # Only compute loss on valid (non-missing) labels
                valid = phenotype_mask.bool()
                if valid.any():
                    loss_bce = F.binary_cross_entropy_with_logits(
                        phenotype_logits[valid],
                        phenotype_targets[valid].float(),
                    )
                else:
                    loss_bce = torch.tensor(0.0, device=visual_emb.device)
            else:
                loss_bce = F.binary_cross_entropy_with_logits(
                    phenotype_logits,
                    phenotype_targets.float(),
                )
        else:
            loss_bce = torch.tensor(0.0, device=visual_emb.device)
        
        total = self.lambda_infonce * loss_infonce + self.lambda_bce * loss_bce
        
        metrics = {
            **metrics_infonce,
            "loss_bce_phenotype": loss_bce.item() if isinstance(loss_bce, torch.Tensor) else loss_bce,
            "loss_merlin_total": total.item(),
        }
        
        return total, metrics
