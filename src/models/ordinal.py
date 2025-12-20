import torch
import torch.nn as nn

class OrdinalSupervisor(nn.Module):
    """
    Ordinal Classifier for Gleason/PI-RADS.
    Architecture: Simple 3D CNN -> Linear output.
    Uses 'coral' (Consistent Rank Logits) or simple binary decomposition approach.
    For simplicity here, we use a standard regression output treated as ordinal rank,
    or a multi-head binary classification (grade > 0, grade > 1, etc.)
    """
    def __init__(self, in_channels=2, num_classes=5): # Classes: 0, 6, 7, 8, 9 -> mapped to 0, 1, 2, 3, 4
        super(OrdinalSupervisor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.AdaptiveAvgPool3d(1)
        )
        # Ordinal output: (num_classes - 1) binary classifiers
        # Output k is prob(y > k)
        self.ordinal_head = nn.Linear(32, num_classes - 1)

    def forward(self, x):
        feat = self.features(x).view(x.size(0), -1)
        logits = self.ordinal_head(feat)
        return logits

def ordinal_loss(logits, targets):
    """
    targets: Integer class labels (0 to K-1).
    logits: (B, K-1).

    For a target class k, we want:
    logits[i] > 0 for all i < k (Prob(y>i) = 1)
    logits[i] < 0 for all i >= k (Prob(y>i) = 0)
    """
    # Create binary targets for each threshold
    # if target is 2 (Gleason 7), then y>0 is True, y>1 is True, y>2 is False...
    batch_size, num_cutpoints = logits.shape

    # Expand targets to match logits shape
    # targets: (B) -> (B, 1)
    targets = targets.unsqueeze(1)

    # Thresholds: 0, 1, 2, ... K-2
    cutpoints = torch.arange(num_cutpoints).to(logits.device).unsqueeze(0)

    # Binary labels: 1 if target > cutpoint, else 0
    binary_targets = (targets > cutpoints).float()

    loss = nn.BCEWithLogitsLoss()(logits, binary_targets)
    return loss
