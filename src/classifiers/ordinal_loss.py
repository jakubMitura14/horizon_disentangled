"""
Ordinal Loss Functions for Deep Learning
Implements CORAL (Consistent Rank Logits) loss for ordinal regression.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoralLoss(nn.Module):
    """
    CORAL (Consistent Rank Logits) Loss for Ordinal Regression.
    
    Decomposes K-class ordinal problem into K-1 binary classification tasks:
    "Is the class > k?" for k = 0, 1, ..., K-2.
    
    Reference: Cao, Mirjalili, Raschka (2020) - Rank consistent ordinal regression
    """
    
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        
    def forward(self, logits, labels):
        """
        Args:
            logits: (B, K-1) raw scores for each binary threshold.
            labels: (B,) integer class labels (0 to K-1).
            
        Returns:
            Scalar loss value.
        """
        batch_size = logits.size(0)
        device = logits.device
        
        # Create binary targets: 1 if label > k, else 0
        # levels: [0, 1, 2, ..., K-2]
        levels = torch.arange(self.num_thresholds, device=device).unsqueeze(0)  # (1, K-1)
        targets = (labels.unsqueeze(1) > levels).float()  # (B, K-1)
        
        # Binary cross-entropy with logits
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')
        
        return loss


class CoralPredictor(nn.Module):
    """
    Convert CORAL logits to predicted class labels.
    """
    
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
    def forward(self, logits):
        """
        Args:
            logits: (B, K-1) raw scores.
            
        Returns:
            (B,) predicted class labels (0 to K-1).
        """
        probs = torch.sigmoid(logits)  # (B, K-1)
        # Predicted class = number of thresholds exceeded (prob > 0.5)
        predictions = (probs > 0.5).sum(dim=1)
        return predictions


class OrdinalHead(nn.Module):
    """
    Output head for ordinal classification using CORAL.
    """
    
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        
        # Single weight vector + K-1 bias terms (cutpoints)
        self.fc = nn.Linear(in_features, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.num_thresholds))
        
    def forward(self, x):
        """
        Args:
            x: (B, in_features) feature vector.
            
        Returns:
            (B, K-1) logits for each threshold.
        """
        # Single projection
        out = self.fc(x)  # (B, 1)
        # Broadcast and add bias (cutpoints)
        logits = out + self.bias  # (B, K-1)
        return logits


class RegressionHead(nn.Module):
    """
    Output head for continuous regression (e.g., PSA).
    """
    
    def __init__(self, in_features):
        super().__init__()
        self.fc = nn.Linear(in_features, 1)
        
    def forward(self, x):
        return self.fc(x).squeeze(-1)


# Test
if __name__ == "__main__":
    print("Testing CORAL Loss...")
    
    # Simulate 5-class ordinal problem (Gleason)
    num_classes = 5
    batch_size = 8
    
    loss_fn = CoralLoss(num_classes)
    predictor = CoralPredictor(num_classes)
    
    # Random logits (B, K-1)
    logits = torch.randn(batch_size, num_classes - 1)
    # Random labels (0 to K-1)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    loss = loss_fn(logits, labels)
    preds = predictor(logits)
    
    print(f"Labels: {labels.tolist()}")
    print(f"Predictions: {preds.tolist()}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test OrdinalHead
    print("\nTesting OrdinalHead...")
    in_features = 512
    head = OrdinalHead(in_features, num_classes)
    x = torch.randn(batch_size, in_features)
    out_logits = head(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out_logits.shape}")  # Should be (B, K-1)
    
    print("\n✓ All tests passed!")
