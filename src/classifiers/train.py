"""
Multi-Task Training Script for Prostate Cancer Classifiers.
Trains T-Stage, Gleason (ordinal), and PSA (regression) from multi-modal imaging.
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

# Local imports
from encoder import ImageEncoder3D, MultiModalEncoder
from ordinal_loss import CoralLoss, CoralPredictor, OrdinalHead, RegressionHead
from dataset import ProstateCancerDataset, collate_fn
from encode_labels import NUM_T_CLASSES, NUM_GLEASON_CLASSES


class MultiTaskClassifier(nn.Module):
    """
    Multi-task model for T-Stage, Gleason, and PSA prediction.
    """
    
    def __init__(
        self,
        modality_names=['CT', 'PET'],
        features_per_modality=256,
        model_depth=10,
    ):
        super().__init__()
        
        # Encoder
        self.encoder = MultiModalEncoder(
            modality_names=modality_names,
            features_per_modality=features_per_modality,
            model_depth=model_depth,
        )
        
        total_features = self.encoder.total_features
        
        # Shared feature projection
        self.shared_fc = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # Task-specific heads
        self.t_stage_head = OrdinalHead(512, NUM_T_CLASSES)      # 9 classes
        self.gleason_head = OrdinalHead(512, NUM_GLEASON_CLASSES) # 5 classes
        self.psa_head = RegressionHead(512)
        
        # Loss functions
        self.t_stage_loss = CoralLoss(NUM_T_CLASSES)
        self.gleason_loss = CoralLoss(NUM_GLEASON_CLASSES)
        self.psa_loss = nn.MSELoss()
        
        # Predictors
        self.t_stage_pred = CoralPredictor(NUM_T_CLASSES)
        self.gleason_pred = CoralPredictor(NUM_GLEASON_CLASSES)
        
    def forward(self, images_dict):
        """
        Args:
            images_dict: dict of {modality: (B, 1, D, H, W) tensor}
            
        Returns:
            dict of logits/predictions
        """
        features = self.encoder(images_dict)
        features = self.shared_fc(features)
        
        return {
            't_stage_logits': self.t_stage_head(features),
            'gleason_logits': self.gleason_head(features),
            'psa_pred': self.psa_head(features),
        }
    
    def compute_loss(self, outputs, labels, loss_weights=(1.0, 1.0, 0.01)):
        """
        Compute weighted multi-task loss.
        
        Args:
            outputs: dict from forward()
            labels: dict with T_label, Gleason_label, PSA_target
            loss_weights: (w_t, w_g, w_psa)
        """
        w_t, w_g, w_psa = loss_weights
        
        total_loss = 0.0
        loss_dict = {}
        
        # T-Stage loss (skip invalid labels = -1)
        t_mask = labels['T_label'] >= 0
        if t_mask.sum() > 0:
            t_loss = self.t_stage_loss(
                outputs['t_stage_logits'][t_mask],
                labels['T_label'][t_mask]
            )
            total_loss += w_t * t_loss
            loss_dict['t_stage'] = t_loss.item()
            
        # Gleason loss
        g_mask = labels['Gleason_label'] >= 0
        if g_mask.sum() > 0:
            g_loss = self.gleason_loss(
                outputs['gleason_logits'][g_mask],
                labels['Gleason_label'][g_mask]
            )
            total_loss += w_g * g_loss
            loss_dict['gleason'] = g_loss.item()
            
        # PSA loss
        psa_mask = ~torch.isnan(labels['PSA_target'])
        if psa_mask.sum() > 0:
            psa_loss = self.psa_loss(
                outputs['psa_pred'][psa_mask],
                labels['PSA_target'][psa_mask]
            )
            total_loss += w_psa * psa_loss
            loss_dict['psa'] = psa_loss.item()
            
        return total_loss, loss_dict


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for images, labels, _ in tqdm(loader, desc="Training"):
        # Move to device
        images = {k: v.to(device) for k, v in images.items()}
        labels = {k: v.to(device) for k, v in labels.items()}
        
        optimizer.zero_grad()
        outputs = model(images)
        loss, _ = model.compute_loss(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
    return total_loss / max(num_batches, 1)


def evaluate(model, loader, device):
    """Evaluate model."""
    model.eval()
    
    all_t_preds, all_t_labels = [], []
    all_g_preds, all_g_labels = [], []
    all_psa_preds, all_psa_labels = [], []
    
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Evaluating"):
            images = {k: v.to(device) for k, v in images.items()}
            labels = {k: v.to(device) for k, v in labels.items()}
            
            outputs = model(images)
            
            # T-Stage
            t_mask = labels['T_label'] >= 0
            if t_mask.sum() > 0:
                t_pred = model.t_stage_pred(outputs['t_stage_logits'][t_mask])
                all_t_preds.extend(t_pred.cpu().tolist())
                all_t_labels.extend(labels['T_label'][t_mask].cpu().tolist())
                
            # Gleason
            g_mask = labels['Gleason_label'] >= 0
            if g_mask.sum() > 0:
                g_pred = model.gleason_pred(outputs['gleason_logits'][g_mask])
                all_g_preds.extend(g_pred.cpu().tolist())
                all_g_labels.extend(labels['Gleason_label'][g_mask].cpu().tolist())
                
            # PSA
            psa_mask = ~torch.isnan(labels['PSA_target'])
            if psa_mask.sum() > 0:
                all_psa_preds.extend(outputs['psa_pred'][psa_mask].cpu().tolist())
                all_psa_labels.extend(labels['PSA_target'][psa_mask].cpu().tolist())
    
    metrics = {}
    
    # T-Stage metrics
    if all_t_labels:
        t_preds = np.array(all_t_preds)
        t_labels = np.array(all_t_labels)
        metrics['t_stage_acc'] = (t_preds == t_labels).mean()
        metrics['t_stage_mae'] = np.abs(t_preds - t_labels).mean()
        
    # Gleason metrics
    if all_g_labels:
        g_preds = np.array(all_g_preds)
        g_labels = np.array(all_g_labels)
        metrics['gleason_acc'] = (g_preds == g_labels).mean()
        metrics['gleason_mae'] = np.abs(g_preds - g_labels).mean()
        
    # PSA metrics
    if all_psa_labels:
        psa_preds = np.array(all_psa_preds)
        psa_labels = np.array(all_psa_labels)
        metrics['psa_rmse'] = np.sqrt(((psa_preds - psa_labels) ** 2).mean())
        metrics['psa_mae'] = np.abs(psa_preds - psa_labels).mean()
        
    return metrics


def main(args):
    print("=" * 50)
    print("Multi-Task Prostate Cancer Classifier Training")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load dataset
    dataset = ProstateCancerDataset(
        csv_path=args.csv_path,
        modalities=args.modalities,
        target_size=(args.vol_size, args.vol_size, args.vol_size),
        require_all_modalities=False,
    )
    
    # Train/Val split
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    
    print(f"Train: {n_train}, Val: {n_val}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    
    # Model
    model = MultiTaskClassifier(
        modality_names=args.modalities,
        features_per_modality=256,
        model_depth=args.model_depth,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        
        scheduler.step()
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val T-Stage Acc: {val_metrics.get('t_stage_acc', 0):.3f}, MAE: {val_metrics.get('t_stage_mae', 0):.3f}")
        print(f"  Val Gleason Acc: {val_metrics.get('gleason_acc', 0):.3f}, MAE: {val_metrics.get('gleason_mae', 0):.3f}")
        print(f"  Val PSA RMSE: {val_metrics.get('psa_rmse', 0):.2f}, MAE: {val_metrics.get('psa_mae', 0):.2f}")
        
        # Save best
        avg_acc = (val_metrics.get('t_stage_acc', 0) + val_metrics.get('gleason_acc', 0)) / 2
        if avg_acc > best_val_acc:
            best_val_acc = avg_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print("  [Saved best model]")
            
    print("\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='dataset_encoded.csv')
    parser.add_argument('--modalities', nargs='+', default=['CT', 'PET'])
    parser.add_argument('--vol_size', type=int, default=48)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model_depth', type=int, default=10, choices=[10, 18, 34])
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    main(args)
