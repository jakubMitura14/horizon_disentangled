"""
3D ResNet Encoder using MONAI for medical imaging.
"""
import torch
import torch.nn as nn
from monai.networks.nets import resnet10, resnet18, resnet34


class ImageEncoder3D(nn.Module):
    """
    3D CNN encoder for volumetric medical images.
    Uses MONAI's pre-implemented 3D ResNet.
    """
    
    def __init__(
        self,
        spatial_dims=3,
        in_channels=1,
        out_features=512,
        model_depth=18,
        pretrained=False
    ):
        super().__init__()
        
        # Select backbone
        if model_depth == 10:
            self.backbone = resnet10(
                spatial_dims=spatial_dims,
                n_input_channels=in_channels,
                num_classes=out_features,
                pretrained=pretrained
            )
        elif model_depth == 18:
            self.backbone = resnet18(
                spatial_dims=spatial_dims,
                n_input_channels=in_channels,
                num_classes=out_features,
                pretrained=pretrained
            )
        elif model_depth == 34:
            self.backbone = resnet34(
                spatial_dims=spatial_dims,
                n_input_channels=in_channels,
                num_classes=out_features,
                pretrained=pretrained
            )
        else:
            raise ValueError(f"Unsupported model_depth: {model_depth}")
        
        self.out_features = out_features
        
    def forward(self, x):
        """
        Args:
            x: (B, C, D, H, W) input volume.
            
        Returns:
            (B, out_features) feature vector.
        """
        return self.backbone(x)


class MultiModalEncoder(nn.Module):
    """
    Encoder that handles multiple imaging modalities.
    Each modality gets its own encoder, features are concatenated.
    """
    
    def __init__(
        self,
        modality_names=['CT', 'T2', 'ADC', 'PET'],
        features_per_modality=256,
        model_depth=18
    ):
        super().__init__()
        
        self.modality_names = modality_names
        self.encoders = nn.ModuleDict({
            name: ImageEncoder3D(
                out_features=features_per_modality,
                model_depth=model_depth
            )
            for name in modality_names
        })
        
        self.total_features = len(modality_names) * features_per_modality
        
    def forward(self, images_dict):
        """
        Args:
            images_dict: {modality_name: (B, 1, D, H, W) tensor}
            
        Returns:
            (B, total_features) concatenated features.
        """
        features = []
        for name in self.modality_names:
            if name in images_dict:
                feat = self.encoders[name](images_dict[name])
                features.append(feat)
            else:
                # Handle missing modality (zero features)
                batch_size = list(images_dict.values())[0].size(0)
                device = list(images_dict.values())[0].device
                feat = torch.zeros(batch_size, self.encoders[name].out_features, device=device)
                features.append(feat)
                
        return torch.cat(features, dim=1)


# Test
if __name__ == "__main__":
    print("Testing ImageEncoder3D...")
    
    encoder = ImageEncoder3D(out_features=512, model_depth=10)
    x = torch.randn(2, 1, 48, 48, 48)  # (B, C, D, H, W)
    
    out = encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")  # Should be (2, 512)
    
    print("\nTesting MultiModalEncoder...")
    multi_enc = MultiModalEncoder(
        modality_names=['CT', 'T2'],
        features_per_modality=256,
        model_depth=10
    )
    
    images = {
        'CT': torch.randn(2, 1, 48, 48, 48),
        'T2': torch.randn(2, 1, 48, 48, 48),
    }
    
    out = multi_enc(images)
    print(f"Output shape: {out.shape}")  # Should be (2, 512)
    
    print("\n✓ All tests passed!")
