"""
PyTorch Dataset for loading NIfTI medical imaging data with clinical labels.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import nibabel as nib

from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Orientation,
    Spacing,
    ScaleIntensity,
    CenterSpatialCrop,
    Resize,
    ToTensor,
)


class ProstateCancerDataset(Dataset):
    """
    Dataset for prostate cancer imaging with clinical labels.
    Loads NIfTI volumes and corresponding labels from dataset_encoded.csv.
    """
    
    def __init__(
        self,
        csv_path,
        modalities=['CT', 'PET'],
        target_size=(48, 48, 48),
        transform=None,
        require_all_modalities=False,
    ):
        """
        Args:
            csv_path: Path to dataset_encoded.csv
            modalities: List of modality names to load ['CT', 'T1', 'T2', 'ADC', 'DWI', 'PET']
            target_size: Resize volumes to this shape (D, H, W)
            transform: Additional MONAI transforms
            require_all_modalities: If True, skip samples missing any modality
        """
        self.df = pd.read_csv(csv_path)
        self.modalities = modalities
        self.target_size = target_size
        self.require_all_modalities = require_all_modalities
        
        # Path column mapping
        self.path_cols = {
            'CT': 'CT_Path',
            'T1': 'T1_Path',
            'T2': 'T2_Paths',  # Note: may have multiple, take first
            'ADC': 'ADC_Path',
            'DWI': 'DWI_Path',
            'PET': 'PET_Path',
        }
        
        # Default transforms
        self.base_transform = Compose([
            ScaleIntensity(minv=0.0, maxv=1.0),
            Resize(spatial_size=target_size),
            ToTensor(),
        ])
        
        self.extra_transform = transform
        
        # Filter valid samples
        self._filter_samples()
        
    def _filter_samples(self):
        """Remove samples without required data."""
        valid_indices = []
        
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            
            # Check labels exist
            if pd.isna(row.get('T_label')) and pd.isna(row.get('Gleason_label')) and pd.isna(row.get('PSA_target')):
                continue
            
            # Check modalities
            if self.require_all_modalities:
                has_all = True
                for mod in self.modalities:
                    path_col = self.path_cols.get(mod)
                    if path_col and (pd.isna(row.get(path_col)) or not row.get(f'{mod}_Available', False)):
                        has_all = False
                        break
                if not has_all:
                    continue
                    
            valid_indices.append(idx)
            
        self.valid_indices = valid_indices
        print(f"Dataset: {len(valid_indices)}/{len(self.df)} samples with valid data")
        
    def __len__(self):
        return len(self.valid_indices)
    
    def _load_volume(self, path):
        """Load a single NIfTI volume."""
        if pd.isna(path) or not os.path.exists(path):
            return None
            
        try:
            nii = nib.load(path)
            data = nii.get_fdata().astype(np.float32)
            
            # Add channel dimension if needed
            if data.ndim == 3:
                data = data[np.newaxis, ...]  # (1, D, H, W)
                
            # Apply transforms
            data = torch.from_numpy(data)
            data = self.base_transform(data)
            
            return data
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None
    
    def __getitem__(self, idx):
        """
        Returns:
            images: dict of {modality: (1, D, H, W) tensor}
            labels: dict of {T_label, Gleason_label, PSA_target}
            patient_id: string
        """
        real_idx = self.valid_indices[idx]
        row = self.df.iloc[real_idx]
        
        # Load images
        images = {}
        for mod in self.modalities:
            path_col = self.path_cols.get(mod)
            if path_col:
                path = row.get(path_col)
                # Handle multiple paths (T2)
                if isinstance(path, str) and ';' in path:
                    path = path.split(';')[0].strip()
                    
                vol = self._load_volume(path)
                if vol is not None:
                    images[mod] = vol
                    
        # Get labels
        labels = {
            'T_label': int(row['T_label']) if pd.notna(row.get('T_label')) else -1,
            'Gleason_label': int(row['Gleason_label']) if pd.notna(row.get('Gleason_label')) else -1,
            'PSA_target': float(row['PSA_target']) if pd.notna(row.get('PSA_target')) else np.nan,
        }
        
        return images, labels, row['Patient_ID']


def collate_fn(batch):
    """
    Custom collate for handling variable modalities.
    """
    images_list, labels_list, patient_ids = zip(*batch)
    
    # Stack images per modality
    all_modalities = set()
    for img_dict in images_list:
        all_modalities.update(img_dict.keys())
        
    stacked_images = {}
    for mod in all_modalities:
        vols = [img_dict.get(mod) for img_dict in images_list]
        # Filter None and stack
        valid_vols = [v for v in vols if v is not None]
        if valid_vols:
            stacked_images[mod] = torch.stack(valid_vols)
            
    # Stack labels
    stacked_labels = {
        'T_label': torch.tensor([l['T_label'] for l in labels_list], dtype=torch.long),
        'Gleason_label': torch.tensor([l['Gleason_label'] for l in labels_list], dtype=torch.long),
        'PSA_target': torch.tensor([l['PSA_target'] for l in labels_list], dtype=torch.float32),
    }
    
    return stacked_images, stacked_labels, patient_ids


# Test
if __name__ == "__main__":
    print("Testing ProstateCancerDataset...")
    
    dataset = ProstateCancerDataset(
        csv_path="dataset_encoded.csv",
        modalities=['CT', 'PET'],
        target_size=(32, 32, 32),  # Smaller for testing
        require_all_modalities=False,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        images, labels, pid = dataset[0]
        print(f"\nPatient: {pid}")
        print(f"Labels: {labels}")
        print(f"Loaded modalities: {list(images.keys())}")
        for mod, vol in images.items():
            print(f"  {mod}: {vol.shape}")
            
    print("\n✓ Dataset test complete!")
