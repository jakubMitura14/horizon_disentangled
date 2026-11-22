"""Multimodal Oncology Dataset — HDF5 Virtual Dataset Reader.

Reads longitudinal multi-visit patient data from the horizon_master.vds.h5
virtual dataset, supporting brain MRI (4-sequence), breast DCE-MRI
(3-timepoint), and I-SPY 2 kinetic maps. Each patient record includes
static clinical features, intervention schedules, and per-visit imaging.

Key classes:
    MultimodalOncologyDataset: PyTorch Dataset with lazy HDF5 file handles.

Key functions:
    worker_init_fn: DataLoader worker initializer for per-process HDF5 handles.

Example:
    >>> from src.pipeline.data.dataset import MultimodalOncologyDataset
    >>> ds = MultimodalOncologyDataset(domain='gbm', split=0)
    >>> sample = ds[0]
"""
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List, Optional, Tuple
from src.pipeline.data.transforms import build_physics_normalized_breast_dce, build_multisequence_brain_mri, standardize_spatial_shape

class MultimodalOncologyDataset(Dataset):
    """
    Multimodal oncology dataset connecting to horizon_master.vds.h5.
    Supports single-visit and longitudinal trajectory access across all 6 cohorts.
    """
    def __init__(
        self,
        vds_path: str = "/DATA/horizon_disentangled/hdf5/horizon_master.vds.h5",
        split: Optional[int] = None,
        domain: Optional[str] = None,
        cohort: Optional[str] = None,
        load_images: bool = True,
        brain_shape: Tuple[int, int, int] = (192, 224, 160),
        breast_shape: Tuple[int, int, int] = (256, 256, 64),
        max_patients: Optional[int] = None
    ):
        self.vds_path = vds_path
        self.load_images = load_images
        self.brain_shape = brain_shape
        self.breast_shape = breast_shape
        
        # Open in master to read registry
        with h5py.File(vds_path, 'r', swmr=True) as master:
            reg = master['registry/patients'][:]
            
        if split is not None:
            reg = reg[reg['split'] == split]
        if domain is not None:
            reg = reg[reg['domain'] == domain.encode('ascii')]
        if cohort is not None:
            reg = reg[reg['cohort'] == cohort.encode('ascii')]
            
        if max_patients is not None:
            reg = reg[:max_patients]
            
        self.registry = reg
        self.h5_file = None

    def __len__(self) -> int:
        return len(self.registry)

    def _get_h5(self):
        if self.h5_file is None:
            import hdf5plugin
            # 256 MB per-worker chunk cache greatly reduces syscall overhead for large 3D volumes.
            # rdcc_nslots: prime × 10 of expected simultaneous chunks (1024 chunks × 101 ≈ prime)
            rdcc_nslots = 1024 * 101          # ~103424 slots (prime for hash distribution)
            rdcc_nbytes = 256 * 1024 * 1024   # 256 MB raw data chunk cache
            rdcc_w0 = 0.75                    # evict 75% dirty chunks (good for read-heavy workloads)
            self.h5_file = h5py.File(
                self.vds_path, 'r', swmr=True,
                rdcc_nslots=rdcc_nslots,
                rdcc_nbytes=rdcc_nbytes,
                rdcc_w0=rdcc_w0,
            )
        return self.h5_file

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.registry[idx]
        uid = row['uid'].decode('ascii')
        domain = row['domain'].decode('ascii')
        cohort = row['cohort'].decode('ascii')
        split = int(row['split'])
        pcr = int(row['pcr'])
        event = int(row['event'])
        os_weeks = float(row['os_weeks']) if np.isfinite(row['os_weeks']) else -1.0
        num_visits = int(row['num_visits'])

        h5 = self._get_h5()
        p_grp = h5[f'patients/{uid}']

        # 1. Static clinical vector (standardized to 16-D float32)
        sc_raw = p_grp['static_clinical'][:] if 'static_clinical' in p_grp else np.zeros((1,), dtype=np.float32)
        sc = np.zeros((16,), dtype=np.float32)
        sc[:min(len(sc_raw), 16)] = sc_raw[:min(len(sc_raw), 16)]

        # 2. Interventions
        interv_data = []
        if 'interventions' in p_grp:
            interv_data = p_grp['interventions'][:]

        # 3. Longitudinal Visits
        v_grp = p_grp.get('visits')
        visits_out = []
        
        if v_grp is not None:
            for v_key in sorted(v_grp.keys()):
                v_sub = v_grp[v_key]
                v_dict = {'key': v_key}

                # Extract t_days & metadata attributes
                t_days = float(v_sub.attrs.get('t_days', 0.0))
                if 'breast_mincrop' in v_sub:
                    t_days = float(v_sub['breast_mincrop'].attrs.get('t_days', t_days))
                elif 'brain_mri' in v_sub:
                    t_days = float(v_sub['brain_mri'].attrs.get('t_days', t_days))
                elif 'ispy2_kinetic' in v_sub:
                    t_days = float(v_sub['ispy2_kinetic'].attrs.get('t_days', t_days))
                v_dict['t_days'] = t_days

                # Tabular features (pyradiomics, FeTS, NACT)
                if 'tabular_features' in v_sub:
                    v_dict['tabular'] = torch.from_numpy(v_sub['tabular_features'][:].astype(np.float32))
                else:
                    v_dict['tabular'] = torch.zeros((1,), dtype=torch.float32)

                # Imaging tensors
                has_image = False
                img_tensor = None
                mask_tensor = None

                if self.load_images:
                    if 'brain_mri' in v_sub:
                        # LUMIERE 4-sequence CT1/T1/T2/FLAIR
                        raw_brain = v_sub['brain_mri'][:]
                        img_norm = build_multisequence_brain_mri(raw_brain, self.brain_shape)
                        img_tensor = torch.from_numpy(img_norm)
                        has_image = True
                        if 'seg_tumor' in v_sub:
                            raw_mask = v_sub['seg_tumor'][:]
                            mask_std = standardize_spatial_shape(raw_mask, self.brain_shape)
                            mask_tensor = torch.from_numpy(mask_std)
                    elif 'breast_mincrop' in v_sub:
                        # BreastDCEDL (3, Y, X, Z) -> build 4-channel tensor
                        raw_crop = v_sub['breast_mincrop'][:] # (3, H, W, Z)
                        s0 = raw_crop[0]
                        s1 = raw_crop[1]
                        s2 = raw_crop[2]
                        dce_norm = build_physics_normalized_breast_dce(s0, s1, s2, target_shape=self.breast_shape)
                        img_tensor = torch.from_numpy(dce_norm)
                        has_image = True
                        if 'seg_tumor' in v_sub:
                            raw_mask = v_sub['seg_tumor'][:]
                            mask_std = standardize_spatial_shape(raw_mask, self.breast_shape)
                            mask_tensor = torch.from_numpy(mask_std)
                    elif 'ispy2_kinetic' in v_sub:
                        # I-SPY 2 (3, Y, X, Z) [SER, PE2, PE5]
                        raw_kin = v_sub['ispy2_kinetic'][:]
                        ser_vol = raw_kin[0]
                        pe2_vol = raw_kin[1]
                        pe5_vol = raw_kin[2]
                        # Create pseudo pre-contrast S0 from PE2 and early signal
                        s0_synth = np.ones_like(pe2_vol, dtype=np.float32)
                        dce_norm = build_physics_normalized_breast_dce(
                            s0_synth, pe2_vol, pe5_vol, ser=ser_vol, target_shape=self.breast_shape
                        )
                        img_tensor = torch.from_numpy(dce_norm)
                        has_image = True
                        if 'seg_tumor' in v_sub:
                            raw_mask = v_sub['seg_tumor'][:]
                            mask_std = standardize_spatial_shape(raw_mask, self.breast_shape)
                            mask_tensor = torch.from_numpy(mask_std)

                v_dict['has_image'] = has_image
                v_dict['image'] = img_tensor
                v_dict['mask'] = mask_tensor
                visits_out.append(v_dict)

        # Baseline imaging (Visit 0)
        baseline_img = None
        baseline_mask = None
        has_baseline_img = False
        if visits_out and visits_out[0]['has_image']:
            baseline_img = visits_out[0]['image']
            baseline_mask = visits_out[0]['mask']
            has_baseline_img = True

        return {
            'uid': uid,
            'domain': domain,
            'cohort': cohort,
            'split': split,
            'pcr': pcr,
            'event': event,
            'os_weeks': os_weeks,
            'num_visits': num_visits,
            'clinical': torch.from_numpy(sc),
            'interventions': interv_data,
            'has_baseline_img': has_baseline_img,
            'baseline_img': baseline_img,
            'baseline_mask': baseline_mask,
            'visits': visits_out
        }

    def close(self):
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

def worker_init_fn(worker_id):
    """Initializes per-worker HDF5 state with registered hdf5plugin decompression filters.

    Also seeds NumPy RNG per worker so stochastic augmentations are independent
    and reproducible (combining global seed with worker id avoids identical sequences).
    """
    import hdf5plugin
    import numpy as np
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        dataset.h5_file = None   # each worker opens its own HDF5 handle
        # Deterministic per-worker seed: combines PyTorch global seed with worker index
        seed = (torch.initial_seed() + worker_id) % (2**32)
        np.random.seed(seed)
