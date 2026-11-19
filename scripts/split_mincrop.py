#!/usr/bin/env python3
"""
Split breast_mincrop.h5 into 3 shards for XNAT upload (nginx 413 limit workaround).
Output: hdf5/breast_mincrop_s00.h5, breast_mincrop_s01.h5, breast_mincrop_s02.h5
"""
import h5py
import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
log = logging.getLogger(__name__)

SRC = Path('/DATA/horizon_disentangled/hdf5/breast_mincrop.h5')
OUT_DIR = Path('/DATA/horizon_disentangled/hdf5')
N_SHARDS = 3

# Patient split points determined by size analysis:
# Shard 0: DUKE_001 -> DUKE_432  (430 patients, ~10.9 GB logical)
# Shard 1: DUKE_433 -> DUKE_873  (441 patients, ~10.9 GB logical)
# Shard 2: DUKE_874 -> SPY1_1239 (210 patients, ~10.9 GB logical)

SHARD_RANGES = [
    ('DUKE_001', 'DUKE_432'),
    ('DUKE_433', 'DUKE_873'),
    ('DUKE_874', None),  # None = last
]


def copy_group(src_group, dst_group):
    """Recursively copy all items from src_group to dst_group."""
    for key in src_group.keys():
        item = src_group[key]
        if isinstance(item, h5py.Dataset):
            src_group.copy(key, dst_group, name=key)
        elif isinstance(item, h5py.Group):
            dst_subgroup = dst_group.require_group(key)
            # Copy group attributes
            for attr_name, attr_val in item.attrs.items():
                dst_subgroup.attrs[attr_name] = attr_val
            copy_group(item, dst_subgroup)
    # Copy group attributes
    for attr_name, attr_val in src_group.attrs.items():
        dst_group.attrs[attr_name] = attr_val


def split_hdf5():
    log.info(f"Opening source: {SRC} ({SRC.stat().st_size/1e9:.2f} GB)")
    
    with h5py.File(SRC, 'r') as src:
        patients = sorted(src['patients'].keys())
        log.info(f"Total patients: {len(patients)}")
        
        # Partition patients into N_SHARDS groups
        shard_size = len(patients) // N_SHARDS
        shards = []
        for i in range(N_SHARDS):
            start = i * shard_size
            end = (i + 1) * shard_size if i < N_SHARDS - 1 else len(patients)
            shards.append(patients[start:end])
            log.info(f"Shard {i}: {len(shards[-1])} patients ({shards[-1][0]} -> {shards[-1][-1]})")
        
        for shard_idx, shard_patients in enumerate(shards):
            out_path = OUT_DIR / f'breast_mincrop_s{shard_idx:02d}.h5'
            if out_path.exists():
                log.info(f"[{shard_idx+1}/{N_SHARDS}] {out_path.name} already exists, skipping.")
                continue
            
            log.info(f"[{shard_idx+1}/{N_SHARDS}] Creating {out_path.name} with {len(shard_patients)} patients...")
            tmp_path = out_path.with_suffix('.h5.tmp')
            
            with h5py.File(tmp_path, 'w') as dst:
                # Copy root attributes
                for attr_name, attr_val in src.attrs.items():
                    dst.attrs[attr_name] = attr_val
                dst.attrs['shard_index'] = shard_idx
                dst.attrs['total_shards'] = N_SHARDS
                dst.attrs['source_file'] = 'breast_mincrop.h5'
                
                patients_grp = dst.require_group('patients')
                
                for j, patient_id in enumerate(shard_patients):
                    if j % 50 == 0:
                        log.info(f"  Progress: {j}/{len(shard_patients)} patients...")
                    src_patient = src[f'patients/{patient_id}']
                    dst_patient = patients_grp.require_group(patient_id)
                    copy_group(src_patient, dst_patient)
            
            # Rename tmp -> final
            tmp_path.rename(out_path)
            sz = out_path.stat().st_size
            log.info(f"  ✓ {out_path.name}: {sz/1e9:.2f} GB")
    
    log.info("Done! Shards created:")
    for i in range(N_SHARDS):
        p = OUT_DIR / f'breast_mincrop_s{i:02d}.h5'
        if p.exists():
            log.info(f"  {p.name}: {p.stat().st_size/1e9:.2f} GB")


if __name__ == '__main__':
    split_hdf5()
