import os
import time
import h5py
import numpy as np
import torch
import hdf5plugin
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Optional

class HorizonClinicalDataset(Dataset):
    """PyTorch Dataset querying multimodal oncology patients from horizon_master.vds.h5."""
    def __init__(self, vds_path: str, split: Optional[int] = None, domain: Optional[str] = None):
        self.vds_path = vds_path
        with h5py.File(vds_path, 'r', swmr=True) as f:
            reg = f['registry/patients'][:]

        if split is not None:
            reg = reg[reg['split'] == split]
        if domain is not None:
            reg = reg[reg['domain'] == domain.encode('ascii')]

        self.registry = reg
        self.h5_file = None

    def __len__(self):
        return len(self.registry)

    def _get_h5(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.vds_path, 'r', swmr=True)
        return self.h5_file

    def __getitem__(self, idx):
        row = self.registry[idx]
        uid = row['uid'].decode('ascii')
        domain = row['domain'].decode('ascii')
        cohort = row['cohort'].decode('ascii')
        split = int(row['split'])
        pcr = int(row['pcr'])
        event = int(row['event'])
        os_weeks = float(row['os_weeks']) if np.isfinite(row['os_weeks']) else -1.0
        num_v = int(row['num_visits'])

        h5 = self._get_h5()
        p_grp = h5[f'patients/{uid}']

        # Standardize clinical vector length to 16 with zero-padding
        sc_raw = p_grp['static_clinical'][:] if 'static_clinical' in p_grp else np.zeros((1,), dtype=np.float32)
        sc = np.zeros((16,), dtype=np.float32)
        sc[:min(len(sc_raw), 16)] = sc_raw[:min(len(sc_raw), 16)]

        return {
            'uid': uid,
            'domain': domain,
            'cohort': cohort,
            'split': split,
            'pcr': pcr,
            'event': event,
            'os_weeks': os_weeks,
            'num_visits': num_v,
            'clinical': torch.from_numpy(sc)
        }

    def close(self):
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

class HorizonVolumetricDataset(Dataset):
    """PyTorch Dataset streaming 3D volumetric imaging from the Master VDS."""
    def __init__(self, vds_path: str, max_samples: Optional[int] = None):
        self.vds_path = vds_path
        with h5py.File(vds_path, 'r', swmr=True) as f:
            reg = f['registry/patients'][:]
            img_cohorts = [b'lumiere', b'breastdcedl_duke', b'breastdcedl_spy1', b'ispy2_ispy2', b'ispy2_acrin_6698']
            self.uids = [r['uid'].decode('ascii') for r in reg if r['cohort'] in img_cohorts]

        if max_samples:
            self.uids = self.uids[:max_samples]
        self.h5_file = None

    def __len__(self):
        return len(self.uids)

    def _get_h5(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.vds_path, 'r', swmr=True)
        return self.h5_file

    def __getitem__(self, idx):
        uid = self.uids[idx]
        h5 = self._get_h5()
        p_grp = h5[f'patients/{uid}']
        v_grp = p_grp['visits']
        v0 = v_grp[list(v_grp.keys())[0]]

        vol_arr = None
        key_found = ''
        for key in ['brain_mri', 'breast_mincrop', 'ispy2_kinetic']:
            if key in v0:
                raw = v0[key]
                C, H, W, Z = raw.shape
                z_mid = Z // 2
                vol_arr = raw[:, :, :, max(0, z_mid-4):min(Z, z_mid+4)]
                key_found = key
                break

        return {
            'uid': uid,
            'key': key_found,
            'shape': vol_arr.shape if vol_arr is not None else None,
            'finite': bool(np.isfinite(vol_arr).all()) if vol_arr is not None else False
        }

    def close(self):
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

def worker_init_fn(worker_id):
    import hdf5plugin
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.h5_file = None

def test_vds_inspection(vds_path: str = "/DATA/horizon_disentangled/hdf5/horizon_master.vds.h5"):
    print("=" * 60)
    print("STEP 1: Inspecting Master VDS Structure & Attributes")
    print("=" * 60)
    assert os.path.exists(vds_path), f"Master VDS file not found: {vds_path}"

    with h5py.File(vds_path, 'r', swmr=True) as master:
        print("Root Attributes:")
        for k, v in master.attrs.items():
            print(f"  {k}: {v}")

        assert "license" in master.attrs, "Missing license attribute"
        assert master.attrs["license"] == "CC-BY-NC-4.0", "Incorrect license attribute"

        assert "registry/patients" in master, "Missing /registry/patients dataset"
        reg = master["registry/patients"][:]
        print(f"\nTotal registered patients: {len(reg)}")

        # Cohort distribution
        cohorts = [r['cohort'].decode('ascii') for r in reg]
        domains = [r['domain'].decode('ascii') for r in reg]
        splits = [r['split'] for r in reg]

        print("\nCohort Patient Counts:")
        for c in sorted(set(cohorts)):
            cnt = cohorts.count(c)
            dom = domains[cohorts.index(c)]
            c_splits = [s for s, ch in zip(splits, cohorts) if ch == c]
            print(f"  {c:20s} ({dom:6s}): {cnt:4d} patients | train={c_splits.count(0)}, val={c_splits.count(1)}, test={c_splits.count(2)}")

        print("\nSplit Distribution Overall:")
        print(f"  Train (0): {splits.count(0)} ({100*splits.count(0)/len(splits):.1f}%)")
        print(f"  Val   (1): {splits.count(1)} ({100*splits.count(1)/len(splits):.1f}%)")
        print(f"  Test  (2): {splits.count(2)} ({100*splits.count(2)/len(splits):.1f}%)")

def test_cohort_queries(vds_path: str = "/DATA/horizon_disentangled/hdf5/horizon_master.vds.h5"):
    print("\n" + "=" * 60)
    print("STEP 2: Testing Transparent Cross-Cohort Patient Queries")
    print("=" * 60)

    cohort_test_prefixes = {
        "CFB-GBM": "CFB_",
        "MU-Glioma-Post": "MUG_",
        "Burdenko-GBM": "BUR_",
        "BreastDCEDL Duke": "DUKE_",
        "BreastDCEDL SPY1": "SPY1_",
        "LUMIERE": "LUM_",
        "I-SPY 2": "ISP2_"
    }

    with h5py.File(vds_path, 'r', swmr=True) as master:
        all_pats = list(master['patients'].keys())
        for cname, pfx in cohort_test_prefixes.items():
            matching = [p for p in all_pats if p.startswith(pfx)]
            assert len(matching) > 0, f"No patients found for {cname} (prefix {pfx})"
            sample_uid = matching[0]
            p_grp = master[f'patients/{sample_uid}']

            print(f"\n[OK] Querying {cname} -> Patient: {sample_uid}")
            print(f"     Keys: {list(p_grp.keys())}")
            if 'static_clinical' in p_grp:
                print(f"     static_clinical shape: {p_grp['static_clinical'].shape}, attrs: {dict(p_grp['static_clinical'].attrs)}")
            if 'visits' in p_grp:
                visits = list(p_grp['visits'].keys())
                print(f"     Visits count: {len(visits)} (samples: {visits[:3]})")
                if visits:
                    v0 = p_grp[f'visits/{visits[0]}']
                    print(f"     Visit 0 datasets: {list(v0.keys())}")
                    for dname in v0.keys():
                        dset = v0[dname]
                        if hasattr(dset, 'shape'):
                            print(f"       -> {dname}: shape={dset.shape}, dtype={dset.dtype}")

def test_pytorch_dataloader(vds_path: str = "/DATA/horizon_disentangled/hdf5/horizon_master.vds.h5"):
    print("\n" + "=" * 60)
    print("STEP 3: Testing PyTorch Multi-Worker DataLoader (swmr=True, 4 workers)")
    print("=" * 60)

    dataset = HorizonClinicalDataset(vds_path, split=0) # Train split
    print(f"Loaded training split: {len(dataset)} patients")

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        worker_init_fn=worker_init_fn,
        pin_memory=False
    )

    t0 = time.time()
    batch_count = 0
    sample_count = 0

    for batch in loader:
        batch_count += 1
        sample_count += len(batch['uid'])
        assert torch.isfinite(batch['clinical']).all(), "Non-finite clinical values encountered"

    elapsed = time.time() - t0
    fps = sample_count / elapsed if elapsed > 0 else 0
    print(f"PyTorch Multi-Worker DataLoader Results:")
    print(f"  Processed {sample_count} patients across {batch_count} batches in {elapsed:.3f}s")
    print(f"  Throughput: {fps:.1f} samples/sec")
    dataset.close()
    print("PyTorch multi-worker data loading verified successfully!")

def test_3d_volumetric_streaming(vds_path: str = "/DATA/horizon_disentangled/hdf5/horizon_master.vds.h5"):
    print("\n" + "=" * 60)
    print("STEP 4: Testing 3D Volumetric Multi-Worker Streaming (4 workers, Blosc2)")
    print("=" * 60)

    dataset = HorizonVolumetricDataset(vds_path, max_samples=64)
    print(f"Loaded volumetric imaging dataset sample: {len(dataset)} patients")

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        worker_init_fn=worker_init_fn,
        collate_fn=lambda b: b
    )

    t0 = time.time()
    streamed = 0
    for b in loader:
        for item in b:
            if item['shape'] is not None:
                streamed += 1
                assert item['finite'], f"Non-finite voxels found in {item['uid']}"

    elapsed = time.time() - t0
    fps = streamed / elapsed if elapsed > 0 else 0
    print(f"3D Volumetric Streaming Results:")
    print(f"  Streamed {streamed}/{len(dataset)} 3D sub-volumes in {elapsed:.3f}s")
    print(f"  Throughput: {fps:.1f} 3D volumes/sec")
    dataset.close()
    print("3D Volumetric streaming verified successfully!")

def run_all_tests():
    vds_path = "/DATA/horizon_disentangled/hdf5/horizon_master.vds.h5"
    test_vds_inspection(vds_path)
    test_cohort_queries(vds_path)
    test_pytorch_dataloader(vds_path)
    test_3d_volumetric_streaming(vds_path)
    print("\n" + "=" * 60)
    print("ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()
