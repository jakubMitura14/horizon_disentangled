import os
import re
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from collections import defaultdict
from src.preprocessing.hdf5_writer import HDF5ShardWriter, compute_xxhash64
from src.preprocessing.ledger import IngestionLedger

def get_foreground_stats(arr: np.ndarray):
    """Computes p01, p50, p99, mean, std for non-zero voxels."""
    fg = arr[arr > 0]
    if len(fg) == 0:
        fg = arr.ravel()
    return {
        "p01": float(np.percentile(fg, 1)),
        "p50": float(np.percentile(fg, 50)),
        "p99": float(np.percentile(fg, 99)),
        "mean": float(np.mean(fg)),
        "std": float(np.std(fg))
    }

def process_duke_patient(pid: str, acq_files: list, duke_meta_row: pd.Series,
                         writer: HDF5ShardWriter, ledger: IngestionLedger, duke_dir: str):
    """Processes a single Duke patient: loads acq0, acq1, acq3, rasterizes bbox, verifies, unlinks raw files."""
    canonical_uid = f"DUKE_{pid.replace('Breast_MRI_', '')}"
    ledger.register_patient(canonical_uid, "breastdcedl_duke", "breast")

    # Acquisition mapping from metadata: pre, post_early, post_late
    pre_idx = int(duke_meta_row.get('pre', 0)) if pd.notna(duke_meta_row.get('pre')) else 0
    early_idx = int(duke_meta_row.get('post_early', 1)) if pd.notna(duke_meta_row.get('post_early')) else 1
    late_idx = int(duke_meta_row.get('post_late', 3)) if pd.notna(duke_meta_row.get('post_late')) else 3

    acq_dict = {acq: os.path.join(duke_dir, fname) for acq, fname in acq_files}
    acq_keys = sorted(acq_dict.keys())

    pre_file = acq_dict.get(pre_idx, acq_dict.get(0))
    early_file = acq_dict.get(early_idx, acq_dict.get(1))
    late_file = acq_dict.get(late_idx)
    if late_file is None:
        # Fallback to last acquisition
        late_file = acq_dict[acq_keys[-1]]

    if pre_file is None or early_file is None:
        ledger.update_status(canonical_uid, "FAILED", error_msg="Missing pre or early acquisition")
        return False

    # Load 3 phases
    try:
        img_pre = nib.load(pre_file)
        arr_pre = img_pre.get_fdata(dtype=np.float32, caching='unchanged')
        img_pre.uncache()

        img_early = nib.load(early_file)
        arr_early = img_early.get_fdata(dtype=np.float32, caching='unchanged')
        img_early.uncache()

        img_late = nib.load(late_file)
        arr_late = img_late.get_fdata(dtype=np.float32, caching='unchanged')
        img_late.uncache()

        # If late acquisition shape does not match pre, try other available acquisitions
        if arr_late.shape != arr_pre.shape:
            for cand_acq, cand_file in sorted(acq_dict.items(), reverse=True):
                if cand_acq not in (pre_idx, early_idx):
                    try:
                        cand_img = nib.load(cand_file)
                        cand_arr = cand_img.get_fdata(dtype=np.float32, caching='unchanged')
                        cand_img.uncache()
                        if cand_arr.shape == arr_pre.shape:
                            arr_late = cand_arr
                            late_file = cand_file
                            break
                    except Exception:
                        pass
    except Exception as e:
        ledger.update_status(canonical_uid, "FAILED", error_msg=f"NIfTI read error: {e}")
        return False

    # Standardize dimensions to (3, H, W, Z)
    if arr_pre.ndim == 3 and arr_pre.shape[1] == arr_pre.shape[2] and arr_pre.shape[0] != arr_pre.shape[1]:
        arr_pre = np.transpose(arr_pre, (1, 2, 0)) # -> (H, W, Z)
        arr_early = np.transpose(arr_early, (1, 2, 0))
        arr_late = np.transpose(arr_late, (1, 2, 0))

    if arr_pre.shape != arr_early.shape or arr_pre.shape != arr_late.shape:
        ledger.update_status(canonical_uid, "FAILED", error_msg=f"Shape mismatch: {arr_pre.shape} vs {arr_early.shape}")
        return False

    H, W, Z = arr_pre.shape
    crop_stack = np.stack([arr_pre, arr_early, arr_late], axis=0) # (3, H, W, Z)
    stats = get_foreground_stats(crop_stack)

    # Rasterize Duke 3D bounding box
    seg_mask = np.zeros((1, H, W, Z), dtype=np.uint8)
    sraw = int(duke_meta_row.get('sraw', 0))
    eraw = int(duke_meta_row.get('eraw', H - 1))
    scol = int(duke_meta_row.get('scol', 0))
    ecol = int(duke_meta_row.get('ecol', W - 1))
    m_start = int(duke_meta_row.get('mask_start', 0))
    m_end = int(duke_meta_row.get('mask_end', Z - 1))

    # Clamp bounds safely
    sraw = max(0, min(H - 1, sraw))
    eraw = max(sraw, min(H - 1, eraw))
    scol = max(0, min(W - 1, scol))
    ecol = max(scol, min(W - 1, ecol))
    m_start = max(0, min(Z - 1, m_start))
    m_end = max(m_start, min(Z - 1, m_end))

    seg_mask[0, sraw:eraw+1, scol:ecol+1, m_start:m_end+1] = 1

    # Clinical covariates
    pcr_val = int(duke_meta_row.get('pCR', -1)) if pd.notna(duke_meta_row.get('pCR')) else -1
    hr_val = float(duke_meta_row.get('HR', -1)) if pd.notna(duke_meta_row.get('HR')) else -1.0
    her2_val = float(duke_meta_row.get('HER2', -1)) if pd.notna(duke_meta_row.get('HER2')) else -1.0
    age_val = float(duke_meta_row.get('age', 0)) if pd.notna(duke_meta_row.get('age')) else 0.0
    split_val = int(duke_meta_row.get('test', 0)) if pd.notna(duke_meta_row.get('test')) else 0

    static_vec = np.array([age_val, float(pcr_val), hr_val, her2_val], dtype=np.float32)

    p_grp = writer.h5.require_group(f"patients/{canonical_uid}")
    writer.write_array(p_grp, "static_clinical", static_vec, attrs={
        "pcr": pcr_val,
        "hr": hr_val,
        "her2": her2_val,
        "split": split_val,
        "domain": "breast",
        "cohort": "breastdcedl_duke"
    })

    v_grp = p_grp.require_group("visits/v_0")
    chunk_spatial = (1, min(128, H), min(128, W), min(16, Z))

    img_attrs = {
        "t_days": 0.0,
        "p01": stats["p01"],
        "p50": stats["p50"],
        "p99": stats["p99"],
        "mean": stats["mean"],
        "std": stats["std"],
        "has_pixel_mask": 0,
        "bbox_sraw": sraw,
        "bbox_eraw": eraw,
        "bbox_scol": scol,
        "bbox_ecol": ecol,
        "bbox_mstart": m_start,
        "bbox_mend": m_end
    }

    _, img_digest = writer.write_array(v_grp, "breast_mincrop", crop_stack, chunks=chunk_spatial, attrs=img_attrs)
    _, seg_digest = writer.write_array(v_grp, "seg_tumor", seg_mask, chunks=chunk_spatial, attrs={"has_pixel_mask": 0})

    writer.flush_and_sync()

    # Read-back verification
    verify_img = writer.verify_dataset(f"patients/{canonical_uid}/visits/v_0/breast_mincrop", img_digest, crop_stack.shape, np.float32)
    verify_seg = writer.verify_dataset(f"patients/{canonical_uid}/visits/v_0/seg_tumor", seg_digest, seg_mask.shape, np.uint8)

    if not (verify_img and verify_seg):
        ledger.update_status(canonical_uid, "FAILED", error_msg="Read-back checksum verification failed")
        return False

    # Unlink raw patient files
    raw_bytes = 0
    for _, fname in acq_files:
        fpath = os.path.join(duke_dir, fname)
        if os.path.exists(fpath):
            raw_bytes += os.path.getsize(fpath)
            os.remove(fpath)

    ledger.update_status(canonical_uid, "RAW_DELETED", shard_file="breast_mincrop.h5",
                         num_visits=1, raw_size_bytes=raw_bytes,
                         checksums={"breast_mincrop": img_digest, "seg_tumor": seg_digest})
    return True

def process_spy1_patient(pid: str, acq_files: list, spy1_meta_row: pd.Series,
                         writer: HDF5ShardWriter, ledger: IngestionLedger,
                         spy1_dir: str, mask_dir: str):
    """Processes a single SPY1 patient: loads acq0, acq1, acq2, attaches true mask, verifies, unlinks raw files."""
    canonical_uid = f"SPY1_{pid.replace('ISPY1_', '')}"
    ledger.register_patient(canonical_uid, "breastdcedl_spy1", "breast")

    acq_dict = {acq: os.path.join(spy1_dir, fname) for acq, fname in acq_files}
    if 0 not in acq_dict or 1 not in acq_dict:
        ledger.update_status(canonical_uid, "FAILED", error_msg="Missing acq0 or acq1")
        return False

    pre_file = acq_dict[0]
    early_file = acq_dict[1]
    late_file = acq_dict.get(2, acq_dict[max(acq_dict.keys())])
    mask_file = os.path.join(mask_dir, f"{pid}_spy1_vis1_mask.nii.gz")

    try:
        img_pre = nib.load(pre_file)
        arr_pre = img_pre.get_fdata(dtype=np.float32, caching='unchanged')
        img_pre.uncache()

        img_early = nib.load(early_file)
        arr_early = img_early.get_fdata(dtype=np.float32, caching='unchanged')
        img_early.uncache()

        img_late = nib.load(late_file)
        arr_late = img_late.get_fdata(dtype=np.float32, caching='unchanged')
        img_late.uncache()

        if os.path.exists(mask_file):
            img_m = nib.load(mask_file)
            arr_m = img_m.get_fdata(dtype=np.float32, caching='unchanged')
            img_m.uncache()
            mask_arr = (arr_m > 0).astype(np.uint8)
        else:
            mask_arr = np.zeros_like(arr_pre, dtype=np.uint8)
    except Exception as e:
        ledger.update_status(canonical_uid, "FAILED", error_msg=f"NIfTI read error: {e}")
        return False

    # Transpose (Z, H, W) -> (H, W, Z)
    if arr_pre.ndim == 3 and arr_pre.shape[1] == arr_pre.shape[2] and arr_pre.shape[0] != arr_pre.shape[1]:
        arr_pre = np.transpose(arr_pre, (1, 2, 0))
        arr_early = np.transpose(arr_early, (1, 2, 0))
        arr_late = np.transpose(arr_late, (1, 2, 0))
        mask_arr = np.transpose(mask_arr, (1, 2, 0))

    if arr_pre.shape != arr_early.shape or arr_pre.shape != arr_late.shape:
        ledger.update_status(canonical_uid, "FAILED", error_msg=f"Shape mismatch: {arr_pre.shape} vs {arr_early.shape}")
        return False

    H, W, Z = arr_pre.shape
    crop_stack = np.stack([arr_pre, arr_early, arr_late], axis=0) # (3, H, W, Z)
    seg_mask = np.expand_dims(mask_arr, axis=0)                   # (1, H, W, Z)
    stats = get_foreground_stats(crop_stack)

    # Clinical covariates
    pcr_val = int(spy1_meta_row.get('pCR', -1)) if pd.notna(spy1_meta_row.get('pCR')) else -1
    hr_val = float(spy1_meta_row.get('HR', -1)) if pd.notna(spy1_meta_row.get('HR')) else -1.0
    her2_val = float(spy1_meta_row.get('HER2', -1)) if pd.notna(spy1_meta_row.get('HER2')) else -1.0
    age_val = float(spy1_meta_row.get('age', 0)) if pd.notna(spy1_meta_row.get('age')) else 0.0
    split_val = 0 # Default train/val split

    static_vec = np.array([age_val, float(pcr_val), hr_val, her2_val], dtype=np.float32)

    p_grp = writer.h5.require_group(f"patients/{canonical_uid}")
    writer.write_array(p_grp, "static_clinical", static_vec, attrs={
        "pcr": pcr_val,
        "hr": hr_val,
        "her2": her2_val,
        "split": split_val,
        "domain": "breast",
        "cohort": "breastdcedl_spy1"
    })

    v_grp = p_grp.require_group("visits/v_0")
    chunk_spatial = (1, min(128, H), min(128, W), min(16, Z))

    img_attrs = {
        "t_days": 0.0,
        "p01": stats["p01"],
        "p50": stats["p50"],
        "p99": stats["p99"],
        "mean": stats["mean"],
        "std": stats["std"],
        "has_pixel_mask": 1
    }

    _, img_digest = writer.write_array(v_grp, "breast_mincrop", crop_stack, chunks=chunk_spatial, attrs=img_attrs)
    _, seg_digest = writer.write_array(v_grp, "seg_tumor", seg_mask, chunks=chunk_spatial, attrs={"has_pixel_mask": 1})

    writer.flush_and_sync()

    # Read-back verification
    verify_img = writer.verify_dataset(f"patients/{canonical_uid}/visits/v_0/breast_mincrop", img_digest, crop_stack.shape, np.float32)
    verify_seg = writer.verify_dataset(f"patients/{canonical_uid}/visits/v_0/seg_tumor", seg_digest, seg_mask.shape, np.uint8)

    if not (verify_img and verify_seg):
        ledger.update_status(canonical_uid, "FAILED", error_msg="Read-back checksum verification failed")
        return False

    # Unlink raw patient files
    raw_bytes = 0
    for _, fname in acq_files:
        fpath = os.path.join(spy1_dir, fname)
        if os.path.exists(fpath):
            raw_bytes += os.path.getsize(fpath)
            os.remove(fpath)
    if os.path.exists(mask_file):
        raw_bytes += os.path.getsize(mask_file)
        os.remove(mask_file)

    ledger.update_status(canonical_uid, "RAW_DELETED", shard_file="breast_mincrop.h5",
                         num_visits=1, raw_size_bytes=raw_bytes,
                         checksums={"breast_mincrop": img_digest, "seg_tumor": seg_digest})
    return True

def run_mincrop_pipeline(target_h5: str = "/DATA/horizon_disentangled/hdf5/breast_mincrop.h5",
                         data_dir: str = "/DATA/horizon_disentangled/data/breastdcedl_ispy2/zenodo_mincrop"):
    ledger = IngestionLedger()
    duke_dir = os.path.join(data_dir, "BreastDCEDL_duke", "duke_dce")
    duke_meta_file = os.path.join(data_dir, "BreastDCEDL_duke", "BrestDCEDL_duke_cropped_metadata.csv")
    spy1_dir = os.path.join(data_dir, "BreastDCEDL_spy1", "spt1_dce")
    mask_dir = os.path.join(data_dir, "BreastDCEDL_spy1", "spy1_mask")
    spy1_meta_file = os.path.join(data_dir, "BreastDCEDL_spy1", "BreastDCEDL_spy1_metadata.csv")

    df_duke = pd.read_csv(duke_meta_file) if os.path.exists(duke_meta_file) else pd.DataFrame()
    df_spy1 = pd.read_csv(spy1_meta_file) if os.path.exists(spy1_meta_file) else pd.DataFrame()

    with HDF5ShardWriter(target_h5) as writer:
        # Process SPY1
        if os.path.exists(spy1_dir):
            spy1_files = os.listdir(spy1_dir)
            spy1_pats = defaultdict(list)
            for f in spy1_files:
                m = re.match(r'(ISPY1_\d+)_spy1_vis1_acq(\d+)\.nii\.gz', f)
                if m:
                    spy1_pats[m.group(1)].append((int(m.group(2)), f))

            print(f"Starting SPY1 MinCrop processing ({len(spy1_pats)} patients)...")
            success_spy1 = 0
            for i, (pid, acqs) in enumerate(sorted(spy1_pats.items())):
                meta_row = df_spy1[df_spy1['pid'] == pid].iloc[0] if len(df_spy1[df_spy1['pid'] == pid]) > 0 else pd.Series()
                if process_spy1_patient(pid, acqs, meta_row, writer, ledger, spy1_dir, mask_dir):
                    success_spy1 += 1
                if (i + 1) % 50 == 0 or (i + 1) == len(spy1_pats):
                    print(f"  Processed {i + 1}/{len(spy1_pats)} SPY1 patients ({success_spy1} verified)")

        # Process Duke
        if os.path.exists(duke_dir):
            duke_files = os.listdir(duke_dir)
            duke_pats = defaultdict(list)
            for f in duke_files:
                m = re.match(r'(Breast_MRI_\d+)_duke_aqc_(\d+)\.nii\.gz', f)
                if m:
                    duke_pats[m.group(1)].append((int(m.group(2)), f))

            print(f"Starting Duke MinCrop processing ({len(duke_pats)} patients)...")
            success_duke = 0
            for i, (pid, acqs) in enumerate(sorted(duke_pats.items())):
                meta_row = df_duke[df_duke['pid'] == pid].iloc[0] if len(df_duke[df_duke['pid'] == pid]) > 0 else pd.Series()
                if process_duke_patient(pid, acqs, meta_row, writer, ledger, duke_dir):
                    success_duke += 1
                if (i + 1) % 100 == 0 or (i + 1) == len(duke_pats):
                    print(f"  Processed {i + 1}/{len(duke_pats)} Duke patients ({success_duke} verified)")

    print("BreastDCEDL MinCrop pipeline finished successfully!")

if __name__ == "__main__":
    run_mincrop_pipeline()
