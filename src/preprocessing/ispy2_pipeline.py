import os
import re
import glob
import json
import hashlib
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
import pydicom
import concurrent.futures
from typing import Dict, Any, List, Optional, Tuple
from src.preprocessing.hdf5_writer import HDF5ShardWriter, compute_xxhash64
from src.preprocessing.ledger import IngestionLedger

def get_slice_z(ds: pydicom.Dataset) -> float:
    """Computes physical coordinate along slice normal."""
    if hasattr(ds, "ImagePositionPatient") and hasattr(ds, "ImageOrientationPatient"):
        iop = ds.ImageOrientationPatient
        normal = np.cross(iop[:3], iop[3:])
        return float(np.dot(ds.ImagePositionPatient, normal))
    elif hasattr(ds, "SliceLocation"):
        return float(ds.SliceLocation)
    elif hasattr(ds, "InstanceNumber"):
        return float(ds.InstanceNumber)
    return 0.0

def load_3d_series(series_dir: str) -> Tuple[Optional[np.ndarray], Dict[str, Any], Optional[pydicom.Dataset]]:
    """Loads all DICOM slices in a series directory, sorts along slice normal, applies RescaleSlope/Intercept."""
    dcm_files = [os.path.join(series_dir, f) for f in os.listdir(series_dir) if f.endswith('.dcm')]
    if not dcm_files:
        return None, {}, None

    slice_meta = []
    first_ds = None
    for fp in dcm_files:
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True)
            if first_ds is None:
                first_ds = ds
            z_pos = get_slice_z(ds)
            slice_meta.append((z_pos, fp))
        except Exception:
            continue

    if not slice_meta:
        return None, {}, None

    slice_meta.sort(key=lambda x: x[0])
    N = len(slice_meta)

    try:
        sample_ds = pydicom.dcmread(slice_meta[0][1])
        rows, cols = sample_ds.Rows, sample_ds.Columns
        slope = float(getattr(sample_ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(sample_ds, 'RescaleIntercept', 0.0))

        vol = np.empty((rows, cols, N), dtype=np.float32)
        for i, (_, fp) in enumerate(slice_meta):
            ds = pydicom.dcmread(fp)
            vol[:, :, i] = ds.pixel_array.astype(np.float32) * slope + intercept

        header_info = {
            "SeriesDescription": str(getattr(sample_ds, 'SeriesDescription', '')),
            "SeriesNumber": int(getattr(sample_ds, 'SeriesNumber', 0)),
            "Manufacturer": str(getattr(sample_ds, 'Manufacturer', '')),
            "ManufacturerModelName": str(getattr(sample_ds, 'ManufacturerModelName', '')),
            "MagneticFieldStrength": float(getattr(sample_ds, 'MagneticFieldStrength', 1.5)) if getattr(sample_ds, 'MagneticFieldStrength', None) else 1.5,
            "PixelSpacing": [float(x) for x in getattr(sample_ds, 'PixelSpacing', [1.0, 1.0])],
            "SliceThickness": float(getattr(sample_ds, 'SliceThickness', 2.0)) if getattr(sample_ds, 'SliceThickness', None) else 2.0,
            "RescaleSlope": slope,
            "RescaleIntercept": intercept,
            "num_slices": N
        }
        return vol, header_info, sample_ds
    except Exception as e:
        print(f"Error loading series {series_dir}: {e}")
        return None, {}, None

def make_json_serializable(obj: Any) -> Any:
    """Recursively converts pydicom/numpy types to pure Python primitives for JSON serialization."""
    if obj is None:
        return None
    if isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj)
    if isinstance(obj, bytes):
        return obj.decode('latin1', errors='ignore')
    if isinstance(obj, (list, tuple, set)):
        return [make_json_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    try:
        val = float(obj)
        if int(val) == val:
            return int(val)
        return val
    except Exception:
        return str(obj)

def extract_private_0117(ds: pydicom.Dataset) -> Dict[str, Any]:
    """Extracts UCSF BIRP 0x0117 private elements into a serializable dictionary."""
    priv = {}
    if ds is None:
        return priv
    for tag in ds.keys():
        if tag.group == 0x0117:
            elem = ds[tag]
            k = f"0x0117_{hex(tag.element)[2:]}"
            val = elem.value
            try:
                if isinstance(val, (int, float, str)):
                    priv[k] = val
                elif hasattr(val, '__iter__') and not isinstance(val, (str, bytes)):
                    priv[k] = [float(x) if isinstance(x, (int, float)) else str(x) for x in val]
                else:
                    priv[k] = str(val)
            except Exception:
                priv[k] = str(val)
    return make_json_serializable(priv)

def index_patient_segs(pat_id: str, collection_id: str, seg_dirs: List[str]) -> Dict[str, str]:
    """Finds all SEG files belonging to this patient and maps StudyInstanceUID -> seg_filepath."""
    seg_map = {}
    for s_root in seg_dirs:
        for c in [collection_id, '']:
            p_dir = os.path.join(s_root, c, pat_id) if c else os.path.join(s_root, pat_id)
            if os.path.exists(p_dir):
                for root, _, files in os.walk(p_dir):
                    for f in files:
                        if f.endswith('.dcm'):
                            fp = os.path.join(root, f)
                            try:
                                ds = pydicom.dcmread(fp, stop_before_pixels=True)
                                suid = getattr(ds, 'StudyInstanceUID', None)
                                if suid and getattr(ds, 'Modality', '') == 'SEG':
                                    seg_map[suid] = fp
                            except Exception:
                                continue
    return seg_map

def parse_dicom_date(date_str: str) -> Optional[datetime]:
    """Parses standard DICOM date string YYYYMMDD."""
    if not date_str or not isinstance(date_str, str) or len(date_str) < 8:
        return None
    try:
        return datetime.strptime(date_str[:8], "%Y%m%d")
    except Exception:
        return None

def process_single_patient(pat_dir: str, pat_id: str, collection_id: str,
                           df_clin: pd.DataFrame, df_nact: pd.DataFrame,
                           writer: HDF5ShardWriter, shard_idx: int,
                           ledger: IngestionLedger, seg_dirs: List[str]) -> bool:
    """Processes a single I-SPY 2 patient across all longitudinal studies/visits."""
    m = re.search(r'(\d{6})', pat_id)
    clean_id = int(m.group(1)) if m else None
    canonical_uid = f"ISP2_{clean_id}" if clean_id else f"ISP2_{pat_id}"
    shard_name = f"breast_ispy2_s0{shard_idx}.h5"

    # Check if already processed & raw deleted
    status = ledger.get_status(canonical_uid)
    if status in ("COMMITTED", "RAW_DELETED"):
        has_dcms = any(f.endswith('.dcm') for _, _, files in os.walk(pat_dir) for f in files)
        if not has_dcms:
            return True

    ledger.register_patient(canonical_uid, f"ispy2_{collection_id}", "breast")

    # Match clinical data
    row_c = None
    if clean_id is not None and len(df_clin) > 0:
        match = df_clin[df_clin['Patient_ID'] == clean_id]
        if len(match) > 0:
            row_c = match.iloc[0]

    pcr_val = int(row_c['pCR']) if row_c is not None and pd.notna(row_c.get('pCR')) else -1
    hr_val = float(row_c['HR']) if row_c is not None and pd.notna(row_c.get('HR')) else -1.0
    her2_val = float(row_c['HER2']) if row_c is not None and pd.notna(row_c.get('HER2')) else -1.0
    mp_val = float(row_c['MP']) if row_c is not None and pd.notna(row_c.get('MP')) else -1.0
    age_val = float(row_c['Age_at_Screening']) if row_c is not None and pd.notna(row_c.get('Age_at_Screening')) else 0.0
    arm_str = str(row_c['Arm']) if row_c is not None and pd.notna(row_c.get('Arm')) else "Unknown"
    race_str = str(row_c['Race']) if row_c is not None and pd.notna(row_c.get('Race')) else "Unknown"

    static_vec = np.array([age_val, float(pcr_val), hr_val, her2_val, mp_val], dtype=np.float32)
    static_mask = np.array([1 if age_val > 0 else 0, 1 if pcr_val >= 0 else 0, 1 if hr_val >= 0 else 0, 1 if her2_val >= 0 else 0, 1 if mp_val >= 0 else 0], dtype=np.uint8)

    p_grp = writer.h5.require_group(f"patients/{canonical_uid}")
    writer.write_array(p_grp, "static_clinical", static_vec, attrs={
        "pcr": pcr_val,
        "hr": hr_val,
        "her2": her2_val,
        "mp": mp_val,
        "arm": arm_str,
        "race": race_str,
        "domain": "breast",
        "cohort": f"ispy2_{collection_id}"
    })
    writer.write_array(p_grp, "static_mask", static_mask)

    # Interventions (Chemotherapy arm)
    intervention_dtype = np.dtype([
        ('t_days', np.float32),
        ('type_id', np.uint8), # 3=Chemo
        ('dose_gy', np.float32),
        ('fractions', np.uint8),
        ('drug_flags', np.uint16)
    ])
    interventions = [(0.0, 3, 0.0, 0, 1)] # T0 Chemo start
    interv_arr = np.array(interventions, dtype=intervention_dtype)
    if "interventions" in p_grp:
        del p_grp["interventions"]
    p_grp.create_dataset("interventions", data=interv_arr)

    # Fast SEG indexing for this patient
    patient_seg_map = index_patient_segs(pat_id, collection_id, seg_dirs)

    # Discover and inspect studies for chronological sorting
    raw_study_dirs = [d for d in os.listdir(pat_dir) if os.path.isdir(os.path.join(pat_dir, d))]
    study_infos = []

    for s_uid in raw_study_dirs:
        full_study_dir = os.path.join(pat_dir, s_uid)
        series_subdirs = [d for d in os.listdir(full_study_dir) if os.path.isdir(os.path.join(full_study_dir, d))]
        study_date = ""
        study_time = ""
        study_desc = ""

        for sdir in series_subdirs:
            full_sdir = os.path.join(full_study_dir, sdir)
            dcms = [f for f in os.listdir(full_sdir) if f.endswith('.dcm')]
            if dcms:
                try:
                    ds_head = pydicom.dcmread(os.path.join(full_sdir, dcms[0]), stop_before_pixels=True)
                    study_date = getattr(ds_head, 'StudyDate', '')
                    study_time = getattr(ds_head, 'StudyTime', '')
                    study_desc = getattr(ds_head, 'StudyDescription', '')
                    break
                except Exception:
                    continue

        study_infos.append({
            "study_uid": s_uid,
            "study_date": str(study_date),
            "study_time": str(study_time),
            "study_desc": str(study_desc),
            "full_path": full_study_dir
        })

    # Sort studies chronologically
    study_infos.sort(key=lambda x: (x["study_date"], x["study_time"], x["study_uid"]))

    t0_dt = parse_dicom_date(study_infos[0]["study_date"]) if study_infos else None

    v_grp = p_grp.require_group("visits")
    digests_to_verify = {}
    matched_seg_paths = set()

    for v_idx, s_info in enumerate(study_infos):
        s_uid = s_info["study_uid"]
        full_study_dir = s_info["full_path"]
        series_dirs = [d for d in os.listdir(full_study_dir) if os.path.isdir(os.path.join(full_study_dir, d))]

        # Classify series
        ser_map = {}
        for sdir in series_dirs:
            full_sdir = os.path.join(full_study_dir, sdir)
            dcms = [f for f in os.listdir(full_sdir) if f.endswith('.dcm')]
            if not dcms:
                continue
            try:
                first_dcm = pydicom.dcmread(os.path.join(full_sdir, dcms[0]), stop_before_pixels=True)
                desc = str(getattr(first_dcm, 'SeriesDescription', '')).lower()
                snum = int(getattr(first_dcm, 'SeriesNumber', 0))

                if 'ser' in desc or snum % 10000 == 1000:
                    ser_map['SER'] = full_sdir
                elif 'pe2' in desc or 'peearly' in desc or snum % 10000 == 1002:
                    ser_map['PE_early'] = full_sdir
                elif 'pe5' in desc or 'pe6' in desc or 'pelate' in desc or snum % 10000 == 1005:
                    ser_map['PE_late'] = full_sdir
                elif 't2' in desc:
                    ser_map['T2'] = full_sdir
                elif 'dwi' in desc:
                    ser_map['DWI'] = full_sdir
                elif 'adc' in desc:
                    ser_map['ADC'] = full_sdir
            except Exception:
                continue

        # Load kinetic maps
        vol_ser, meta_ser, ds_ser = load_3d_series(ser_map['SER']) if 'SER' in ser_map else (None, {}, None)
        vol_pe2, meta_pe2, ds_pe2 = load_3d_series(ser_map['PE_early']) if 'PE_early' in ser_map else (None, {}, None)
        vol_pe_late, meta_pelate, ds_pelate = load_3d_series(ser_map['PE_late']) if 'PE_late' in ser_map else (None, {}, None)

        ref_vol = vol_ser if vol_ser is not None else (vol_pe2 if vol_pe2 is not None else vol_pe_late)
        ref_meta = meta_ser if vol_ser is not None else (meta_pe2 if vol_pe2 is not None else meta_pelate)
        ref_ds = ds_ser if ds_ser is not None else (ds_pe2 if ds_pe2 is not None else ds_pelate)

        if ref_vol is None:
            continue

        Y, X, Z = ref_vol.shape
        kinetic_stack = np.zeros((3, Y, X, Z), dtype=np.float32)
        modality_mask = 0

        if vol_ser is not None and vol_ser.shape == (Y, X, Z):
            kinetic_stack[0] = vol_ser
            modality_mask |= 1
        if vol_pe2 is not None and vol_pe2.shape == (Y, X, Z):
            kinetic_stack[1] = vol_pe2
            modality_mask |= 2
        if vol_pe_late is not None and vol_pe_late.shape == (Y, X, Z):
            kinetic_stack[2] = vol_pe_late
            modality_mask |= 4

        # DICOM SEG mask for this study
        seg_mask = np.zeros((2, Y, X, Z), dtype=np.uint8)
        seg_file = patient_seg_map.get(s_uid)
        if seg_file is not None and os.path.exists(seg_file):
            matched_seg_paths.add(seg_file)
            try:
                seg_dcm = pydicom.dcmread(seg_file)
                p_arr = seg_dcm.pixel_array
                if p_arr.ndim == 3:
                    p_trans = np.transpose(p_arr, (1, 2, 0))
                    if p_trans.shape == (Y, X, Z):
                        seg_mask[0] = p_trans.astype(np.uint8)
                        seg_mask[1] = (((p_trans & 33) == 33) & ((p_trans & 64) == 0)).astype(np.uint8)
            except Exception as e:
                print(f"Warning: error decoding SEG {seg_file}: {e}")

        # Compute t_days
        curr_dt = parse_dicom_date(s_info["study_date"])
        if curr_dt and t0_dt:
            t_days = float((curr_dt - t0_dt).days)
        else:
            t_days = float(v_idx * 21.0)

        v_sub = v_grp.require_group(f"v_{v_idx}")
        chunk_spatial = (1, min(128, Y), min(128, X), min(16, Z))

        img_attrs = {
            "t_days": t_days,
            "study_uid": s_uid,
            "study_date": s_info["study_date"],
            "study_desc": s_info["study_desc"],
            "modality_mask": int(modality_mask),
            "is_bilateral": 1 if (X < 384 or 'bi-lateral' in ref_meta.get('SeriesDescription', '').lower()) else 0,
            "field_strength": float(ref_meta.get('MagneticFieldStrength', 1.5))
        }

        _, kin_digest = writer.write_array(v_sub, "ispy2_kinetic", kinetic_stack, chunks=chunk_spatial, attrs=img_attrs)
        _, seg_digest = writer.write_array(v_sub, "seg_tumor", seg_mask, chunks=chunk_spatial)

        # Tabular NACT features if available
        if clean_id is not None and len(df_nact) > 0:
            match_n = df_nact[df_nact['CLINICAL-TRIAL-SUBJECT-ID'] == clean_id]
            if len(match_n) > 0:
                row_n = match_n.iloc[0]
                ftv_col = f"VOLUME_TUM_BLU_V{v_idx+1}0"
                spher_col = f"SPHERICITY_T{v_idx}"
                ld_col = f"LD_T{v_idx}"
                bpe_col = f"BPE_5slice_mean_T{v_idx}"

                ftv = float(row_n[ftv_col]) if ftv_col in row_n and pd.notna(row_n[ftv_col]) else 0.0
                spher = float(row_n[spher_col]) if spher_col in row_n and pd.notna(row_n[spher_col]) else 0.0
                ld = float(row_n[ld_col]) if ld_col in row_n and pd.notna(row_n[ld_col]) else 0.0
                bpe = float(row_n[bpe_col]) if bpe_col in row_n and pd.notna(row_n[bpe_col]) else 0.0

                nact_vec = np.array([ftv, spher, ld, bpe], dtype=np.float32)
                writer.write_array(v_sub, "tabular_features", nact_vec, attrs={
                    "ftv": ftv, "sphericity": spher, "longest_diameter": ld, "bpe": bpe
                })

        # Save complete DICOM tags dictionary and private 0x0117 tags
        priv_tags = extract_private_0117(ref_ds)
        meta_dump = {
            "study_uid": s_uid,
            "patient_id": pat_id,
            "study_date": s_info["study_date"],
            "study_desc": s_info["study_desc"],
            "private_0117": priv_tags,
            "series": {k: ref_meta for k in ser_map.keys()}
        }
        writer.write_json_blob(v_sub, "dicom_tags_json", make_json_serializable(meta_dump))

        digests_to_verify[f"patients/{canonical_uid}/visits/v_{v_idx}/ispy2_kinetic"] = (kin_digest, kinetic_stack.shape, np.float32)
        digests_to_verify[f"patients/{canonical_uid}/visits/v_{v_idx}/seg_tumor"] = (seg_digest, seg_mask.shape, np.uint8)

    writer.flush_and_sync()

    # Read-back verification
    all_ok = True
    for dpath, (exp_digest, exp_shape, exp_dtype) in digests_to_verify.items():
        if not writer.verify_dataset(dpath, exp_digest, exp_shape, exp_dtype):
            all_ok = False
            break

    if not all_ok:
        ledger.update_status(canonical_uid, "FAILED", error_msg="Read-back verification failed")
        return False

    # Unlink raw patient imaging files (MR and matched SEG)
    raw_bytes = 0
    for root, _, files in os.walk(pat_dir):
        for f in files:
            if f.endswith('.dcm'):
                fp = os.path.join(root, f)
                raw_bytes += os.path.getsize(fp)
                try:
                    os.remove(fp)
                except Exception:
                    pass

    for s_fp in matched_seg_paths:
        if os.path.exists(s_fp):
            raw_bytes += os.path.getsize(s_fp)
            try:
                os.remove(s_fp)
            except Exception:
                pass

    ledger.update_status(canonical_uid, "RAW_DELETED", shard_file=shard_name,
                         num_visits=len(study_infos), raw_size_bytes=raw_bytes)
    return True

def process_shard_worker(shard_idx: int, pat_tasks: List[Tuple[str, str, str]],
                         hdf5_dir: str, data_dir: str) -> int:
    """Worker process dedicated to exactly one HDF5 shard."""
    if not pat_tasks:
        return 0

    ledger = IngestionLedger()
    clin_file = os.path.join(data_dir, "metadata", "ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx")
    nact_file = os.path.join(data_dir, "metadata", "Multi-feature-MRI-NACT-Data.xlsx")

    df_clin = pd.read_excel(clin_file) if os.path.exists(clin_file) else pd.DataFrame()
    df_nact = pd.read_excel(nact_file) if os.path.exists(nact_file) else pd.DataFrame()

    seg_dirs = [
        os.path.join(data_dir, "dicom_seg_longitudinal"),
        os.path.join(data_dir, "dicom_seg_50")
    ]

    shard_path = os.path.join(hdf5_dir, f"breast_ispy2_s0{shard_idx}.h5")
    writer = HDF5ShardWriter(shard_path)
    success = 0
    try:
        for i, (full_p, pid, coll) in enumerate(pat_tasks):
            if process_single_patient(full_p, pid, coll, df_clin, df_nact, writer, shard_idx, ledger, seg_dirs):
                success += 1
            if (i + 1) % 10 == 0 or (i + 1) == len(pat_tasks):
                print(f"Shard s0{shard_idx}: {i+1}/{len(pat_tasks)} patients processed ({success} verified)")
    finally:
        writer.close()
    return success

def run_ispy2_pipeline(hdf5_dir: str = "/DATA/horizon_disentangled/hdf5",
                       data_dir: str = "/DATA/horizon_disentangled/data/ispy2_longitudinal",
                       max_workers: int = 4):
    mr_dir = os.path.join(data_dir, "dicom_mr")
    collections = ['acrin_6698', 'ispy2']
    all_pat_tasks = []

    for coll in collections:
        coll_dir = os.path.join(mr_dir, coll)
        if os.path.exists(coll_dir):
            for p in sorted(os.listdir(coll_dir)):
                full_p = os.path.join(coll_dir, p)
                if os.path.isdir(full_p):
                    all_pat_tasks.append((full_p, p, coll))

    # Partition tasks into 8 buckets by deterministic hash
    shard_buckets = [[] for _ in range(8)]
    for full_p, pat_id, coll in all_pat_tasks:
        m = re.search(r'(\d{6})', pat_id)
        clean_id = int(m.group(1)) if m else None
        canonical_uid = f"ISP2_{clean_id}" if clean_id else f"ISP2_{pat_id}"
        shard_idx = int(hashlib.md5(canonical_uid.encode()).hexdigest(), 16) % 8
        shard_buckets[shard_idx].append((full_p, pat_id, coll))

    print(f"Starting parallel I-SPY 2 ingestion across 8 shards with {max_workers} workers...")
    for i in range(8):
        print(f"  Shard s0{i}: {len(shard_buckets[i])} patients")

    total_verified = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_shard_worker, s_idx, shard_buckets[s_idx], hdf5_dir, data_dir): s_idx
            for s_idx in range(8)
        }
        for fut in concurrent.futures.as_completed(futures):
            s_idx = futures[fut]
            try:
                cnt = fut.result()
                total_verified += cnt
                print(f"Shard s0{s_idx} fully finished with {cnt} verified patients.")
            except Exception as e:
                print(f"Shard s0{s_idx} failed with error: {e}")

    print(f"All I-SPY 2 shards complete! Total verified: {total_verified}/{len(all_pat_tasks)} patients.")

if __name__ == "__main__":
    run_ispy2_pipeline(max_workers=4)
