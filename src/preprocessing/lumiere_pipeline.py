import os
import re
import json
import glob
import shutil
import numpy as np
import pandas as pd
import nibabel as nib
from src.preprocessing.hdf5_writer import HDF5ShardWriter, compute_xxhash64
from src.preprocessing.ledger import IngestionLedger

def get_rano_ordinal(rating_str: str) -> int:
    """Maps LUMIERE expert RANO rating string to ordinal integer."""
    if not isinstance(rating_str, str):
        return -1
    r = rating_str.strip()
    if r == "CR":
        return 0
    elif r == "PR":
        return 1
    elif r == "SD":
        return 2
    elif r == "PD":
        return 3
    elif "Pre-Op" in r:
        return -1
    elif "Post-Op" in r:
        return -2
    return -1

def process_lumiere_patient(pat_dir: str, pat_id: str,
                           df_demo: pd.DataFrame, df_expert: pd.DataFrame,
                           writer: HDF5ShardWriter, ledger: IngestionLedger) -> bool:
    """Processes a single LUMIERE patient across all longitudinal visits."""
    canonical_uid = f"LUM_{pat_id.replace('Patient-', '')}"
    ledger.register_patient(canonical_uid, "lumiere", "gbm")

    # Demographics
    p_demo = df_demo[df_demo['Patient'] == pat_id]
    age = 0.0
    sex_str = "unknown"
    sex_val = -1.0
    idh_val = -1.0
    mgmt_val = -1.0
    survival_w = np.nan
    has_event = 0

    if len(p_demo) > 0:
        row_d = p_demo.iloc[0]
        if pd.notna(row_d.get('Age at surgery (years)')):
            age = float(row_d['Age at surgery (years)'])
        if pd.notna(row_d.get('Sex')):
            sex_str = str(row_d['Sex']).strip().lower()
            sex_val = 1.0 if sex_str == 'male' else (0.0 if sex_str == 'female' else -1.0)
        if pd.notna(row_d.get('IDH (WT: wild type)')):
            idh_s = str(row_d['IDH (WT: wild type)']).strip()
            idh_val = 1.0 if 'mut' in idh_s.lower() else (0.0 if 'wt' in idh_s.lower() else -1.0)
        if pd.notna(row_d.get('MGMT qualitative')):
            mgmt_s = str(row_d['MGMT qualitative']).strip().lower()
            mgmt_val = 1.0 if 'methylated' in mgmt_s and 'not' not in mgmt_s else (0.0 if 'not methylated' in mgmt_s else -1.0)
        if pd.notna(row_d.get('Survival time (weeks)')):
            try:
                survival_w = float(row_d['Survival time (weeks)'])
                has_event = 1
            except ValueError:
                pass

    static_vec = np.array([age, sex_val, idh_val, mgmt_val, survival_w if np.isfinite(survival_w) else -1.0], dtype=np.float32)
    static_mask = np.array([1 if age > 0 else 0, 1 if sex_val >= 0 else 0, 1 if idh_val >= 0 else 0, 1 if mgmt_val >= 0 else 0, has_event], dtype=np.uint8)

    p_grp = writer.h5.require_group(f"patients/{canonical_uid}")
    writer.write_array(p_grp, "static_clinical", static_vec, attrs={
        "sex": sex_str,
        "age": age,
        "survival_weeks": survival_w,
        "os_weeks": survival_w,
        "event": has_event,
        "domain": "gbm",
        "cohort": "lumiere"
    })
    writer.write_array(p_grp, "static_mask", static_mask)

    # Interventions (surgery at week 000-2)
    intervention_dtype = np.dtype([
        ('t_days', np.float32),
        ('type_id', np.uint8), # 1=RT, 2=Surgery
        ('dose_gy', np.float32),
        ('fractions', np.uint8),
        ('drug_flags', np.uint16)
    ])
    interventions = [(0.0, 2, 0.0, 0, 0)] # Baseline surgical resection
    interv_arr = np.array(interventions, dtype=intervention_dtype)
    if "interventions" in p_grp:
        del p_grp["interventions"]
    p_grp.create_dataset("interventions", data=interv_arr)

    # Longitudinal visits
    v_dirs = sorted([d for d in os.listdir(pat_dir) if os.path.isdir(os.path.join(pat_dir, d))])
    v_grp = p_grp.require_group("visits")
    digests_to_verify = {}

    for v_idx, v_name in enumerate(v_dirs):
        full_v_dir = os.path.join(pat_dir, v_name)
        # Parse timepoint
        is_postop = 0
        if v_name == "week-000-1":
            t_days = 0.0
            is_postop = 0
        elif v_name == "week-000-2":
            t_days = 0.0
            is_postop = 1
        else:
            m = re.match(r'week-(\d+)', v_name)
            t_days = float(int(m.group(1)) * 7) if m else float(v_idx * 30.0)

        # Expert RANO rating for this visit
        p_exp = df_expert[(df_expert['Patient'] == pat_id) & (df_expert['Date'] == v_name)]
        rano_val = -1
        if len(p_exp) > 0:
            rating_col = [c for c in df_expert.columns if 'Rating' in c and 'rationale' not in c][0]
            rano_val = get_rano_ordinal(str(p_exp.iloc[0].get(rating_col, '')))

        # Sequences: CT1, T1, T2, FLAIR
        seq_names = ['CT1', 'T1', 'T2', 'FLAIR']
        loaded_seqs = {}
        ref_shape = None
        ref_affine = None

        modality_mask = 0
        for s_idx, s in enumerate(seq_names):
            s_file = os.path.join(full_v_dir, f"{s}.nii.gz")
            if os.path.exists(s_file):
                try:
                    img = nib.load(s_file)
                    img_ras = nib.as_closest_canonical(img)
                    arr = img_ras.get_fdata(dtype=np.float32, caching='unchanged')
                    img.uncache()
                    loaded_seqs[s] = arr
                    modality_mask |= (1 << s_idx)
                    if ref_shape is None:
                        ref_shape = arr.shape
                        ref_affine = img_ras.affine
                except Exception as e:
                    print(f"Warning: error reading {s_file}: {e}")

        if ref_shape is None:
            continue

        H, W, D = ref_shape
        seq_stack = np.zeros((4, H, W, D), dtype=np.float32)
        for s_idx, s in enumerate(seq_names):
            if s in loaded_seqs:
                arr = loaded_seqs[s]
                if arr.shape == ref_shape:
                    seq_stack[s_idx] = arr

        # Segmentations: DeepBraTumIA and HD-GLIO
        seg_stack = np.zeros((2, H, W, D), dtype=np.uint8)

        # DeepBraTumIA
        deep_fp = os.path.join(full_v_dir, "DeepBraTumIA-segmentation", "native", "segmentation", "ct1_seg_mask.nii.gz")
        if not os.path.exists(deep_fp):
            deep_fp = os.path.join(full_v_dir, "DeepBraTumIA-segmentation", "atlas", "segmentation", "seg_mask.nii.gz")
        if os.path.exists(deep_fp):
            try:
                m_img = nib.load(deep_fp)
                m_ras = nib.as_closest_canonical(m_img)
                m_arr = m_ras.get_fdata(dtype=np.float32, caching='unchanged')
                m_img.uncache()
                if m_arr.shape == ref_shape:
                    seg_stack[0] = m_arr.astype(np.uint8)
            except Exception:
                pass

        # HD-GLIO
        hd_fp = os.path.join(full_v_dir, "HD-GLIO-AUTO-segmentation", "native", "segmentation_CT1_origspace.nii.gz")
        if not os.path.exists(hd_fp):
            hd_fp = os.path.join(full_v_dir, "HD-GLIO-AUTO-segmentation", "registered", "segmentation.nii.gz")
        if os.path.exists(hd_fp):
            try:
                h_img = nib.load(hd_fp)
                h_ras = nib.as_closest_canonical(h_img)
                h_arr = h_ras.get_fdata(dtype=np.float32, caching='unchanged')
                h_img.uncache()
                if h_arr.shape == ref_shape:
                    seg_stack[1] = h_arr.astype(np.uint8)
            except Exception:
                pass

        # Write visit dataset
        v_sub = v_grp.require_group(f"v_{v_idx}")
        chunk_4d = (1, min(64, H), min(64, W), min(64, D))

        img_attrs = {
            "t_days": t_days,
            "modality_mask": int(modality_mask),
            "rano_ordinal": rano_val,
            "is_postop_jump": is_postop,
            "affine_ras": ref_affine if ref_affine is not None else np.eye(4)
        }

        _, img_digest = writer.write_array(v_sub, "brain_mri", seq_stack, chunks=chunk_4d, attrs=img_attrs)
        _, seg_digest = writer.write_array(v_sub, "seg_tumor", seg_stack, chunks=chunk_4d)

        # Measured volume JSON
        json_fp = os.path.join(full_v_dir, "DeepBraTumIA-segmentation", "atlas", "segmentation", "measured_volumes_in_mm3.json")
        if os.path.exists(json_fp):
            try:
                with open(json_fp) as jf:
                    vol_data = json.load(jf)
                writer.write_json_blob(v_sub, "deepbratumia_volumes", vol_data)
            except Exception:
                pass

        digests_to_verify[f"patients/{canonical_uid}/visits/v_{v_idx}/brain_mri"] = (img_digest, seq_stack.shape, np.float32)
        digests_to_verify[f"patients/{canonical_uid}/visits/v_{v_idx}/seg_tumor"] = (seg_digest, seg_stack.shape, np.uint8)

    writer.flush_and_sync()

    # Read-back verification
    all_ok = True
    for dpath, (exp_digest, exp_shape, exp_dtype) in digests_to_verify.items():
        if not writer.verify_dataset(dpath, exp_digest, exp_shape, exp_dtype):
            all_ok = False
            break

    if not all_ok:
        ledger.update_status(canonical_uid, "FAILED", error_msg="Read-back verification failed for visit datasets")
        return False

    # Unlink raw NIfTI files inside pat_dir
    raw_bytes = 0
    for root, dirs, files in os.walk(pat_dir):
        for f in files:
            if f.endswith(".nii.gz"):
                fp = os.path.join(root, f)
                raw_bytes += os.path.getsize(fp)
                os.remove(fp)

    ledger.update_status(canonical_uid, "RAW_DELETED", shard_file="gbm_lumiere.h5",
                         num_visits=len(v_dirs), raw_size_bytes=raw_bytes)
    return True

def run_lumiere_pipeline(target_h5: str = "/DATA/horizon_disentangled/hdf5/gbm_lumiere.h5",
                        data_dir: str = "/DATA/horizon_disentangled/data/lumiere"):
    ledger = IngestionLedger()
    img_dir = os.path.join(data_dir, "Imaging")
    df_demo = pd.read_csv(os.path.join(data_dir, "LUMIERE-Demographics_Pathology.csv"))
    df_expert = pd.read_csv(os.path.join(data_dir, "LUMIERE-ExpertRating-v202211.csv"))

    pat_dirs = sorted([d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))])
    print(f"Starting LUMIERE preprocessing ({len(pat_dirs)} patients)...")

    with HDF5ShardWriter(target_h5) as writer:
        success = 0
        for i, pid in enumerate(pat_dirs):
            full_pat_dir = os.path.join(img_dir, pid)
            if process_lumiere_patient(full_pat_dir, pid, df_demo, df_expert, writer, ledger):
                success += 1
            if (i + 1) % 10 == 0 or (i + 1) == len(pat_dirs):
                print(f"  Processed {i + 1}/{len(pat_dirs)} LUMIERE patients ({success} verified)")

    print(f"LUMIERE pipeline finished: {success}/{len(pat_dirs)} patients successfully processed and verified!")

if __name__ == "__main__":
    run_lumiere_pipeline()
