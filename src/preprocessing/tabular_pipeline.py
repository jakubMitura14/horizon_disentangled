import os
import pandas as pd
import numpy as np
import h5py
from src.preprocessing.hdf5_writer import HDF5ShardWriter
from src.preprocessing.ledger import IngestionLedger

def process_cfb_gbm(h5_writer: HDF5ShardWriter, ledger: IngestionLedger, data_dir: str):
    """Processes CFB-GBM clinical, treatment, RANO, and PyRadiomics tables."""
    print("Processing CFB-GBM tabular data...")
    meta_dir = os.path.join(data_dir, "cfb_gbm", "metadata")
    df_clin = pd.read_csv(os.path.join(meta_dir, "CFB-GBM_clinical_data_v03_20260619.tsv"), sep="\t")
    df_treat = pd.read_csv(os.path.join(meta_dir, "CFB-GBM_treatment_data_v02_20260129.tsv"), sep="\t")
    df_rano = pd.read_csv(os.path.join(meta_dir, "CFB-GBM_rano_criteria_v03_20260619.tsv"), sep="\t")
    df_mri = pd.read_csv(os.path.join(meta_dir, "CFB-GBM_mri_availability_v02_20260129.tsv"), sep="\t")
    df_ct = pd.read_csv(os.path.join(meta_dir, "CFB-GBM_ct_availability_v02_20260129.tsv"), sep="\t")
    df_treat_img = pd.read_csv(os.path.join(meta_dir, "CFB-GBM_treatment_imaging_availability_v03_20260619.tsv"), sep="\t")
    df_pyr = pd.read_csv(os.path.join(meta_dir, "CFB_GBM_features_extraction_pyradiomics_v03_20260619.tsv"), sep="\t")

    # Map RANO string to ordinal int: CR=0, PR=1, MR=2, SD=3, PD=4
    rano_map = {
        'Complete Response (CR)': 0,
        'Partial Response (PR)': 1,
        'Minor Response (MR)': 2,
        'Stable Disease (SD)': 3,
        'Progressive Disease (PD)': 4
    }

    pyr_feature_cols = [c for c in df_pyr.columns if c not in ['Patient', 'Temporality', 'Image', 'Mask', 'Label name', 'Sequence']]
    for c in pyr_feature_cols:
        df_pyr[c] = pd.to_numeric(df_pyr[c], errors='coerce').fillna(0.0)

    pat_ids = sorted(df_clin['id_patient'].unique())
    print(f"Total CFB-GBM patients: {len(pat_ids)}")

    intervention_dtype = np.dtype([
        ('t_days', np.float32),
        ('type_id', np.uint8),       # 1=Radiotherapy, 2=Surgery, 3=Chemo
        ('dose_gy', np.float32),
        ('fractions', np.uint8),
        ('drug_flags', np.uint16)
    ])

    for pid in pat_ids:
        canonical_uid = f"CFB_{pid:03d}"
        ledger.register_patient(canonical_uid, "cfb_gbm", "gbm")
        p_grp = h5_writer.h5.require_group(f"patients/{canonical_uid}")

        p_clin = df_clin[df_clin['id_patient'] == pid].iloc[0]
        age = float(p_clin['age_at_t0 (years)']) if pd.notna(p_clin['age_at_t0 (years)']) else 0.0
        gender = str(p_clin['gender'])
        sex_val = 1.0 if gender.upper() == 'M' else 0.0
        who_ps = float(p_clin['who_performance_status']) if pd.notna(p_clin['who_performance_status']) else -1.0
        survival_w = float(p_clin['survival (weeks)']) if pd.notna(p_clin['survival (weeks)']) else np.nan
        h_cm = float(p_clin['height (cm)']) if pd.notna(p_clin['height (cm)']) else 0.0
        w_kg = float(p_clin['weight (kg)']) if pd.notna(p_clin['weight (kg)']) else 0.0
        bmi = w_kg / ((h_cm / 100.0) ** 2) if h_cm > 0 else 0.0

        # static_clinical vector: [age, sex, who_ps, height, weight, bmi]
        static_vec = np.array([age, sex_val, who_ps, h_cm, w_kg, bmi], dtype=np.float32)
        static_mask = np.array([1, 1, 1 if who_ps >= 0 else 0, 1 if h_cm > 0 else 0, 1 if w_kg > 0 else 0, 1 if bmi > 0 else 0], dtype=np.uint8)

        h5_writer.write_array(p_grp, "static_clinical", static_vec, attrs={
            "sex": gender,
            "who_performance_status": who_ps,
            "survival_weeks": survival_w,
            "os_weeks": survival_w,
            "event": 1 if np.isfinite(survival_w) else 0,
            "domain": "gbm",
            "cohort": "cfb_gbm"
        })
        h5_writer.write_array(p_grp, "static_mask", static_mask)

        # Interventions (Radiotherapy)
        p_treat = df_treat[df_treat['id_patient'] == pid]
        interventions = []
        if len(p_treat) > 0:
            row_t = p_treat.iloc[0]
            delay_w = float(row_t['delay_t0_to_radiotherapy (weeks)']) if pd.notna(row_t['delay_t0_to_radiotherapy (weeks)']) else 0.0
            dose = float(row_t['dose (Gy)']) if pd.notna(row_t['dose (Gy)']) else 0.0
            fractions = int(row_t['fractions_number']) if pd.notna(row_t['fractions_number']) else 0
            interventions.append((delay_w * 7.0, 1, dose, fractions, 0))

        interv_arr = np.array(interventions, dtype=intervention_dtype)
        if "interventions" in p_grp:
            del p_grp["interventions"]
        p_grp.create_dataset("interventions", data=interv_arr)

        # Visits (t0, t1, t2)
        v_grp = p_grp.require_group("visits")
        p_mri = df_mri[df_mri['id_patient'] == pid]
        p_rano = df_rano[df_rano['id_patient'] == pid]

        # Determine available timepoints
        temps = sorted(p_mri['temporality'].unique()) if len(p_mri) > 0 else ['t0']
        for v_idx, temp in enumerate(temps):
            v_sub = v_grp.require_group(f"v_{v_idx}")
            mri_row = p_mri[p_mri['temporality'] == temp]
            time_diff = float(mri_row['time_diff_t0 (weeks)'].iloc[0]) * 7.0 if len(mri_row) > 0 and pd.notna(mri_row['time_diff_t0 (weeks)'].iloc[0]) else float(v_idx * 30.0)
            mri_mach = str(mri_row['mri_machine'].iloc[0]) if len(mri_row) > 0 and pd.notna(mri_row['mri_machine'].iloc[0]) else "Unknown"
            field_str = 3.0 if "3T" in mri_mach or "VIDA" in mri_mach else 1.5

            # RANO score for visit
            rano_val = -1
            if len(p_rano) > 0:
                r_row = p_rano.iloc[0]
                if temp == 't1' and pd.notna(r_row['rano_t0_to_t1']):
                    rano_val = rano_map.get(str(r_row['rano_t0_to_t1']), -1)
                elif temp == 't2' and pd.notna(r_row['rano_t0_to_t2']):
                    rano_val = rano_map.get(str(r_row['rano_t0_to_t2']), -1)

            # PyRadiomics features for this patient & temporality
            p_pyr_v = df_pyr[(df_pyr['Patient'] == pid) & (df_pyr['Temporality'] == temp)]
            if len(p_pyr_v) > 0:
                pyr_mat = p_pyr_v[pyr_feature_cols].values.astype(np.float32)
                pyr_flat = pyr_mat.mean(axis=0) # average across sequences if multiple, or flattened
            else:
                pyr_flat = np.zeros(len(pyr_feature_cols), dtype=np.float32)

            h5_writer.write_array(v_sub, "tabular_features", pyr_flat, attrs={
                "t_days": time_diff,
                "temporality": temp,
                "rano_ordinal": rano_val,
                "mri_machine": mri_mach,
                "field_strength": field_str
            })

        h5_writer.flush_and_sync()
        ledger.update_status(canonical_uid, "COMMITTED", shard_file="gbm_tabular_cohorts.h5", num_visits=len(temps))

    print(f"CFB-GBM processed successfully: {len(pat_ids)} patients.")

def process_mu_glioma(h5_writer: HDF5ShardWriter, ledger: IngestionLedger, data_dir: str):
    """Processes MU-Glioma-Post clinical and FeTS segmentation volume tables."""
    print("Processing MU-Glioma-Post tabular data...")
    meta_dir = os.path.join(data_dir, "mu_glioma_post")
    df_vol = pd.read_excel(os.path.join(meta_dir, "MU-Glioma-Post_Segmentation_Volumes.xlsx"))

    # Parse clinical file descriptor
    pats = sorted(df_vol['Patient ID'].unique())
    print(f"Total MU-Glioma-Post patients with volume measurements: {len(pats)}")

    for pid in pats:
        canonical_uid = f"MUG_{pid.replace('PatientID_', '')}"
        ledger.register_patient(canonical_uid, "mu_glioma_post", "gbm")
        p_grp = h5_writer.h5.require_group(f"patients/{canonical_uid}")

        p_vols = df_vol[df_vol['Patient ID'] == pid]

        # static_clinical placeholder
        static_vec = np.array([0.0], dtype=np.float32)
        h5_writer.write_array(p_grp, "static_clinical", static_vec, attrs={
            "domain": "gbm",
            "cohort": "mu_glioma_post"
        })

        v_grp = p_grp.require_group("visits")
        # Store longitudinal volume entries
        for v_idx, (_, row) in enumerate(p_vols.iterrows()):
            v_sub = v_grp.require_group(f"v_{v_idx}")
            vol_mm3 = float(row['Volume (mm^3)']) if pd.notna(row['Volume (mm^3)']) else 0.0
            mean_t1c = float(row['Image mean (brain_t1c)']) if pd.notna(row['Image mean (brain_t1c)']) else 0.0
            std_t1c = float(row['Image stdev (brain_t1c)']) if pd.notna(row['Image stdev (brain_t1c)']) else 0.0
            mean_t1n = float(row['Image mean (brain_t1n)']) if pd.notna(row['Image mean (brain_t1n)']) else 0.0
            std_t1n = float(row['Image stdev (brain_t1n)']) if pd.notna(row['Image stdev (brain_t1n)']) else 0.0
            mean_t2f = float(row['Image mean (brain_t2f)']) if pd.notna(row['Image mean (brain_t2f)']) else 0.0
            std_t2f = float(row['Image stdev (brain_t2f)']) if pd.notna(row['Image stdev (brain_t2f)']) else 0.0
            mean_t2w = float(row['Image mean (brain_t2w)']) if pd.notna(row['Image mean (brain_t2w)']) else 0.0
            std_t2w = float(row['Image stdev (brain_t2w)']) if pd.notna(row['Image stdev (brain_t2w)']) else 0.0

            feats = np.array([vol_mm3, mean_t1c, std_t1c, mean_t1n, std_t1n, mean_t2f, std_t2f, mean_t2w, std_t2w], dtype=np.float32)
            h5_writer.write_array(v_sub, "tabular_features", feats, attrs={
                "t_days": float(v_idx * 90.0),
                "volume_mm3": vol_mm3,
                "label_id": int(row['Label Id']) if pd.notna(row['Label Id']) else 1
            })

        h5_writer.flush_and_sync()
        ledger.update_status(canonical_uid, "COMMITTED", shard_file="gbm_tabular_cohorts.h5", num_visits=len(p_vols))

    print(f"MU-Glioma-Post processed successfully: {len(pats)} patients.")

def process_burdenko_gbm(h5_writer: HDF5ShardWriter, ledger: IngestionLedger, data_dir: str):
    """Processes Burdenko-GBM clinical CSV and follow-up response trajectory table."""
    print("Processing Burdenko-GBM tabular data...")
    meta_dir = os.path.join(data_dir, "burdenko_gbm")
    df_clin = pd.read_csv(os.path.join(meta_dir, "Burdenko-GBM-clinical-data_20230410.csv"))

    # Map response strings to integer code:
    # response=0, stable=1, pseudo progression=2, progression=3, distant progression=4
    resp_map = {
        'response': 0,
        'stable': 1,
        'pseudo progression': 2,
        'progression': 3,
        'distant progression': 4,
        'response, distant progression': 4,
        'stable, distant progression': 4,
        'response, pseudo progression': 2
    }

    intervention_dtype = np.dtype([
        ('t_days', np.float32),
        ('type_id', np.uint8),       # 1=Radiotherapy, 2=Surgery
        ('dose_gy', np.float32),
        ('fractions', np.uint8),
        ('drug_flags', np.uint16)
    ])

    for _, row in df_clin.iterrows():
        raw_pid = str(row['AnonymPatientID']).strip()
        canonical_uid = f"BUR_{raw_pid.replace('Burdenko-GBM-', '')}"
        ledger.register_patient(canonical_uid, "burdenko_gbm", "gbm")
        p_grp = h5_writer.h5.require_group(f"patients/{canonical_uid}")

        age = float(row['AgeAtStudyDate']) if pd.notna(row['AgeAtStudyDate']) else 0.0
        sex = str(row['Sex']).strip().upper()
        sex_val = 1.0 if sex == 'M' else 0.0
        idh_str = str(row['IDH1/2']).strip().lower()
        idh_val = 1.0 if 'pos' in idh_str else (0.0 if 'neg' in idh_str else -1.0)
        mgmt_str = str(row['MGMT']).strip().lower()
        mgmt_val = 1.0 if 'pos' in mgmt_str else (0.0 if 'neg' in mgmt_str else -1.0)

        death_str = str(row['AnonymDeath']).strip()
        has_death = len(death_str) > 0 and death_str != 'nan' and death_str != ' '

        static_vec = np.array([age, sex_val, idh_val, mgmt_val], dtype=np.float32)
        static_mask = np.array([1 if age > 0 else 0, 1, 1 if idh_val >= 0 else 0, 1 if mgmt_val >= 0 else 0], dtype=np.uint8)

        h5_writer.write_array(p_grp, "static_clinical", static_vec, attrs={
            "sex": sex,
            "idh": idh_str,
            "mgmt": mgmt_str,
            "event": 1 if has_death else 0,
            "domain": "gbm",
            "cohort": "burdenko_gbm"
        })
        h5_writer.write_array(p_grp, "static_mask", static_mask)

        # Interventions: Surgery and RT
        interventions = []
        interventions.append((0.0, 2, 0.0, 0, 0)) # Surgery at t=0
        interventions.append((30.0, 1, 60.0, 30, 0)) # RT standard Stupp
        interv_arr = np.array(interventions, dtype=intervention_dtype)
        if "interventions" in p_grp:
            del p_grp["interventions"]
        p_grp.create_dataset("interventions", data=interv_arr)

        # Longitudinal follow-ups
        v_grp = p_grp.require_group("visits")
        v_count = 0
        for fu_idx in range(1, 11):
            col_resp = f"Response_{fu_idx}{'st' if fu_idx==1 else ('nd' if fu_idx==2 else ('rd' if fu_idx==3 else 'th'))}_fup"
            if col_resp in row and pd.notna(row[col_resp]):
                resp_text = str(row[col_resp]).strip().lower()
                resp_code = resp_map.get(resp_text, 1)
                t_days = float(fu_idx * 60.0) # approx 2-month intervals
                v_sub = v_grp.require_group(f"v_{v_count}")
                feats = np.array([float(resp_code)], dtype=np.float32)
                h5_writer.write_array(v_sub, "tabular_features", feats, attrs={
                    "t_days": t_days,
                    "response_code": resp_code,
                    "response_text": resp_text
                })
                v_count += 1

        h5_writer.flush_and_sync()
        ledger.update_status(canonical_uid, "COMMITTED", shard_file="gbm_tabular_cohorts.h5", num_visits=max(1, v_count))

    print(f"Burdenko-GBM processed successfully: {len(df_clin)} patients.")

def run_all_tabular(target_h5: str = "/DATA/horizon_disentangled/hdf5/gbm_tabular_cohorts.h5",
                    data_dir: str = "/DATA/horizon_disentangled/data"):
    ledger = IngestionLedger()
    with HDF5ShardWriter(target_h5) as writer:
        process_cfb_gbm(writer, ledger, data_dir)
        process_mu_glioma(writer, ledger, data_dir)
        process_burdenko_gbm(writer, ledger, data_dir)
    print("All tabular cohorts successfully written and synchronized to HDF5!")

if __name__ == "__main__":
    run_all_tabular()
