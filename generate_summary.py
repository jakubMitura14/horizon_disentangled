import os
import pandas as pd
import glob

# Paths
DATA_DIR = "./data"
CLIND_PATH = "/media/jm/hddData/projects_new/horizon_disentangled/src/data/clind.xlsx"
OUTPUT_CSV = "dataset_summary.csv"

# --- 1. SCAN NIFTI DATASET ---

patient_data = []

modality_map = {
    "Fixed_CT_Volume": "CT",
    "T1_TSE_TRA_Volume_Original": "T1_TRA",
    "T2_TSE_TRA": "T2_TRA", 
    "T2W_TSE_cor": "T2_COR",
    "T2W_TSE_sag": "T2_SAG",
    "ddADC_Volume_Original": "ADC",
    "DWI_3B_Volume_Original": "DWI",
    "SUV_PET_Image": "PET"
}

print("Scanning data directory...")
patient_folders = glob.glob(os.path.join(DATA_DIR, "Pat*"))

for folder in patient_folders:
    pat_id = os.path.basename(folder)
    
    # Initialize record
    record = {
        "Patient_ID": pat_id,
        "Path_Study": os.path.abspath(folder),
        "CT_Available": False,
        "T1_Available": False,
        "T2_Available": False,
        "ADC_Available": False,
        "DWI_Available": False,
        "PET_Available": False,
        "CT_Path": None,
        "T1_Path": None,
        "T2_Paths": [],
        "ADC_Path": None,
        "DWI_Path": None,
        "PET_Path": None
    }
    
    # List files
    files = glob.glob(os.path.join(folder, "*.nii.gz"))
    
    for f in files:
        fname = os.path.basename(f)
        fpath = os.path.abspath(f)
        
        if "Fixed_CT_Volume" in fname:
            record["CT_Available"] = True
            record["CT_Path"] = fpath
        elif "T1_TSE_TRA" in fname:
            record["T1_Available"] = True
            record["T1_Path"] = fpath
        elif "T2" in fname:
            record["T2_Available"] = True
            record["T2_Paths"].append(fpath)
        elif "ddADC" in fname or "ADC" in fname:
            record["ADC_Available"] = True
            record["ADC_Path"] = fpath
        elif "DWI" in fname:
            record["DWI_Available"] = True
            record["DWI_Path"] = fpath
        elif "SUV_PET" in fname or "PET" in fname:
            record["PET_Available"] = True
            record["PET_Path"] = fpath

    # Flatten list paths for CSV readability
    record["T2_Paths"] = "; ".join(record["T2_Paths"])
    
    # User Request: Verify SUV image is available each time
    record["SUV_Available"] = False
    if record["PET_Available"] and "SUV" in record["PET_Path"]:
        record["SUV_Available"] = True
        
    patient_data.append(record)

df_imaging = pd.DataFrame(patient_data)
print(f"Found {len(df_imaging)} patients in imaging dataset.")

# Verify SUV Availability
suv_available_count = df_imaging["SUV_Available"].sum()
print(f"Status Verify: SUV Image available for {suv_available_count}/{len(df_imaging)} patients.")
if suv_available_count < len(df_imaging):
    print("WARNING: Some patients are missing SUV images!")
    missing_suv = df_imaging[~df_imaging["SUV_Available"]]["Patient_ID"].tolist()
    print(f"Patients missing SUV: {missing_suv}")
else:
    print("SUCCESS: SUV Image available for ALL patients.")

# --- 2. LOAD & CLEAN CLINICAL DATA ---

print("Loading clinical data (Advanced Mode)...")
xl = pd.ExcelFile(CLIND_PATH)

# Target 'bimodal' sheet
target_sheet = "bimodal"
print(f"Targeting sheet: '{target_sheet}'")

try:
    # 2.1 FIND HEADER & ID COLUMN
    header_row = 0
    df_preview = pd.read_excel(CLIND_PATH, sheet_name=target_sheet, header=None, nrows=10)
    found_header = False
    for idx, row in df_preview.iterrows():
        row_str = " ".join([str(val).lower() for val in row.values])
        if "epoch" in row_str and "immun" in row_str:
            header_row = idx
            found_header = True
            break
            
    if not found_header:
        print("WARNING: Could not find 'epoch r immunchemotherapie' in first 10 rows. Defaulting to row 0.")
        
    df_clinical_raw = pd.read_excel(CLIND_PATH, sheet_name=target_sheet, header=header_row)
    
    # Identify Patient ID column
    pat_id_col = None
    for col in df_clinical_raw.columns:
        c_str = str(col).lower()
        if "epoch" in c_str and "immun" in c_str:
            pat_id_col = col
            break
    
    if not pat_id_col:
        pat_id_col = df_clinical_raw.columns[0]
        print(f"WARNING: ID column not found. Using first column: '{pat_id_col}'")
    else:
        print(f"Using Patient ID column: '{pat_id_col}'")

    # 2.2 NORMALIZE PATIENT ID
    def normalize_pat_id(val):
        if pd.isna(val): return "UNKNOWN"
        s = str(val).strip()
        s_clean = s.replace("Pat", "").replace("pat", "").strip()
        s_num = ''.join(filter(str.isdigit, s_clean))
        if s_num: return f"Pat{s_num}"
        return f"Pat{s}"

    df_clinical_raw["Merge_ID"] = df_clinical_raw[pat_id_col].apply(normalize_pat_id)

    # 2.3 SELECT & RENAME COLUMNS (Notebook Logic)
    # Mapping based on User's Notebook and bimodal dump inspection
    # 'bimodal' column -> 'Notebook' name
    selected_columns = {
        'Age': 'Age',
        'Tumor Marker': 'PSA',
        'Gleason': 'Gl_pet',
        'AHT': 'AHT',
        'LK-Mx': 'LK_Mx',
        'Retroper': 'Retroper',
        'Gabel': 'Gabel',
        'Ext': 'Ext',
        'Int': 'Int',
        'Comm': 'Comm',
        'Rectum': 'Rectum',
        # 'Seminal vesicles' : 'Samenblase', # Dump showed 'Seminal vesicles' AND 'Samenblase', let's prioritize
        'Seminal vesicles': 'Samenblase', 
        'SUVmax Prostata': 'SUVmax',
        'MTV Prostata': 'MTV',
        'TLG-Last Prostata': 'TLG',
        'Tumor Volume': 'Tumor_vol',
        'OS-Mx': 'OS_Mx',
        'Infiltration': 'Infiltration',
        'GRADING': 'GRADING',
        'KLIN_P_T': 'KLIN_P_T',
        'KLIN_T': 'KLIN_T',
        'KLIN_P_N': 'KLIN_P_N',
        'KLIN_N': 'KLIN_N',
        'KLIN_P_M': 'KLIN_P_M',
        'KLIN_MET': 'KLIN_MET',
        'KLIN_STADIUM': 'KLIN_STADIUM',
        'PSA_PRIMAER': 'PSA_PRIMAER',
        'POSTTH_PSA': 'POSTTH_PSA',
        'GLEASON_SCORE': 'gl_b',
        'ANZAHL_OPERATIONEN': 'number_operations',
        'ANZAHL_INNERE': 'number_internal_treatments',
        'ANZAHL_BESTRAHLUNGEN': 'number_radiations',
        'ZEITPUNKT_TUMORFREIHEIT': 'tumor_freedom',
        'DATUM_ERSTES_REZIDIV': 'first_recurrence',
        'ERSTES_LOK_REZIDIVDATUM': 'first_local_resurrence',
        'ERSTES_LOK_REZIDIV_SITZ': 'site_first_recurrence',
        'DATUM_ERSTE_PROGRESSION': 'date_first_progression',
        'LETZTER_STATUS_DATUM': 'date_last_status',
        'LETZTER_STATUS_GESAMT': 'overall_last_status',
        'LETZTER_STATUS_TUMOR': 'last_status_tumour',
        'LETZTER_STATUS_LYMPHKNOTEN': 'last_status_lymph_nodes',
        'LETZTER_STATUS_METASTASEN': 'last_status_metastases',
        'Date of PET/CT': 'date_pet_ct',
        'Date of birth': 'date_of_birth'
    }

    # Handle 'Samenblase' fallback if 'Seminal vesicles' missing
    if 'Seminal vesicles' not in df_clinical_raw.columns and 'Samenblase' in df_clinical_raw.columns:
        selected_columns['Samenblase'] = 'Samenblase'
    
    # Extract existing columns
    cols_to_use = ["Merge_ID"]
    for old_col, new_col in selected_columns.items():
        if old_col in df_clinical_raw.columns:
            cols_to_use.append(old_col)
        else:
            # Create empty placeholder if critical column missing
            # print(f"Warning: {old_col} missing, creating empty.")
            df_clinical_raw[old_col] = pd.NA
            cols_to_use.append(old_col)
            
    df_selected = df_clinical_raw.loc[:, cols_to_use].copy()
    
    # Rename columns using the mapping
    # Note: Rename only works if key exists.
    # We invert the dict for rename: old -> new
    df_selected.rename(columns=selected_columns, inplace=True)

    # 2.4 DATA CLEANING & TYPE CONVERSION
    import numpy as np
    
    # Numeric conversions (handle commas)
    numeric_cols = ['PSA', 'SUVmax', 'MTV', 'TLG', 'Tumor_vol', 'PSA_PRIMAER', 'POSTTH_PSA']
    for col in numeric_cols:
        if col in df_selected.columns:
            df_selected[col] = df_selected[col].astype(str).str.replace(',', '.', regex=False)
            df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce')

    # Boolean/Integer conversions
    bool_like_cols = ['AHT', 'LK_Mx', 'Samenblase', 'Rectum', 'Int', 'Ext', 'Comm', 'Gabel', 'Retroper', 'Infiltration', 'OS_Mx']
    for col in bool_like_cols:
        if col in df_selected.columns:
             df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce').fillna(0).astype(int)

    # Date conversions
    date_cols = ['tumor_freedom', 'first_recurrence', 'first_local_resurrence', 'date_first_progression', 'date_last_status', 'date_pet_ct', 'date_of_birth']
    for col in date_cols:
        if col in df_selected.columns:
            df_selected[col] = pd.to_datetime(df_selected[col], format='%d.%m.%Y', errors='coerce')
            # Fallback for Mixed formats
            mask = df_selected[col].isna()
            if mask.any(): # try standard parser
                 df_selected.loc[mask, col] = pd.to_datetime(df_selected.loc[mask, col], errors='coerce')

    # Gleason Cleaning (from Notebook)
    # '7 bzw 8' -> '8', '7a'->7, '7b'->7
    # Note: 'Gl_pet' is from 'Gleason' column
    df_selected['Gl_pet_orig'] = df_selected['Gl_pet']
    df_selected['Gl_pet'] = df_selected['Gl_pet'].replace({'7 bzw 8': '8', '7a': '7', '7b': '7'})
    
    def clean_gleason(val):
        if pd.isna(val): return np.nan
        s = str(val).replace(',', '.')
        try:
            return float(s)
        except:
            return np.nan
            
    df_selected['Gl_pet'] = df_selected['Gl_pet'].apply(clean_gleason)

    # 2.5 FIX KLIN_T (Notebook Logic)
    # Infers T stage from PET findings if missing
    def fix_klin_t(row):
        val = row.get('KLIN_T', pd.NA)
        if pd.isna(val) or str(val).strip() in ['0', '', 'nan']:
            # Logic from notebook
            if row['Retroper'] == 1 or row['Rectum'] == 1:
                return 'T3b' # Treat rectum/retroper as advanced
            elif row['Samenblase'] == 1:
                return 'T4' # Wait, notebook said Samenblase -> T4? "Samenblase => T4" comment
                # Wait, standard is T3b for SVI. Notebook code says:
                # "2) Samenblase => T4" - CHECK THIS. 
                # Notebook logic: "if value of ... Samenblase is 1 ... then KLIN_T should be T4"
                # OK, following Notebook strictly.
            elif str(row.get('KLIN_STADIUM', '')).upper() == 'I':
                return 'T1'
            elif str(row.get('KLIN_STADIUM', '')).upper() in ['III', 'IV']:
                return 'T3b'
            elif str(row.get('KLIN_STADIUM', '')).upper() == 'II':
                return 'T1'
            return val
        return val
        
    df_selected['KLIN_T'] = df_selected.apply(fix_klin_t, axis=1)

    # 2.6 CAPRA-S CALCULATION
    def map_psa_to_capra(psa):
        if pd.isna(psa): return 0
        if psa <= 6: return 0
        elif psa <= 10: return 1
        elif psa <= 20: return 2
        else: return 3
    
    def map_svi_to_capra(svi): return 2 if svi == 1 else 0
    def map_ece_to_capra(ece): return 1 if ece == 1 else 0
    def map_lni_to_capra(lni): return 1 if lni == 1 else 0
    
    def map_gleason_to_capra(g_val):
        # Uses original string or cleaned value?
        # Notebook uses 'Gl_pet_orig' for granular logic but fallback to int
        # Implementing simpler version based on int value as 'orig' logic was mixed
        if pd.isna(g_val): return 0
        val = int(g_val)
        if val <= 6: return 0
        elif val == 7: return 1 # 3+4 vs 4+3 not distiguishable from single int
        elif val >= 8: return 3
        return 0

    df_selected['capra_s_psa'] = df_selected['PSA'].apply(map_psa_to_capra)
    df_selected['capra_s_svi'] = df_selected['Samenblase'].apply(map_svi_to_capra)
    df_selected['capra_s_gleason'] = df_selected['Gl_pet'].apply(map_gleason_to_capra)
    df_selected['capra_s_ece'] = df_selected['Infiltration'].apply(map_ece_to_capra)
    df_selected['capra_s_lni'] = df_selected['LK_Mx'].apply(map_lni_to_capra)
    df_selected['capra_s_sm'] = 1 # Assuming mean (0+2)/2 as per notebook

    df_selected['capra_s_score'] = (
        df_selected['capra_s_psa'] + df_selected['capra_s_sm'] +
        df_selected['capra_s_svi'] + df_selected['capra_s_gleason'] +
        df_selected['capra_s_ece'] + df_selected['capra_s_lni']
    )

    # 2.7 J-CAPRA CALCULATION
    def map_jcapra_psa(psa):
        if pd.isna(psa): return 0
        if psa <= 20: return 0
        elif psa <= 100: return 1
        elif psa <= 500: return 2
        else: return 3

    def map_jcapra_gleason(g_val):
        if pd.isna(g_val): return 0
        if g_val <= 6: return 0
        elif g_val == 7: return 1
        elif g_val >= 8: return 2
        return 0

    def map_jcapra_t(t_str):
        if pd.isna(t_str): return 0
        t_low = str(t_str).lower()
        if '4' in t_low: return 3
        elif '3b' in t_low: return 2
        elif '1' in t_low or '2a' in t_low: return 0
        return 1

    def map_jcapra_n(n_val):
        # Notebook: "if '0' return 0, else 1 if int(n)==1"
        # KLIN_N might be string '0', '1', 'N0', 'N1'
        s = str(n_val).strip()
        if s == '1' or 'N1' in s: return 1
        return 0
        
    def map_jcapra_m(m_val):
        s = str(m_val).strip()
        if s == '1' or 'M1' in s: return 3 # Wait, M1 -> 3 points?
        # Notebook: "return 3" if not 0/NaN
        return 0

    df_selected['jcapra_psa'] = df_selected['PSA'].apply(map_jcapra_psa)
    df_selected['jcapra_gleason'] = df_selected['Gl_pet'].apply(map_jcapra_gleason)
    df_selected['jcapra_t'] = df_selected['KLIN_T'].apply(map_jcapra_t)
    df_selected['jcapra_n'] = df_selected['KLIN_N'].apply(map_jcapra_n)
    df_selected['jcapra_m'] = df_selected['KLIN_MET'].apply(map_jcapra_m)

    # Note: "add 100 to differentiate" logic from notebook
    # Notebook: sum + 101 ?
    # "df_filtered['j_capra_score'] = ... + 101"
    df_selected['j_capra_score'] = (
        df_selected['jcapra_psa'] + df_selected['jcapra_gleason'] +
        df_selected['jcapra_t'] + df_selected['jcapra_n'] +
        df_selected['jcapra_m'] + 101
    )

    # Select Score based on AHT
    df_selected['capra_score'] = np.where(
        df_selected['AHT'] == 1,
        df_selected['j_capra_score'],
        df_selected['capra_s_score']
    )

    # 2.8 OUTCOME & DAYS DIFF
    # "days_diff" = date_last_status - date_pet_ct
    df_selected['days_diff'] = (df_selected['date_last_status'] - df_selected['date_pet_ct']).dt.days
    
    # Outcome Mapping
    status_mapping = {
        "Keine Änderung": "No_Change",
        "Progression": "Progression",
        "Teilremission": "Partial_Remission",
        "Vollremission": "Full_Remission",
        "klinische Besserung": "Clinical_Improvement",
        "unbekannt": "Unknown"
    }
    df_selected['overall_last_status'] = df_selected['overall_last_status'].map(status_mapping).fillna("Unknown")
    
    mapping_binary = {
        "No_Change": 0, "Progression": 1, "Partial_Remission": 0,
        "Full_Remission": 0, "Clinical_Improvement": 0, "Unknown": 0
    }
    df_selected['is_progression'] = df_selected['overall_last_status'].map(mapping_binary)

    # 2.9 DISCRETIZATION (Binning)
    # Tumor_vol: > 4
    df_selected['Tumor_vol_binned'] = (df_selected['Tumor_vol'] > 4).astype(int)
    # SUVmax: > 12 (Notebook threshold)
    df_selected['SUVmax_binned'] = (df_selected['SUVmax'] > 12).astype(int)

    # 2.10 FINAL MERGE SPECS
    # Columns to include in final output
    final_clinical_cols = [
        'Merge_ID', 'Age', 'PSA', 'Gl_pet', 'AHT', 
        'Samenblase', 'Retroper', 'Rectum', 'Gabel', 
        'SUVmax', 'SUVmax_binned', 'MTV', 'TLG', 'Tumor_vol', 'Tumor_vol_binned',
        'OS_Mx', 'Infiltration', 'KLIN_T', 
        'weight', 'capra_s_score', 'j_capra_score', 'capra_score', 
        'days_diff', 'overall_last_status', 'is_progression', 'tumor_freedom'
    ]
    # Filter only existing
    final_clinical_cols = [c for c in final_clinical_cols if c in df_selected.columns]
    
    df_clinical_subset = df_selected[final_clinical_cols].copy()
    
    # Merge
    print("Merging datasets...")
    df_final = pd.merge(df_imaging, df_clinical_subset, left_on="Patient_ID", right_on="Merge_ID", how="left")
    
    if "Merge_ID" in df_final.columns:
        df_final.drop(columns=["Merge_ID"], inplace=True)
    
    # Save
    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"Summary saved to {OUTPUT_CSV}")
    print("Columns:", df_final.columns.tolist())
    
    # Validation
    matched = df_final["capra_score"].notna().sum()
    print(f"Clinical data matched (CAPRA calculated) for {matched}/{len(df_final)} patients.")

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"ERROR: Failed to process clinical data: {e}")
