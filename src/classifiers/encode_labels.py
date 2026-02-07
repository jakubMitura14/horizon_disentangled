"""
Label Encoding for Ordinal Classification
Converts raw clinical values to ordinal integer labels.
"""
import pandas as pd
import numpy as np

# T-Stage Mapping (9 classes: 0-8)
T_STAGE_MAP = {
    # T1 variants
    '1a': 0, 't1a': 0, 'T1a': 0,
    '1b': 1, 't1b': 1, 'T1b': 1,
    '1c': 2, 't1c': 2, 'T1c': 2,
    # T2 variants
    '2a': 3, 't2a': 3, 'T2a': 3,
    '2b': 4, 't2b': 4, 'T2b': 4,
    '2c': 5, 't2c': 5, 'T2c': 5,
    # T3 variants
    '3a': 6, 't3a': 6, 'T3a': 6,
    '3b': 7, 't3b': 7, 'T3b': 7,
    # T4
    '4': 8, 't4': 8, 'T4': 8,
}

# Gleason Mapping (5 classes: 0-4)
GLEASON_MAP = {
    6: 0, 6.0: 0,
    7: 1, 7.0: 1,
    8: 2, 8.0: 2,
    9: 3, 9.0: 3,
    10: 4, 10.0: 4,
}

NUM_T_CLASSES = 9
NUM_GLEASON_CLASSES = 5


def encode_t_stage(value):
    """Convert T-stage string to ordinal label (0-8)."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    # Direct lookup
    if s in T_STAGE_MAP:
        return T_STAGE_MAP[s]
    # Try lowercase
    if s.lower() in T_STAGE_MAP:
        return T_STAGE_MAP[s.lower()]
    return np.nan


def encode_gleason(value):
    """Convert Gleason score to ordinal label (0-4)."""
    if pd.isna(value):
        return np.nan
    try:
        g = float(value)
        if g in GLEASON_MAP:
            return GLEASON_MAP[g]
        # Handle edge cases (floor to nearest)
        if g < 6:
            return 0
        if g > 10:
            return 4
        return GLEASON_MAP.get(int(g), np.nan)
    except (ValueError, TypeError):
        return np.nan


def encode_labels(df):
    """
    Add encoded label columns to the dataframe.
    
    Args:
        df: DataFrame with 'KLIN_T', 'Gl_pet', 'PSA' columns.
        
    Returns:
        DataFrame with new columns: 'T_label', 'Gleason_label', 'PSA_target'.
    """
    df = df.copy()
    
    # Encode T-Stage
    df['T_label'] = df['KLIN_T'].apply(encode_t_stage)
    
    # Encode Gleason
    df['Gleason_label'] = df['Gl_pet'].apply(encode_gleason)
    
    # PSA is already numeric (use as-is for regression)
    df['PSA_target'] = pd.to_numeric(df['PSA'], errors='coerce')
    
    return df


if __name__ == "__main__":
    # Test encoding on dataset
    CSV_PATH = "dataset_summary.csv"
    df = pd.read_csv(CSV_PATH)
    df = encode_labels(df)
    
    print("=== T-Stage Encoding ===")
    print(df[['KLIN_T', 'T_label']].dropna().value_counts().sort_index())
    print(f"\nValid T-Stage labels: {df['T_label'].notna().sum()}/{len(df)}")
    
    print("\n=== Gleason Encoding ===")
    print(df[['Gl_pet', 'Gleason_label']].dropna().value_counts().sort_index())
    print(f"\nValid Gleason labels: {df['Gleason_label'].notna().sum()}/{len(df)}")
    
    print("\n=== PSA Stats ===")
    print(df['PSA_target'].describe())
    
    # Save encoded version
    df.to_csv("dataset_encoded.csv", index=False)
    print("\nSaved encoded dataset to dataset_encoded.csv")
