import pandas as pd

CSV_PATH = "dataset_summary.csv"

try:
    df = pd.read_csv(CSV_PATH)
    print("--- Dataset Verification ---")
    print(f"Total Rows: {len(df)}")
    
    # Check CAPRA Score distribution
    if "capra_score" in df.columns:
        print("\nCAPRA Score Stats:")
        print(df["capra_score"].describe())
        print("Unique Values:", sorted(df["capra_score"].dropna().unique()))
        
    # Check Outcome
    if "is_progression" in df.columns:
        print("\nProgression Counts:")
        print(df["is_progression"].value_counts())
        
    if "overall_last_status" in df.columns:
        print("\nOverall Status:")
        print(df["overall_last_status"].value_counts())
        
    # Check T Stage
    if "KLIN_T" in df.columns:
         print("\nCombined T Stage (KLIN_T):")
         print(df["KLIN_T"].value_counts())

    # Check Completeness
    missing_scores = df["capra_score"].isna().sum()
    print(f"\nMissing CAPRA scores: {missing_scores}")

except Exception as e:
    print(f"Error: {e}")
