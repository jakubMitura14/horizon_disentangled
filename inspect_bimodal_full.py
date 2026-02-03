import pandas as pd

CLIND_PATH = "/media/jm/hddData/projects_new/horizon_disentangled/src/data/clind.xlsx"

try:
    print("Checking 'bimodal' sheet full columns...")
    header_row = 0
    # Find header row again as done in summary script
    df_preview = pd.read_excel(CLIND_PATH, sheet_name="bimodal", header=None, nrows=10)
    for idx, row in df_preview.iterrows():
        row_str = " ".join([str(val).lower() for val in row.values])
        if "epoch" in row_str and "immun" in row_str:
            header_row = idx
            break
            
    df = pd.read_excel(CLIND_PATH, sheet_name="bimodal", header=header_row)
    print("Columns:", df.columns.tolist())
    
    # Check for keywords
    keywords = ["KLIN", "GRADING", "PSA", "GLEASON", "operation"]
    found = [c for c in df.columns if any(k in str(c).upper() for k in keywords)]
    print("\nFound significant columns:", found)
    
except Exception as e:
    print(f"Error: {e}")
