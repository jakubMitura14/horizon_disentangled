import pandas as pd

file_path = "/media/jm/hddData/projects_new/horizon_disentangled/src/data/clind.xlsx"

try:
    df = pd.read_excel(file_path)
    print("Columns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Check for Patient ID column
    possible_id_cols = [c for c in df.columns if "pat" in c.lower() or "id" in c.lower() or "epoch" in c.lower()]
    print("\nPossible Patient ID columns:", possible_id_cols)
    
except Exception as e:
    print(f"Error reading file: {e}")
