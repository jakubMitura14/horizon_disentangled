import pandas as pd

file_path = "/media/jm/hddData/projects_new/horizon_disentangled/src/data/clind.xlsx"

try:
    xl = pd.ExcelFile(file_path)
    print("Sheet names:", xl.sheet_names)
    
    # Try reading the first sheet with no header to see the raw structure
    df_raw = pd.read_excel(file_path, header=None, nrows=5)
    print("\nFirst 5 rows (raw):")
    print(df_raw)
    
    # Check if there's a sheet that matches the user's description
    for sheet in xl.sheet_names:
        df_sheet = pd.read_excel(file_path, sheet_name=sheet)
        cols = [str(c).lower() for c in df_sheet.columns]
        if any("epoch" in c for c in cols) or any("befund" in c for c in cols):
            print(f"\nFOUND POTENTIAL MATCH in sheet: {sheet}")
            print("Columns:", df_sheet.columns.tolist())

except Exception as e:
    print(f"Error reading file: {e}")
