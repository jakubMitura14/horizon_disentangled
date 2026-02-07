import pandas as pd

CLIND_PATH = "/media/jm/hddData/projects_new/horizon_disentangled/src/data/clind.xlsx"

try:
    print("Checking 'Tabellenblatt3'...")
    df = pd.read_excel(CLIND_PATH, sheet_name="Tabellenblatt3", header=0) # Assume header at 0 for now
    print("Columns:", df.columns.tolist())
    
    # Check for keywords from the user's notebook
    keywords = ["KLIN", "GRADING", "PSA", "GLEASON", "operation"]
    found = [c for c in df.columns if any(k in str(c).upper() for k in keywords)]
    print("\nFound significant columns:", found)
    
except Exception as e:
    print(f"Error: {e}")
