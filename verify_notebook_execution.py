import json
import pandas as pd
import numpy as np

# This script will simulate running the notebook by executing its code cells in order.

with open('project_analysis.ipynb', 'r') as f:
    notebook = json.load(f)

# A global namespace to execute all the code in
execution_namespace = {}

for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        # Join the source lines and execute
        source_code = "".join(cell['source'])
        try:
            exec(source_code, globals(), execution_namespace)
        except Exception as e:
            print(f"Error executing cell:\\n{source_code}\\n")
            print(f"Error: {e}")
            exit(1)

print("--- Notebook Verification Successful ---")
print("The code in project_analysis.ipynb was executed without errors.")

# Perform a final check on the dataframes created
df_total = execution_namespace['df_alloc_total']
print("\nFinal PM allocation per WP:")
print(df_total.sum())
