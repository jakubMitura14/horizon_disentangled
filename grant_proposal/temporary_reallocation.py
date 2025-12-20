# temporary_reallocation.py
import pandas as pd
import numpy as np
import json

# Read the notebook
with open('project_analysis.ipynb', 'r') as f:
    notebook = json.load(f)

# Extract the relevant cell source
personnel_data_source = notebook['cells'][3]['source']
wp_data_source = notebook['cells'][5]['source']
alloc_y1_source = notebook['cells'][7]['source']
alloc_y2_source = notebook['cells'][8]['source']
alloc_y3_source = notebook['cells'][9]['source']

# Execute the source code to get the dataframes
exec("".join(personnel_data_source))
exec("".join(wp_data_source))
exec("".join(alloc_y1_source))
exec("".join(alloc_y2_source))
exec("".join(alloc_y3_source))


# Add WP9 and re-balance
df_alloc_y1.loc['Senior Researcher', 'WP2'] -= 2
df_alloc_y1.loc['Senior Researcher', 'WP3'] -= 2
df_alloc_y1['WP9'] = 0
df_alloc_y1.loc['Senior Researcher', 'WP9'] = 4

df_alloc_y2.loc['Senior Researcher', 'WP2'] -= 2
df_alloc_y2.loc['Senior Researcher', 'WP3'] -= 2
df_alloc_y2['WP9'] = 0
df_alloc_y2.loc['Senior Researcher', 'WP9'] = 4

df_alloc_y3.loc['Senior Researcher', 'WP4'] -= 4
df_alloc_y3['WP9'] = 0
df_alloc_y3.loc['Senior Researcher', 'WP9'] = 4


# Create new wp_data
wp_data = {
    'Work_Package': ['WP1', 'WP2', 'WP3', 'WP4', 'WP5', 'WP6', 'WP7', 'WP8', 'WP9'],
    'Year_1_PM': df_alloc_y1.sum(axis=0).values.tolist(),
    'Year_2_PM': df_alloc_y2.sum(axis=0).values.tolist(),
    'Year_3_PM': df_alloc_y3.sum(axis=0).values.tolist(),
}


# Create the new source strings
new_wp_data_source = f"wp_data = {wp_data}"
new_alloc_y1_source = f"df_alloc_y1 = pd.DataFrame({df_alloc_y1.to_dict()})"
new_alloc_y2_source = f"df_alloc_y2 = pd.DataFrame({df_alloc_y2.to_dict()})"
new_alloc_y3_source = f"df_alloc_y3 = pd.DataFrame({df_alloc_y3.to_dict()})"


# Update the notebook object
notebook['cells'][5]['source'] = [new_wp_data_source]
notebook['cells'][7]['source'] = [new_alloc_y1_source]
notebook['cells'][8]['source'] = [new_alloc_y2_source]
notebook['cells'][9]['source'] = [new_alloc_y3_source]


# Write the new notebook
with open('project_analysis.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook updated successfully.")
