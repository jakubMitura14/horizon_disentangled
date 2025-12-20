import json
import pandas as pd

# --- 1. Load the existing notebook ---
with open('project_analysis.ipynb', 'r') as f:
    notebook = json.load(f)

# --- 2. Extract and parse the existing allocation dataframes ---
def parse_df_from_notebook_source(source_list):
    source_str = "".join(source_list)
    match = re.search(r'pd\.DataFrame\((?P<dict>\{.*?\})\)', source_str, re.DOTALL)
    if not match:
        raise ValueError("Could not find a pandas DataFrame definition in the source.")
    dict_str = match.group('dict')
    return pd.DataFrame(eval(dict_str))

# Find the cells with the allocation data
y1_cell = next(c for c in notebook['cells'] if 'df_alloc_y1 =' in c['source'])
y2_cell = next(c for c in notebook['cells'] if 'df_alloc_y2 =' in c['source'])
y3_cell = next(c for c in notebook['cells'] if 'df_alloc_y3 =' in c['source'])

df_alloc_y1 = parse_df_from_notebook_source(y1_cell['source'])
df_alloc_y2 = parse_df_from_notebook_source(y2_cell['source'])
df_alloc_y3 = parse_df_from_notebook_source(y3_cell['source'])

# --- 3. Perform the reallocation ---
# Add WP9 column to all dataframes
df_alloc_y1['WP9'] = 0
df_alloc_y2['WP9'] = 0
df_alloc_y3['WP9'] = 0

# Allocate 18 PMs from Senior Researcher to WP9, spread over 3 years
# 6 PMs per year
df_alloc_y1.loc['Senior Researcher', 'WP9'] = 6
df_alloc_y2.loc['Senior Researcher', 'WP9'] = 6
df_alloc_y3.loc['Senior Researcher', 'WP9'] = 6

# Remove 18 PMs from Senior Researcher from other WPs to keep budget neutral
# Let's take it from WP3 and WP4 where they have high allocation
df_alloc_y1.loc['Senior Researcher', 'WP3'] -= 6 # Was 16, now 10
df_alloc_y2.loc['Senior Researcher', 'WP3'] -= 6 # Was 16, now 10
df_alloc_y3.loc['Senior Researcher', 'WP4'] -= 6 # Was 18, now 12

# --- 4. Update the notebook content ---
# This is tricky because we need to format the dataframe back into the notebook source string.
# A simple way is to use to_json, but the original was a dict of dicts.
def df_to_dict_of_dicts_str(df):
    return json.dumps(df.to_dict())

y1_cell['source'] = [f"df_alloc_y1 = pd.DataFrame({df_to_dict_of_dicts_str(df_alloc_y1)})"]
y2_cell['source'] = [f"df_alloc_y2 = pd.DataFrame({df_to_dict_of_dicts_str(df_alloc_y2)})"]
y3_cell['source'] = [f"df_alloc_y3 = pd.DataFrame({df_to_dict_of_dicts_str(df_alloc_y3)})"]

# Also update the WP requirements list
wp_req_cell = next(c for c in notebook['cells'] if 'wp_data =' in c['source'])
wp_req_cell['source'] = [
    "wp_data = {\\n    'Work_Package': ['WP1', 'WP2', 'WP3', 'WP4', 'WP5', 'WP6', 'WP7', 'WP8', 'WP9'],\\n"
    "    'Total_PM': [78, 58, 46, 52, 56, 16, 36, 12, 18]\\n}\\n" # Adjusted WP3 and WP4
    "df_wp_req = pd.DataFrame(wp_data).set_index('Work_Package')"
]

# Update the verification cell to check the new WP requirements
verification_cell = next(c for c in notebook['cells'] if 'VERIFICATION RESULTS' in c['source'])
verification_cell['source'] = [
    "df_alloc_total = df_alloc_y1.add(df_alloc_y2, fill_value=0).add(df_alloc_y3, fill_value=0)\\n"
    "personnel_check = df_alloc_total.sum(axis=1) == df_personnel_avail['Total_PM']\\n"
    "wp_check = df_alloc_total.sum() == df_wp_req['Total_PM']\\n"
    "print('--- VERIFICATION RESULTS ---')\\n"
    "print('Personnel Allocation Correct:\\n', personnel_check)\\n"
    "print('\\nWP Allocation Correct:\\n', wp_check)"
]


# --- 5. Write the new notebook back to the file ---
with open('project_analysis.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("project_analysis.ipynb has been updated with WP9.")
