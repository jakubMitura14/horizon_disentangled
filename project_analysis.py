import pandas as pd
import re

# --- Step 1: Define the Ground Truth Allocation Plan ---

# Available PMs (Core Team)
personnel_list = [
    {'Role': 'Principal Investigator', 'Available_PM': 36},
    {'Role': 'Senior Researcher', 'Available_PM': 72},
    {'Role': 'Clinical Investigator/Consultant', 'Available_PM': 108},
    {'Role': 'Mathematician', 'Available_PM': 30},
    {'Role': 'Data Scientist', 'Available_PM': 36},
    {'Role': 'Programmer', 'Available_PM': 36},
    {'Role': 'Technician', 'Available_PM': 36},
    {'Role': 'Project Manager', 'Available_PM': 18},
    {'Role': 'Secretary', 'Available_PM': 36}
]
df_available = pd.DataFrame(personnel_list).set_index('Role')

# Required PMs (Reduced Scope)
required_list = [
    {'Work_Package': 'WP1', 'Required_PM': 85},
    {'Work_Package': 'WP2', 'Required_PM': 65},
    {'Work_Package': 'WP3', 'Required_PM': 65},
    {'Work_Package': 'WP4', 'Required_PM': 65},
    {'Work_Package': 'WP5', 'Required_PM': 64},
    {'Work_Package': 'WP6', 'Required_PM': 16},
    {'Work_Package': 'WP7', 'Required_PM': 36},
    {'Work_Package': 'WP8', 'Required_PM': 12}
]
df_required = pd.DataFrame(required_list).set_index('Work_Package')


# --- Step 2: Extract Data from main_horizon.tex to Verify ---

def extract_data_from_tex(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Extract WP PMs from descriptions
    wp_desc_pms = {}
    pattern = re.compile(r'\\textbf{WP(\\d+):.*?\\((\\d+)\\s*PMs\\):}')
    matches = pattern.findall(content)
    for match in matches:
        wp_desc_pms[f'WP{match[0]}'] = int(match[1])

    # Extract WP PMs from summary table
    table_pms = {}
    table_pattern = re.compile(r'\\begin{table}.*?\\label{tab:person_months}(.*?)\\end{table}', re.DOTALL)
    table_match = table_pattern.search(content)
    if table_match:
        numbers_row_pattern = re.compile(r'\\textbf{Person-Months}\\s*&\\s*(.*?)\\\\s*')
        numbers_match = numbers_row_pattern.search(table_match.group(1))
        if numbers_match:
            numbers_str = numbers_match.group(1).strip()
            pm_values = [int(re.search(r'(\\d+)', val).group(1)) for val in numbers_str.split('&')]
            for i, pm in enumerate(pm_values[:-1], 1):
                table_pms[f'WP{i}'] = pm
            table_pms['Total'] = pm_values[-1]

    return wp_desc_pms, table_pms

# --- Step 3: Perform Verification ---

print('--- Final Verification of main_horizon.tex ---')
wp_from_desc, wp_from_table = extract_data_from_tex('main_horizon.tex')

all_ok = True

# Verify WP descriptions
for wp, pm in df_required['Required_PM'].items():
    if wp_from_desc.get(wp) != pm:
        print(f'ERROR: WP Description mismatch for {wp}. Expected {pm}, found {wp_from_desc.get(wp)}')
        all_ok = False

# Verify summary table
for wp, pm in df_required['Required_PM'].items():
    if wp_from_table.get(wp) != pm:
        print(f'ERROR: Summary table mismatch for {wp}. Expected {pm}, found {wp_from_table.get(wp)}')
        all_ok = False

# Verify table total
if wp_from_table.get('Total') != df_required['Required_PM'].sum():
    print(f'ERROR: Summary table total is incorrect. Expected {df_required["Required_PM"].sum()}, found {wp_from_table.get("Total")}')
    all_ok = False

if all_ok:
    print('\\nSUCCESS: All numbers in main_horizon.tex are consistent with the verified plan.')
else:
    print('\\nFAILURE: Inconsistencies found in main_horizon.tex.')
