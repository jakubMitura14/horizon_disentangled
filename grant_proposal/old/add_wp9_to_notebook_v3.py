import json
import pandas as pd

# --- 1. Load the existing notebook ---
with open('project_analysis.ipynb', 'r') as f:
    notebook = json.load(f)

# --- 2. Modify the Work Package Definitions ---
# Find the cell defining the work packages
wp_cell = next(c for c in notebook['cells'] if any('wp_data =' in s for s in c['source']))

# Modify the source code lines in that cell
wp_source_lines = wp_cell['source']
# Add 'WP9' to the list of work packages
wp_source_lines[1] = wp_source_lines[1].replace("]", ", 'WP9']")
# Add PMs for WP9 (0 for Y1/Y2, 18 for Y3)
wp_source_lines[2] = wp_source_lines[2].replace("]", ", 0]") # Year 1
wp_source_lines[3] = wp_source_lines[3].replace("]", ", 0]") # Year 2
wp_source_lines[4] = wp_source_lines[4].replace("]", ", 18]")# Year 3
# Update the cell
wp_cell['source'] = wp_source_lines

# --- 3. Modify the Allocation Dataframes ---
alloc_cell = next(c for c in notebook['cells'] if any('df_alloc_y1 =' in s for s in c['source']))
alloc_source_lines = alloc_cell['source']

# Add WP9 column with 0 values to the dataframe definitions
new_lines = []
for line in alloc_source_lines:
    if ".loc" in line:
        line = line.replace("]", ", 0]")
    new_lines.append(line)
alloc_cell['source'] = new_lines

# Now, add the specific allocations for WP9 (6 PM per year for Senior Researcher)
# and remove the PMs from other WPs to keep budget neutral.
# This requires carefully editing the specific lines.

def modify_line(lines, identifier, new_line):
    for i, line in enumerate(lines):
        if identifier in line:
            lines[i] = new_line
            return
    raise ValueError(f"Identifier '{identifier}' not found in lines.")

# Year 1: SR gets 6 in WP9, loses 6 from WP2
modify_line(alloc_cell['source'],
            "df_alloc_y1.loc['Senior Researcher']",
            "df_alloc_y1.loc['Senior Researcher']                = [0, 8, 10, 0, 0, 0, 0, 0, 6]\\n")

# Year 2: SR gets 6 in WP9, loses 6 from WP2
modify_line(alloc_cell['source'],
            "df_alloc_y2.loc['Senior Researcher']",
            "df_alloc_y2.loc['Senior Researcher']                = [0, 4, 14, 0, 0, 0, 0, 0, 6]\\n")

# Year 3: SR gets 6 in WP9, loses 6 from WP4
modify_line(alloc_cell['source'],
            "df_alloc_y3.loc['Senior Researcher']",
            "df_alloc_y3.loc['Senior Researcher']                = [0, 0, 0, 10, 8, 0, 0, 0, 6]\\n")


# --- 4. Write the new notebook back to the file ---
with open('project_analysis.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("project_analysis.ipynb has been updated with WP9.")
