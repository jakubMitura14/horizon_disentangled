import json
import pandas as pd

# --- 1. Load the existing notebook ---
with open('project_analysis.ipynb', 'r') as f:
    notebook = json.load(f)

# --- 2. Modify the Work Package Definitions ---
# Find the cell defining the work packages
wp_cell = next(c for c in notebook['cells'] if 'wp_data =' in c['source'])

# Modify the source code lines in that cell
wp_source_lines = wp_cell['source']
# Add 'WP9' to the list of work packages
wp_source_lines[1] = wp_source_lines[1].replace("]", ", 'WP9']")
# Add PMs for WP9 (0 for now, will be filled by allocation)
wp_source_lines[2] = wp_source_lines[2].replace("]", ", 0]") # Year 1
wp_source_lines[3] = wp_source_lines[3].replace("]", ", 0]") # Year 2
wp_source_lines[4] = wp_source_lines[4].replace("]", ", 18]")# Year 3
# Update the cell
wp_cell['source'] = wp_source_lines

# --- 3. Modify the Allocation Dataframes ---
alloc_cell = next(c for c in notebook['cells'] if 'df_alloc_y1 =' in c['source'])
alloc_source_lines = alloc_cell['source']

# Add WP9 column with 0 values to the dataframe definitions
new_lines = []
for line in alloc_source_lines:
    if ".loc" in line:
        line = line.replace("]", ", 0]")
    new_lines.append(line)
alloc_cell['source'] = new_lines

# Now, add the specific allocations for WP9
# And remove the PMs from other WPs
# Senior Researcher: +18 in WP9, -18 elsewhere
# Let's take from WP2 and WP3
new_lines_y1 = []
for line in alloc_cell['source']:
    if "'Senior Researcher'" in line and "df_alloc_y1" in line:
         # Reduce WP2 by 8, WP3 by 10
        line = "df_alloc_y1.loc['Senior Researcher']                = [0, 6, 0, 0, 0, 0, 0, 0, 6]\\n"
    new_lines_y1.append(line)
alloc_cell['source'] = new_lines_y1

new_lines_y2 = []
for line in alloc_cell['source']:
     if "'Senior Researcher'" in line and "df_alloc_y2" in line:
        # Reduce WP2 by 10, WP3 by 4
        line = "df_alloc_y2.loc['Senior Researcher']                = [0, 0, 10, 0, 0, 0, 0, 0, 6]\\n"
     new_lines_y2.append(line)
alloc_cell['source'] = new_lines_y2

new_lines_y3 = []
for line in alloc_cell['source']:
    if "'Senior Researcher'" in line and "df_alloc_y3" in line:
        # Reduce WP4 by 16 -> 10, add 6 to WP9
        line = "df_alloc_y3.loc['Senior Researcher']                = [0, 0, 0, 10, 8, 0, 0, 0, 6]\\n"
    new_lines_y3.append(line)
alloc_cell['source'] = new_lines_y3


# --- 4. Write the new notebook back to the file ---
with open('project_analysis.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("project_analysis.ipynb has been updated with WP9.")
