import pandas as pd
import json

# This script now reads the corrected allocation directly from the notebook
# to ensure it's always in sync.

# --- 1. Load Data from Corrected Notebook ---
with open('project_analysis.ipynb', 'r') as f:
    notebook = json.load(f)

# Extract the relevant dataframes from the notebook cells' source code
# Note: This is a simplified parser. It assumes the dataframe definitions are in specific cells.
personnel_source = next(c['source'] for c in notebook['cells'] if 'personnel_data' in c['source'])
y1_source = next(c['source'] for c in notebook['cells'] if 'df_alloc_y1 =' in c['source'])
y2_source = next(c['source'] for c in notebook['cells'] if 'df_alloc_y2 =' in c['source'])
y3_source = next(c['source'] for c in notebook['cells'] if 'df_alloc_y3 =' in c['source'])

# Clean the source strings before executing
personnel_source_clean = "".join(personnel_source)
y1_source_clean = "".join(y1_source)
y2_source_clean = "".join(y2_source)
y3_source_clean = "".join(y3_source)


# Execute the source code to define the dataframes
exec(personnel_source_clean)
df_personnel_avail = pd.DataFrame(personnel_data).set_index('Role')
df_personnel_avail['Total_PM'] = df_personnel_avail.sum(axis=1)

# A new namespace to execute the code in, to avoid overwriting variables
ns = {}
exec(y1_source_clean, ns)
exec(y2_source_clean, ns)
exec(y3_source_clean, ns)
df_alloc_y1_loaded = ns['df_alloc_y1']
df_alloc_y2_loaded = ns['df_alloc_y2']
df_alloc_y3_loaded = ns['df_alloc_y3']


df_alloc_total = df_alloc_y1_loaded.add(df_alloc_y2_loaded, fill_value=0).add(df_alloc_y3_loaded, fill_value=0)
wp_person_months = df_alloc_total.sum()
total_person_months = wp_person_months.sum()


# --- 2. Define Final Budget Numbers (Based on User Input) ---
final_total_budget = 3998114.25
indirect_costs_rate = 0.25

# Back-calculate other totals to be perfectly consistent
total_direct_costs = final_total_budget / (1 + indirect_costs_rate)
indirect_costs = total_direct_costs * indirect_costs_rate
total_other_direct_costs = 278900.00 # This is fixed from the tables
total_personnel_cost = total_direct_costs - total_other_direct_costs


# --- 3. Calculation ---

# Calculate the average cost per person-month based on the consistent totals
avg_cost_per_pm = total_personnel_cost / total_person_months

# Calculate the personnel cost for each WP
wp_personnel_costs = wp_person_months * avg_cost_per_pm

# Distribute other direct and indirect costs proportionally to personnel costs
wp_personnel_cost_proportion = wp_personnel_costs / total_personnel_cost
wp_other_direct_costs = wp_personnel_cost_proportion * total_other_direct_costs
wp_indirect_costs = wp_personnel_cost_proportion * indirect_costs

# Calculate final costs per WP
wp_direct_costs_final = wp_personnel_costs + wp_other_direct_costs
wp_total_costs_final = wp_direct_costs_final + wp_indirect_costs


# --- 4. Format for LaTeX Table ---

wp_names = {
    'WP1': 'Data Curation', 'WP2': 'Supervisor Models', 'WP3': 'Causal VAE',
    'WP4': 'Temporal Modeling', 'WP5': 'Validation', 'WP6': 'Dissemination',
    'WP7': 'Project Management', 'WP8': 'Portfolio Activities'
}

print("--- Final LaTeX Budget Table (in EUR) ---")
for wp_code, wp_name in wp_names.items():
    direct = wp_direct_costs_final[wp_code]
    indirect = wp_indirect_costs[wp_code]
    total = wp_total_costs_final[wp_code]
    # SISetup format is without commas for easy copy-pasting
    print(f"WP{wp_names[wp_code].split()[0][0]}{wp_names[wp_code].split()[1][0]}: {wp_name} & {direct:,.2f} & {indirect:,.2f} & {total:,.2f} \\\\")

print("\\midrule")
direct_sum = wp_direct_costs_final.sum()
indirect_sum = wp_indirect_costs.sum()
total_sum = wp_total_costs_final.sum()
print(f"\\textbf{{Total}} & \\textbf{{{direct_sum:,.2f}}} & \\textbf{{{indirect_sum:,.2f}}} & \\textbf{{{total_sum:,.2f}}} \\\\")

print("\n--- Verification ---")
print(f"Final calculated total: €{total_sum:,.2f}")
print(f"Target total:           €{final_total_budget:,.2f}")
print(f"Difference:             €{total_sum - final_total_budget:.4f}")
