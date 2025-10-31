import pandas as pd
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from allocation_logic import get_final_allocations

def calculate_budget_with_correct_logic():
    """
    Calculates the detailed budget using the correct, user-specified business logic
    for cost distribution, addressing all issues from the code review.
    """
    script_dir = os.path.dirname(__file__)

    # --- 1. Load Data ---
    df_alloc_y1, df_alloc_y2, df_alloc_y3 = get_final_allocations()
    all_roles = list(df_alloc_y1.index) + ['Student/Research Assistant']
    df_alloc_total = df_alloc_y1.add(df_alloc_y2, fill_value=0).add(df_alloc_y3, fill_value=0).reindex(all_roles, fill_value=0)

    df_personnel_costs = pd.read_csv(os.path.join(script_dir, 'personnel_costs.csv')).set_index('Role')
    with open(os.path.join(script_dir, 'parsed_tex_data.json'), 'r') as f:
        tex_data = json.load(f)

    # --- 2. Config & Ground Truth ---
    other_direct_costs_totals = tex_data['other_direct_costs']
    grand_total_personnel_cost = tex_data['total_personnel_cost']

    role_to_category = {
        'Principal Investigator': 'SENIOR SCIENTISTS', 'Senior Researcher': 'SENIOR SCIENTISTS',
        'Clinical Investigator/Consultant': 'SENIOR SCIENTISTS', 'Mathematician': 'SENIOR SCIENTISTS',
        'Data Scientist': 'SENIOR SCIENTISTS', 'PhD Student': 'SENIOR SCIENTISTS',
        'Programmer': 'TECHNICAL PERSONNEL', 'Technician': 'TECHNICAL PERSONNEL',
        'Project Manager': 'TECHNICAL PERSONNEL', 'Secretary': 'TECHNICAL PERSONNEL',
        'Student/Research Assistant': 'TECHNICAL PERSONNEL'
    }

    # --- 3. Calculate Adjusted Personnel Costs ---
    cost_per_pm = (df_personnel_costs['Annual_Cost'] / 12).to_dict()
    df_raw_costs = df_alloc_total.apply(lambda row: row * cost_per_pm.get(row.name, 0), axis=1)
    raw_total_personnel_cost = df_raw_costs.sum().sum()
    adjustment_ratio = grand_total_personnel_cost / raw_total_personnel_cost if raw_total_personnel_cost > 0 else 0
    df_final_personnel_costs = df_raw_costs * adjustment_ratio

    # --- 4. Distribute Other Direct Costs Based on Explicit Rules ---
    wp_personnel_cost_totals = df_final_personnel_costs.sum()
    total_personnel_for_dist = wp_personnel_cost_totals.sum()
    fixed_costs = {k: v for k, v in other_direct_costs_totals.items() if k in ['UK Biobank Access', 'Subject Insurance']}
    proportional_costs = {k: v for k, v in other_direct_costs_totals.items() if k not in fixed_costs}
    df_other_costs_final = pd.DataFrame(0.0, index=df_alloc_total.columns, columns=other_direct_costs_totals.keys())

    for wp, personnel_cost in wp_personnel_cost_totals.items():
        proportion = personnel_cost / total_personnel_for_dist if total_personnel_for_dist > 0 else 0
        for item_name, total_value in proportional_costs.items():
            df_other_costs_final.loc[wp, item_name] = total_value * proportion
    for item_name, total_value in fixed_costs.items():
        if 'WP1' in df_other_costs_final.index:
            df_other_costs_final.loc['WP1', item_name] += total_value

    # --- 5. Assemble Final CSV ---
    df_final_personnel_costs['Category'] = df_final_personnel_costs.index.map(role_to_category)
    wp_personnel_costs_by_cat = df_final_personnel_costs.groupby('Category').sum()

    df_alloc_total_categorized = df_alloc_total.copy()
    df_alloc_total_categorized['Category'] = df_alloc_total_categorized.index.map(role_to_category)
    wp_person_months_by_cat = df_alloc_total_categorized.groupby('Category').sum()

    output_rows = []
    wps_sorted = sorted(df_alloc_total.columns.tolist())

    for wp in wps_sorted:
        for category in sorted(wp_personnel_costs_by_cat.index.unique()):
            total_pms = wp_person_months_by_cat.loc[category, wp]
            total_cost = wp_personnel_costs_by_cat.loc[category, wp]
            cost_per_item = total_cost / total_pms if total_pms > 0 else 0
            output_rows.append({'Work Package': wp, 'COST CATEGORY': 'A. DIRECT PERSONNEL COSTS', 'ITEMS': f"{total_pms:.2f}", 'COST PER ITEM': f"{cost_per_item:.2f}", 'BE TOTAL COSTS': f"{total_cost:.2f}"})

        for item in sorted(df_other_costs_final.columns):
            be_total_costs = df_other_costs_final.loc[wp, item]
            grand_total = other_direct_costs_totals.get(item, 0)
            output_rows.append({'Work Package': wp, 'COST CATEGORY': 'C. DIRECT PURCHASE COSTS', 'ITEMS': '~', 'COST PER ITEM': f"{grand_total:.2f}", 'BE TOTAL COSTS': f"{be_total_costs:.2f}"})

    df_final = pd.DataFrame(output_rows)
    df_final.to_csv(os.path.join(script_dir, 'detailed_wp_budgets.csv'), index=False)

    print("Successfully generated CSV with correct business logic.")
    print("\n--- Final CSV Preview ---")
    print(df_final.head(10))
    print("-------------------------")

if __name__ == "__main__":
    calculate_budget_with_correct_logic()
