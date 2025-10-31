import pandas as pd
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from allocation_logic import get_final_allocations

def calculate_and_generate_final_budget():
    """
    Calculates the detailed budget using backward-calculation from WP totals,
    including personnel cost capping to prevent negative values. This is the
    correct and final logic.
    """
    script_dir = os.path.dirname(__file__)

    # --- 1. Load Data ---
    df_alloc_y1, df_alloc_y2, df_alloc_y3 = get_final_allocations()
    df_alloc_total = df_alloc_y1.add(df_alloc_y2, fill_value=0).add(df_alloc_y3, fill_value=0)
    df_personnel_costs = pd.read_csv(os.path.join(script_dir, 'personnel_costs.csv')).set_index('Role')
    with open(os.path.join(script_dir, 'parsed_tex_data.json'), 'r') as f:
        tex_data = json.load(f)

    # --- 2. Ground Truth & Config ---
    grand_total_personnel_cost = tex_data['total_personnel_cost']
    wp_total_budgets = tex_data['budget_per_wp']
    other_direct_costs_grand_totals = tex_data.get('other_direct_costs', {})
    total_other_direct_costs = sum(other_direct_costs_grand_totals.values())
    role_to_category = {
        'Principal Investigator': 'SENIOR SCIENTISTS', 'Senior Researcher': 'SENIOR SCIENTISTS',
        'Clinical Investigator/Consultant': 'SENIOR SCIENTISTS', 'Mathematician': 'SENIOR SCIENTISTS',
        'Data Scientist': 'SENIOR SCIENTISTS', 'Programmer': 'TECHNICAL PERSONNEL',
        'Technician': 'TECHNICAL PERSONNEL', 'Project Manager': 'TECHNICAL PERSONNEL', 'Secretary': 'TECHNICAL PERSONNEL',
    }

    # --- 3. Adjusted Personnel Cost Calculation ---
    cost_per_pm = (df_personnel_costs['Annual_Cost'] / 12).to_dict()
    df_raw_costs = df_alloc_total.apply(lambda row: row * cost_per_pm.get(row.name, 0), axis=1)
    raw_total_personnel_cost = df_raw_costs.sum().sum()
    adjustment_ratio = grand_total_personnel_cost / raw_total_personnel_cost if raw_total_personnel_cost > 0 else 0
    df_adjusted_personnel_costs = df_raw_costs * adjustment_ratio

    # --- 4. Capping Logic ---
    wp_total_direct_costs = {wp: total / 1.25 for wp, total in wp_total_budgets.items()}
    wp_personnel_totals_adjusted = df_adjusted_personnel_costs.sum()

    df_final_personnel_costs = df_adjusted_personnel_costs.copy()

    for wp, total_direct in wp_total_direct_costs.items():
        personnel_cost = wp_personnel_totals_adjusted.get(wp, 0)
        if personnel_cost > total_direct:
            capping_ratio = total_direct / personnel_cost
            df_final_personnel_costs[wp] = df_final_personnel_costs[wp] * capping_ratio

    # --- 5. Final Categorization and Aggregation ---
    df_final_personnel_costs['Category'] = df_final_personnel_costs.index.map(role_to_category)
    wp_personnel_costs_by_cat = df_final_personnel_costs.groupby('Category').sum()

    df_alloc_total_categorized = df_alloc_total.copy()
    df_alloc_total_categorized['Category'] = df_alloc_total_categorized.index.map(role_to_category)
    wp_person_months_by_cat = df_alloc_total_categorized.groupby('Category').sum()

    # --- 6. Final Other Direct Costs Calculation (Backward Calculation) ---
    wp_final_personnel_totals = wp_personnel_costs_by_cat.sum()
    wp_other_direct_costs_totals = {
        wp: wp_total_direct_costs.get(wp, 0) - wp_final_personnel_totals.get(wp, 0)
        for wp in wp_total_budgets.keys()
    }
    total_calculated_other_direct = sum(wp_other_direct_costs_totals.values())

    df_other_costs_final = pd.DataFrame(index=list(wp_total_budgets.keys()))
    for item, grand_total in other_direct_costs_grand_totals.items():
        proportion = grand_total / total_other_direct_costs if total_other_direct_costs > 0 else 0
        df_other_costs_final[item] = [v * proportion for v in wp_other_direct_costs_totals.values()]

    # --- 7. Assemble CSV ---
    output_rows = []
    wps_sorted = sorted(wp_total_budgets.keys())

    for wp in wps_sorted:
        for category in sorted(wp_personnel_costs_by_cat.index.unique()):
            total_pms = wp_person_months_by_cat.loc[category, wp]
            total_cost = wp_personnel_costs_by_cat.loc[category, wp]
            cost_per_item = total_cost / total_pms if total_pms > 0 else 0
            output_rows.append({'Work Package': wp, 'COST CATEGORY': 'A. DIRECT PERSONNEL COSTS', 'ITEMS': f"{total_pms:.2f}", 'COST PER ITEM': f"{cost_per_item:.2f}", 'BE TOTAL COSTS': f"{total_cost:.2f}"})

        for item in sorted(df_other_costs_final.columns):
            be_total_costs = df_other_costs_final.loc[wp, item]
            grand_total_for_item = other_direct_costs_grand_totals.get(item, 0)
            output_rows.append({'Work Package': wp, 'COST CATEGORY': 'C. DIRECT PURCHASE COSTS', 'ITEMS': '~', 'COST PER ITEM': f"{grand_total_for_item:.2f}", 'BE TOTAL COSTS': f"{be_total_costs:.2f}"})

    df_final = pd.DataFrame(output_rows)
    df_final.to_csv(os.path.join(script_dir, 'detailed_wp_budgets.csv'), index=False)

    print("Successfully generated final CSV with backward-calculation and capping logic.")
    print("\n--- Final CSV Preview ---")
    print(df_final.head(10))
    print("-------------------------")

if __name__ == "__main__":
    calculate_and_generate_final_budget()
