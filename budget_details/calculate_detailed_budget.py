import pandas as pd
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from allocation_logic import get_final_allocations

def calculate_final_budget_with_all_corrections():
    """
    Final, fully corrected version.
    - Removes hardcoded registration fee total.
    """
    script_dir = os.path.dirname(__file__)

    # --- 1. Load Data ---
    df_alloc_y1, df_alloc_y2, df_alloc_y3 = get_final_allocations()
    df_yearly_costs = pd.read_csv(os.path.join(script_dir, 'yearly_personnel_costs.csv')).set_index('Role')
    with open(os.path.join(script_dir, 'parsed_tex_data.json'), 'r') as f:
        tex_data = json.load(f)

    # --- 2. Ground Truth & Config ---
    wp_ground_truth_direct_costs = tex_data['direct_costs_per_wp']
    other_direct_costs_grand_totals = tex_data.get('other_direct_costs', {})
    role_to_granular_category = {
        'Principal Investigator': 'SENIOR SCIENTISTS (or equivalent in the private sector)',
        'Senior Researcher': 'SENIOR SCIENTISTS (or equivalent in the private sector)',
        'Clinical Investigator/Consultant': 'SENIOR SCIENTISTS (or equivalent in the private sector)',
        'Mathematician': 'SENIOR SCIENTISTS (or equivalent in the private sector)',
        'Data Scientist': 'SENIOR SCIENTISTS (or equivalent in the private sector)',
        'Programmer': 'TECHNICAL PERSONNEL (or equivalent in the private sector)',
        'Technician': 'TECHNICAL PERSONNEL (or equivalent in the private sector)',
        'Project Manager': 'TECHNICAL PERSONNEL (or equivalent in the private sector)',
        'Secretary': 'ADMINISTRATIVE PERSONNEL (or equivalent in the private sector)'
    }

    # --- 3. Personnel Cost Calculation ---
    cost_per_pm_y1 = (df_yearly_costs['y1_cost'] / 12).to_dict()
    cost_per_pm_y2 = (df_yearly_costs['y2_cost'] / 12).to_dict()
    cost_per_pm_y3 = (df_yearly_costs['y3_cost'] / 12).to_dict()

    df_cost_y1 = df_alloc_y1.apply(lambda row: row * cost_per_pm_y1.get(row.name, 0), axis=1)
    df_cost_y2 = df_alloc_y2.apply(lambda row: row * cost_per_pm_y2.get(row.name, 0), axis=1)
    df_cost_y3 = df_alloc_y3.apply(lambda row: row * cost_per_pm_y3.get(row.name, 0), axis=1)
    df_total_personnel_cost = df_cost_y1.add(df_cost_y2, fill_value=0).add(df_cost_y3, fill_value=0)

    # --- 4. Handle Student Costs ---
    student_costs = df_yearly_costs.loc['Student/Research Assistant'].sum()
    other_direct_costs_grand_totals['Student/Research Assistant'] = student_costs

    # --- 5. Backward Calculation with Capping ---
    wp_personnel_totals = df_total_personnel_cost.sum()
    df_final_personnel_costs = df_total_personnel_cost.copy()

    for wp, total_direct in wp_ground_truth_direct_costs.items():
        personnel_cost = wp_personnel_totals.get(wp, 0)
        if personnel_cost > total_direct:
            capping_ratio = total_direct / personnel_cost
            df_final_personnel_costs[wp] = df_final_personnel_costs[wp] * capping_ratio

    # --- 6. Final ODC Distribution ---
    wp_final_personnel_totals = df_final_personnel_costs.sum()
    wp_odc_pool = {
        wp: wp_ground_truth_direct_costs.get(wp, 0) - wp_final_personnel_totals.get(wp, 0)
        for wp in wp_ground_truth_direct_costs.keys()
    }

    df_other_costs_final = pd.DataFrame(index=list(wp_ground_truth_direct_costs.keys()))
    registration_total = other_direct_costs_grand_totals.get('Registration', 0.0)
    registration_split = registration_total / 2.0

    df_other_costs_final['Registration'] = 0.0
    if 'WP7' in wp_odc_pool and wp_odc_pool['WP7'] > registration_split:
        df_other_costs_final.loc['WP7', 'Registration'] = registration_split
        wp_odc_pool['WP7'] -= registration_split
    elif 'WP7' in wp_odc_pool:
        df_other_costs_final.loc['WP7', 'Registration'] = wp_odc_pool['WP7']
        wp_odc_pool['WP7'] = 0.0

    if 'WP9' in wp_odc_pool and wp_odc_pool['WP9'] > registration_split:
        df_other_costs_final.loc['WP9', 'Registration'] = registration_split
        wp_odc_pool['WP9'] -= registration_split
    elif 'WP9' in wp_odc_pool:
        df_other_costs_final.loc['WP9', 'Registration'] = wp_odc_pool['WP9']
        wp_odc_pool['WP9'] = 0.0

    proportional_odc = {k: v for k, v in other_direct_costs_grand_totals.items() if k != 'Registration'}
    total_proportional_odc = sum(proportional_odc.values())

    for wp, pool_amount in wp_odc_pool.items():
        if pool_amount > 0:
            for item, grand_total in proportional_odc.items():
                proportion = grand_total / total_proportional_odc if total_proportional_odc > 0 else 0
                if item not in df_other_costs_final.columns:
                    df_other_costs_final[item] = 0.0
                df_other_costs_final.loc[wp, item] = pool_amount * proportion

    # --- 7. Assemble CSV ---
    odc_to_granular_category = {
        'Travel': 'C.1 Travel and subsistence', 'Publication Fees': 'Publication fees',
        'UK Biobank Access': 'D.3 Transnational access to research infrastructure unit costs (if mentioned as eligible in the topic specific conditions)',
        'Software Licenses': 'Other (shipment, insurance, translation, etc.)',
        'Long-Term Data Storage': 'Other (shipment, insurance, translation, etc.)',
        'External Expert Consultations': 'Other (shipment, insurance, translation, etc.)',
        'Registration': 'Other (shipment, insurance, translation, etc.)',
        'Student/Research Assistant': 'Other (shipment, insurance, translation, etc.)'
    }

    output_rows = []
    df_alloc_total = df_alloc_y1.add(df_alloc_y2, fill_value=0).add(df_alloc_y3, fill_value=0)

    for wp in sorted(wp_ground_truth_direct_costs.keys()):
        for role, cost_series in df_final_personnel_costs.iterrows():
            total_pms = df_alloc_total.loc[role, wp] if role in df_alloc_total.index else 0
            total_cost = cost_series[wp]
            if abs(total_pms) > 0.001 or abs(total_cost) > 0.001:
                cost_per_item = total_cost / total_pms if total_pms > 0 else 0
                category = role_to_granular_category.get(role, 'OTHERS')
                output_rows.append({'Work Package': wp, 'COST CATEGORY': category, 'ITEMS': f"{total_pms:.2f} PMs ({role})", 'COST PER ITEM': f"{cost_per_item:.2f}", 'BE TOTAL COSTS': f"{total_cost:.2f}"})

        for item in sorted(df_other_costs_final.columns):
            be_total_costs = df_other_costs_final.loc[wp, item]
            if abs(be_total_costs) > 0.001:
                grand_total = other_direct_costs_grand_totals.get(item, 0)
                category = odc_to_granular_category.get(item, 'Other (shipment, insurance, translation, etc.)')
                output_rows.append({'Work Package': wp, 'COST CATEGORY': category, 'ITEMS': item, 'COST PER ITEM': f"{grand_total:.2f}", 'BE TOTAL COSTS': f"{be_total_costs:.2f}"})

    df_final = pd.DataFrame(output_rows)
    df_final.to_csv(os.path.join(script_dir, 'detailed_wp_budgets.csv'), index=False)

    print("Successfully generated final CSV with all corrections.")

if __name__ == "__main__":
    calculate_final_budget_with_all_corrections()
