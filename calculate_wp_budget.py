
import pandas as pd
from allocation_logic import get_final_allocations

def calculate_budget():
    """
    Calculates the detailed budget for each work package, ensuring alignment with the
    main grant proposal's summary table and preventing negative cost items.
    """
    # --- 1. Hardcoded Monthly Personnel Costs ---
    costs_y1 = {
        'Principal Investigator': 99448.89 / 12, 'Senior Researcher': 90410.93 / 12,
        'Clinical Investigator/Consultant': (42320.15 / 6), 'Mathematician': 90410.93 / 12,
        'Data Scientist': 90410.93 / 12, 'Programmer': 68382.89 / 12,
        'Technician': 66527.32 / 12, 'Project Manager': (42320.15 / 12) * 2,
        'Secretary': 62674.92 / 12, 'Student/Research Assistant': 32961.20 / 12,
    }
    costs_y2 = {
        'Principal Investigator': 104421.33 / 12, 'Senior Researcher': 94931.48 / 12,
        'Clinical Investigator/Consultant': 88872.31 / 12, 'Mathematician': 94931.48 / 12,
        'Data Scientist': 94931.48 / 12, 'Programmer': 71802.04 / 12,
        'Technician': 69853.68 / 12, 'Project Manager': (44436.15 / 12) * 2,
        'Secretary': 65808.66 / 12, 'Student/Research Assistant': 34609.26 / 12,
    }
    costs_y3 = {
        'Principal Investigator': 113547.29 / 12, 'Senior Researcher': 105198.96 / 12,
        'Clinical Investigator/Consultant': (49041.11 / 6), 'Mathematician': (63935.00 / 6),
        'Data Scientist': 127869.99 / 12, 'Programmer': 75392.14 / 12,
        'Technician': 69853.68 / 12, 'Project Manager': (49041.11 / 12) * 2,
        'Secretary': 127897.77 / 12, 'Student/Research Assistant': 36339.72 / 12,
    }

    # --- 2. Target Direct Costs & Itemized Other Costs ---
    target_direct_costs = {
        'WP1': 825417.14, 'WP2': 412708.57, 'WP3': 343923.81, 'WP4': 498689.52,
        'WP5': 361120.00, 'WP6': 137569.52, 'WP7': 309531.43, 'WP8': 154765.71,
        'WP9': 154765.71
    }

    fixed_costs_wp1 = { 'UK Biobank Access': 12000, 'Subject Insurance': 20000, }
    distributable_other_costs = {
        'Travel and subsistence': 32500, 'Publication fees': 15000,
        'Other': (81600 - 20000) + 36000 + 75800 + 26000,
    }

    # --- 3. Get Allocations & Calculate Core Personnel Costs ---
    df_alloc_y1, df_alloc_y2, df_alloc_y3 = get_final_allocations()
    allocations = {'Y1': df_alloc_y1, 'Y2': df_alloc_y2, 'Y3': df_alloc_y3}
    costs = {'Y1': costs_y1, 'Y2': costs_y2, 'Y3': costs_y3}

    df_alloc_total = df_alloc_y1 + df_alloc_y2 + df_alloc_y3

    wp_budgets = {}
    total_personnel_costs_all_wps = 0
    for wp in df_alloc_y1.columns:
        personnel_cost_wp = 0
        personnel_breakdown = {}
        for year_str, df_alloc in allocations.items():
            cost_map = costs[year_str]
            for role in df_alloc.index:
                pm = df_alloc.loc[role, wp]
                if pm > 0:
                    cost = pm * cost_map[role]
                    personnel_cost_wp += cost
                    if role not in personnel_breakdown: personnel_breakdown[role] = {'cost': 0, 'pm': 0}
                    personnel_breakdown[role]['cost'] += cost
                    personnel_breakdown[role]['pm'] += pm
        wp_budgets[wp] = {'personnel_costs': personnel_cost_wp, 'personnel_breakdown': personnel_breakdown}
        total_personnel_costs_all_wps += personnel_cost_wp

    # --- 4. Calculate and Distribute Student Costs ---
    student_total_cost = (costs_y1['Student/Research Assistant'] * 12) + (costs_y2['Student/Research Assistant'] * 12) + (costs_y3['Student/Research Assistant'] * 12)
    wp1_personnel = wp_budgets['WP1']['personnel_costs']
    wp5_personnel = wp_budgets['WP5']['personnel_costs']
    student_wp1_share = wp1_personnel / (wp1_personnel + wp5_personnel)

    wp_budgets['WP1']['personnel_costs'] += student_total_cost * student_wp1_share
    wp_budgets['WP5']['personnel_costs'] += student_total_cost * (1-student_wp1_share)
    total_personnel_costs_all_wps += student_total_cost

    if 'Student/Research Assistant' not in wp_budgets['WP1']['personnel_breakdown']: wp_budgets['WP1']['personnel_breakdown']['Student/Research Assistant'] = {'cost': 0, 'pm': 0}
    wp_budgets['WP1']['personnel_breakdown']['Student/Research Assistant']['cost'] += student_total_cost * student_wp1_share
    if 'Student/Research Assistant' not in wp_budgets['WP5']['personnel_breakdown']: wp_budgets['WP5']['personnel_breakdown']['Student/Research Assistant'] = {'cost': 0, 'pm': 0}
    wp_budgets['WP5']['personnel_breakdown']['Student/Research Assistant']['cost'] += student_total_cost * (1-student_wp1_share)


    # --- 5. Distribute Other Costs and Adjust Personnel if Necessary ---
    for wp in wp_budgets:
        personnel_costs = wp_budgets[wp]['personnel_costs']
        target_direct = target_direct_costs[wp]

        if personnel_costs > target_direct:
            # Personnel costs exceed the total budget for this WP. Reduce them.
            surplus = personnel_costs - target_direct
            reduction_ratio = 1 - (surplus / personnel_costs)

            # Reduce cost of each role in the breakdown
            for role in wp_budgets[wp]['personnel_breakdown']:
                wp_budgets[wp]['personnel_breakdown'][role]['cost'] *= reduction_ratio

            # Update total personnel cost to match the target
            wp_budgets[wp]['personnel_costs'] = target_direct

            # Set other costs to zero
            distributed_costs = {key: 0 for key in distributable_other_costs}
            if wp == 'WP1': # Still need to account for this possibility
                distributed_costs['UK Biobank Access'] = 0
        else:
            # Proceed as before
            proportion = personnel_costs / total_personnel_costs_all_wps if total_personnel_costs_all_wps > 0 else 0
            distributed_costs = {key: value * proportion for key, value in distributable_other_costs.items()}
            if wp == 'WP1':
                distributed_costs['UK Biobank Access'] = fixed_costs_wp1['UK Biobank Access']
                distributed_costs['Other'] += fixed_costs_wp1['Subject Insurance']

            total_distributed_other_costs = sum(distributed_costs.values())
            calculated_direct_cost = personnel_costs + total_distributed_other_costs
            adjustment = target_direct - calculated_direct_cost

            if adjustment < 0:
                positive_costs_sum = sum(v for v in distributed_costs.values() if v > 0)
                if positive_costs_sum > 0:
                    for key in distributed_costs:
                        if distributed_costs[key] > 0: distributed_costs[key] += adjustment * (distributed_costs[key] / positive_costs_sum)
            else:
                if 'Other' not in distributed_costs: distributed_costs['Other'] = 0
                distributed_costs['Other'] += adjustment

        wp_budgets[wp]['other_costs_breakdown'] = distributed_costs
        wp_budgets[wp]['direct_costs'] = target_direct
        wp_budgets[wp]['indirect_costs'] = target_direct * 0.25
        wp_budgets[wp]['total_costs'] = wp_budgets[wp]['direct_costs'] + wp_budgets[wp]['indirect_costs']

    return wp_budgets, df_alloc_total

def generate_detailed_csv(budgets, df_alloc_total):
    role_to_category = {
        'Principal Investigator': 'SENIOR SCIENTISTS', 'Senior Researcher': 'SENIOR SCIENTISTS',
        'Clinical Investigator/Consultant': 'SENIOR SCIENTISTS', 'Mathematician': 'SENIOR SCIENTISTS',
        'Data Scientist': 'SENIOR SCIENTISTS', 'Programmer': 'TECHNICAL PERSONNEL',
        'Technician': 'TECHNICAL PERSONNEL', 'Project Manager': 'TECHNICAL PERSONNEL',
        'Secretary': 'TECHNICAL PERSONNEL', 'Student/Research Assistant': 'TECHNICAL PERSONNEL'
    }

    total_pm_by_category = {'SENIOR SCIENTISTS': 0, 'TECHNICAL PERSONNEL': 0}
    for role, category in role_to_category.items():
        if role == 'Student/Research Assistant' or role not in df_alloc_total.index: continue
        total_pm_by_category[category] += df_alloc_total.loc[role].sum()

    csv_data = []

    for wp_name, budget_data in budgets.items():
        pm_by_cat = {'SENIOR SCIENTISTS': 0, 'TECHNICAL PERSONNEL': 0}
        cost_by_cat = {'SENIOR SCIENTISTS': 0, 'TECHNICAL PERSONNEL': 0}
        for role, data in budget_data['personnel_breakdown'].items():
            category = role_to_category.get(role)
            if category in pm_by_cat:
                cost_by_cat[category] += data['cost']
                if role != 'Student/Research Assistant':
                    pm_by_cat[category] += data['pm']

        effort_ratio_senior = pm_by_cat['SENIOR SCIENTISTS'] / total_pm_by_category['SENIOR SCIENTISTS'] if total_pm_by_category['SENIOR SCIENTISTS'] > 0 else 0
        effort_ratio_tech = pm_by_cat['TECHNICAL PERSONNEL'] / total_pm_by_category['TECHNICAL PERSONNEL'] if total_pm_by_category['TECHNICAL PERSONNEL'] > 0 else 0

        other_costs = budget_data['other_costs_breakdown']
        rows = [
            ('A. DIRECT PERSONNEL COSTS', 'SENIOR SCIENTISTS (or equivalent in the private sector)', f"{effort_ratio_senior:.4f}", f"{cost_by_cat['SENIOR SCIENTISTS']:.2f}"),
            ('A. DIRECT PERSONNEL COSTS', 'JUNIOR SCIENTISTS (or equivalent in the private sector)', '0.0000', '0.00'),
            ('A. DIRECT PERSONNEL COSTS', 'TECHNICAL PERSONNEL (or equivalent in the private sector)', f"{effort_ratio_tech:.4f}", f"{cost_by_cat['TECHNICAL PERSONNEL']:.2f}"),
            ('A. DIRECT PERSONNEL COSTS', 'ADMINISTRATIVE PERSONNEL (or equivalent in the private sector)', '0.0000', '0.00'),
            ('B. DIRECT SUBCONTRACTING COSTS', '~', '0.0000', '0.00'),
            ('C. DIRECT PURCHASE COSTS', 'C.1 Travel and subsistence', '~', f"{other_costs.get('Travel and subsistence', 0):.2f}"),
            ('C. DIRECT PURCHASE COSTS', 'C.3 Publication fees', '~', f"{other_costs.get('Publication fees', 0):.2f}"),
            ('C. DIRECT PURCHASE COSTS', 'C.3 Other (shipment, insurance, translation, etc.)', '~', f"{other_costs.get('Other', 0):.2f}"),
            ('D. OTHER COST CATEGORIES', 'D.3 Transnational access to research infrastructure unit costs', '~', f"{other_costs.get('UK Biobank Access', 0):.2f}"),
            ('TOTALS', 'TOTAL DIRECT COSTS', '~', f"{budget_data['direct_costs']:.2f}"),
            ('TOTALS', 'INDIRECT COSTS', '~', f"{budget_data['indirect_costs']:.2f}"),
            ('TOTALS', 'TOTAL COSTS', '~', f"{budget_data['total_costs']:.2f}"),
        ]

        for cat, item, cost_per, total in rows:
            csv_data.append({'Work Package': wp_name, 'COST CATEGORY': cat, 'ITEMS': item, 'COST PER ITEM': cost_per, 'BE TOTAL COSTS': total})

    df = pd.DataFrame(csv_data)
    df.to_csv('detailed_wp_budgets.csv', index=False)

    summary_data = [{'Work Package': wp, 'Direct Costs': f"{d['direct_costs']:.2f}", 'Indirect Costs': f"{d['indirect_costs']:.2f}", 'Total Cost': f"{d['total_costs']:.2f}"} for wp, d in budgets.items()]
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('budget_per_wp.csv', index=False)


if __name__ == '__main__':
    budgets, df_alloc_total = calculate_budget()
    generate_detailed_csv(budgets, df_alloc_total)
    print("CSV files 'detailed_wp_budgets.csv' and 'budget_per_wp.csv' generated successfully.")
