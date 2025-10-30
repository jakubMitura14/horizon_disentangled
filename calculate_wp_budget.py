
import pandas as pd
from allocation_logic import get_final_allocations

def calculate_budget():
    """
    Calculates the detailed budget for each work package, ensuring alignment with the
    main grant proposal's summary table.
    """
    # --- 1. Hardcoded Monthly Personnel Costs (per full-time equivalent) ---
    # Extracted from the 'andere Förderer' column in budget_tables.tex for each year.
    costs_y1 = {
        'Principal Investigator': 99448.89 / 12,
        'Senior Researcher': 90410.93 / 12,
        'Clinical Investigator/Consultant': (42320.15 / 6), # Cost is for 6 months in table
        'Mathematician': 90410.93 / 12,
        'Data Scientist': 90410.93 / 12,
        'Programmer': 68382.89 / 12,
        'Technician': 66527.32 / 12,
        'Project Manager': (42320.15 / 12) * 2, # Cost is for 50% in table
        'Secretary': 62674.92 / 12,
        'Student/Research Assistant': 32961.20 / 12,
    }

    costs_y2 = {
        'Principal Investigator': 104421.33 / 12,
        'Senior Researcher': 94931.48 / 12,
        'Clinical Investigator/Consultant': 88872.31 / 12,
        'Mathematician': 94931.48 / 12,
        'Data Scientist': 94931.48 / 12,
        'Programmer': 71802.04 / 12,
        'Technician': 69853.68 / 12,
        'Project Manager': (44436.15 / 12) * 2, # Cost is for 50% in table
        'Secretary': 65808.66 / 12,
        'Student/Research Assistant': 34609.26 / 12,
    }

    costs_y3 = {
        'Principal Investigator': 113547.29 / 12,
        'Senior Researcher': 105198.96 / 12,
        'Clinical Investigator/Consultant': (49041.11 / 6), # Cost is for 6 months in table
        'Mathematician': (63935.00 / 6), # Cost is for 6 months in table
        'Data Scientist': 127869.99 / 12,
        'Programmer': 75392.14 / 12,
        'Technician': 69853.68 / 12, # From 2028 table, as it's the same in 2029
        'Project Manager': (49041.11 / 12) * 2, # Cost is for 50% in table
        'Secretary': 127897.77 / 12, # Using the total from the table
        'Student/Research Assistant': 36339.72 / 12,
    }

    # --- 2. Target Direct Costs per Work Package ---
    # From the summary table in main_horizon.tex
    target_direct_costs = {
        'WP1': 825417.14, 'WP2': 412708.57, 'WP3': 343923.81, 'WP4': 498689.52,
        'WP5': 361120.00, 'WP6': 137569.52, 'WP7': 309531.43, 'WP8': 154765.71,
        'WP9': 154765.71
    }

    # --- 3. Get Person-Month Allocations ---
    df_alloc_y1, df_alloc_y2, df_alloc_y3 = get_final_allocations()
    allocations = {'Y1': df_alloc_y1, 'Y2': df_alloc_y2, 'Y3': df_alloc_y3}
    costs = {'Y1': costs_y1, 'Y2': costs_y2, 'Y3': costs_y3}

    # --- 4. Calculate Personnel Costs and Other Direct Costs ---
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
                    if role not in personnel_breakdown:
                        personnel_breakdown[role] = 0
                    personnel_breakdown[role] += cost

        wp_budgets[wp] = {'personnel_costs': personnel_cost_wp, 'personnel_breakdown': personnel_breakdown}
        total_personnel_costs_all_wps += personnel_cost_wp

    # --- 5. Distribute Other Direct Costs to Match Target ---
    total_other_direct_costs = sum(target_direct_costs.values()) - total_personnel_costs_all_wps

    for wp in wp_budgets:
        personnel_costs = wp_budgets[wp]['personnel_costs']
        # Distribute other costs based on the proportion of personnel costs
        proportion = personnel_costs / total_personnel_costs_all_wps if total_personnel_costs_all_wps > 0 else 0
        other_costs = total_other_direct_costs * proportion

        # Adjust to match the target direct costs exactly
        calculated_direct_cost = personnel_costs + other_costs
        adjustment = target_direct_costs[wp] - calculated_direct_cost
        final_other_costs = other_costs + adjustment

        wp_budgets[wp]['other_costs'] = final_other_costs
        wp_budgets[wp]['direct_costs'] = target_direct_costs[wp]
        wp_budgets[wp]['indirect_costs'] = target_direct_costs[wp] * 0.25
        wp_budgets[wp]['total_costs'] = wp_budgets[wp]['direct_costs'] + wp_budgets[wp]['indirect_costs']

    return wp_budgets

def generate_latex_tables(budgets):
    """
    Generates LaTeX tables from the calculated budget data.
    """

    # Mapping from role names to the categories in the LaTeX template
    role_to_category = {
        'Principal Investigator': 'SENIOR SCIENTISTS',
        'Senior Researcher': 'SENIOR SCIENTISTS',
        'Clinical Investigator/Consultant': 'SENIOR SCIENTISTS',
        'Mathematician': 'SENIOR SCIENTISTS',
        'Data Scientist': 'SENIOR SCIENTISTS',
        'Programmer': 'TECHNICAL PERSONNEL',
        'Technician': 'TECHNICAL PERSONNEL',
        'Project Manager': 'TECHNICAL PERSONNEL',
        'Secretary': 'TECHNICAL PERSONNEL',
        'Student/Research Assistant': 'TECHNICAL PERSONNEL'
    }

    latex_string = ""
    for wp_name, budget_data in budgets.items():
        # Initialize cost categories
        costs = {
            'SENIOR SCIENTISTS': 0,
            'JUNIOR SCIENTISTS': 0,
            'TECHNICAL PERSONNEL': 0,
            'ADMINISTRATIVE PERSONNEL': 0,
            'OTHERS': 0
        }

        # Aggregate costs by category
        for role, cost in budget_data['personnel_breakdown'].items():
            category = role_to_category.get(role, 'OTHERS')
            costs[category] += cost

        # Format costs to two decimal places for LaTeX
        for category in costs:
            costs[category] = f"{costs[category]:.2f}"

        total_personnel = budget_data['personnel_costs']
        total_direct = budget_data['direct_costs']
        indirect_costs = budget_data['indirect_costs']
        total_costs = budget_data['total_costs']
        other_costs = budget_data['other_costs']


        table_string = f"""
\\begin{{table}}[H]
\\centering
\\caption{{Budget for {wp_name.replace('_', ' ')}}}
\\begin{{tabular}}{{|l|l|l|l|}}
\\hline
00000001-0001-0001-0001-000000000001 & Otto-von-Guericke-Universität Magdeburg & ~ & ~ \\\\ \\hline
COST CATEGORY & ITEMS & COST PER ITEM & BE TOTAL COSTS \\\\ \\hline
~ & ~ & ~ & ~ \\\\ \\hline
COSTS WORK PACKAGE {wp_name.split('P')[1]}: {wp_name.replace('WP' + wp_name.split('P')[1] + ': ', '')} & ~ & ~ & ~ \\\\ \\hline
~ & ~ & ~ & ~ \\\\ \\hline
A. DIRECT PERSONNEL COSTS & ~ & ~ & {total_personnel:.2f} \\\\ \\hline
A.1 Employees (or equivalent) & ~ & ~ & ~ \\\\ \\hline
SENIOR SCIENTISTS (or equivalent in the private sector) & ~ & ~ & {costs['SENIOR SCIENTISTS']} \\\\ \\hline
JUNIOR SCIENTISTS (or equivalent in the private sector) & ~ & ~ & {costs['JUNIOR SCIENTISTS']} \\\\ \\hline
TECHNICAL PERSONNEL (or equivalent in the private sector) & ~ & ~ & {costs['TECHNICAL PERSONNEL']} \\\\ \\hline
ADMINISTRATIVE PERSONNEL (or equivalent in the private sector) & ~ & ~ & {costs['ADMINISTRATIVE PERSONNEL']} \\\\ \\hline
OTHERS & ~ & ~ & {costs['OTHERS']} \\\\ \\hline
A.2 Natural Persons under direct contract & ~ & ~ & 0.00 \\\\ \\hline
A.3 Seconded Persons & ~ & ~ & 0.00 \\\\ \\hline
A.4 SME owners and natural person beneficiaries & ~ &  & 0.00 \\\\ \\hline
B. DIRECT SUBCONTRACTING COSTS & ~ & ~ & 0.00\\\\ \\hline
~ & ~ & ~ & 0.00 \\\\ \\hline
C. DIRECT PURCHASE COSTS & ~ & ~ & {other_costs:.2f} \\\\ \\hline
C.1 Travel and subsistence & ~ &  & 0.00 \\\\ \\hline
C.2 Equipment (complete 'Depreciation costs' sheet) & ~ & ~ & ~ \\\\ \\hline
Equipment & ~ &  & 0.00 \\\\ \\hline
Infrastructure & ~ &  & 0.00 \\\\ \\hline
Other assets & ~ &  & 0.00 \\\\ \\hline
C.3 Other goods, works and services & ~ & ~ & ~ \\\\ \\hline
Consumables & ~ &  & 0.00 \\\\ \\hline
Services for meetings, seminars & ~ & ~ & 0.00 \\\\ \\hline
Services for dissemination activities (including website) & ~ & ~ & 0.00 \\\\ \\hline
Publication fees & ~ & 15000 & 0.00 \\\\ \\hline
Other (shipment, insurance, translation, etc.) & ~ & 0 & 0.00 \\\\ \\hline
D. OTHER COST CATEGORIES & ~ & ~ & 0.00\\\\ \\hline
D.1 Financial support to third parties (if applicable in the topic specific conditions) & ~ & ~ & 0.00 \\\\ \\hline
D.2 Internally invoiced goods and services & ~ & ~ & 0.00 \\\\ \\hline
D.3 Transnational access to research infrastructure unit costs (if mentioned as eligible in the topic specific conditions) & ~ & ~ & 0.00 \\\\ \\hline
D.4 Virtual access to research infrastructure unit costs (if mentioned as eligible in the topic specific conditions) & ~ & ~ & 0.00 \\\\ \\hline
D.5 PCP/PPI procurement costs (if mentioned as eligible in the topic specific conditions) & ~ & ~ & 0.00 \\\\ \\hline
~ & ~ & ~ & ~ \\\\ \\hline
TOTAL DIRECT PERSONNEL COSTS AND PURCHASE COSTS (A+C) & ~ & ~ & {total_direct:.2f} \\\\ \\hline
TOTAL DIRECT COSTS (A+B+C+D) & ~ & ~ & {total_direct:.2f} \\\\ \\hline
~ & ~ & ~ & ~ \\\\ \\hline
E. INDIRECT COSTS (25\\% * (A+C)) & ~ & ~ & {indirect_costs:.2f} \\\\ \\hline
~ & ~ & ~ & ~ \\\\ \\hline
F. TOTAL COSTS (A+B+C+D+E) & ~ & ~ & {total_costs:.2f} \\\\ \\hline
\\end{{tabular}}
\\end{{table}}
"""
        latex_string += table_string
    return latex_string

if __name__ == '__main__':
    budgets = calculate_budget()
    latex_output = generate_latex_tables(budgets)
    print(latex_output)

