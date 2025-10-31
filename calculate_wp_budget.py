
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

    itemized_other_costs = {
        'Travel and subsistence': 32500, 'Publication fees': 15000,
        'Other': 81600 + 12000 + 36000 + 75800 + 26000
    }

    # --- 3. Get Allocations & Calculate Personnel Costs ---
    df_alloc_y1, df_alloc_y2, df_alloc_y3 = get_final_allocations()
    allocations = {'Y1': df_alloc_y1, 'Y2': df_alloc_y2, 'Y3': df_alloc_y3}
    costs = {'Y1': costs_y1, 'Y2': costs_y2, 'Y3': costs_y3}

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
                    if role not in personnel_breakdown: personnel_breakdown[role] = 0
                    personnel_breakdown[role] += cost
        wp_budgets[wp] = {'personnel_costs': personnel_cost_wp, 'personnel_breakdown': personnel_breakdown}
        total_personnel_costs_all_wps += personnel_cost_wp

    # --- 4. Distribute Other Costs with new adjustment logic ---
    for wp in wp_budgets:
        personnel_costs = wp_budgets[wp]['personnel_costs']
        proportion = personnel_costs / total_personnel_costs_all_wps if total_personnel_costs_all_wps > 0 else 0

        distributed_costs = {key: value * proportion for key, value in itemized_other_costs.items()}
        total_distributed_other_costs = sum(distributed_costs.values())

        calculated_direct_cost = personnel_costs + total_distributed_other_costs
        adjustment = target_direct_costs[wp] - calculated_direct_cost

        if adjustment < 0:
            # Distribute negative adjustment proportionally across positive cost items
            positive_costs_sum = sum(distributed_costs.values())
            if positive_costs_sum > 0:
                for key in distributed_costs:
                    distributed_costs[key] += adjustment * (distributed_costs[key] / positive_costs_sum)
        else:
            distributed_costs['Other'] += adjustment

        wp_budgets[wp]['other_costs_breakdown'] = distributed_costs
        wp_budgets[wp]['direct_costs'] = target_direct_costs[wp]
        wp_budgets[wp]['indirect_costs'] = target_direct_costs[wp] * 0.25
        wp_budgets[wp]['total_costs'] = wp_budgets[wp]['direct_costs'] + wp_budgets[wp]['indirect_costs']

    return wp_budgets

def generate_latex_tables(budgets):
    role_to_category = {
        'Principal Investigator': 'SENIOR SCIENTISTS', 'Senior Researcher': 'SENIOR SCIENTISTS',
        'Clinical Investigator/Consultant': 'SENIOR SCIENTISTS', 'Mathematician': 'SENIOR SCIENTISTS',
        'Data Scientist': 'SENIOR SCIENTISTS', 'Programmer': 'TECHNICAL PERSONNEL',
        'Technician': 'TECHNICAL PERSONNEL', 'Project Manager': 'TECHNICAL PERSONNEL',
        'Secretary': 'TECHNICAL PERSONNEL', 'Student/Research Assistant': 'TECHNICAL PERSONNEL'
    }

    wp_names = {
        'WP1': 'Data Curation', 'WP2': 'Supervisor Models', 'WP3': 'Causal VAE',
        'WP4': 'Temporal Modeling', 'WP5': 'Validation', 'WP6': 'Dissemination',
        'WP7': 'Project Management', 'WP8': 'Portfolio Activities', 'WP9': 'Compliance'
    }

    latex_string = ""
    for wp_name, budget_data in budgets.items():
        costs = {
            'SENIOR SCIENTISTS': 0, 'JUNIOR SCIENTISTS': 0, 'TECHNICAL PERSONNEL': 0,
            'ADMINISTRATIVE PERSONNEL': 0, 'OTHERS': 0
        }
        for role, cost in budget_data['personnel_breakdown'].items():
            category = role_to_category.get(role, 'OTHERS')
            costs[category] += cost
        for category in costs:
            costs[category] = f"{costs[category]:.2f}"

        total_personnel = budget_data['personnel_costs']
        total_direct = budget_data['direct_costs']
        indirect_costs = budget_data['indirect_costs']
        total_costs = budget_data['total_costs']
        other_costs_breakdown = budget_data['other_costs_breakdown']

        table_string = f"""
\\begin{{table}}[H]
\\centering
\\caption{{Budget for {wp_names.get(wp_name, wp_name)}}}
\\begin{{tabular}}{{|l|l|l|l|}}
\\hline
00000001-0001-0001-0001-000000000001 & Otto-von-Guericke-Universität Magdeburg & ~ & ~ \\\\ \\hline
COST CATEGORY & ITEMS & COST PER ITEM & BE TOTAL COSTS \\\\ \\hline
~ & ~ & ~ & ~ \\\\ \\hline
COSTS WORK PACKAGE {wp_name.split('P')[1]}: {wp_names.get(wp_name, wp_name)} & ~ & ~ & ~ \\\\ \\hline
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
C. DIRECT PURCHASE COSTS & ~ & ~ & {sum(other_costs_breakdown.values()):.2f} \\\\ \\hline
C.1 Travel and subsistence & ~ &  & {other_costs_breakdown['Travel and subsistence']:.2f} \\\\ \\hline
C.2 Equipment (complete 'Depreciation costs' sheet) & ~ & ~ & ~ \\\\ \\hline
Equipment & ~ &  & 0.00 \\\\ \\hline
Infrastructure & ~ &  & 0.00 \\\\ \\hline
Other assets & ~ &  & 0.00 \\\\ \\hline
C.3 Other goods, works and services & ~ & ~ & ~ \\\\ \\hline
Consumables & ~ &  & 0.00 \\\\ \\hline
Services for meetings, seminars & ~ & ~ & 0.00 \\\\ \\hline
Services for dissemination activities (including website) & ~ & ~ & 0.00 \\\\ \\hline
Publication fees & ~ & 15000 & {other_costs_breakdown['Publication fees']:.2f} \\\\ \\hline
Other (shipment, insurance, translation, etc.) & ~ & 0 & {other_costs_breakdown['Other']:.2f} \\\\ \\hline
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

def generate_csv(budgets):
    """
    Generates a CSV file from the calculated budget data.
    """
    csv_data = []
    for wp_name, budget_data in budgets.items():
        csv_data.append({
            'Work Package': wp_name,
            'Direct Costs': f"{budget_data['direct_costs']:.2f}",
            'Indirect Costs': f"{budget_data['indirect_costs']:.2f}",
            'Total Cost': f"{budget_data['total_costs']:.2f}"
        })
    df = pd.DataFrame(csv_data)
    df.to_csv('budget_per_wp.csv', index=False)

if __name__ == '__main__':
    budgets = calculate_budget()

    # Generate LaTeX and print to stdout
    latex_output = generate_latex_tables(budgets)
    print(latex_output)

    # Generate CSV file
    generate_csv(budgets)
