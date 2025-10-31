
import pandas as pd
import re
from allocation_logic import get_final_allocations

def calculate_and_generate_final_budgets():
    """
    Calculates and generates the final budget CSVs by precisely following the
    user's step-by-step logic for year-by-year cost calculation.
    """
    # Step 1: Use hardcoded annual costs
    annual_costs = {
        'Y1': {'Principal Investigator': 99448.89, 'Senior Researcher': 90410.93, 'Clinical Investigator/Consultant': 84640.30, 'Mathematician': 90410.93, 'Data Scientist': 90410.93, 'Programmer': 68382.89, 'Technician': 66527.32, 'Project Manager': 42320.15, 'Secretary': 62674.92, 'Student/Research Assistant': 32961.20},
        'Y2': {'Principal Investigator': 104421.33, 'Senior Researcher': 94931.48, 'Clinical Investigator/Consultant': 88872.31, 'Mathematician': 94931.48, 'Data Scientist': 94931.48, 'Programmer': 71802.04, 'Technician': 69853.68, 'Project Manager': 44436.15, 'Secretary': 65808.66, 'Student/Research Assistant': 34609.26},
        'Y3': {'Principal Investigator': 113547.29, 'Senior Researcher': 105198.96, 'Clinical Investigator/Consultant': 98082.22, 'Mathematician': 127870.00, 'Data Scientist': 127869.99, 'Programmer': 75392.14, 'Technician': 69853.68, 'Project Manager': 49041.11, 'Secretary': 127897.77, 'Student/Research Assistant': 36339.72}
    }
    role_to_category = {
        'Principal Investigator': 'SENIOR SCIENTISTS', 'Senior Researcher': 'SENIOR SCIENTISTS', 'Clinical Investigator/Consultant': 'SENIOR SCIENTISTS', 'Mathematician': 'SENIOR SCIENTISTS', 'Data Scientist': 'SENIOR SCIENTISTS',
        'Programmer': 'TECHNICAL PERSONNEL', 'Technician': 'TECHNICAL PERSONNEL', 'Project Manager': 'TECHNICAL PERSONNEL', 'Secretary': 'TECHNICAL PERSONNEL', 'Student/Research Assistant': 'TECHNICAL PERSONNEL'
    }

    # Step 2: Get Person-Months
    df_alloc_y1, df_alloc_y2, df_alloc_y3 = get_final_allocations()
    allocations = {'Y1': df_alloc_y1, 'Y2': df_alloc_y2, 'Y3': df_alloc_y3}

    # Steps 3 & 4: Calculate total costs per WP with year-by-year logic
    wp_budgets = {}
    total_personnel_cost_project = 0
    for wp in df_alloc_y1.columns:
        wp_personnel_data = {'SENIOR SCIENTISTS': {'pm': 0.0, 'cost': 0.0}, 'TECHNICAL PERSONNEL': {'pm': 0.0, 'cost': 0.0}}

        for year, df_alloc in allocations.items():
            for role in df_alloc.index:
                pm = df_alloc.loc[role, wp]
                if pm > 0:
                    cost = (annual_costs[year][role] / 12) * pm
                    category = role_to_category[role]
                    wp_personnel_data[category]['pm'] += pm
                    wp_personnel_data[category]['cost'] += cost

        wp_total_personnel = wp_personnel_data['SENIOR SCIENTISTS']['cost'] + wp_personnel_data['TECHNICAL PERSONNEL']['cost']
        total_personnel_cost_project += wp_total_personnel
        wp_budgets[wp] = {'personnel_data': wp_personnel_data, 'total_personnel_cost': wp_total_personnel}

    # Step 7: Populate CSV
    detailed_data, summary_data = [], []
    other_direct_total = 278900.00
    other_direct_proportions = {'Travel': 32500.00, 'Publication': 15000.00, 'Other': 231400.00}

    for wp, budget in wp_budgets.items():
        proportion = budget['total_personnel_cost'] / total_personnel_cost_project if total_personnel_cost_project > 0 else 0
        other_direct_wp = other_direct_total * proportion

        direct_cost = budget['total_personnel_cost'] + other_direct_wp
        indirect_cost = direct_cost * 0.25
        total_cost = direct_cost + indirect_cost

        senior_data = budget['personnel_data']['SENIOR SCIENTISTS']
        tech_data = budget['personnel_data']['TECHNICAL PERSONNEL']

        # Step 6: Get average costs per person month for this specific WP
        avg_cost_senior_wp = senior_data['cost'] / senior_data['pm'] if senior_data['pm'] > 0 else 0
        avg_cost_tech_wp = tech_data['cost'] / tech_data['pm'] if tech_data['pm'] > 0 else 0

        detailed_data.extend([
            {'Work Package': wp, 'COST CATEGORY': 'A. DIRECT PERSONNEL COSTS', 'ITEMS': f"{senior_data['pm']:.2f}", 'COST PER ITEM': f"{avg_cost_senior_wp:.2f}", 'BE TOTAL COSTS': f"{senior_data['cost']:.2f}"},
            {'Work Package': wp, 'COST CATEGORY': 'A. DIRECT PERSONNEL COSTS', 'ITEMS': '0.00', 'COST PER ITEM': '0.00', 'BE TOTAL COSTS': '0.00'}, # Junior
            {'Work Package': wp, 'COST CATEGORY': 'A. DIRECT PERSONNEL COSTS', 'ITEMS': f"{tech_data['pm']:.2f}", 'COST PER ITEM': f"{avg_cost_tech_wp:.2f}", 'BE TOTAL COSTS': f"{tech_data['cost']:.2f}"},
            {'Work Package': wp, 'COST CATEGORY': 'C. DIRECT PURCHASE COSTS', 'ITEMS': '~', 'COST PER ITEM': f"{other_direct_proportions['Travel']:.2f}", 'BE TOTAL COSTS': f"{other_direct_wp * (other_direct_proportions['Travel'] / other_direct_total):.2f}"},
            {'Work Package': wp, 'COST CATEGORY': 'C. DIRECT PURCHASE COSTS', 'ITEMS': '~', 'COST PER ITEM': f"{other_direct_proportions['Publication']:.2f}", 'BE TOTAL COSTS': f"{other_direct_wp * (other_direct_proportions['Publication'] / other_direct_total):.2f}"},
            {'Work Package': wp, 'COST CATEGORY': 'C. DIRECT PURCHASE COSTS', 'ITEMS': '~', 'COST PER ITEM': f"{other_direct_proportions['Other']:.2f}", 'BE TOTAL COSTS': f"{other_direct_wp * (other_direct_proportions['Other'] / other_direct_total):.2f}"},
            {'Work Package': wp, 'COST CATEGORY': 'TOTALS', 'ITEMS': 'TOTAL DIRECT COSTS', 'COST PER ITEM': '~', 'BE TOTAL COSTS': f"{direct_cost:.2f}"},
            {'Work Package': wp, 'COST CATEGORY': 'TOTALS', 'ITEMS': 'INDIRECT COSTS', 'COST PER ITEM': '~', 'BE TOTAL COSTS': f"{indirect_cost:.2f}"},
            {'Work Package': wp, 'COST CATEGORY': 'TOTALS', 'ITEMS': 'TOTAL COSTS', 'COST PER ITEM': '~', 'BE TOTAL COSTS': f"{total_cost:.2f}"},
        ])
        summary_data.append({'Work Package': wp, 'Direct Costs': f"{direct_cost:.2f}", 'Indirect Costs': f"{indirect_cost:.2f}", 'Total Cost': f"{total_cost:.2f}"})

    pd.DataFrame(detailed_data).to_csv('detailed_wp_budgets.csv', index=False)
    pd.DataFrame(summary_data).to_csv('budget_per_wp.csv', index=False)

if __name__ == '__main__':
    calculate_and_generate_final_budgets()
    print("Final budget CSVs generated with precise year-by-year cost calculation.")
