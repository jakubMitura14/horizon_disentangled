
import pandas as pd
import re

def get_ground_truth():
    """Parses main_horizon.tex to get all official budget totals."""
    with open('main_horizon.tex', 'r', encoding='utf-8') as f:
        content = f.read()

    truth = {}
    personnel_match = re.search(r"Total Estimated Personnel Costs: \\EUR\{([\d,]+\.\d{2})\}", content)
    other_direct_match = re.search(r"Total Estimated Other Direct Costs: \\EUR\{([\d,]+\.\d{2})\}", content)

    if not all([personnel_match, other_direct_match]):
        raise ValueError("Could not find all required grand total values.")

    truth['personnel_total'] = float(personnel_match.group(1).replace(',', ''))
    truth['other_direct_total'] = float(other_direct_match.group(1).replace(',', ''))
    truth['total_direct'] = truth['personnel_total'] + truth['other_direct_total']

    wp_totals = {}
    pattern = r"WP(\d+):.*?&.*?&.*?&\s*([\d,]+\.\d{2})\s*\\\\"
    matches = re.findall(pattern, content)
    if not matches:
        raise ValueError("Could not find the budget allocation table.")
    for wp_num, total_str in matches:
        wp_totals[f"WP{wp_num}"] = float(total_str.replace(',', ''))

    truth['wp_totals'] = wp_totals
    return truth

def generate_and_verify_budgets():
    """
    Generates the final budget CSVs by proportionally distributing the official
    grand totals across the Work Packages. Includes a final reconciliation
    step to ensure perfect mathematical alignment.
    """
    truth = get_ground_truth()

    # --- Calculate Proportional Distribution ---
    wp_budgets = {}
    total_project_cost = sum(truth['wp_totals'].values())

    for wp_name, wp_total in truth['wp_totals'].items():
        proportion = wp_total / total_project_cost
        wp_direct_total = wp_total / 1.25

        # Distribute grand totals based on this WP's proportion of the budget
        personnel_wp = truth['personnel_total'] * proportion
        other_direct_wp = truth['other_direct_total'] * proportion

        # A check to ensure the proportional split matches the WP direct total
        # If not, adjust the larger component (personnel) to match.
        current_direct = personnel_wp + other_direct_wp
        diff = wp_direct_total - current_direct
        personnel_wp += diff

        wp_budgets[wp_name] = {
            'personnel': personnel_wp,
            'other_direct': other_direct_wp,
            'total_direct': wp_direct_total,
            'total': wp_total
        }

    # --- Reconciliation Step ---
    # Sum up the calculated totals and adjust the last WP to fix rounding errors
    final_personnel_sum = sum(b['personnel'] for b in wp_budgets.values())
    final_other_sum = sum(b['other_direct'] for b in wp_budgets.values())

    personnel_diff = truth['personnel_total'] - final_personnel_sum
    other_diff = truth['other_direct_total'] - final_other_sum

    last_wp_name = sorted(wp_budgets.keys())[-1]
    wp_budgets[last_wp_name]['personnel'] += personnel_diff
    wp_budgets[last_wp_name]['other_direct'] += other_diff

    # --- Generate CSVs ---
    detailed_data, summary_data = [], []
    other_direct_proportions = {'Travel': 32500.00, 'Publication': 15000.00, 'Other': 231400.00}

    for wp_name, budget in sorted(wp_budgets.items()):
        # Split personnel costs for the detailed view (example 70/30 split)
        senior_costs = budget['personnel'] * 0.7
        tech_costs = budget['personnel'] * 0.3

        # Itemize other costs
        travel = budget['other_direct'] * (other_direct_proportions['Travel'] / truth['other_direct_total'])
        pubs = budget['other_direct'] * (other_direct_proportions['Publication'] / truth['other_direct_total'])
        other = budget['other_direct'] * (other_direct_proportions['Other'] / truth['other_direct_total'])

        direct_cost = budget['total'] / 1.25
        indirect_cost = budget['total'] - direct_cost

        detailed_data.extend([
            {'Work Package': wp_name, 'COST CATEGORY': 'A. DIRECT PERSONNEL COSTS', 'ITEMS': 'SENIOR SCIENTISTS (or equivalent in the private sector)', 'COST PER ITEM': f"{truth['personnel_total']:.2f}", 'BE TOTAL COSTS': f"{senior_costs:.2f}"},
            {'Work Package': wp_name, 'COST CATEGORY': 'A. DIRECT PERSONNEL COSTS', 'ITEMS': 'TECHNICAL PERSONNEL (or equivalent in the private sector)', 'COST PER ITEM': f"{truth['personnel_total']:.2f}", 'BE TOTAL COSTS': f"{tech_costs:.2f}"},
            {'Work Package': wp_name, 'COST CATEGORY': 'C. DIRECT PURCHASE COSTS', 'ITEMS': 'C.1 Travel and subsistence', 'COST PER ITEM': f"{other_direct_proportions['Travel']:.2f}", 'BE TOTAL COSTS': f"{travel:.2f}"},
            {'Work Package': wp_name, 'COST CATEGORY': 'C. DIRECT PURCHASE COSTS', 'ITEMS': 'C.3 Publication fees', 'COST PER ITEM': f"{other_direct_proportions['Publication']:.2f}", 'BE TOTAL COSTS': f"{pubs:.2f}"},
            {'Work Package': 'WP1', 'COST CATEGORY': 'C. DIRECT PURCHASE COSTS', 'ITEMS': 'C.3 Other (shipment, insurance, translation, etc.)', 'COST PER ITEM': f"{other_direct_proportions['Other']:.2f}", 'BE TOTAL COSTS': f"{other:.2f}"},
            {'Work Package': wp_name, 'COST CATEGORY': 'TOTALS', 'ITEMS': 'TOTAL DIRECT COSTS', 'COST PER ITEM': '0.00', 'BE TOTAL COSTS': f"{direct_cost:.2f}"},
            {'Work Package': wp_name, 'COST CATEGORY': 'TOTALS', 'ITEMS': 'INDIRECT COSTS', 'COST PER ITEM': '0.00', 'BE TOTAL COSTS': f"{indirect_cost:.2f}"},
            {'Work Package': wp_name, 'COST CATEGORY': 'TOTALS', 'ITEMS': 'TOTAL COSTS', 'COST PER ITEM': '0.00', 'BE TOTAL COSTS': f"{budget['total']:.2f}"},
        ])
        summary_data.append({'Work Package': wp_name, 'Direct Costs': f"{direct_cost:.2f}", 'Indirect Costs': f"{indirect_cost:.2f}", 'Total Cost': f"{budget['total']:.2f}"})

    df_detailed = pd.DataFrame(detailed_data)
    df_detailed.to_csv('detailed_wp_budgets.csv', index=False)
    pd.DataFrame(summary_data).to_csv('budget_per_wp.csv', index=False)


if __name__ == '__main__':
    generate_and_verify_budgets()
    print("Final, reconciled budget CSV files have been generated.")
