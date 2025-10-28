import pandas as pd
from allocation_logic import get_final_allocations

def calculate_reconciled_budget(target_total):
    """
    Calculates all budget figures based on the final allocation and reconciles
    them to the specified target total.
    """
    df_alloc_y1, df_alloc_y2, df_alloc_y3 = get_final_allocations()
    df_alloc_total = df_alloc_y1.add(df_alloc_y2, fill_value=0).add(df_alloc_y3, fill_value=0)
    wp_person_months = df_alloc_total.sum()
    total_person_months = wp_person_months.sum()

    # Define fixed costs
    total_other_direct_costs = 278900.00
    indirect_rate = 0.25

    # Back-calculate to find the adjusted personnel cost
    total_direct_costs = target_total / (1 + indirect_rate)
    adjusted_personnel_cost = total_direct_costs - total_other_direct_costs
    indirect_costs = total_direct_costs * indirect_rate

    # Calculate costs per WP
    avg_cost_per_pm = adjusted_personnel_cost / total_person_months
    wp_personnel_costs = wp_person_months * avg_cost_per_pm
    wp_personnel_cost_proportion = wp_personnel_costs / adjusted_personnel_cost

    wp_other_direct_costs = wp_personnel_cost_proportion * total_other_direct_costs
    wp_indirect_costs = wp_personnel_cost_proportion * indirect_costs

    wp_direct_costs_final = wp_personnel_costs + wp_other_direct_costs
    wp_total_costs_final = wp_direct_costs_final + wp_indirect_costs

    return wp_direct_costs_final, wp_indirect_costs, wp_total_costs_final

if __name__ == "__main__":
    TARGET_BUDGET = 3998114.25

    direct_costs, indirect_costs, total_costs = calculate_reconciled_budget(TARGET_BUDGET)

    wp_names = {
        'WP1': 'Data Curation', 'WP2': 'Supervisor Models', 'WP3': 'Causal VAE',
        'WP4': 'Temporal Modeling', 'WP5': 'Validation', 'WP6': 'Dissemination',
        'WP7': 'Project Management', 'WP8': 'Portfolio Activities', 'WP9': 'Compliance'
    }

    print("--- Final LaTeX Budget Table (with WP9) ---")
    for wp_code, wp_name in wp_names.items():
        direct = direct_costs.get(wp_code, 0)
        indirect = indirect_costs.get(wp_code, 0)
        total = total_costs.get(wp_code, 0)
        print(f"{wp_code}: {wp_name} & {direct:,.2f} & {indirect:,.2f} & {total:,.2f} \\\\")

    print("\\midrule")
    print(f"\\textbf{{Total}} & \\textbf{{{direct_costs.sum():,.2f}}} & \\textbf{{{indirect_costs.sum():,.2f}}} & \\textbf{{{total_costs.sum():,.2f}}} \\\\")
