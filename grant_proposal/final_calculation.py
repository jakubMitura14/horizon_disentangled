import pandas as pd
from allocation_logic import get_final_allocations

def calculate_reconciled_budget(target_total):
    df_alloc_y1, df_alloc_y2, df_alloc_y3 = get_final_allocations()
    df_alloc_total = df_alloc_y1.add(df_alloc_y2, fill_value=0).add(df_alloc_y3, fill_value=0)
    wp_person_months = df_alloc_total.sum()
    total_person_months = wp_person_months.sum()

    total_other_direct_costs = 278900.00
    indirect_rate = 0.25

    total_direct_costs = target_total / (1 + indirect_rate)
    adjusted_personnel_cost = total_direct_costs - total_other_direct_costs
    indirect_costs = total_direct_costs * indirect_rate

    avg_cost_per_pm = adjusted_personnel_cost / total_person_months
    wp_personnel_costs = wp_person_months * avg_cost_per_pm
    wp_personnel_cost_proportion = wp_personnel_costs / adjusted_personnel_cost

    wp_other_direct_costs = wp_personnel_cost_proportion * total_other_direct_costs
    wp_indirect_costs = wp_personnel_cost_proportion * indirect_costs

    wp_direct_costs_final = wp_personnel_costs + wp_other_direct_costs
    wp_total_costs_final = wp_direct_costs_final + wp_indirect_costs

    return wp_direct_costs_final, wp_indirect_costs, wp_total_costs_final, wp_person_months

if __name__ == "__main__":
    TARGET_BUDGET = 3998114.25

    direct_costs, indirect_costs, total_costs, pm_per_wp = calculate_reconciled_budget(TARGET_BUDGET)

    wp_names = {
        'WP1': 'Data Curation', 'WP2': 'Supervisor Models', 'WP3': 'Causal VAE',
        'WP4': 'Temporal Modeling', 'WP5': 'Validation', 'WP6': 'Dissemination',
        'WP7': 'Project Management', 'WP8': 'Portfolio Activities', 'WP9': 'Compliance'
    }

    print("--- Final Person-Months Table ---")
    pm_header = " & ".join([f"\\textbf{{{wp}}}" for wp in wp_names.keys()])
    pm_values = " & ".join([str(int(pm_per_wp[wp])) for wp in wp_names.keys()])
    print(f"& {pm_header} & \\textbf{{Total}} \\\\")
    print(f"\\textbf{{Person-Months}} & {pm_values} & \\textbf{{{int(pm_per_wp.sum())}}} \\\\")


    print("\n--- Final LaTeX Budget Table (with WP9) ---")
    for wp_code in wp_names.keys():
        direct = direct_costs.get(wp_code, 0)
        indirect = indirect_costs.get(wp_code, 0)
        total = total_costs.get(wp_code, 0)
        print(f"{wp_code}: {wp_names[wp_code]} & {direct:,.2f} & {indirect:,.2f} & {total:,.2f} \\\\")

    print("\\midrule")
    print(f"\\textbf{{Total}} & \\textbf{{{direct_costs.sum():,.2f}}} & \\textbf{{{indirect_costs.sum():,.2f}}} & \\textbf{{{total_costs.sum():,.2f}}} \\\\")
