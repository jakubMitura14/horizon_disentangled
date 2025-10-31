import pandas as pd
import json
import sys
import os

def run_final_verification_on_aggregated_csv():
    """
    Final verification script, updated to handle the fully aggregated CSV format.
    """
    TOLERANCE = 0.05
    GROUND_TRUTH_GRAND_TOTAL_DIRECT = 3198491.40

    script_dir = os.path.dirname(__file__)

    with open(os.path.join(script_dir, 'parsed_tex_data.json'), 'r') as f:
        tex_data = json.load(f)

    ground_truth_pms = tex_data.get('person_months_per_wp', {})
    ground_truth_direct_costs = tex_data.get('direct_costs_per_wp', {})

    try:
        df_generated = pd.read_csv(os.path.join(script_dir, 'detailed_wp_budgets.csv'))
    except FileNotFoundError:
        print("ERROR: 'detailed_wp_budgets.csv' not found.")
        sys.exit(1)

    # --- Verification a: Person-Months ---
    print("--- Verification Step 1: Person-Months per Work Package ---")

    personnel_categories = [
        'SENIOR SCIENTISTS (or equivalent in the private sector)',
        'TECHNICAL PERSONNEL (or equivalent in the private sector)',
        'ADMINISTRATIVE PERSONNEL (or equivalent in the private sector)'
    ]
    df_personnel = df_generated[df_generated['COST CATEGORY'].isin(personnel_categories)].copy()

    if df_personnel.empty:
        print("CRITICAL ERROR: No personnel rows found in the CSV. Filtering has failed.")
        sys.exit(1)

    df_personnel['ITEMS'] = pd.to_numeric(df_personnel['ITEMS'], errors='coerce').fillna(0)
    generated_pms = df_personnel.groupby('Work Package')['ITEMS'].sum().to_dict()

    all_pms_ok = True
    for wp, truth_pm in ground_truth_pms.items():
        gen_pm = generated_pms.get(wp, 0)
        if abs(truth_pm - gen_pm) > TOLERANCE:
            print(f"FAILED: {wp} PMs do not match. Ground Truth: {truth_pm:.2f}, Generated: {gen_pm:.2f}")
            all_pms_ok = False
        else:
            print(f"OK: {wp} PMs match. Ground Truth: {truth_pm:.2f}, Generated: {gen_pm:.2f}")

    if all_pms_ok:
        print("✅ SUCCESS: All Work Package Person-Months match the ground truth.")
    else:
        print("❌ FAILED: Person-Month verification failed.")

    print("-" * 60)

    # --- Verification b & c (Cost checks) ---
    print("--- Verification Step 2: Total DIRECT Cost per Work Package ---")
    df_generated['BE TOTAL COSTS'] = pd.to_numeric(df_generated['BE TOTAL COSTS'], errors='coerce').fillna(0)
    generated_direct_costs = df_generated.groupby('Work Package')['BE TOTAL COSTS'].sum().to_dict()
    all_costs_ok = True
    for wp, truth_cost in ground_truth_direct_costs.items():
        gen_cost = generated_direct_costs.get(wp, 0)
        if abs(truth_cost - gen_cost) > TOLERANCE:
            print(f"FAILED: {wp} Total Direct Cost does not match. Ground Truth: {truth_cost:,.2f}, Generated: {gen_cost:,.2f}")
            all_costs_ok = False
        else:
            print(f"OK: {wp} Total Direct Cost matches. Ground Truth: {truth_cost:,.2f}, Generated: {gen_cost:,.2f}")
    if all_costs_ok:
        print("✅ SUCCESS: All Work Package Total Direct Costs match the ground truth.")
    else:
        print("❌ FAILED: Total Direct Cost verification failed.")
    print("-" * 60)

    print("--- Verification Step 3: Grand Total of All Direct Costs ---")
    calculated_grand_total = df_generated['BE TOTAL COSTS'].sum()
    if abs(GROUND_TRUTH_GRAND_TOTAL_DIRECT - calculated_grand_total) > TOLERANCE:
        print(f"FAILED: Grand Total Direct Cost does not match. Ground Truth: {GROUND_TRUTH_GRAND_TOTAL_DIRECT:,.2f}, Calculated: {calculated_grand_total:,.2f}")
        all_costs_ok = False
    else:
        print(f"OK: Grand Total Direct Cost matches. Ground Truth: {GROUND_TRUTH_GRAND_TOTAL_DIRECT:,.2f}, Calculated: {calculated_grand_total:,.2f}")
        print("✅ SUCCESS: The sum of all generated direct costs matches the project's grand total.")
    print("-" * 60)

    if not all_pms_ok or not all_costs_ok:
        sys.exit(1)

if __name__ == "__main__":
    run_final_verification_on_aggregated_csv()
