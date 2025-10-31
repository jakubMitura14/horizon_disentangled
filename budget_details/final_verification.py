import pandas as pd
import json
import sys
import os

def run_final_verification():
    """
    Performs the final verification of the generated budget CSV against ground truth data.
    """
    TOLERANCE = 0.05  # 5 cents

    script_dir = os.path.dirname(__file__)

    # --- 1. Load Ground Truth Data ---
    with open(os.path.join(script_dir, 'parsed_tex_data.json'), 'r') as f:
        tex_data = json.load(f)

    ground_truth_pms = tex_data.get('person_months_per_wp', {})
    ground_truth_costs = tex_data.get('budget_per_wp', {})

    # --- 2. Load Generated Data ---
    try:
        df_generated = pd.read_csv(os.path.join(script_dir, 'detailed_wp_budgets.csv'))
    except FileNotFoundError:
        print("ERROR: The file 'detailed_wp_budgets.csv' was not found. Please generate it first.")
        sys.exit(1)

    # --- 3. Perform Verification a: Person-Months ---
    print("--- Verification Step 1: Person-Months per Work Package ---")

    # Calculate PMs from the generated CSV
    df_personnel = df_generated[df_generated['COST CATEGORY'] == 'A. DIRECT PERSONNEL COSTS'].copy()
    # The 'ITEMS' column is the PMs, but it's a string. Convert to numeric.
    df_personnel['ITEMS'] = pd.to_numeric(df_personnel['ITEMS'], errors='coerce').fillna(0)
    generated_pms = df_personnel.groupby('Work Package')['ITEMS'].sum().to_dict()

    all_pms_ok = True
    all_wps = sorted(list(set(ground_truth_pms.keys()) | set(generated_pms.keys())))

    for wp in all_wps:
        truth_pm = ground_truth_pms.get(wp, 0)
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

    # --- 4. Perform Verification b: Total Cost per Work Package ---
    print("--- Verification Step 2: Total Cost per Work Package ---")

    # Calculate total direct costs from CSV
    df_generated['BE TOTAL COSTS'] = pd.to_numeric(df_generated['BE TOTAL COSTS'], errors='coerce').fillna(0)
    generated_direct_costs = df_generated.groupby('Work Package')['BE TOTAL COSTS'].sum()

    # Add 25% indirect costs to get the total
    generated_total_costs = (generated_direct_costs * 1.25).to_dict()

    all_costs_ok = True
    all_wps_costs = sorted(list(set(ground_truth_costs.keys()) | set(generated_total_costs.keys())))

    for wp in all_wps_costs:
        truth_cost = ground_truth_costs.get(wp, 0)
        gen_cost = generated_total_costs.get(wp, 0)

        if abs(truth_cost - gen_cost) > TOLERANCE:
            print(f"FAILED: {wp} Total Cost does not match. Ground Truth: {truth_cost:,.2f}, Generated: {gen_cost:,.2f}")
            all_costs_ok = False
        else:
            print(f"OK: {wp} Total Cost matches. Ground Truth: {truth_cost:,.2f}, Generated: {gen_cost:,.2f}")

    if all_costs_ok:
        print("✅ SUCCESS: All Work Package Total Costs match the ground truth.")
    else:
        print("❌ FAILED: Total Cost verification failed.")

    print("-" * 60)

    if not all_pms_ok or not all_costs_ok:
        sys.exit(1) # Exit with an error code if any verification fails

if __name__ == "__main__":
    run_final_verification()
