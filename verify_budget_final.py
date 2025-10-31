
import pandas as pd
import re

def verify_budget():
    """
    Verifies that the aggregated totals from the detailed CSV match the
    summary figures mentioned in the main_horizon.tex file.
    """
    # --- 1. Read and Aggregate Data from the Detailed CSV ---
    try:
        df = pd.read_csv('detailed_wp_budgets.csv')
    except FileNotFoundError:
        print("Error: 'detailed_wp_budgets.csv' not found. Please generate it first.")
        return

    # Convert cost column to numeric, coercing errors
    df['BE TOTAL COSTS'] = pd.to_numeric(df['BE TOTAL COSTS'], errors='coerce')

    # Calculate aggregated totals from the CSV
    total_personnel_csv = df[df['COST CATEGORY'] == 'A. DIRECT PERSONNEL COSTS']['BE TOTAL COSTS'].sum()
    total_travel_csv = df[df['ITEMS'] == 'C.1 Travel and subsistence']['BE TOTAL COSTS'].sum()
    total_direct_csv = df[df['ITEMS'] == 'TOTAL DIRECT COSTS']['BE TOTAL COSTS'].sum()
    total_indirect_csv = df[df['ITEMS'] == 'INDIRECT COSTS']['BE TOTAL COSTS'].sum()
    total_project_csv = df[df['ITEMS'] == 'TOTAL COSTS']['BE TOTAL COSTS'].sum()


    # --- 2. Extract Target Totals from main_horizon.tex ---
    try:
        with open('main_horizon.tex', 'r', encoding='utf-8') as f:
            tex_content = f.read()
    except FileNotFoundError:
        print("Error: 'main_horizon.tex' not found.")
        return

    def extract_cost(pattern):
        match = re.search(pattern, tex_content)
        if match:
            # Extract number, remove commas/currency symbols, and convert to float
            return float(match.group(1).replace(',', '').replace('\\EUR{', '').replace('}', ''))
        return 0.0

    target_personnel = extract_cost(r"Total Estimated Personnel Costs: \\EUR\{([\d,.]+)\}")
    target_travel = extract_cost(r"Total Estimated Travel Costs: \\EUR\{([\d,.]+)\}")
    target_direct = extract_cost(r"Total Direct Costs \(A\): \\EUR\{([\d,.]+)\}")
    target_indirect = extract_cost(r"Indirect Costs \(B = 25\\% of A\): \\EUR\{([\d,.]+)\}")
    target_total = extract_cost(r"Total Estimated Project Cost \(A \+ B\): \\EUR\{([\d,.]+)\}")

    # --- 3. Compare and Print Verification Report ---
    results = {
        "Total Personnel Costs": (total_personnel_csv, target_personnel),
        "Total Travel Costs": (total_travel_csv, target_travel),
        "Total Direct Costs": (total_direct_csv, target_direct),
        "Total Indirect Costs": (total_indirect_csv, target_indirect),
        "Total Project Cost": (total_project_csv, target_total),
    }

    print("--- Budget Verification Report ---")
    all_match = True
    for category, (csv_val, tex_val) in results.items():
        match = "MATCH" if abs(csv_val - tex_val) < 0.01 else "MISMATCH"
        if match == "MISMATCH":
            all_match = False
        print(f"{category:<25} | CSV Total: {csv_val:12.2f} | TeX Total: {tex_val:12.2f} | Status: {match}")

    print("\n--- Verification Summary ---")
    if all_match:
        print("SUCCESS: All aggregated costs from the CSV match the totals in main_horizon.tex.")
    else:
        print("FAILURE: One or more aggregated costs from the CSV do not match the totals in main_horizon.tex.")


if __name__ == '__main__':
    verify_budget()
