
import pandas as pd
import re
from allocation_logic import get_final_allocations

def parse_main_horizon_for_verification():
    """Parses main_horizon.tex to get person-month and WP total cost targets."""
    with open('main_horizon.tex', 'r', encoding='utf-8') as f:
        content = f.read()

    targets = {}

    # --- Person-Months per WP ---
    pm_totals = {}
    table_match = re.search(r"\\begin\{tabular\}\{lccccccccc\|c\}(.*?)\\end\{tabular\}", content, re.DOTALL)
    if not table_match:
        raise ValueError("Could not find the Person-Months per Work Package table.")

    table_content = table_match.group(1)
    pm_row_match = re.search(r"\\textbf\{Person-Months\}\s*&(.*?)\s*\\\\", table_content)
    if not pm_row_match:
        raise ValueError("Could not find the 'Person-Months' row in the table.")

    pm_values = [float(v.strip()) for v in pm_row_match.group(1).split('&')[:-1]]
    for i, pm in enumerate(pm_values, 1):
        pm_totals[f"WP{i}"] = pm
    targets['pm_totals'] = pm_totals

    # --- WP Total Costs ---
    cost_totals = {}
    pattern = r"WP(\d+):.*?&.*?&.*?&\s*([\d,]+\.\d{2})\s*\\\\"
    matches = re.findall(pattern, content)
    if not matches:
        raise ValueError("Could not find the budget allocation table.")
    for wp_num, total_str in matches:
        cost_totals[f"WP{wp_num}"] = float(total_str.replace(',', ''))
    targets['cost_totals'] = cost_totals

    return targets

def final_verification():
    """
    Performs two final checks with a 5-cent tolerance:
    1. Verifies person-month consistency.
    2. Verifies WP total costs.
    """
    errors = []
    TOLERANCE = 0.05 # 5-cent tolerance

    # --- 1. Person-Month Verification ---
    tex_targets = parse_main_horizon_for_verification()

    df_alloc_y1, df_alloc_y2, df_alloc_y3 = get_final_allocations()
    df_alloc_total = df_alloc_y1 + df_alloc_y2 + df_alloc_y3
    script_pm_totals = df_alloc_total.sum().to_dict()

    df_csv = pd.read_csv('detailed_wp_budgets.csv')
    csv_pm_totals = df_csv[df_csv['COST CATEGORY'] == 'A. DIRECT PERSONNEL COSTS'].groupby('Work Package')['ITEMS'].apply(lambda x: pd.to_numeric(x, errors='coerce').sum()).to_dict()

    for wp, tex_pm in tex_targets['pm_totals'].items():
        # Check against source script
        script_pm = script_pm_totals.get(wp, 0.0)
        if abs(tex_pm - script_pm) > 0.01:
             errors.append(f"Person-Month Mismatch (Source vs Tex) for {wp}: TEX={tex_pm}, SCRIPT={script_pm}")
        # Check against final CSV
        csv_pm = csv_pm_totals.get(wp, 0.0)
        if abs(tex_pm - csv_pm) > 0.01:
            errors.append(f"Person-Month Mismatch (CSV vs Tex) for {wp}: TEX={tex_pm}, CSV={csv_pm}")

    # --- 2. WP Total Cost Verification ---
    csv_cost_totals = df_csv[df_csv['ITEMS'] == 'TOTAL COSTS'].groupby('Work Package')['BE TOTAL COSTS'].sum().astype(float).to_dict()

    for wp, tex_cost in tex_targets['cost_totals'].items():
        csv_cost = csv_cost_totals.get(wp)
        if csv_cost is None:
            errors.append(f"WP Total Cost Mismatch: {wp} is in .tex but not in CSV.")
            continue
        if not abs(tex_cost - csv_cost) < TOLERANCE:
            errors.append(f"WP Total Cost Mismatch for {wp}: TEX={tex_cost:.2f}, CSV={csv_cost:.2f}")

    # --- Final Report ---
    if errors:
        print("Final Verification FAILED:")
        for error in errors:
            print(f"- {error}")
    else:
        print("Final Verification PASSED: Person-months and WP totals are correct.")

if __name__ == "__main__":
    try:
        final_verification()
    except (ValueError, FileNotFoundError) as e:
        print(f"An error occurred during verification: {e}")
