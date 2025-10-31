import re
import json
import os

def parse_main_horizon():
    """
    Parses main_horizon.tex to extract ground truth budget data. This version
    now extracts the DIRECT COSTS per WP as the ground truth.
    """
    with open('main_horizon.tex', 'r', encoding='utf-8') as f:
        content = f.read()

    data = {}

    # --- 1. Parse Person-Months per Work Package ---
    pm_table_match = re.search(r'\\caption\{Total Person-Months per Work Package\.\}.*?\\begin{tabular}{.*?}(.*?)\\end{tabular}', content, re.DOTALL)
    if pm_table_match:
        table_content = pm_table_match.group(1)
        header_line_match = re.search(r'\\toprule\s*&(.+?)\s*\\\\', table_content)
        value_line_match = re.search(r'Person-Months}\s*&(.+?)\s*\\\\', table_content)

        if header_line_match and value_line_match:
            headers = [re.sub(r'\\textbf\{|\}', '', h).strip() for h in header_line_match.group(1).split('&')]
            values_str = value_line_match.group(1).split('&')
            values = [float(re.sub(r'\\textbf\{|\}|\\s*', '', v).strip()) for v in values_str]
            data['person_months_per_wp'] = {h: v for h, v in zip(headers, values) if h.startswith('WP')}

    # --- 2. Parse DIRECT COSTS per Work Package from Table 11 ---
    budget_table_match = re.search(r'\\caption\{Estimated Budget Allocation per Work Package\.*?}.*?\\begin{tabular}{.*?}(.*?)\\end{tabular}', content, re.DOTALL)
    if budget_table_match:
        table_content = budget_table_match.group(1)
        direct_costs_per_wp = {}
        # Regex to find a WP row and capture the FIRST number (Direct Costs)
        matches = re.findall(r'^\s*(WP\d+):.*?&\s*([\d,\s]+\.\d{2})\s*&', table_content, re.MULTILINE)
        for wp, direct_cost_str in matches:
            # Clean up the string by removing spaces and commas
            cost = float(direct_cost_str.replace(',', '').replace(' ', ''))
            direct_costs_per_wp[wp] = cost
        data['direct_costs_per_wp'] = direct_costs_per_wp

    # --- 3. Parse Other Direct Costs and Grand Personnel Total ---
    # The logic for this remains the same as it was already correct
    other_costs = {}
    personnel_total_match = re.search(r'Total Estimated Personnel Costs: \\EUR\{([\d,]+\.\d{2})\}', content)
    if personnel_total_match:
        data['total_personnel_cost'] = float(personnel_total_match.group(1).replace(',', ''))

    a4_section_match = re.search(r'\\subsubsection\*\{A\.4 Other Direct Costs\}(.*?)\\begin\{itemize\}(.*?)\\end\{itemize\}', content, re.DOTALL)
    if a4_section_match:
        items_content = a4_section_match.group(2)
        item_matches = re.findall(r'\\item \\textbf\{(.*?):\}\s*\\EUR\{([\d,]+)\}', items_content)
        for name, cost_str in item_matches:
            normalized_name = name.replace('Open Access ', '').replace(' (DIZ)', '').strip()
            other_costs[normalized_name] = float(cost_str.replace(',', ''))

    travel_match = re.search(r'Total Estimated Travel Costs: \\EUR\{([\d,]+)\}', content)
    if travel_match:
        other_costs['Travel'] = float(travel_match.group(1).replace(',', ''))

    # Add the new fixed registration cost
    registration_match = re.search(r'Pre-CE Clinical Investigation.*?\\textbf\{\\EUR\{([\d,]+)\}\}', content)
    if registration_match:
        other_costs['Registration'] = float(registration_match.group(1).replace(',', ''))

    data['other_direct_costs'] = other_costs

    return data

if __name__ == "__main__":
    parsed_data = parse_main_horizon()
    script_dir = os.path.dirname(__file__)
    output_path = os.path.join(script_dir, 'parsed_tex_data.json')
    with open(output_path, 'w') as f:
        json.dump(parsed_data, f, indent=4)
    print(f"Successfully parsed data and saved to {output_path}")

    print("\n--- Parsed Data Verification ---")
    print(json.dumps(parsed_data, indent=2))
    print("--------------------------------")
