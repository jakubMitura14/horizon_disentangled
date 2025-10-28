import re

def parse_budget(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # --- Personnel Costs ---
    personnel_costs = []
    # Regex to find lines that are table rows but NOT the blue total rows.
    # It looks for lines containing '&' and ending with '\\' but uses a negative lookahead
    # to exclude lines containing the color code for the total rows.
    rows = re.findall(r'^(?!.*rowcolor\[HTML\]\{0070C0\}).*&.*\\\\$', content, re.MULTILINE)

    for row in rows:
        # For each valid row, find all Euro values. The one we want is the last one in the line.
        found_values = re.findall(r'([\d\.,]+)\s*€', row)

        if found_values:
            # The last numerical value found corresponds to the "andere Förderer" column.
            cost_str = found_values[-1]
            # Convert German number format (e.g., "1.234,56") to float
            cost = float(cost_str.replace('.', '').replace(',', '.'))
            personnel_costs.append(cost)

    total_personnel_costs = sum(personnel_costs)

    # --- Other Direct Costs ---
    other_costs = []
    # Isolate the content of the "Other costs" table
    other_costs_section_match = re.search(r'\\section\{Other costs \}(.*?)\\end\{table\}', content, re.DOTALL)
    if other_costs_section_match:
        other_costs_content = other_costs_section_match.group(1)
        # Find all numbers that are clearly costs (e.g., 4+ digits)
        found_other_costs = re.findall(r'(\d{4,})', other_costs_content)
        for cost_str in found_other_costs:
            other_costs.append(float(cost_str))

    total_other_direct_costs = sum(other_costs)

    # --- Final Calculations ---
    total_direct_costs = total_personnel_costs + total_other_direct_costs
    indirect_costs = total_direct_costs * 0.25
    total_budget = total_direct_costs * 1.25

    print("--- Budget Calculation Results ---")
    print(f"Total Personnel Costs:      {total_personnel_costs:15,.2f} €")
    print(f"Total Other Direct Costs:   {total_other_direct_costs:15,.2f} €")
    print("-----------------------------------------")
    print(f"Total Direct Costs:         {total_direct_costs:15,.2f} €")
    print(f"Indirect Costs (25%):       {indirect_costs:15,.2f} €")
    print("=========================================")
    print(f"Total Project Budget:       {total_budget:15,.2f} €")
    print("-----------------------------------------")
    print("(These numbers will be used to update main_horizon.tex)")

if __name__ == "__main__":
    parse_budget('budget_tables.tex')
