import pandas as pd

# Data manually extracted from budget_tables.tex
# All values are from the "andere Förderer" column

# Main Personnel Costs
main_personnel_2027 = {
    'Investigator 1': 42320.15,
    'Investigator 2': 42320.15,
    'Investigator 3': 42320.15,
    'Project Manager': 42320.15,
    'Mathematician': 90410.93,
    'Data Scientist': 90410.93,
    'Secretary': 62674.92,
    'Senior Researcher 1': 90410.93,
    'Senior Researcher 2': 90410.93,
    'PI': 99448.89,
}

main_personnel_2028 = {
    'Investigator 1': 88872.31,
    'Investigator 2': 88872.31,
    'Investigator 3': 88872.31,
    'Project Manager': 44436.15,
    'Mathematician': 94931.48,
    'Data Scientist': 94931.48,
    'Secretary': 65808.66,
    'Senior Researcher 1': 94931.48,
    'Senior Researcher 2': 94931.48,
    'PI': 104421.33,
}

main_personnel_2029 = {
    'Investigator 1': 49041.11,
    'Investigator 2': 49041.11,
    'Investigator 3': 49041.11,
    'Project Manager': 49041.11,
    'Mathematician': 63935.00,
    'Data Scientist': 127869.99,
    'Secretary': 127897.77,
    'Senior Researcher 1': 105198.96,
    'Senior Researcher 2': 105198.96,
    'PI': 113547.29,
}

# Technical Personnel Costs
tech_personnel_2027 = {
    'Programmer': 68382.89,
    'Technician': 66527.32,
}

tech_personnel_2028 = {
    'Programmer': 71802.04,
    'Technician': 69853.68,
}

tech_personnel_2029 = {
    'Programmer': 75392.14,
    'Technician': 69853.68, # Note: value is the same as 2028 in the table
}


# Student Assistant Costs
student_2027 = {'Assistant': 32961.20}
student_2028 = {'Assistant': 34609.26}
student_2029 = {'Assistant': 36339.72}

# --- Calculations ---

# Sum personnel costs for each year
total_2027 = sum(main_personnel_2027.values()) + sum(tech_personnel_2027.values()) + sum(student_2027.values())
total_2028 = sum(main_personnel_2028.values()) + sum(tech_personnel_2028.values()) + sum(student_2028.values())
total_2029 = sum(main_personnel_2029.values()) + sum(tech_personnel_2029.values()) + sum(student_2029.values())

# Total Personnel Cost over 3 years
total_personnel_cost = total_2027 + total_2028 + total_2029

# Other Direct Costs (from "Other costs" table)
travel = 32500
pre_ce_investigation = 9900 + 6700 + 20000 + 5000 + 15000 + 12000 + 5000 + 3000
other_direct_items = 15000 + 12000 + 36000 + 75800 + 26000
total_other_direct_costs = travel + pre_ce_investigation + other_direct_items

# Total Direct Costs
total_direct_costs = total_personnel_cost + total_other_direct_costs

# Indirect Costs (25% of Total Direct Costs)
indirect_costs = total_direct_costs * 0.25

# Final Total Project Budget
total_project_budget = total_direct_costs + indirect_costs

# --- Verification Output ---
print("--- Budget Verification based on budget_tables.tex ---")
print(f"Personnel Costs 2027: €{total_2027:,.2f}")
print(f"Personnel Costs 2028: €{total_2028:,.2f}")
print(f"Personnel Costs 2029: €{total_2029:,.2f}")
print("-" * 20)
print(f"Total Personnel Costs (3 years): €{total_personnel_cost:,.2f}")
print(f"Total Other Direct Costs: €{total_other_direct_costs:,.2f}")
print("-" * 20)
print(f"Total Direct Costs: €{total_direct_costs:,.2f}")
print(f"Indirect Costs (25%): €{indirect_costs:,.2f}")
print("-" * 20)
print(f"FINAL TOTAL PROJECT BUDGET: €{total_project_budget:,.2f}")
print("-" * 50)
