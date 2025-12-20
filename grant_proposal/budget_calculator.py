#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json

# --- Configuration ---
# Based on the user's input and the main_horizon.tex file
PERSONNEL_DATA = {
    # Role: [Cost for entire project duration for given FTEs, total Person-Months for given FTEs, FTE count]
    'Principal Investigator': [263141, 36, 1],
    'Senior Researcher': [789423, 108, 3],
    'PhD Student': [513000, 108, 3],
    'Mathematician': [263141, 36, 1],
    'Data Scientist': [263141, 36, 1],
    'Programmer': [405966, 72, 2],
    'Project Manager': [202984, 36, 1],
    'Secretary': [121790, 36, 1],
    'Student/Research Assistant': [33040, 36, 1] # Note: This is for one part-time person over 36 months
}

# Values for other direct costs extracted from main_horizon.tex (as of last update)
OTHER_DIRECT_COSTS = {
    'Travel': 32500,
    'Publications': 10000, # Deduced from the total in the .tex file
    'UK Biobank Access': 12000,
    'Software Licenses': 36000,
    'Long-Term Data Storage': 75800,
    'Other Work-Related Expenses': 100000,
    'External Expert Consultations': 15000
}

TOTAL_BUDGET_CEILING = 4000000
INDIRECT_COST_RATE = 0.25

# --- Step 1: Calculate cost per person-month and per FTE-year ---
print("--- Step 1: Cost Analysis per Role ---")
# Using a dictionary to store the calculated costs to avoid recalculating
cost_per_pm = {}
for role, data in PERSONNEL_DATA.items():
    cost, pms, fte = data
    cost_per_pm[role] = cost / pms
    cost_per_year = (cost / pms) * 12
    print(f"{role:<28}: €{cost_per_year:,.2f} per year")
print("-" * 50)


# --- Step 2: Update personnel list as per user request ---
print("\n--- Step 2: Updating Personnel Counts ---")
# Keep 3 Senior Researchers
PERSONNEL_DATA['Senior Researcher'][2] = 3
# Set to 2 PhD students
PERSONNEL_DATA['PhD Student'][2] = 2

print("Updated counts:")
print(f"Senior Researchers: {PERSONNEL_DATA['Senior Researcher'][2]}")
print(f"PhD Students: {PERSONNEL_DATA['PhD Student'][2]} (for 24 months each)")
print("-" * 50)


# --- Step 3: Recalculate Total Personnel Costs ---
print("\n--- Step 3: Recalculating Total Personnel Costs ---")
total_personnel_cost = 0
for role, data in PERSONNEL_DATA.items():
    fte_count = data[2]

    # Default duration is 36 months, but PhDs are for 24 months
    duration_in_months = 36
    if role == 'PhD Student':
        duration_in_months = 24

    cost_per_fte_for_duration = cost_per_pm[role] * duration_in_months
    role_total_cost = cost_per_fte_for_duration * fte_count
    total_personnel_cost += role_total_cost
    print(f"{role:<28}: {fte_count} FTE(s) for {duration_in_months} months = €{role_total_cost:,.2f}")

print(f"\n{'NEW TOTAL PERSONNEL COST:':<28} €{total_personnel_cost:,.2f}")
print("-" * 50)


# --- Step 4: Calculate Other Direct Costs and Indirect Costs ---
print("\n--- Step 4: Calculating Other Costs ---")
total_other_direct_costs = sum(OTHER_DIRECT_COSTS.values())
print("Itemized Other Direct Costs:")
for item, cost in OTHER_DIRECT_COSTS.items():
    print(f"- {item:<27}: €{cost:,.2f}")
print(f"\n{'TOTAL OTHER DIRECT COSTS:':<28} €{total_other_direct_costs:,.2f}")

# This is the subtotal of all direct costs EXCEPT for compute
subtotal_direct_costs = total_personnel_cost + total_other_direct_costs
print(f"{'SUBTOTAL DIRECT COSTS (w/o Compute):':<35} €{subtotal_direct_costs:,.2f}")

# Calculate indirect costs on this subtotal
subtotal_indirect_costs = subtotal_direct_costs * INDIRECT_COST_RATE
print(f"{'INDIRECT COSTS (on Subtotal):':<35} €{subtotal_indirect_costs:,.2f}")
print("-" * 50)


# --- Step 5: Calculate Remaining Budget for Compute ---
print("\n--- Step 5: Calculating Remaining Budget for Compute ---")
# The total budget ceiling includes its own indirect costs.
# TotalBudget = (DirectCosts) + (DirectCosts * 0.25)
# TotalBudget = DirectCosts * 1.25
# So, TotalDirectCosts = TotalBudget / 1.25
total_direct_costs_allowed = TOTAL_BUDGET_CEILING / (1 + INDIRECT_COST_RATE)

print(f"{'Total Budget Ceiling:':<35} €{TOTAL_BUDGET_CEILING:,.2f}")
print(f"{'Max Allowable Total Direct Costs:':<35} €{total_direct_costs_allowed:,.2f}")

# The remaining amount for direct compute cost is the difference
direct_compute_cost = total_direct_costs_allowed - subtotal_direct_costs
print(f"{'Subtotal Direct Costs (from Step 4):':<35} €{subtotal_direct_costs:,.2f}")
print(f"\n{'REMAINING FOR DIRECT COMPUTE COSTS:':<35} €{direct_compute_cost:,.2f}")

# The indirect cost associated with compute is
compute_indirect_cost = direct_compute_cost * INDIRECT_COST_RATE
print(f"{'Indirect Costs on Compute (@25%):':<35} €{compute_indirect_cost:,.2f}")
print(f"{'Total Compute Cost (Direct + Indirect):':<35} €{direct_compute_cost + compute_indirect_cost:,.2f}")
print("-" * 50)


# --- Final Summary ---
print("\n--- FINAL BUDGET SUMMARY ---")
final_total_direct_costs = subtotal_direct_costs + direct_compute_cost
final_total_indirect_costs = subtotal_indirect_costs + compute_indirect_cost
final_total_cost = final_total_direct_costs + final_total_indirect_costs

print(f"{'Total Personnel Costs:':<25} €{total_personnel_cost:,.2f}")
print(f"{'Total Other Direct Costs:':<25} €{total_other_direct_costs:,.2f}")
print(f"{'Direct Compute Costs:':<25} €{direct_compute_cost:,.2f}")
print("="*40)
print(f"{'TOTAL DIRECT COSTS:':<25} €{final_total_direct_costs:,.2f}")
print(f"{'TOTAL INDIRECT COSTS:':<25} €{final_total_indirect_costs:,.2f}")
print("="*40)
print(f"{'TOTAL PROJECT COST:':<25} €{final_total_cost:,.2f}")
print(f"({round((final_total_cost/TOTAL_BUDGET_CEILING)*100, 2)}% of €{TOTAL_BUDGET_CEILING:,.2f} ceiling)")

# --- Verification Step ---
print("\n--- VERIFICATION ---")
manual_total = total_personnel_cost + total_other_direct_costs + direct_compute_cost + final_total_indirect_costs
assert round(manual_total) == TOTAL_BUDGET_CEILING, f"Verification failed! Manual total {manual_total} does not match ceiling {TOTAL_BUDGET_CEILING}"
print(f"✅ Verification successful: Manual sum of all costs is €{manual_total:,.2f}, matching the €{TOTAL_BUDGET_CEILING:,.2f} ceiling.")
