# calculate_totals.py
# This script calculates the final budget totals for the grant proposal.
# The raw data is manually extracted from the 'andere Förderer' column
# in `budget_tables.tex` and the itemized list of other costs.
# This script serves as a verifiable record of the calculation.

# --- Manually Extracted Data ---

personnel_costs = {
    "main_2027": 693048.10,
    "main_2028": 861008.97,
    "main_2029": 839812.41,
    "tech_2027": 134910.21,
    "tech_2028": 141655.72,
    "tech_2029": 145245.82,
    "student_2027": 32961.20,
    "student_2028": 34609.26,
    "student_2029": 36339.72,
}

other_direct_costs_items = {
    # A.2 Travel Costs
    "Project meetings and conferences": 32500,
    # A.3 Pre-CE Clinical Investigation
    "Regulatory Authorisation Fees (BfArM)": 9900,
    "Ethics Committee Review (Medical Faculty OVGU)": 6700,
    "Subject Insurance (Trial-participant)": 20000,
    "Legal and Administrative Services (KKS)": 5000,
    # Operational Study Costs
    "Sub-item: Clinical departments (OVGU/UMMD)": 15000,
    "Sub-item: KKS Magdeburg (Monitoring, safety, GCP)": 12000,
    "Sub-item: Institute for Medical Data Science (IMDS)": 5000,
    "Sub-item: External service providers (Validation, QM)": 5000,
    "Sub-item: Data Integration Centre (DIZ) (IT, SAE, storage)": 3000,
    # A.4 Other Direct Costs
    "Open Access Publication Fees (5 publications)": 15000,
    "UK Biobank Access": 12000,
    "Software Licenses (AI tools, literature platforms, LLMs)": 36000,
    "Long-Term Data Storage (DIZ, 200 TB)": 75800,
    "External Expert Consultations (Technical or medical)": 26000,
}


# --- Calculations ---

total_personnel_costs = sum(personnel_costs.values())
total_other_direct_costs = sum(other_direct_costs_items.values())

total_direct_costs = total_personnel_costs + total_other_direct_costs
indirect_costs = total_direct_costs * 0.25
total_project_cost = total_direct_costs + indirect_costs

# --- Verification Output ---

print("--- Budget Verification ---")
print(f"Total Personnel Costs: {total_personnel_costs:,.2f} €")
print(f"Total Other Direct Costs: {total_other_direct_costs:,.2f} €")
print("-----------------------------------")
print(f"Total Direct Costs (A): {total_direct_costs:,.2f} €")
print(f"Indirect Costs (B = 25% of A): {indirect_costs:,.2f} €")
print("===================================")
print(f"Total Project Cost (A + B): {total_project_cost:,.2f} €")
print("===================================")
