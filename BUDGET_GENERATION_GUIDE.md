# Guide to Generating the Detailed Budget CSV

## 1. Primary Goal

The main objective is to create a detailed, itemized budget breakdown in a CSV file named `detailed_wp_budgets.csv`. This file must accurately reflect the project's budget allocation per Work Package (WP). The final numbers must be internally consistent and verifiable against several sources of truth within the repository.

This task has proven to be highly complex due to the need to reconcile dynamically calculated costs (based on person-month allocations) with pre-defined, fixed totals in the main grant document.

## 2. Sources of Truth

There are three primary sources of information that must be used for calculation and verification:

### a. `allocation_logic.py`
- **Purpose:** This script is the definitive source for **person-month (PM) allocations**.
- **Data:** It contains three pandas DataFrames (`df_alloc_y1`, `df_alloc_y2`, `df_alloc_y3`), which detail the PMs assigned to each personnel role for each Work Package, for each of the three project years.

### b. `budget_tables.tex`
- **Purpose:** This LaTeX file is the source for the **annual cost of each personnel role**.
- **Data:** It contains tables that specify the total cost for each role (e.g., 'Principal Investigator') for Year 1, Year 2, and Year 3.
- **Note:** Parsing this file directly is brittle. For stability, the following correct values should be used:
  ```python
  annual_costs = {
      'Y1': {'Principal Investigator': 99448.89, 'Senior Researcher': 90410.93, ...},
      'Y2': {'Principal Investigator': 104421.33, 'Senior Researcher': 94931.48, ...},
      'Y3': {'Principal Investigator': 113547.29, 'Senior Researcher': 105198.96, ...}
  }
  ```

### c. `main_horizon.tex`
- **Purpose:** The main grant proposal document contains the **final, non-negotiable totals** that must be matched.
- **Data:**
    1.  **Grand Total Costs (in the text):**
        - `Total Estimated Personnel Costs: \EUR{2,919,591.40}`
        - `Total Estimated Other Direct Costs: \EUR{278,900.00}`
    2.  **Person-Months per Work Package (Table \ref{tab:person_months}):** A summary table showing the total PMs for each WP (e.g., WP1: 96 PMs, WP2: 48 PMs, etc.).
    3.  **Budget Allocation per Work Package (Table \ref{tab:budget_wp}):** A summary table showing the final, total cost for each WP (e.g., WP1 Total: €1,031,771.42).

## 3. Required Calculation Logic

The following is the precise, step-by-step logic required to generate the `detailed_wp_budgets.csv`:

### Step 1: Parse Costs and Define Mappings
- Get the annual cost for each personnel role from the source data (`budget_tables.tex` or the hardcoded dictionary).
- Define the mapping of each role to its personnel category ('SENIOR SCIENTISTS' or 'TECHNICAL PERSONNEL').

### Step 2: Get Person-Months
- Import the three yearly person-month allocation DataFrames from `allocation_logic.py`.

### Step 3 & 4: Calculate Total Personnel Cost per WP (Year-by-Year)
- For each Work Package:
  - Create a data structure to hold the costs and PMs for 'SENIOR SCIENTISTS' and 'TECHNICAL PERSONNEL'.
  - Iterate through each of the three years (`Y1`, `Y2`, `Y3`).
  - For each role, get the person-months for that specific role, in that specific year, for the current WP.
  - If the PMs are greater than zero, calculate the cost for that slice: `cost = (annual_cost_for_role_in_year / 12) * person_months`.
  - Add the calculated cost and PMs to the correct personnel category for the current WP.
- After iterating through all years and roles, you will have the total cost and total PMs for each personnel category within each WP.

### Step 5 & 6: Calculate PMs and Average Costs per WP
- For each Work Package and each personnel category:
  - The **total person-months** is the sum calculated in the previous step.
  - The **average cost per person-month** is `(total cost for this category in this WP) / (total person-months for this category in this WP)`.

### Step 7: Finalize CSV Data
- For each row in the final CSV:
  - **Personnel Rows:**
    - `ITEMS`: The total person-months for that category in that WP.
    - `COST PER ITEM`: The average cost per person-month for that category in that WP.
    - `BE TOTAL COSTS`: The total cost for that category in that WP.
  - **Other Direct Cost Rows:**
    - The grand total for "Other Direct Costs" (€278,900.00) should be distributed across the WPs proportionally to the total personnel cost of each WP.
    - `COST PER ITEM` should be the grand total for that specific item (e.g., €32,500.00 for Travel).
    - `BE TOTAL COSTS` should be the proportionally distributed amount for that WP.
  - **Total Rows:**
    - Calculate the `TOTAL DIRECT COSTS` for the WP.
    - Calculate the `INDIRECT COSTS` (25% of direct).
    - Calculate the `TOTAL COSTS`.

## 4. Verification Steps

A final verification script must perform two checks (allowing for a 5-cent tolerance on financial figures):

### a. Person-Month Verification
- The sum of person-months in the `ITEMS` column for each WP in the generated `detailed_wp_budgets.csv` **must match** the total person-months for that WP as defined in the summary table in `main_horizon.tex`.

### b. Work Package Total Cost Verification
- For each WP in the generated CSV, the final `TOTAL COSTS` value **must match** the total cost for that WP as defined in the budget summary table in `main_horizon.tex`. This is the most critical check. The calculation is `(Sum of all BE TOTAL COSTS for direct items) * 1.25`.
