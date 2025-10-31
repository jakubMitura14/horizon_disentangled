# Detailed Budget Generation Pipeline

This directory contains the scripts and data files necessary to generate a detailed, itemized budget breakdown for the grant proposal. The primary goal of this pipeline is to produce a CSV file (`detailed_wp_budgets.csv`) that is internally consistent and perfectly aligned with the financial constraints defined in the main grant document (`main_horizon.tex`).

## Workflow Overview

The process is executed in a sequence of scripts:

1.  **`parse_tex.py`**: Extracts ground truth financial data from the main LaTeX file.
2.  **`calculate_detailed_budget.py`**: Performs the core budget calculation using multiple data sources and complex business logic.
3.  **`final_verification.py`**: Verifies that the output of the calculation script is correct and consistent.

## Data Sources (Inputs)

The calculation relies on three primary sources of truth:

1.  **`allocation_logic.py` (Repo Root)**: This file is the definitive source for **person-month (PM) allocations**. It contains three pandas DataFrames (`df_alloc_y1`, `df_alloc_y2`, `df_alloc_y3`) detailing the PMs assigned to each role for each Work Package (WP), for each of the three project years.

2.  **`budget_details/yearly_personnel_costs.csv`**: This file contains the **year-by-year cost** for a full-time equivalent (FTE) for each personnel role. This data is used for the detailed personnel cost calculation.

3.  **`main_horizon.tex` (Repo Root)**: The main grant proposal document contains the **final, non-negotiable direct cost totals** that the final output must match. The `parse_tex.py` script extracts this information into `budget_details/parsed_tex_data.json` for easy access.

## Core Calculation Logic (`calculate_detailed_budget.py`)

The calculation is a multi-step process designed to reconcile the bottom-up allocation data with the top-down budget constraints:

1.  **Year-by-Year Personnel Cost Calculation**: The script first calculates the personnel costs on a year-by-year basis. It calculates a cost per person-month for each year by dividing the annual cost from `yearly_personnel_costs.csv` by 12. This is then multiplied by the person-months allocated for each role in that year.

2.  **Handling of Special Costs**:
    *   **Student/Research Assistant**: The total cost for this role is calculated and treated as an "Other Direct Cost" (ODC), and is not included in the core personnel-month calculations.
    *   **Registration Fee**: A fixed fee of €81,600 is handled as a fixed "Other Direct Cost", allocated specifically to WP7 and WP9 (50/50 split, where possible).

3.  **Backward Calculation & Cost Capping**: This is the most critical step. The script works backward from the ground truth direct cost total for each Work Package (from `main_horizon.tex`).
    *   It compares the calculated personnel cost for a WP against the total available direct cost for that WP.
    *   If the personnel cost exceeds the available budget, it is **capped** at the maximum available amount. This prevents negative "Other Direct Costs".
    *   The remaining budget for each WP after personnel costs are accounted for is then allocated to the ODC pool for that WP.

4.  **ODC Distribution**: The calculated ODC pool for each WP is then distributed among the various ODC items (Travel, Publications, etc.) based on the specified business rules (proportional distribution or fixed allocation). The script includes logic to prevent negative costs if a fixed cost (like Registration) exceeds the available pool for a given WP.

## Final CSV Structure (`detailed_wp_budgets.csv`)

The final output CSV contains the following columns:
*   `Work Package`: The Work Package identifier (e.g., "WP1").
*   `COST CATEGORY`: A granular cost category as specified by the user (e.g., "SENIOR SCIENTISTS (or equivalent in the private sector)", "C.1 Travel and subsistence").
*   `ITEMS`: A description of the cost item. For personnel, this includes the person-months and the role (e.g., "36.00 PMs (Data Scientist)"). For ODCs, it is the name of the item (e.g., "Travel").
*   `COST PER ITEM`: For personnel, the calculated average cost per person-month. For ODCs, the grand total for that item across the entire project.
*   `BE TOTAL COSTS`: The final, calculated direct cost for that specific item within that specific WP.

## Verification (`final_verification.py`)

The final script performs three critical checks to ensure the generated `detailed_wp_budgets.csv` is correct:

1.  **Person-Month Verification**: It confirms that the total person-months for each WP in the final CSV match the ground truth totals from `allocation_logic.py`.
2.  **Per-WP Direct Cost Verification**: It confirms that the sum of all calculated direct costs (Personnel + ODC) for each WP in the CSV matches the ground truth direct cost for that WP from `main_horizon.tex` (with a 5-cent tolerance for rounding).
3.  **Grand Total Direct Cost Verification**: It confirms that the grand total of all direct costs in the CSV matches the project's total direct cost of €3,198,491.40 (with a 5-cent tolerance).
