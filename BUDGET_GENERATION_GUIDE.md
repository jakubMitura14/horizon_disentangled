# Guide to Generating the Detailed Budget CSV

## 1. Primary Goal

The main objective is to create a detailed, itemized budget breakdown in a CSV file named `detailed_wp_budgets.csv`. This file must accurately reflect the project's budget allocation per Work Package (WP). The final numbers must be internally consistent and verifiable against several sources of truth within the repository.

This task is highly complex because the final budget for each WP must dynamically emerge from the person-month allocations while still perfectly matching the fixed WP totals and grand totals specified in `main_horizon.tex`.

## 2. Output CSV Structure (`detailed_wp_budgets.csv`)

The output CSV file must have the following columns: `Work Package`, `COST CATEGORY`, `ITEMS`, `COST PER ITEM`, `BE TOTAL COSTS`.

### For Personnel Rows:
- **`Work Package`**: e.g., "WP1"
- **`COST CATEGORY`**: "A. DIRECT PERSONNEL COSTS"
- **`ITEMS`**: The total **Person-Months (PMs)** for this personnel category (e.g., 'SENIOR SCIENTISTS') in this specific WP. This must be a floating-point number.
- **`COST PER ITEM`**: The **average cost per person-month** for this personnel category, calculated specifically for this WP. This is a floating-point number.
- **`BE TOTAL COSTS`**: The total cost for this personnel category in this WP. This value is the result of `(ITEMS) * (COST PER ITEM)`.

### For Other Direct Cost Rows:
- **`Work Package`**: e.g., "WP1"
- **`COST CATEGORY`**: "C. DIRECT PURCHASE COSTS"
- **`ITEMS`**: This column is not applicable for these rows and should be left blank or contain a placeholder like `~`.
- **`COST PER ITEM`**: The **grand total** for this specific cost item across the entire project (e.g., 32500.00 for Travel).
- **`BE TOTAL COSTS`**: The portion of the grand total that is allocated to this specific WP.

## 3. Sources of Truth

There are three primary sources of information that must be used for calculation and verification:

### a. `allocation_logic.py`
- **Purpose:** This script is the definitive source for **person-month (PM) allocations**.
- **Data:** It contains three pandas DataFrames (`df_alloc_y1`, `df_alloc_y2`, `df_alloc_y3`).

### b. `budget_tables.tex`
- **Purpose:** This is the source for the **annual cost of each personnel role**.
- **Note:** Parsing this file is brittle. The known correct values should be used directly.

### c. `main_horizon.tex`
- **Purpose:** The main grant proposal document contains the **final, non-negotiable totals** that must be matched.
- **Data:**
    1.  Grand Total Costs (in the text).
    2.  Total Person-Months per Work Package (a summary table).
    3.  Total Budget per Work Package (a summary table).

## 4. Personnel Role Categorization

- **SENIOR SCIENTISTS:** 'Principal Investigator', 'Senior Researcher', 'Clinical Investigator/Consultant', 'Mathematician', 'Data Scientist'.
- **TECHNICAL PERSONNEL:** 'Programmer', 'Technician', 'Project Manager', 'Secretary', 'Student/Research Assistant'.

## 5. "Other Direct Costs" Breakdown and Constraints

- **Proportionally Distributed Costs:** `Travel`, `Publication fees`, and `Other` should be distributed across WPs in proportion to each WP's share of the total personnel cost.
- **Fixed Costs:** Costs like `UK Biobank Access` and `Subject Insurance` must be allocated entirely to **WP1**.

## 6. Required Calculation Logic

The script must implement a year-by-year cost calculation as described in previous sections to arrive at the total cost and total PMs for each personnel category within each WP.

## 7. Detailed Testing and Verification

A final verification script (`final_verification.py`) must perform two critical checks, allowing for a **5-cent tolerance** on all financial comparisons.

### a. Person-Month Verification (Step-by-Step)
This check ensures the person-months (effort) are consistent across all data sources.

1.  **Extract from `main_horizon.tex`:**
    - Parse the "Total Person-Months per Work Package" table.
    - Create a dictionary mapping each WP to its target PM total (e.g., `{'WP1': 96.0, 'WP2': 48.0, ...}`). This is the **ground truth**.
2.  **Calculate from `allocation_logic.py`:**
    - Import the three yearly allocation DataFrames.
    - Sum them together (`df_total = df_alloc_y1 + df_alloc_y2 + df_alloc_y3`).
    - Calculate the column sums of `df_total` to get the total PMs per WP from the source data.
3.  **Calculate from the final CSV (`detailed_wp_budgets.csv`):**
    - Read the generated CSV file.
    - Filter the rows where `COST CATEGORY` is "A. DIRECT PERSONNEL COSTS".
    - Group the filtered data by `Work Package`.
    - For each group, sum the values in the `ITEMS` column. This gives the total PMs per WP from the final output.
4.  **Compare:** For each Work Package, assert that the PM total from the LaTeX ground truth is equal to the PM total from the CSV.

### b. Work Package Total Cost Verification
This check ensures the final, all-inclusive cost for each WP matches the grant document.

1.  **Extract from `main_horizon.tex`:**
    - Parse the "Budget Allocation per Work Package" table to get the target total cost for each WP (e.g., `{'WP1': 1031771.42, ...}`).
2.  **Calculate from the final CSV:**
    - Read the `detailed_wp_budgets.csv` file.
    - Group the data by `Work Package`.
    - For each WP group, sum the `BE TOTAL COSTS` for all rows that are **direct costs** (i.e., Personnel and Other Direct Costs).
    - Multiply this sum by 1.25 to include the 25% indirect costs.
3.  **Compare:** For each Work Package, assert that this calculated total matches the target total from `main_horizon.tex` within the 5-cent tolerance.
