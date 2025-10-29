import json
from allocation_logic import get_final_allocations
import pandas as pd

def format_df_to_loc_statements(df, df_name, personnel_df, wp_df):
    """Formats a DataFrame into a series of .loc assignment statements for the notebook."""
    lines = [f"{df_name} = pd.DataFrame(0, index={personnel_df}.index, columns={wp_df}.index)\\n"]
    for index, row in df.iterrows():
        if row.sum() > 0:
            line = f"{df_name}.loc['{index}']{' ' * (35 - len(index))}= {row.values.tolist()}\\n"
            lines.append(line)
    return lines

def create_notebook():
    """Creates a new, clean Jupyter notebook from the ground truth data."""

    df_alloc_y1, df_alloc_y2, df_alloc_y3 = get_final_allocations()
    df_alloc_total = df_alloc_y1.add(df_alloc_y2, fill_value=0).add(df_alloc_y3, fill_value=0)

    # --- Cell Definitions ---

    cell_intro = {
        "cell_type": "markdown", "metadata": {},
        "source": ["# CausalPCa Grant Proposal: Resource Allocation and Verification (Corrected)\n\nThis notebook serves as the single source of truth for the project's person-month (PM) allocation."]
    }

    cell_imports = {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": ["import pandas as pd\n", "import numpy as np"]
    }

    cell_personnel = {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": [
            "personnel_data = { \n",
            "    'Role': ['Principal Investigator', 'Senior Researcher', 'Clinical Investigator/Consultant', 'Mathematician', 'Data Scientist', 'Programmer', 'Technician', 'Project Manager', 'Secretary'],\n",
            "    'Year_1_PM': [12, 24, 18, 12, 12, 12, 12, 6, 12],\n",
            "    'Year_2_PM': [12, 24, 36, 12, 12, 12, 12, 6, 12],\n",
            "    'Year_3_PM': [12, 24, 18, 6, 12, 12, 12, 6, 12]\n",
            "}\n",
            "df_personnel_avail = pd.DataFrame(personnel_data).set_index('Role')\n",
            "df_personnel_avail['Total_PM'] = df_personnel_avail.sum(axis=1)"
        ]
    }

    wp_req_pm = df_alloc_total.sum()
    cell_wp_req = {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": [
            "wp_data = {\\\n",
            "    'Work_Package': " + str(wp_req_pm.index.tolist()) + ",\\\n",
            "    'Total_PM': " + str(wp_req_pm.values.tolist()) + "\\\n",
            "}\\\n",
            "df_wp_req = pd.DataFrame(wp_data).set_index('Work_Package')"
        ]
    }

    y1_lines = format_df_to_loc_statements(df_alloc_y1, 'df_alloc_y1', 'df_personnel_avail', 'df_wp_req')
    y2_lines = format_df_to_loc_statements(df_alloc_y2, 'df_alloc_y2', 'df_personnel_avail', 'df_wp_req')
    y3_lines = format_df_to_loc_statements(df_alloc_y3, 'df_alloc_y3', 'df_personnel_avail', 'df_wp_req')

    cell_allocations = {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": y1_lines + ["\n"] + y2_lines + ["\n"] + y3_lines
    }

    cell_verification = {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": [
            "df_alloc_total = df_alloc_y1.add(df_alloc_y2, fill_value=0).add(df_alloc_y3, fill_value=0)\n",
            "personnel_check = df_alloc_total.sum(axis=1) == df_personnel_avail['Total_PM']\n",
            "wp_check = df_alloc_total.sum() == df_wp_req['Total_PM']\n",
            "print('--- VERIFICATION RESULTS ---')\n",
            "print('All Personnel Allocations Correct: ', personnel_check.all())\n",
            "print('All WP Allocations Correct: ', wp_check.all())"
        ]
    }

    # --- Assemble Notebook ---
    notebook = {
        "cells": [
            cell_intro, cell_imports,
            {"cell_type": "markdown", "metadata": {}, "source": ["### Step 1: Define Personnel Availability"]},
            cell_personnel,
            {"cell_type": "markdown", "metadata": {}, "source": ["### Step 2: Define Work Package Requirements"]},
            cell_wp_req,
            {"cell_type": "markdown", "metadata": {}, "source": ["### Step 3: Define Allocation Plan"]},
            cell_allocations,
            {"cell_type": "markdown", "metadata": {}, "source": ["### Step 4: Verification"]},
            cell_verification
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version":"3.9.7"}
        },
        "nbformat": 4, "nbformat_minor": 4
    }

    with open('project_analysis.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)

if __name__ == "__main__":
    create_notebook()
    print("Clean project_analysis.ipynb has been generated.")
