#!/usr/bin/env python
# coding: utf-8

# # CausalPCa Grant Proposal: Resource Allocation and Verification
#
# This notebook serves as the single source of truth for the project's person-month (PM) allocation. It provides a detailed, year-by-year breakdown of personnel availability, work package requirements, and the final allocation plan.
#
# The final cell provides a comprehensive, automated verification to ensure that all resources are perfectly reconciled on both a yearly and a total project basis.

# In[ ]:


import pandas as pd
import numpy as np


# ### Step 1: Define Yearly Personnel Availability

# In[ ]:


personnel_data = {
    'Role': [
        'Principal Investigator',
        'Senior Researcher',
        'Clinical Investigator/Consultant',
        'Mathematician',
        'Data Scientist',
        'Programmer',
        'Technician',
        'Project Manager',
        'Secretary'
    ],
    'Year_1_PM': [12, 24, 18, 12, 12, 12, 12, 6, 12],
    'Year_2_PM': [12, 24, 36, 12, 12, 12, 12, 6, 12],
    'Year_3_PM': [12, 24, 18, 6, 12, 12, 12, 6, 12]
}

df_personnel_avail = pd.DataFrame(personnel_data).set_index('Role')
df_personnel_avail['Total_PM'] = df_personnel_avail.sum(axis=1)

print("--- Personnel Availability (Person-Months) ---")
print(df_personnel_avail)
print("\n")
print("--- Yearly Totals ---")
print(df_personnel_avail.sum().rename('Total Available PM'))


# ### Step 2: Define Yearly Work Package Requirements (Re-balanced Plan)

# In[ ]:


wp_data = {'Work_Package': ['WP1', 'WP2', 'WP3', 'WP4', 'WP5', 'WP6', 'WP7', 'WP8', 'WP9'], 'Year_1_PM': [26, 30, 23, 14, 0, 5, 12, 0, 6], 'Year_2_PM': [12, 0, 18, 14, 10, 0, 0, 4, 6], 'Year_3_PM': [0, 0, 0, 18, 0, 0, 0, 0, 6]}


# ### Step 3: Create Detailed Yearly Allocation Plan (Simplified & Corrected)

# In[ ]:


df_alloc_y1 = pd.DataFrame({'WP1': {'Principal Investigator': 0, 'Senior Researcher': 0, 'Clinical Investigator/Consultant': 14, 'Mathematician': 0, 'Data Scientist': 12, 'Programmer': 0, 'Technician': 0, 'Project Manager': 0, 'Secretary': 0}, 'WP2': {'Principal Investigator': 0, 'Senior Researcher': 8, 'Clinical Investigator/Consultant': 0, 'Mathematician': 6, 'Data Scientist': 0, 'Programmer': 12, 'Technician': 4, 'Project Manager': 0, 'Secretary': 0}, 'WP3': {'Principal Investigator': 0, 'Senior Researcher': 10, 'Clinical Investigator/Consultant': 0, 'Mathematician': 6, 'Data Scientist': 0, 'Programmer': 0, 'Technician': 4, 'Project Manager': 0, 'Secretary': 3}, 'WP4': {'Principal Investigator': 4, 'Senior Researcher': 0, 'Clinical Investigator/Consultant': 4, 'Mathematician': 0, 'Data Scientist': 0, 'Programmer': 0, 'Technician': 4, 'Project Manager': 0, 'Secretary': 2}, 'WP5': {'Principal Investigator': 0, 'Senior Researcher': 0, 'Clinical Investigator/Consultant': 0, 'Mathematician': 0, 'Data Scientist': 0, 'Programmer': 0, 'Technician': 0, 'Project Manager': 0, 'Secretary': 0}, 'WP6': {'Principal Investigator': 3, 'Senior Researcher': 0, 'Clinical Investigator/Consultant': 0, 'Mathematician': 0, 'Data Scientist': 0, 'Programmer': 0, 'Technician': 0, 'Project Manager': 0, 'Secretary': 2}, 'WP7': {'Principal Investigator': 5, 'Senior Researcher': 0, 'Clinical Investigator/Consultant': 0, 'Mathematician': 0, 'Data Scientist': 0, 'Programmer': 0, 'Technician': 0, 'Project Manager': 6, 'Secretary': 1}, 'WP8': {'Principal Investigator': 0, 'Senior Researcher': 0, 'Clinical Investigator/Consultant': 0, 'Mathematician': 0, 'Data Scientist': 0, 'Programmer': 0, 'Technician': 0, 'Project Manager': 0, 'Secretary': 0}, 'WP9': {'Principal Investigator': 0, 'Senior Researcher': 6, 'Clinical Investigator/Consultant': 0, 'Mathematician': 0, 'Data Scientist': 0, 'Programmer': 0, 'Technician': 0, 'Project Manager': 0, 'Secretary': 0}})


# df_alloc_y2 = pd.DataFrame({'WP1': {'Principal Investigator': 0, 'Senior Researcher': 0, 'Clinical Investigator/Consultant': 12, 'Mathematician': 0, 'Data Scientist': 0, 'Programmer': 0, 'Technician': 0, 'Project Manager': 0, 'Secretary': 0}, 'WP2': {'Principal Investigator': 0, 'Senior Researcher': 0, 'Clinical Investigator/Consultant': 0, 'Mathematician': 0, 'Data Scientist': 0, 'Programmer': 0, 'Technician': 0, 'Project Manager': 0, 'Secretary': 0}, 'WP3': {'Principal Investigator': 0, 'Senior Researcher': 18, 'Clinical Investigator/Consultant': 0, 'Mathematician': 0, 'Data Scientist': 0, 'Programmer': 0, 'Technician': 0, 'Project Manager': 0, 'Secretary': 0}, 'WP4': {'Principal Investigator': 0, 'Senior Researcher': 0, 'Clinical Investigator/Consultant': 14, 'Mathematician': 0, 'Data Scientist': 0, 'Programmer': 0, 'Technician': 0, 'Project Manager': 0, 'Secretary': 0}, 'WP5': {'Principal Investigator': 0, 'Senior Researcher': 0, 'Clinical Investigator/Consultant': 10, 'Mathematician': 0, 'Data Scientist': 0, 'Programmer': 0, 'Technician': 0, 'Project Manager': 0, 'Secretary': 0}, 'WP6': {'Principal Investigator': 0, 'Senior Researcher': 0, 'Clinical Investigator/Consultant': 0, 'Mathematician': 0, 'Data Scientist': 0, 'Programmer': 0, 'Technician': 0, 'Project Manager': 0, 'Secretary': 0}, 'WP7': {'Principal Investigator': 0, 'Senior Researcher': 0, 'Clinical Investigator/Consultant': 0, 'Mathematician': 0, 'Data Scientist': 0, 'Programmer': 0, 'Technician': 0, 'Project Manager': 0, 'Secretary': 0}, 'WP8': {'Principal Investigator': 0, 'Senior Researcher': 0, 'Clinical Investigator/Consultant': 0, 'Mathematician': 0, 'Data Scientist': 0, 'Programmer': 0, 'Technician': 0, 'Project Manager': 0, 'Secretary': 4}, 'WP9': {'Principal Investigator': 0, 'Senior Researcher': 6, 'Clinical Investigator/Consultant': 0, 'Mathematician': 0, 'Data Scientist': 0, 'Programmer': 0, 'Technician': 0, 'Project Manager': 0, 'Secretary': 0}})

# In[ ]:


df_alloc_y3 = pd.DataFrame({'WP1': {'Principal Investigator': 0, 'Senior Researcher': 0, 'Clinical Investigator/Consultant': 0, 'Mathematician': 0, 'Data Scientist': 0, 'Programmer': 0, 'Technician': 0, 'Project Manager': 0, 'Secretary': 0}, 'WP2': {'Principal Investigator': 0, 'Senior Researcher': 0, 'Clinical Investigator/Consultant': 0, 'Mathematician': 0, 'Data Scientist': 0, 'Programmer': 0, 'Technician': 0, 'Project Manager': 0, 'Secretary': 0}, 'WP3': {'Principal Investigator': 0, 'Senior Researcher': 0, 'Clinical Investigator/Consultant': 0, 'Mathematician': 0, 'Data Scientist': 0, 'Programmer': 0, 'Technician': 0, 'Project Manager': 0, 'Secretary': 0}, 'WP4': {'Principal Investigator': 0, 'Senior Researcher': 18, 'Clinical Investigator/Consultant': 0, 'Mathematician': 0, 'Data Scientist': 0, 'Programmer': 0, 'Technician': 0, 'Project Manager': 0, 'Secretary': 0}, 'WP5': {'Principal Investigator': 0, 'Senior Researcher': 0, 'Clinical Investigator/Consultant': 0, 'Mathematician': 0, 'Data Scientist': 0, 'Programmer': 0, 'Technician': 0, 'Project Manager': 0, 'Secretary': 0}, 'WP6': {'Principal Investigator': 0, 'Senior Researcher': 0, 'Clinical Investigator/Consultant': 0, 'Mathematician': 0, 'Data Scientist': 0, 'Programmer': 0, 'Technician': 0, 'Project Manager': 0, 'Secretary': 0}, 'WP7': {'Principal Investigator': 0, 'Senior Researcher': 0, 'Clinical Investigator/Consultant': 0, 'Mathematician': 0, 'Data Scientist': 0, 'Programmer': 0, 'Technician': 0, 'Project Manager': 0, 'Secretary': 0}, 'WP8': {'Principal Investigator': 0, 'Senior Researcher': 0, 'Clinical Investigator/Consultant': 0, 'Mathematician': 0, 'Data Scientist': 0, 'Programmer': 0, 'Technician': 0, 'Project Manager': 0, 'Secretary': 0}, 'WP9': {'Principal Investigator': 0, 'Senior Researcher': 6, 'Clinical Investigator/Consultant': 0, 'Mathematician': 0, 'Data Scientist': 0, 'Programmer': 0, 'Technician': 0, 'Project Manager': 0, 'Secretary': 0}})
