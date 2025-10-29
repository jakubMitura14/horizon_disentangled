import pandas as pd

def get_final_allocations():
    """
    Returns the final, corrected, and perfectly balanced dataframes for the allocation plan.
    This is the single source of truth. All totals have been manually verified to be correct.
    """
    roles = ['Principal Investigator', 'Senior Researcher', 'Clinical Investigator/Consultant',
             'Mathematician', 'Data Scientist', 'Programmer', 'Technician',
             'Project Manager', 'Secretary']
    wps = ['WP1', 'WP2', 'WP3', 'WP4', 'WP5', 'WP6', 'WP7', 'WP8', 'WP9']

    # --- YEAR 1 ALLOCATION --- Total = 120
    data_y1 = [
        [0, 0, 0, 4, 0, 3, 5, 0, 0],    # PI: 12
        [0, 8, 10, 0, 0, 0, 0, 0, 6],   # Senior Researcher: 24
        [12, 6, 0, 0, 0, 0, 0, 0, 0],   # Clinical Investigator: 18
        [0, 6, 6, 0, 0, 0, 0, 0, 0],    # Mathematician: 12
        [12, 0, 0, 0, 0, 0, 0, 0, 0],   # Data Scientist: 12
        [0, 12, 0, 0, 0, 0, 0, 0, 0],   # Programmer: 12
        [12, 0, 0, 0, 0, 0, 0, 0, 0],   # Technician: 12
        [0, 0, 0, 0, 0, 0, 6, 0, 0],    # Project Manager: 6
        [0, 4, 0, 0, 0, 2, 2, 4, 0]     # Secretary: 12
    ]
    df_alloc_y1 = pd.DataFrame(data_y1, index=roles, columns=wps)

    # --- YEAR 2 ALLOCATION --- Total = 138
    data_y2 = [
        [0, 0, 0, 0, 0, 2, 4, 6, 0],    # PI: 12
        [0, 0, 18, 0, 0, 0, 0, 0, 6],   # Senior Researcher: 24
        [12, 0, 0, 12, 12, 0, 0, 0, 0],  # Clinical Investigator: 36
        [0, 12, 0, 0, 0, 0, 0, 0, 0],    # Mathematician: 12
        [12, 0, 0, 0, 0, 0, 0, 0, 0],   # Data Scientist: 12
        [0, 0, 6, 6, 0, 0, 0, 0, 0],    # Programmer: 12
        [12, 0, 0, 0, 0, 0, 0, 0, 0],   # Technician: 12
        [0, 0, 0, 0, 0, 0, 6, 0, 0],    # Project Manager: 6
        [0, 0, 0, 0, 0, 4, 4, 4, 0]     # Secretary: 12
    ]
    df_alloc_y2 = pd.DataFrame(data_y2, index=roles, columns=wps)

    # --- YEAR 3 ALLOCATION --- Total = 114
    data_y3 = [
        [0, 0, 0, 0, 5, 3, 0, 4, 0],    # PI: 12
        [0, 0, 0, 18, 0, 0, 0, 0, 6],   # Senior Researcher: 24
        [0, 0, 0, 0, 18, 0, 0, 0, 0],   # Clinical Investigator: 18
        [0, 0, 0, 6, 0, 0, 0, 0, 0],    # Mathematician: 6
        [12, 0, 0, 0, 0, 0, 0, 0, 0],   # Data Scientist: 12
        [0, 0, 0, 12, 0, 0, 0, 0, 0],   # Programmer: 12
        [12, 0, 0, 0, 0, 0, 0, 0, 0],   # Technician: 12
        [0, 0, 0, 0, 0, 0, 6, 0, 0],    # Project Manager: 6
        [0, 0, 0, 0, 7, 2, 3, 0, 0]     # Secretary: 12
    ]
    df_alloc_y3 = pd.DataFrame(data_y3, index=roles, columns=wps)

    return df_alloc_y1, df_alloc_y2, df_alloc_y3
