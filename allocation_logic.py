import pandas as pd

def get_final_allocations():
    """
    Returns the final, corrected dataframes for the allocation plan, including WP9.
    This function contains the hardcoded, verified allocation data and serves as the
    single source of truth for the project's person-month distribution.
    """
    roles = ['Principal Investigator', 'Senior Researcher', 'Clinical Investigator/Consultant',
             'Mathematician', 'Data Scientist', 'Programmer', 'Technician',
             'Project Manager', 'Secretary']
    wps = ['WP1', 'WP2', 'WP3', 'WP4', 'WP5', 'WP6', 'WP7', 'WP8', 'WP9']

    # --- YEAR 1 ALLOCATION ---
    data_y1 = [
        [0, 0, 0, 0, 0, 3, 5, 4, 0],    # PI
        [0, 8, 10, 0, 0, 0, 0, 0, 6],   # Senior Researcher (adjusted)
        [10, 0, 0, 8, 0, 0, 0, 0, 0],   # Clinical Investigator
        [0, 6, 6, 0, 0, 0, 0, 0, 0],    # Mathematician
        [12, 0, 0, 0, 0, 0, 0, 0, 0],   # Data Scientist
        [0, 12, 0, 0, 0, 0, 0, 0, 0],   # Programmer
        [4, 4, 4, 0, 0, 0, 0, 0, 0],    # Technician
        [0, 0, 0, 0, 0, 0, 6, 0, 0],    # Project Manager
        [0, 0, 3, 6, 0, 2, 1, 0, 0]     # Secretary
    ]
    df_alloc_y1 = pd.DataFrame(data_y1, index=roles, columns=wps)

    # --- YEAR 2 ALLOCATION ---
    data_y2 = [
        [0, 0, 0, 0, 3, 3, 2, 4, 0],    # PI
        [0, 4, 14, 0, 0, 0, 0, 0, 6],   # Senior Researcher (adjusted)
        [10, 0, 0, 16, 10, 0, 0, 0, 0],  # Clinical Investigator
        [0, 6, 6, 0, 0, 0, 0, 0, 0],    # Mathematician
        [12, 0, 0, 0, 0, 0, 0, 0, 0],   # Data Scientist
        [0, 6, 5, 1, 0, 0, 0, 0, 0],    # Programmer
        [4, 0, 1, 3, 0, 0, 4, 0, 0],    # Technician
        [0, 0, 0, 0, 0, 0, 6, 0, 0],    # Project Manager
        [0, 0, 9, 0, 0, 3, 0, 0, 0]     # Secretary
    ]
    df_alloc_y2 = pd.DataFrame(data_y2, index=roles, columns=wps)

    # --- YEAR 3 ALLOCATION ---
    data_y3 = [
        [0, 0, 0, 0, 5, 3, 0, 4, 0],    # PI
        [0, 0, 0, 10, 8, 0, 0, 0, 6],   # Senior Researcher (adjusted)
        [10, 0, 0, 8, 0, 0, 0, 0, 0],   # Clinical Investigator
        [0, 0, 0, 0, 6, 0, 0, 0, 0],    # Mathematician
        [12, 0, 0, 0, 0, 0, 0, 0, 0],   # Data Scientist
        [0, 0, 0, 0, 12, 0, 0, 0, 0],   # Programmer
        [4, 0, 0, 0, 5, 0, 3, 0, 0],    # Technician
        [0, 0, 0, 0, 0, 0, 6, 0, 0],    # Project Manager
        [0, 0, 0, 0, 7, 2, 3, 0, 0]     # Secretary
    ]
    df_alloc_y3 = pd.DataFrame(data_y3, index=roles, columns=wps)

    return df_alloc_y1, df_alloc_y2, df_alloc_y3

if __name__ == '__main__':
    # This allows the script to be run directly for verification.
    df1, df2, df3 = get_final_allocations()
    df_total = df1.add(df2, fill_value=0).add(df3, fill_value=0)

    print("--- TOTAL ALLOCATION (PERSON-MONTHS) ---")
    print(df_total)
    print("\n--- PMs Allocated per Role ---")
    print(df_total.sum(axis=1))
    print("\n--- PMs Allocated per WP ---")
    print(df_total.sum(axis=0))
