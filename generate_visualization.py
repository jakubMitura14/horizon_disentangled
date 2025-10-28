
import pandas as pd
import matplotlib.pyplot as plt
import mpld3

# --- Data from project_analysis.ipynb ---

# Personnel Roles and Work Packages
roles = [
    'Principal Investigator', 'Senior Researcher', 'Clinical Investigator/Consultant',
    'Mathematician', 'Data Scientist', 'Programmer', 'Technician',
    'Project Manager', 'Secretary'
]
work_packages = ['WP1', 'WP2', 'WP3', 'WP4', 'WP5', 'WP6', 'WP7', 'WP8']

# Year 1 Allocation
data_y1 = {
    'WP1': [0, 0, 10, 0, 12, 0, 4, 0, 0], 'WP2': [0, 14, 0, 6, 0, 12, 4, 0, 0],
    'WP3': [0, 10, 0, 6, 0, 0, 4, 0, 3], 'WP4': [0, 0, 8, 0, 0, 0, 0, 0, 6],
    'WP5': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'WP6': [3, 0, 0, 0, 0, 0, 0, 0, 2],
    'WP7': [5, 0, 0, 0, 0, 0, 0, 6, 1], 'WP8': [4, 0, 0, 0, 0, 0, 0, 0, 0]
}
df_alloc_y1 = pd.DataFrame(data_y1, index=roles)

# Year 2 Allocation
data_y2 = {
    'WP1': [0, 0, 10, 0, 12, 0, 4, 0, 0], 'WP2': [0, 10, 0, 6, 0, 6, 0, 0, 0],
    'WP3': [0, 14, 0, 6, 0, 5, 1, 0, 9], 'WP4': [0, 0, 16, 0, 0, 1, 3, 0, 0],
    'WP5': [3, 0, 10, 0, 0, 0, 0, 0, 0], 'WP6': [3, 0, 0, 0, 0, 0, 0, 0, 3],
    'WP7': [2, 0, 0, 0, 0, 0, 4, 6, 0], 'WP8': [4, 0, 0, 0, 0, 0, 0, 0, 0]
}
df_alloc_y2 = pd.DataFrame(data_y2, index=roles)

# Year 3 Allocation
data_y3 = {
    'WP1': [0, 0, 10, 0, 12, 0, 4, 0, 0], 'WP2': [0, 0, 0, 0, 0, 0, 0, 0, 0],
    'WP3': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'WP4': [0, 16, 8, 0, 0, 0, 0, 0, 0],
    'WP5': [5, 8, 0, 6, 0, 12, 5, 0, 7], 'WP6': [3, 0, 0, 0, 0, 0, 0, 0, 2],
    'WP7': [0, 0, 0, 0, 0, 0, 3, 6, 3], 'WP8': [4, 0, 0, 0, 0, 0, 0, 0, 0]
}
df_alloc_y3 = pd.DataFrame(data_y3, index=roles)


# --- Visualization ---

# Create a figure and a grid of subplots
fig, axes = plt.subplots(
    nrows=len(roles),
    ncols=len(work_packages),
    figsize=(18, 14),
    sharex=True,
    sharey=True
)
plt.subplots_adjust(wspace=0.1, hspace=0.1)

# Set a general title
fig.suptitle('Person-Month Allocation per Role and Work Package', fontsize=16)

# Iterate over each role and work package to create the bar charts
for i, role in enumerate(roles):
    for j, wp in enumerate(work_packages):
        ax = axes[i, j]

        # Get PM data for the 3 years
        y1_pm = df_alloc_y1.loc[role, wp]
        y2_pm = df_alloc_y2.loc[role, wp]
        y3_pm = df_alloc_y3.loc[role, wp]

        # Plot the data if any of the values are non-zero
        if y1_pm > 0 or y2_pm > 0 or y3_pm > 0:
            ax.bar(['Y1', 'Y2', 'Y3'], [y1_pm, y2_pm, y3_pm], color=['#4e79a7', '#f28e2b', '#e15759'])
            ax.set_ylim(0, max(y1_pm, y2_pm, y3_pm, 1) * 1.2) # Dynamic Y-axis

        # Clean up the plot
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Set column and row labels
        if i == 0:
            ax.set_title(wp, fontsize=12, pad=15)
        if j == 0:
            ax.set_ylabel(role, rotation=0, labelpad=90, ha='right', va='center', fontsize=10)

# Remove the x-axis labels from the last row for a cleaner look
for ax in axes[-1, :]:
    ax.tick_params(labelbottom=False)


# --- Save Output ---
plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
output_filename = 'allocation_visualization.html'
with open(output_filename, 'w') as f:
    f.write(mpld3.fig_to_html(fig))

print(f"Visualization saved to {output_filename}")
plt.close(fig)
