import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from project_analysis import df_alloc_y1, df_alloc_y2, df_alloc_y3

roles = df_alloc_y1.index
wps = df_alloc_y1.columns
years = ['Year 1', 'Year 2', 'Year 3']
alloc_data = {'Year 1': df_alloc_y1, 'Year 2': df_alloc_y2, 'Year 3': df_alloc_y3}

# --- Visualization ---

# Create a figure with one subplot for each role
fig, axes = plt.subplots(len(roles), 1, figsize=(12, 20), sharex=True)
plt.subplots_adjust(hspace=0.5)
fig.suptitle('Person-Month Allocation by Role and Work Package', fontsize=18, y=0.95)

# Define a color map for the Work Packages
colors = plt.cm.get_cmap('tab10', len(wps))

for i, role in enumerate(roles):
    ax = axes[i]
    ax.set_title(role, loc='left', fontsize=12, fontweight='bold')

    bottoms = np.zeros(len(years))

    for j, wp in enumerate(wps):
        values = [alloc_data[year].loc[role, wp] for year in years]

        # Add bars for each year
        bars = ax.bar(years, values, bottom=bottoms, label=wp, color=colors(j))

        # Add text labels inside the bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + height / 2,
                        f'{height:.0f}', ha='center', va='center', color='white', fontsize=9)

        bottoms += values

    # Formatting for each subplot
    ax.set_ylabel('Person-Months')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# Common legend
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', title='Work Packages', bbox_to_anchor=(0.95, 0.92))

plt.savefig('allocation_visualization.png', dpi=300, bbox_inches='tight')
print("New visualization saved as allocation_visualization.png")
