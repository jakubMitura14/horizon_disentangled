import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from project_analysis import df_alloc_y1, df_alloc_y2, df_alloc_y3

roles = df_alloc_y1.index
wps = df_alloc_y1.columns

# --- Visualization ---

fig, axes = plt.subplots(len(roles), len(wps), figsize=(20, 15), sharex=True, sharey=True)
plt.subplots_adjust(wspace=0.1, hspace=0.1)

bar_width = 0.25
index = np.arange(1)

for i, role in enumerate(roles):
    for j, wp in enumerate(wps):
        ax = axes[i, j]

        y1_val = df_alloc_y1.loc[role, wp]
        y2_val = df_alloc_y2.loc[role, wp]
        y3_val = df_alloc_y3.loc[role, wp]

        # Plot bars only if there's data
        if y1_val > 0:
            ax.bar(index - bar_width, y1_val, bar_width, label='Year 1', color='skyblue')
        if y2_val > 0:
            ax.bar(index, y2_val, bar_width, label='Year 2', color='lightgreen')
        if y3_val > 0:
            ax.bar(index + bar_width, y3_val, bar_width, label='Year 3', color='salmon')

        ax.set_xticks([])
        ax.set_yticks(np.arange(0, 21, 5))
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Hide spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Labels
        if j == 0:
            ax.set_ylabel(role, rotation=0, ha='right', va='center', fontsize=10)
        if i == len(roles) - 1:
            ax.set_xlabel(wp, fontsize=10)

# Set common labels
fig.text(0.5, 0.04, 'Work Packages', ha='center', va='center', fontsize=14)
fig.text(0.08, 0.5, 'Personnel Roles', ha='center', va='center', rotation='vertical', fontsize=14)
fig.suptitle('Person-Month Allocation per Role and Work Package', fontsize=18)

# Legend
handles, labels = [], []
for ax in fig.axes:
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in labels:
            handles.append(h)
            labels.append(l)
fig.legend(handles, labels, loc='upper right')

plt.savefig('allocation_visualization.png', dpi=300, bbox_inches='tight')
print("Visualization saved as allocation_visualization.png")
