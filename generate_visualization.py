import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from allocation_logic import get_final_allocations

# --- 1. Get Data Directly from the Logic Script ---
df_alloc_y1, df_alloc_y2, df_alloc_y3 = get_final_allocations()

roles = df_alloc_y1.index
wps = df_alloc_y1.columns

# --- 2. Create Visualization Grid ---
fig, axes = plt.subplots(len(roles), len(wps), figsize=(22, 14), sharex=True, sharey=True)
fig.suptitle('Personnel Allocation (Person-Months per Year)', fontsize=24, y=0.95)

bar_width = 0.25
index = np.arange(1)

for i, role in enumerate(roles):
    for j, wp in enumerate(wps):
        ax = axes[i, j]

        y1_val = df_alloc_y1.loc[role, wp]
        y2_val = df_alloc_y2.loc[role, wp]
        y3_val = df_alloc_y3.loc[role, wp]

        if y1_val > 0:
            ax.bar(index - bar_width, [y1_val], bar_width, label='Year 1', color='skyblue')
        if y2_val > 0:
            ax.bar(index, [y2_val], bar_width, label='Year 2', color='steelblue')
        if y3_val > 0:
            ax.bar(index + bar_width, [y3_val], bar_width, label='Year 3', color='royalblue')

        ax.set_xticks([])
        ax.set_yticks(np.arange(0, 37, 12))
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)

for ax, col in zip(axes[0], wps):
    ax.set_title(col, fontsize=14, pad=10)

for ax, row in zip(axes[:,0], roles):
    ax.set_ylabel(row, rotation=0, size='large', labelpad=100, ha='center', va='center')
    ax.yaxis.set_label_coords(-0.8, 0.5)

handles, labels = [], []
for ax in fig.axes:
    h, l = ax.get_legend_handles_labels()
    for handle, label in zip(h, l):
        if label not in labels:
            labels.append(label)
            handles.append(handle)

fig.legend(handles, labels, loc='upper right', fontsize=12, bbox_to_anchor=(0.95, 0.93))

plt.tight_layout(rect=[0.05, 0, 0.95, 0.93])
plt.savefig('allocation_visualization.png', dpi=300, bbox_inches='tight')
print("Generated allocation_visualization.png with WP9.")
