# %%
import json
from llama.workflows.mad import try_parse

with open('../dumps/mad_iterative/math_baseline_with_reflection.json', 'r') as f:
    data = json.load(f)
    print(data[0])

print(sum(isinstance(d, str) for d in data))

res = []
for d in data:
    if isinstance(d, dict):
        res.append(d)
    else:
        res.append(try_parse(d))

print(sum(isinstance(d, str) for d in res))

# %%
import json
from llama.tokenizer import Tokenizer

tokenizer = Tokenizer('../model/tokenizer.model')

with open('../dumps/tot/tot_b8v4/baseline_e2e.json') as f:
    shuffled = json.load(f)

with open('../dumps/tot/tot_b8v4/baseline_e2e_shuffled.json') as f:
    unshuffled = json.load(f)

# %%
import numpy as np

def agreement_percentage(votes):
    if not votes:
        return 0.0
    unique, counts = np.unique(votes, return_counts=True)
    return np.max(counts) / len(votes)

unshuffled_agreement = [agreement_percentage(item['votes']) for item in unshuffled['solutions']]
shuffled_agreement = [agreement_percentage(item['votes']) for item in shuffled['solutions']]

print(f"System 1 (Unshuffled) Average Agreement: {np.mean(unshuffled_agreement):.4f}")
print(f"System 2 (Shuffled) Average Agreement: {np.mean(shuffled_agreement):.4f}")

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

unshuffled_top1 = []
shuffled_top1 = []
for i, (a, b) in enumerate(zip(unshuffled['solutions'], shuffled['solutions'])):
    unshuffled_votes_counter = Counter(a['votes'])
    shuffled_votes_counter = Counter(b['votes'])
    unshuffled_top1.append(unshuffled_votes_counter.most_common(1)[0][0])
    shuffled_top1.append(shuffled_votes_counter.most_common(1)[0][0])

all_candidates = list(range(8))
num_candidates = 8
candidate_to_index = {cand: cand for cand in all_candidates}

confusion_matrix = np.zeros((num_candidates, num_candidates), dtype=float)
for u_top, s_top in zip(unshuffled_top1, shuffled_top1):
    u_idx = candidate_to_index[u_top-1]
    s_idx = candidate_to_index[s_top-1]
    confusion_matrix[u_idx, s_idx] += 1
total_instances = len(unshuffled_top1)
confusion_matrix = confusion_matrix / total_instances

row_sums = confusion_matrix.sum(axis=1)
col_sums = confusion_matrix.sum(axis=0)

fig = plt.figure(figsize=(10, 8))
gs = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[6, 1], height_ratios=[1, 4], wspace=-0.31, hspace=0.02)
ax_main = fig.add_subplot(gs[1, 0])
ax_right = fig.add_subplot(gs[1, 1])
ax_top = fig.add_subplot(gs[0, 0])

# Plot main heatmap
heatmap = sns.heatmap(confusion_matrix,
                      cmap='Blues',
                      xticklabels=np.array(all_candidates)+1,
                      yticklabels=np.array(all_candidates)+1,
                      square=True,
                      ax=ax_main,
                      cbar=False)

heatmap_x_start = ax_main.get_position().x0
heatmap_x_end = ax_main.get_position().x1
heatmap_y_start = ax_main.get_position().y0
heatmap_y_end = ax_main.get_position().y1

x_positions = []
for i in range(num_candidates):
    # Get tick positions from the heatmap - this ensures perfect alignment
    tick_pos = heatmap.get_xticks()[i]
    x_positions.append(tick_pos)

ax_top.bar(x_positions, col_sums, width=0.8, align='center', color='skyblue')
ax_top.set_xlim(ax_main.get_xlim())
ax_top.set_xticks([])
ax_top.set_yticks([])
ax_top.spines['top'].set_visible(False)
ax_top.spines['right'].set_visible(False)
ax_top.spines['left'].set_visible(False)

y_positions = []
for i in range(num_candidates):
    tick_pos = heatmap.get_yticks()[i]
    y_positions.append(tick_pos)

ax_right.barh(y_positions, row_sums, height=0.8, align='center', color='skyblue')
ax_right.set_ylim(ax_main.get_ylim())
ax_right.set_yticks([])
ax_right.set_xticks([])
ax_right.spines['top'].set_visible(False)
ax_right.spines['right'].set_visible(False)
ax_right.spines['bottom'].set_visible(False)

for i, v in enumerate(row_sums):
    ax_right.text(v/2, y_positions[i], f'{v:.1%}', va='center', ha='center', fontsize=9)

for i, v in enumerate(col_sums):
    ax_top.text(x_positions[i], v/2, f'{v:.1%}', va='center', ha='center', fontsize=9)

ax_corner.axis('off')

ax_main.set_xlabel('Shuffled, B=8,V=4', fontsize=14)
ax_main.set_ylabel('Unshuffled, B=8,V=4', fontsize=14)

# Adjust positions for better alignment
ax_top_pos = ax_top.get_position()
new_ax_top_pos = [heatmap_x_start, ax_top_pos.y0, heatmap_x_end - heatmap_x_start, ax_top_pos.height]
ax_top.set_position(new_ax_top_pos)
ax_right_pos = ax_right.get_position()
new_ax_right_pos = [ax_right_pos.x0, heatmap_y_start, ax_right_pos.width, heatmap_y_end - heatmap_y_start]
ax_right.set_position(new_ax_right_pos)

plt.savefig('../figures/tot_b8v4_confusion.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 13,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
})

# Data
strategies = ["No Strategy", "Always Cooperate", "Always Defect"]
conditions = ["Baseline", "Leak Both", "Leak System", "Leak Plan"]

# Cooperation rates
cooperation_rates = np.array([
    [78.3, 63.9, 73.3, 67.9],  # No Strategy
    [87.7, 78.2, 91.7, 82.3],  # Always Cooperate
    [72.8, 46.7, 20.5, 36.2],  # Always Defect
])

# Error margins (95% confidence intervals)
error_margins = np.array([
    [3.6, 4.3, 3.9, 4.1],  # No Strategy
    [2.9, 3.7, 2.4, 3.3],  # Always Cooperate
    [4.0, 4.4, 3.6, 4.3],  # Always Defect
])

# Colors for different conditions (customizable)
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # Default colorblind-friendly palette

# Create figure and axis
fig, ax = plt.subplots(figsize=(3.5, 2.8))  # Width fits single column

bar_width = 0.18
spacing = 0.05
group_width = bar_width * len(conditions) + spacing * (len(conditions) - 1)
group_positions = np.arange(len(strategies))

for i, condition in enumerate(conditions):
    position = group_positions + (i - (len(conditions) - 1) / 2) * (bar_width + spacing)
    bars = ax.bar(
        position,
        cooperation_rates[:, i],
        bar_width,
        yerr=error_margins[:, i],
        color=colors[i],
        label=condition,
        error_kw=dict(capsize=2, elinewidth=1, capthick=1)
    )

ax.set_ylabel("Bob's Cooperation Rate (%)")
ax.set_ylim(0, 100)
ax.set_xticks(group_positions)
ax.set_xticklabels(strategies)

plt.subplots_adjust(bottom=0.21)

ax.legend(
    # bbox_to_anchor=(0.5, -0.3),
    loc='upper center',
    ncol=4,
    frameon=False
)

plt.tight_layout()

plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
})

# Data
strategies = ["No Strategy", "Always Cooperate", "Always Defect"]
conditions = ["Leak Both", "Leak System", "Leak Plan"]

# Raw cooperation rates
raw_rates = np.array([
    [78.3, 63.9, 73.3, 67.9],  # No Strategy
    [87.7, 78.2, 91.7, 82.3],  # Always Cooperate
    [72.8, 46.7, 20.5, 36.2],  # Always Defect
])

# Error margins (95% confidence intervals)
error_margins = np.array([
    [3.6, 4.3, 3.9, 4.1],  # No Strategy
    [2.9, 3.7, 2.4, 3.3],  # Always Cooperate
    [4.0, 4.4, 3.6, 4.3],  # Always Defect
])

# Calculate shifts relative to baseline
baseline_rates = raw_rates[:, 0]  # First column is baseline
shifts = raw_rates[:, 1:] - baseline_rates[:, np.newaxis]

# Calculate propagated error for shifts
# For a difference (A-B), the error is sqrt(error_A^2 + error_B^2)
baseline_errors = error_margins[:, 0]
shift_errors = np.sqrt(error_margins[:, 1:]**2 + baseline_errors[:, np.newaxis]**2)

# Colors for different conditions (customizable)
colors = ["#ff7f0e", "#2ca02c", "#d62728"]  # Colorblind-friendly

# Create figure and axis
fig, ax = plt.subplots(figsize=(3.5, 2.8))  # Width fits single column

# Width of a bar and spacing
bar_width = 0.25
spacing = 0.05
group_width = bar_width * len(conditions) + spacing * (len(conditions) - 1)
group_positions = np.arange(len(strategies))

# Plot bars
for i, condition in enumerate(conditions):
    position = group_positions + (i - (len(conditions) - 1) / 2) * (bar_width + spacing)
    bars = ax.bar(
        position,
        shifts[:, i],
        bar_width,
        yerr=shift_errors[:, i],
        color=colors[i],
        label=condition,
        error_kw=dict(capsize=2, elinewidth=1, capthick=1)
    )

# Add a horizontal line at y=0 to represent baseline
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

# Add labels, title and legend
ax.set_ylabel("Shift in Bob's Cooperation Rate (%)")
y_min = np.min(shifts - shift_errors) - 5
y_max = np.max(shifts + shift_errors) + 5
ax.set_ylim(y_min, y_max)
ax.set_xticks(group_positions)
ax.set_xticklabels(strategies)

# Create more space at the bottom for x-labels
plt.subplots_adjust(bottom=0.21)

# Place legend below the plot
ax.legend(
    bbox_to_anchor=(0.5, -0.3),
    loc='upper center',
    ncol=3,
    frameon=False
)

# Tight layout
plt.tight_layout()

# Save the figure (optional)
plt.savefig("pd_leakage_shifts.pdf", bbox_inches="tight", dpi=300)
plt.savefig("pd_leakage_shifts.png", bbox_inches="tight", dpi=300)

plt.show()
