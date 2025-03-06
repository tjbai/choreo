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
