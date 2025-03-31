'''
# %%
import json
from llama.workflows.mad import load_ciar

with open('improved_ciar_cached.json') as f:
    ours = json.load(f)

with open('ciar_baseline_without_reflection.json') as f:
    baseline = json.load(f)

print(sum(b is None for b in baseline)) # why so many None?

ciar = load_ciar('data/CIAR', start=0, end=50)

for attempt, b, problem in zip(ours, baseline, ciar):
    if attempt['decision'] is None or b is None:
        continue
    print('Ours', attempt['decision']['Answer'])
    print('Baseline', b['Answer'])
    print('Ground truth', problem['answer'])
    print('###')

# %%
# what is percent well-formed for baseline? around 471/500 ~= 94%
'''

# %%
import json
import numpy as np
import matplotlib.pyplot as plt

with open('dumps/mad/perf.json') as f:
    data = json.load(f)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.titlesize': 14,
    'savefig.dpi': 300,
    'figure.dpi': 300,
})

fig, ax = plt.subplots(figsize=(4, 4.5))

x = np.array([d.get('tokens', i) for i, d in enumerate(data['baseline'])])
baseline_times = np.array([b['cuda_time'] * 1000 for b in data['baseline']])
cached_times = np.array([c['cuda_time'] * 1000 for c in data['cached']])
# diffs = baseline_times - cached_times
diffs = baseline_times / cached_times

mean_abs_diff = np.mean(diffs)
colors = ['#ff6666' if diff < 0 else '#66cc66' for diff in diffs]
edge_colors = ['#cc0000' if diff < 0 else '#009900' for diff in diffs]
scatter = ax.scatter(x, diffs, c=colors, edgecolor=edge_colors, s=80, alpha=0.9)

ax.set_xlabel('Tokens generated')
ax.set_ylabel('Speedup')
ax.set_title('Iter. Multi-agent Debate')
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.3)

plt.tight_layout()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('figures/mad_perf_scatter.png')
