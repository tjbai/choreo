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
# import json
# import numpy as np
# import matplotlib.pyplot as plt

# with open('dumps/mad/perf.json') as f:
#     data = json.load(f)

# plt.rcParams.update({
#     'font.family': 'serif',
#     'font.size': 14,
#     'axes.titlesize': 14,
#     'savefig.dpi': 300,
#     'figure.dpi': 300,
# })

# fig, ax = plt.subplots(figsize=(4, 4.5))

# x = np.array([d.get('tokens', i) for i, d in enumerate(data['baseline'])])
# baseline_times = np.array([b['wall_time'] * 1000 for b in data['baseline']])
# cached_times = np.array([c['wall_time'] * 1000 for c in data['cached']])
# diffs = baseline_times / cached_times

# mean_abs_diff = np.mean(diffs)
# colors = ['#ff6666' if diff < 0 else '#66cc66' for diff in diffs]
# edge_colors = ['#cc0000' if diff < 0 else '#009900' for diff in diffs]
# scatter = ax.scatter(x, diffs, c=colors, edgecolor=edge_colors, s=80, alpha=0.9)

# ax.set_xlabel('Tokens generated')
# ax.set_ylabel('Speedup')
# ax.set_title('Iter. Multi-agent Debate')
# ax.axhline(y=1, color='gray', linestyle='--', alpha=0.3)

# plt.tight_layout()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.savefig('figures/mad_perf_scatter.png')
# plt.show()

# %%
import numpy as np
def bootstrap_ci(baseline, comparison, n_bootstrap=10000):
    n = len(baseline)
    rats = []
    print(np.mean([a/b for a, b in zip(baseline, comparison)]))
    for _ in range(n_bootstrap):
        indices = np.random.choice(range(n), size=n, replace=True)
        baseline_sample = np.array([baseline[i] for i in indices])
        comparison_sample = np.array([comparison[i] for i in indices])
        rats.append(np.mean(baseline_sample / comparison_sample))
    lower = np.percentile(rats, 2.5)
    upper = np.percentile(rats, 97.5)
    return lower, upper

import json
with open('dumps/madpar/perf_A-3.json') as f:
    data = json.load(f)
print(bootstrap_ci([d['wall_time'] for d in data['baseline']], [d['wall_time'] for d in data['cached']]))
print(bootstrap_ci([d['ttft'] for d in data['baseline']], [d['ttft'] for d in data['cached']]))

# %%
import json
with open('dumps/mad/baseline_correct.json') as f:
    baseline = json.load(f)
with open('dumps/mad/baseline_correct_p2.json') as f:
    baseline.extend(json.load(f))
with open('dumps/mad/choreo_correct.json') as f:
    choreo = json.load(f)
with open('dumps/mad/choreo_correct_p2.json') as f:
    choreo.extend(json.load(f))
with open('dumps/mad/choreo_ft_correct.json') as f:
    choreo_ft = json.load(f)
with open('dumps/simple/from_mad_correct.json') as f:
    distilled = json.load(f)

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
exact2x2 = importr('exact2x2')
def mcnemar_exact(control, treatment):
    control_r = ro.FloatVector(control)
    treatment_r = ro.FloatVector(treatment)
    n = len(treatment)
    x = sum((treatment_r[i] == 1) & (control_r[i] == 0) for i in range(n))
    y = sum((treatment_r[i] == 0) & (control_r[i] == 1) for i in range(n))
    m = x + y
    result = exact2x2.mcnemarExactDP(x=x, m=m, n=n)
    print(result)

mcnemar_exact(baseline, choreo)
mcnemar_exact(baseline, choreo_ft)
mcnemar_exact(baseline, distilled)

# %%
