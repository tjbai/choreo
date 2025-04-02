'''
library(jsonlite)
library(exact2x2)

control <- fromJSON("/Users/bai/argo/projects/choreo/llama3/dumps/madpar/math_baseline_preft_test_correct.json")
treatment <- fromJSON("/Users/bai/argo/projects/choreo/llama3/dumps/madpar/math_cached_postft_test_correct.json")

control_num <- as.numeric(control)
treatment_num <- as.numeric(treatment)

n <- length(treatment_num)
x <- sum(treatment_num == 1 & control_num == 0)
y <- sum(treatment_num == 0 & control_num == 1)
m <- x + y

result <- mcnemarExactDP(x=x, m=m, n=n)
print(result)
'''

'''
# %%
import json
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

jsonlite = importr('jsonlite')
exact2x2 = importr('exact2x2')

with open('dumps/triviaqa/eval_large.json') as f:
    data = json.load(f)
    baseline = [(a and b) for a, b in zip(data['correct']['baseline'][0], data['correct']['baseline'][1])]
    choreo = [(a and b) for a, b in zip(data['correct']['choreographed'][0], data['correct']['choreographed'][1])]
    choreo_lin = [(a and b) for a, b in zip(data['correct']['choreographed+linearized'][0], data['correct']['choreographed+linearized'][1])]
    choreo_ft = [(a and b) for a, b in zip(data['correct']['choreographed+finetuned'][0], data['correct']['choreographed+finetuned'][1])]

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
mcnemar_exact(baseline, choreo_lin)
mcnemar_exact(baseline, choreo_ft)

import json
with open('dumps/tot/baseline_correct.json') as f:
    baseline = json.load(f)
with open('dumps/tot/choreographed_correct.json') as f:
    choreo = json.load(f)
with open('dumps/tot/choreographed_ft_correct.json') as f:
    choreo_ft = json.load(f)

with open('dumps/madpar/math_baseline_preft_test_correct.json') as f:
    baseline = json.load(f)
with open('dumps/madpar/math_cached_preft_test_correct.json') as f:
    choreo = json.load(f)
with open('dumps/madpar/math_cached_postft_test_correct.json') as f:
    choreo_ft = json.load(f)
with open('dumps/simple/from_madpar_correct.json') as f:
    distilled = json.load(f)
'''

import json
import numpy as np
import matplotlib.pyplot as plt

with open('dumps/madpar/perf_A-3.json') as f:
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
baseline_times = np.array([b['wall_time'] * 1000 for b in data['baseline']])
cached_times = np.array([c['wall_time'] * 1000 for c in data['cached']])
diffs = baseline_times / cached_times

mean_abs_diff = np.mean(diffs)
colors = ['#ff6666' if diff < 0 else '#66cc66' for diff in diffs]
edge_colors = ['#cc0000' if diff < 0 else '#009900' for diff in diffs]
scatter = ax.scatter(x, diffs, c=colors, edgecolor=edge_colors, s=80, alpha=0.9)

ax.set_xlabel('Tokens generated')
ax.set_ylabel('Speedup')
ax.set_title('Par. Multi-agent Debate')
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.3)

plt.tight_layout()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('figures/madpar_perf_scatter.png')
plt.show()

# %%
import json
with open('dumps/triviaqa/eval_large.json') as f:
    data = json.load(f)['correct']
    baseline = data['baseline']
    choreo = data['choreographed']
    choreo_lin = data['choreographed+linearized']
    choreo_ft = data['choreographed+finetuned']

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

print('Q1')
mcnemar_exact(baseline[0], choreo[0])
mcnemar_exact(baseline[0], choreo_lin[0])
mcnemar_exact(baseline[0], choreo_ft[0])

print('Q2')
mcnemar_exact(baseline[1], choreo[1])
mcnemar_exact(baseline[1], choreo_lin[1])
mcnemar_exact(baseline[1], choreo_ft[1])

def to_both(jwn):
    return [c and d for c, d in zip(jwn[0], jwn[1])]

print('both')
mcnemar_exact(to_both(baseline), to_both(choreo))
mcnemar_exact(to_both(baseline), to_both(choreo_lin))
mcnemar_exact(to_both(baseline), to_both(choreo_ft))

# %%
import json
import numpy as np

for path in [
    'perf_A-2.json',
    'perf_A-3.json',
    'perf_A-4.json',
    'perf_A-5.json',
    'perf_A-6.json',
    'perf_A-8.json',
]:
    with open(f'dumps/madpar/{path}') as f:
        data = json.load(f)
    baseline = np.array([d['wall_time'] for d in data['baseline']])
    cached = np.array([d['wall_time'] for d in data['cached']])
    print('wall time', np.mean(baseline / cached))
    baseline = np.array([d['ttft'] for d in data['baseline']])
    cached = np.array([d['ttft'] for d in data['cached']])
    print('ttft', np.mean(baseline / cached))
