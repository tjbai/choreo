# %%
import json
from llama.tokenizer import Tokenizer
import numpy as np

tokenizer = Tokenizer('model/tokenizer.model')
with open('dumps/bsm/baseline_e2e.json') as f:
    data = [d['outputs'] for d in json.load(f)]

covered = []
for d in data:
    covered.append(
        sum(
            concept.lower() in tokenizer.decode(d['merge_tokens'][0])
            for concept in d['concept_groups'][0]
        )
    )

print(np.mean(covered) / 15, np.std(np.array(covered) / 15))

# %%
with open('dumps/bsm/initial_eval.json') as f:
    data = json.load(f)['raw_data']

for d in data:

# %%
import json
import numpy as np

with open('dumps/bsm/preft_eval.json') as f:
    pre = json.load(f)
    baseline = pre['raw_data']['baseline']['group2_coverage']
    choreo = pre['raw_data']['cached']['group2_coverage']
    choreo_lin = pre['raw_data']['cached_compact']['group2_coverage']

with open('dumps/bsm/postft_eval.json') as f:
    post = json.load(f)
    choreo_ft = post['lora_step-104']['raw_data']['group2_coverage']

def bootstrap_ci(baseline, comparison, n_bootstrap=10000):
    n = len(baseline)
    diffs = []
    orig_diff = np.mean(comparison) - np.mean(baseline)
    for _ in range(n_bootstrap):
        indices = np.random.choice(range(n), size=n, replace=True)
        baseline_sample = [baseline[i] for i in indices]
        comparison_sample = [comparison[i] for i in indices]
        diffs.append(np.mean(comparison_sample) - np.mean(baseline_sample))
    lower = np.percentile(diffs, 2.5)
    upper = np.percentile(diffs, 97.5)
    return orig_diff, (lower, upper)

choreo_diff, choreo_ci = bootstrap_ci(baseline, choreo)
print(choreo_diff, choreo_ci)
choreo_lin_diff, choreo_lin_ci = bootstrap_ci(baseline, choreo_lin)
print(choreo_lin_diff, choreo_lin_ci)
choreo_ft_diff, choreo_ft_ci = bootstrap_ci(baseline, choreo_ft)
print(choreo_ft_diff, choreo_ft_ci)

# %%
def bootstrap_test(baseline, comparison, n_bootstrap=10000, alpha=0.05):
    n = len(baseline)
    mean_diff = np.mean(comparison) - np.mean(baseline)
    pooled = baseline + comparison
    count = 0
    for _ in range(n_bootstrap):
        indices = np.random.choice(range(2*n), size=2*n, replace=True)
        boot_baseline = [pooled[i] for i in indices[:n]]
        boot_comparison = [pooled[i] for i in indices[n:]]
        boot_diff = np.mean(boot_comparison) - np.mean(boot_baseline)
        if boot_diff <= mean_diff:
            count += 1
    p_value = count / n_bootstrap
    is_lower = p_value <= alpha
    return p_value, is_lower, mean_diff

with open('dumps/bsm/preft_eval.json') as f:
    pre = json.load(f)
    print(bootstrap_test(pre['raw_data']['baseline']['group1_coverage'], pre['raw_data']['baseline']['group2_coverage']))

# %%
