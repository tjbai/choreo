# %%
import json

with open('benchmark_tot.json') as f:
    data = json.load(f)

print(data.keys())

# %%
from collections import defaultdict
voter_to_len = defaultdict(list)

for run in data['runs']:
    idx = run['problem_idx']
    branches = run['branches']
    voters = run['voters']
    baseline_mean = run['baseline']['mean']
    cached_mean = run['cached']['mean']
    abs = baseline_mean - cached_mean
    rel = abs / baseline_mean
    trial = run['baseline']['outputs'][0]
    branch_context_len = sum(len(t) for t in trial['proposal_tokens'])
    voter_to_len[run['voters']].append((branch_context_len + 250, abs)) # for system prompt, headers, etc.

# %%
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from collections import defaultdict

def plot_scaling(voter_to_len):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    fig, ax = plt.subplots(figsize=(10, 6))

    for batch in sorted(voter_to_len.keys()):
        seq_lens, times = zip(*voter_to_len[batch])
        ax.scatter(seq_lens, times, s=100, label=f'Voters={batch}')

    ax.set_xscale('symlog', base=2)
    ax.set_yscale('symlog', base=2)
    ax.grid(True, which="both", ls="-", alpha=0.2)

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.set_xlabel('Sum of CoT Length')
    ax.set_ylabel('Time (ms)')
    ax.set_title('ToT Performance Gap')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    return fig

fig = plot_scaling(voter_to_len)
plt.savefig('initial_benchmark.png')

# %%
bsz_4 = {}
bsz_4[4] = voter_to_len[4]
fig = plot_scaling(bsz_4)

baseline = {
    128: 49.450207,
    256: 90.050149,
    512: 169.244862,
    1024: 336.142778,
}

ax = fig.axes[0]
seq_lens, times = zip(*baseline.items())
seq_lens, times = zip(*baseline.items())
ax.plot(seq_lens, times, alpha=0.7, label='Baseline', color='orange')
ax.scatter(seq_lens, times, s=100, marker='x', color='orange')
plt.savefig('scaling_comp.png')

# %%
import json
from collections import defaultdict

def trick_rate(votes_list, trick_indices_list):
    dist = defaultdict(int)
    tricked = 0

    for votes, trick_indices in zip(votes_list, trick_indices_list):
        for vote in votes:
            dist[vote] += 1
        tricked += bool(set(votes) & set(trick_indices))

    return tricked, dist

# %%
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("muted")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

y_max = max(max(dist_baseline.values()), max(dist_cached.values()))
y_min = min(min(dist_baseline.values()), min(dist_cached.values()))
ax1.set_ylim(y_min, y_max)
ax2.set_ylim(y_min, y_max)

ax1.bar(dist_baseline.keys(), dist_baseline.values(), color=sns.color_palette("muted")[0])
ax2.bar(dist_cached.keys(), dist_cached.values(), color=sns.color_palette("muted")[1])

ax1.set_title('Baseline Distribution', pad=20)
ax2.set_title('Cached Distribution', pad=20)

plt.tight_layout()
plt.savefig('vote_distribution.png')

# %%
import json
from collections import defaultdict

for ckpt in [0, 99, 199, 299, 399]:
    baseline_tricked = 0
    baseline_tricked_tot = 0
    cached_tricked = 0
    cached_tricked_tot = 0
    tricked_dist = defaultdict(int)
    correct_dist = defaultdict(int)

    with open(f'dumps/checkpoint-{ckpt}_trick_results.json') as f:
        data = json.load(f)

    for ex in data:
        if ex['baseline']:
            for vote in ex['baseline_votes']:
                if (vote - 1) in ex['trick_indices']:
                    tricked_dist[vote] += 1
                    baseline_tricked_tot += 1
                else:
                    correct_dist[vote] += 1
        if ex['cached']:
            for vote in ex['cached_votes']:
                if (vote - 1) in ex['trick_indices']:
                    cached_tricked_tot += 1
        baseline_tricked += ex['baseline'] if ex['baseline'] else 0
        cached_tricked += ex['cached']

    print('consensus', baseline_tricked, cached_tricked)
    print('total', baseline_tricked_tot, cached_tricked_tot)
    print()

# %%
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("muted")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
y_max = max(max(tricked_dist.values()), max(correct_dist.values()))
y_min = min(min(tricked_dist.values()), min(correct_dist.values()))
ax1.set_ylim(y_min, y_max)
ax2.set_ylim(y_min, y_max)

ax1.bar(correct_dist.keys(), correct_dist.values(), color=sns.color_palette("muted")[2])
ax2.bar(tricked_dist.keys(), tricked_dist.values(), color=sns.color_palette("muted")[3])

ax1.set_title('Baseline Correct Distribution', pad=20)
ax2.set_title('Baseline Tricked Distribution', pad=20)

plt.tight_layout()
plt.savefig('trick_vs_correct_baseline_distribution.png')

# %%
tot = 0
N = 0
for k, v in correct_dist.items():
    tot += k * v
    N += v
print(tot / N)

tot = 0
N = 0
for k, v in tricked_dist.items():
    tot += k * v
    N += v
print(tot / N)

# %%
import json

def format_dump(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    md_output = []
    for example in data['examples']:
        md_output.append('# Example\n')
        for i, proposal in enumerate(example['result']['proposal_content'], 1):
            md_output.append(f'## Proposal {i}\n')
            md_output.append(f'{proposal}\n')
            md_output.append('---\n')

        md_output.append('\n## Votes\n')
        for i, vote in enumerate(example['result']['vote_content'], 1):
            md_output.append(f'### Vote {i}\n')
            md_output.append(f'{vote}\n')
            md_output.append('---\n')

        md_output.append('\n## Final Decision\n')
        md_output.append(f'{example["result"]["final_content"]}\n')
        md_output.append('\n\n=================\n\n')

    return '\n'.join(md_output)

with open('dumps/formatted_dump.md', 'w') as f:
    f.write(format_dump('dumps/tot_training_data.json'))

# %%
import json

with open('dumps/prisoners_baseline.jsonl') as f:
    baseline_data = [json.loads(line) for line in f]

with open('dumps/prisoners_cached.jsonl') as f:
    cached_data = [json.loads(line) for line in f]

# %%
from llama.tokenizer import Tokenizer
tokenizer = Tokenizer('model/tokenizer.model')

for thing in cached_data[205]['outputs']['alice_context']:
    print(tokenizer.decode(thing['tokens']))

# %%
import json
from llama.tokenizer import Tokenizer
tokenizer = Tokenizer('model/tokenizer.model')

with open('dumps/tot_2/lora_epoch-0_step-595.pt_e2e.json') as f:
    cached_data = json.load(f)

with open('dumps/baseline_e2e.json') as f:
    baseline_data = json.load(f)

# %%
thing = cached_data[100]
for proposal in thing['proposal_tokens']:
    print(tokenizer.decode(proposal))
for vote in thing['vote_tokens']:
    print(tokenizer.decode(vote))
print(tokenizer.decode(thing['final_tokens']))

# %%
thing = baseline_data[100]
for proposal in thing['proposal_content']:
    print(proposal)
for vote in thing['vote_content']:
    print(vote)
print(thing['final_content'])

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data = pd.DataFrame({
    'Samples': ['Baseline', '0', '400', '800', '1200'],
    'Accuracy': [40.7, 26.4, 31.4, 38.5, 39.6]
})

data['Samples_Numeric'] = pd.to_numeric(data['Samples'].replace('Baseline', '0'))

plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

sns.lineplot(data=data[data['Samples'] != 'Baseline'],
            x='Samples_Numeric', y='Accuracy',
            marker='o', linewidth=2, markersize=8)

plt.axhline(y=data[data['Samples'] == 'Baseline']['Accuracy'].values[0],
           color='red', linestyle='--', alpha=0.7,
           label='Baseline Performance')

plt.xlabel('Number of Fine-tuning Samples', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Performance Recovery with Fine-tuning', fontsize=14, pad=20)
plt.legend()

plt.xticks([0, 400, 800, 1200])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('tot_performance_recovery.png')

# %%
import os
import json
from llama.util import run_mcnemars_test

ckpts = []
for path in [
    'dumps/tot_3/lora_epoch-0_step-395.pt_e2e.json',
    'dumps/tot_3/lora_epoch-0_step-795.pt_e2e.json',
    'dumps/tot_3/lora_epoch-1_step-295.pt_e2e.json',
]:
    with open(path, 'r') as f:
            ckpts.append(json.load(f)['all_correct'])

with open('dumps/tot_3/baseline_e2e.json') as f:
    baseline = json.load(f)['all_correct']

for ckpt in ckpts:
    print(sum(ckpt))
    print(run_mcnemars_test(ckpt, baseline))

# %%
import json
from llama.tokenizer import Tokenizer

tokenizer = Tokenizer('model/tokenizer.model')

with open('dumps/prisoners_cached_3.jsonl') as f:
    data = [json.loads(line) for line in f]

cached = data[200:]
thing = cached[90]

for context in thing['outputs']['alice_context']:
    print('####')
    print(tokenizer.decode(context['tokens']))

# %%
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

with open('dumps/mad_iterative/ciar_e2e.json') as f:
    data = json.load(f)

time_diffs = (np.array(data['mad_baseline_times']) - np.array(data['cached_times']))

plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
sns.histplot(time_diffs, bins=20, kde=True)
plt.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='No Difference')
plt.axvline(x=np.mean(time_diffs), color='green', linestyle='--', alpha=0.5, label=f'Mean Difference ({np.mean(time_diffs):.1f}s)')

plt.title('CIAR 50, Baseline vs. Choreographed Absolute Wall-Clock Difference')
plt.xlabel('Time Difference (s)')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig('figures/ciar_50_absolute.png')

# %%
baseline_tot = 0
for item in data['mad_baseline']:
    for message in item['moderator_context']:
        baseline_tot += len(message['tokens'])
baseline_time = sum(data['mad_baseline_times'])
print(baseline_tot, baseline_time, baseline_tot / baseline_time)

cached_tot = 0
for item in data['cached']:
    for message in item['moderator_context']:
        cached_tot += len(message['tokens'])
cached_time = sum(data['cached_times'])
print(cached_tot, cached_time, cached_tot / cached_time)

# %%
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

with open('dumps/mad_iterative/translate_e2e_throughput.json') as f:
    data = json.load(f)

time_diffs = (np.array(data['baseline_times']) - np.array(data['cached_times']))

baseline_time = sum(data['baseline_times'])
cached_time = sum(data['cached_times'])

plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
sns.histplot(time_diffs, bins=20, kde=True)
plt.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='No Difference')
plt.axvline(x=np.mean(time_diffs), color='green', linestyle='--', alpha=0.5, label=f'Mean Difference ({np.mean(time_diffs):.1f}s)')

plt.title('CommonMT 50, Baseline vs. Choreographed Absolute Difference')
plt.xlabel('Time Difference (s)')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig('figures/commonmt_200_absolute.png')

# %%
baseline_tot = 0
for item in data['baseline_res']:
    for message in item['moderator_context']:
        baseline_tot += len(message['tokens'])

cached_tot = 0
for item in data['cached_res']:
    for message in item['moderator_context']:
        cached_tot += len(message['tokens'])

print(baseline_tot, baseline_time, baseline_tot / baseline_time)
print(cached_tot, cached_time, cached_tot / cached_time)

# %%
import json

with open('dumps/prisoners/prisoners_baseline.jsonl') as f:
    data = [json.loads(line) for line in f]

data[0]['outputs'].keys()

# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns

sns.set_style('whitegrid')
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial']
mpl.rcParams['axes.edgecolor'] = '#333333'
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['xtick.color'] = '#333333'
mpl.rcParams['ytick.color'] = '#333333'

strategies = ['Baseline', 'Always cooperate', 'Always defect']
settings = ['Leak Everything', 'Leak System Prompt', 'Leak Private CoT']

shifts = {
    'Leak Everything': [-0.11, -0.10, -0.29],
    'Leak System Prompt': [-0.09, 0.07, -0.48],
    'Leak Private CoT': [-0.13, -0.06, -0.35]
}

fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

x = np.arange(len(strategies))
width = 0.2
multiplier = 0

colors = {
    'Leak Everything': '#3182bd',
    'Leak System Prompt': '#e6550d',
    'Leak Private CoT': '#31a354'
}

for setting in settings:
    offset = width * multiplier - width
    setting_shifts = shifts[setting]

    rects = ax.bar(x + offset, setting_shifts, width * 0.9,
                   label=setting, color=colors[setting],
                   edgecolor='white', linewidth=0.7,
                   alpha=0.85)

    for rect, value in zip(rects, setting_shifts):
        height = rect.get_height() if value >= 0 else rect.get_height() - 0.05
        ax.annotate(f'{value:.2f}',
                   xy=(rect.get_x() + rect.get_width() / 2, height),
                   xytext=(0, 5 if value >= 0 else 0),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=10,
                   color='#333333', fontweight='medium')

    multiplier += 1

ax.axhline(y=0, color='#666666', linestyle='-', linewidth=1, alpha=0.7)
ax.set_title("Information Leakage Ablations", fontsize=18, pad=20, fontweight='bold', color='#333333')
ax.set_ylabel('Bob Cooperate Rate Shift', fontsize=14, labelpad=10, color='#333333')
ax.set_xlabel('Alice Strategy', fontsize=14, labelpad=10, color='#333333')

ax.set_xticks(x)
ax.set_xticklabels([s.replace(' ', '\n') for s in strategies], fontsize=12)

ax.set_ylim(-0.55, 0.15)
ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='#cccccc')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

legend = ax.legend(title='Leakage Condition', title_fontsize=12,
                  loc='lower left', frameon=True, framealpha=0.95,
                  edgecolor='#cccccc')

ax.annotate('Only leak system prompt increases cooperation?',
           xy=(1.05, 0.07), xytext=(1.4, 0.12),
           arrowprops=dict(arrowstyle='->', color='#666666', lw=1),
           fontsize=11, ha='center', color='#333333')

for spine in ['left', 'bottom']:
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_color('#666666')
    ax.spines[spine].set_linewidth(0.8)

plt.tight_layout()
plt.savefig('figuresinformation_leakage_effects.png')
plt.show()

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style('whitegrid')
sns.set_context("paper", font_scale=1.2)
colors = sns.color_palette("Blues_d", 4)

n2 = dict([(0, 39), (1, 33)])
n4 = dict([(0, 40), (1, 34), (2, 32), (3, 35)])
n8 = dict([(0, 41), (1, 35), (2, 24), (3, 20), (4, 30), (5, 24), (6, 22), (7, 17)])
n16 = dict([(0, 34), (1, 31), (2, 15), (3, 12), (4, 18), (5, 7), (6, 5), (7, 5), (8, 3), (9, 3), (10, 2), (11, 1), (12, 2), (13, 1), (14, 2), (15, 2)])

def normalize_data(data, n):
    return [(data.get(i, 0) / 50) * 100 for i in range(n)]

fig, axs = plt.subplots(2, 2, figsize=(12, 9), dpi=100)
fig.suptitle('QA Correctness (%) by Position, Parallel Fine-tuned', fontsize=20, y=1.0, fontweight='bold')

def style_axis(ax, positions, counts, color, title, show_all_xticks=True):
    bars = ax.bar(positions, counts, color=color, width=0.7, edgecolor='white', linewidth=1)
    ax.set_title(title, fontsize=16, pad=10)

    ax.set_ylim(0, 100)
    ax.axhline(y=0, color='grey', linestyle='-', linewidth=0.5, alpha=0.3)

    if show_all_xticks:
        ax.set_xticks(positions)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    return bars

positions2 = list(range(1, 2+1))
counts2 = normalize_data(n2, 2)
style_axis(axs[0, 0], positions2, counts2, colors[0], 'n=2')

positions4 = list(range(1, 4+1))
counts4 = normalize_data(n4, 4)
style_axis(axs[0, 1], positions4, counts4, colors[1], 'n=4')

positions8 = list(range(1, 8+1))
counts8 = normalize_data(n8, 8)
style_axis(axs[1, 0], positions8, counts8, colors[2], 'n=8')

positions16 = list(range(1, 16+1))
counts16 = normalize_data(n16, 16)
style_axis(axs[1, 1], positions16, counts16, colors[3], 'n=16', show_all_xticks=False)
axs[1, 1].set_xticks(positions16)

plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.08)
plt.savefig('figures/qa_parallel_by_position.png', bbox_inches='tight')

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style('whitegrid')
sns.set_context("paper", font_scale=1.2)

colors = sns.color_palette("Blues_d", 4)

n2 = {1: 34, 2: 37}
n4 = {3: 36, 2: 41, 4: 31, 1: 29}
n8 = {3: 36, 7: 29, 8: 34, 2: 39, 4: 35, 5: 40, 6: 38, 1: 31}
n16 = {3: 31, 7: 29, 8: 35, 9: 39, 10: 32, 11: 33, 12: 37, 13: 36, 14: 32, 15: 35, 2: 38, 5: 40, 6: 36, 16: 35, 4: 30, 1: 31}

def normalize_data(data, n):
    return [(data.get(i, 0) / 50) * 100 for i in range(1, n+1)]

fig, axs = plt.subplots(2, 2, figsize=(12, 9), dpi=100)
fig.suptitle('QA Correctness (%) by Position, Baseline', fontsize=20, y=1.0, fontweight='bold')

def style_axis(ax, positions, counts, color, title, show_all_xticks=True):
    bars = ax.bar(positions, counts, color=color, width=0.7, edgecolor='white', linewidth=1)
    ax.set_title(title, fontsize=16, pad=10)

    ax.set_ylim(0, 100)
    ax.axhline(y=0, color='grey', linestyle='-', linewidth=0.5, alpha=0.3)

    if show_all_xticks:
        ax.set_xticks(positions)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    return bars

positions2 = list(range(1, 2+1))
counts2 = normalize_data(n2, 2)
style_axis(axs[0, 0], positions2, counts2, colors[0], 'n=2')

positions4 = list(range(1, 4+1))
counts4 = normalize_data(n4, 4)
style_axis(axs[0, 1], positions4, counts4, colors[1], 'n=4')

positions8 = list(range(1, 8+1))
counts8 = normalize_data(n8, 8)
style_axis(axs[1, 0], positions8, counts8, colors[2], 'n=8')

positions16 = list(range(1, 16+1))
counts16 = normalize_data(n16, 16)
style_axis(axs[1, 1], positions16, counts16, colors[3], 'n=16', show_all_xticks=False)
axs[1, 1].set_xticks(positions16)

plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.08)
plt.savefig('figures/qa_baseline_by_position.png', bbox_inches='tight')
