# %%
import json

with open('../dumps/prisoners/prisoners_ft_eval_baseline.jsonl') as f:
    data = [json.loads(line) for line in f]

data[301]['ckpt_path']

# %%
import json

with open('dumps/prisoners/prisoners_baseline_train.jsonl') as f:
    data = [json.loads(line) for line in f]

len(data)

# %%
import json
from llama.tokenizer import Tokenizer

tokenizer = Tokenizer('model/tokenizer.model')

with open('dumps/prisoners/prisoners_cached_paired_predict.jsonl') as f:
    data = [json.loads(line) for line in f]

for strategy in [None, 'always_cooperate', 'always_defect']:
    subset = [d for d in data if d['strategy'] == strategy]
    alice_coop_pred = ['always defect' in tokenizer.decode(d['outputs']['bob_predictions'][-1]).lower() for d in subset]
    print(sum(alice_coop_pred))
    alice_coop_pred = ['COOPERATE' in tokenizer.decode(d['outputs']['final_prediction'])[:20].upper() for d in subset]
    alice_defect_pred = [not pred for pred in alice_coop_pred]
    bob_defect = ['DEFECT' in d['bob_final'].upper() for d in subset]
    p_defect_given_coop_pred = sum(b and a for b, a in zip(bob_defect, alice_coop_pred)) / (sum(alice_coop_pred) or 1)
    p_defect_given_defect_pred = sum(b and a for b, a in zip(bob_defect, alice_defect_pred)) / (sum(alice_defect_pred) or 1)
    exploitation_index = sum(b and a for b, a in zip(bob_defect, alice_coop_pred)) / (sum(alice_coop_pred) or 1)
    defense_index = sum(b and a for b, a in zip(bob_defect, alice_defect_pred)) / (sum(alice_defect_pred) or 1)
    print(f"=== {strategy} ===")
    print(f"P(Bob defects | predicts cooperation): {p_defect_given_coop_pred:.2f}")
    print(f"P(Bob defects | predicts defection): {p_defect_given_defect_pred:.2f}")
    print(f"Exploitation Index: {exploitation_index:.2f}")
    print(f"Defense Index: {defense_index:.2f}")

for strategy in [None, 'always_cooperate', 'always_defect']:
    subset = [d for d in data if d['strategy'] == strategy and d['alice_first']]
    alice_cooperate_pred = ['COOPERATE' in tokenizer.decode(d['outputs']['bob_predictions'][0])[:50].upper() for d in subset]
    alice_actual = ['COOPERATE' in d['alice_final'].upper() for d in subset]
    bob_defect_actual = ['DEFECT' in d['bob_final'].upper() for d in subset]
    exploited = sum(pred and defect for pred, defect in zip(alice_cooperate_pred, bob_defect_actual))
    alice_defect_pred = [not pred for pred in alice_cooperate_pred]
    defended = sum(pred and defect for pred, defect in zip(alice_defect_pred, bob_defect_actual))
    print(f'=== {strategy} ===')
    print('actual cooperate', sum(alice_actual))
    print('predicted cooperate', sum(alice_cooperate_pred))
    print('correct prediction', sum(a == b for a, b in zip(alice_cooperate_pred, alice_actual)))
    print('times exploited', exploited)
    print('times defended', defended)

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

def get_coop(data):
    return ['cooperate' in d['bob_final'].lower() for d in data]

# baseline
with open('dumps/prisoners/prisoners_baseline_large.jsonl') as f:
    data = [json.loads(line) for line in f][0]
    baseline_pre = get_coop([d for d in data if d['strategy'] is None])
    cooperate_pre = get_coop([d for d in data if d['strategy'] == 'always_cooperate'])
    defect_pre = get_coop([d for d in data if d['strategy'] == 'always_defect'])

with open('dumps/prisoners/prisoners_cached_paired_large.jsonl') as f:
    data = [json.loads(line) for line in f]
    baseline_untrained = get_coop([d for d in data if d['strategy'] is None])
    cooperate_untrained = get_coop([d for d in data if d['strategy'] == 'always_cooperate'])
    defect_untrained = get_coop([d for d in data if d['strategy'] == 'always_defect'])

mcnemar_exact(baseline_pre, baseline_untrained)
mcnemar_exact(cooperate_pre, cooperate_untrained)
mcnemar_exact(defect_pre, defect_untrained)

# after fine-tuning
with open('dumps/prisoners/prisoners_ft_eval_baseline_large.jsonl') as f:
    data = [json.loads(line) for line in f]
    baseline_post = get_coop([
        d for d in data
        if d['ckpt_path'] == '/scratch4/jeisner1/tjbai/checkpoints/prisoners/baseline/lora_epoch-1_step-155.pt'
    ])
with open('dumps/prisoners/prisoners_ft_eval_always_defect_large.jsonl') as f:
    data = [json.loads(line) for line in f]
    defect_post = get_coop([
        d for d in data
        if d['ckpt_path'] == '/scratch4/jeisner1/tjbai/checkpoints/prisoners/always_defect/lora_epoch-0_step-95.pt'
    ])
with open('dumps/prisoners/prisoners_ft_eval_always_cooperate_large.jsonl') as f:
    data = [json.loads(line) for line in f]
    cooperate_post = get_coop([
        d for d in data
        if d['ckpt_path'] == '/scratch4/jeisner1/tjbai/checkpoints/prisoners/always_cooperate/lora_epoch-0_step-195.pt'
    ])

mcnemar_exact(baseline_pre, baseline_post)
mcnemar_exact(cooperate_pre, cooperate_post)
mcnemar_exact(defect_pre, defect_post)

# %%
with open('dumps/prisoners/prisoners_baseline_large.jsonl') as f:
    baseline_data = [json.loads(line) for line in f][0]

with open('dumps/prisoners/prisoners_cached_paired_ablations.jsonl') as f:
    data = [json.loads(line) for line in f]
    sys_leak = [d for d in data if d['leak_setting'] == [True, False]]
    plan_leak = [d for d in data if d['leak_setting'] == [False, True]]

def align(a, b):
    in_a = set(d['seed'] for d in a)
    in_b = set(d['seed'] for d in b)
    res_a = [d for d in a if d['seed'] in in_a and d['seed'] in in_b]
    res_b = [d for d in b if d['seed'] in in_a and d['seed'] in in_b]
    return sorted(res_a, key=lambda d: d['seed']), sorted(res_b, key=lambda d: d['seed'])

import statistics
for strategy in [None, 'always_cooperate', 'always_defect']:
    for data in [sys_leak, plan_leak]:
        baseline = [d for d in baseline_data if d['strategy'] == strategy]
        leak = [d for d in data if d['strategy'] == strategy]
        baseline, leak = align(baseline, leak)
        assert len(baseline) == len(leak)
        print(strategy)
        print(statistics.mean(get_coop(baseline)), statistics.mean(get_coop(leak)))
        mcnemar_exact(get_coop(baseline), get_coop(leak))
        print('###')
