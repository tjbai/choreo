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
    # alice_coop_pred = ['COOPERATE' in tokenizer.decode(d['outputs']['final_prediction'])[:20].upper() for d in subset]
    # alice_defect_pred = [not pred for pred in alice_coop_pred]
    # bob_defect = ['DEFECT' in d['bob_final'].upper() for d in subset]
    # p_defect_given_coop_pred = sum(b and a for b, a in zip(bob_defect, alice_coop_pred)) / (sum(alice_coop_pred) or 1)
    # p_defect_given_defect_pred = sum(b and a for b, a in zip(bob_defect, alice_defect_pred)) / (sum(alice_defect_pred) or 1)
    # exploitation_index = sum(b and a for b, a in zip(bob_defect, alice_coop_pred)) / (sum(alice_coop_pred) or 1)
    # defense_index = sum(b and a for b, a in zip(bob_defect, alice_defect_pred)) / (sum(alice_defect_pred) or 1)
    # print(f"=== {strategy} ===")
    # print(f"P(Bob defects | predicts cooperation): {p_defect_given_coop_pred:.2f}")
    # print(f"P(Bob defects | predicts defection): {p_defect_given_defect_pred:.2f}")
    # print(f"Exploitation Index: {exploitation_index:.2f}")
    # print(f"Defense Index: {defense_index:.2f}")

# for strategy in [None, 'always_cooperate', 'always_defect']:
#     subset = [d for d in data if d['strategy'] == strategy and d['alice_first']]
#     alice_cooperate_pred = ['COOPERATE' in tokenizer.decode(d['outputs']['bob_predictions'][0])[:50].upper() for d in subset]
#     alice_actual = ['COOPERATE' in d['alice_final'].upper() for d in subset]
#     bob_defect_actual = ['DEFECT' in d['bob_final'].upper() for d in subset]
#     exploited = sum(pred and defect for pred, defect in zip(alice_cooperate_pred, bob_defect_actual))
#     alice_defect_pred = [not pred for pred in alice_cooperate_pred]
#     defended = sum(pred and defect for pred, defect in zip(alice_defect_pred, bob_defect_actual))
#     print(f'=== {strategy} ===')
#     print('actual cooperate', sum(alice_actual))
#     print('predicted cooperate', sum(alice_cooperate_pred))
#     print('correct prediction', sum(a == b for a, b in zip(alice_cooperate_pred, alice_actual)))
#     print('times exploited', exploited)
#     print('times defended', defended)
