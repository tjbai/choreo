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

with open('dumps/prisoners/prisoners_baseline_predict.jsonl') as f:
    data = [json.loads(line) for line in f]

for strategy in [None, 'always_cooperate', 'always_defect']:
    subset = [d for d in data if d['strategy'] == strategy]
    predictions = ['COOPERATE' in tokenizer.decode(d['outputs']['final_prediction'])[:20].upper() for d in subset]
    alt_predictions = ['DEFECT' in tokenizer.decode(d['outputs']['final_prediction'])[:20].upper() for d in subset]
    actual = ['COOPERATE' in d['alice_final'].upper() for d in subset]
    print(f'=== {strategy} ===')
    print('predicted cooperate', sum(predictions))
    print('predicted defect', sum(alt_predictions))
    print('actual cooperate', sum(actual))
    print('correct prediction', sum(a == b for a, b in zip(predictions, actual)))
