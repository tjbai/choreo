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
