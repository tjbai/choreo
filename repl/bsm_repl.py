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
