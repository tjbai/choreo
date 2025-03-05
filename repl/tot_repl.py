# %%
import json
from llama.workflows.mad import try_parse

with open('../dumps/mad_iterative/math_baseline_with_reflection.json', 'r') as f:
    data = json.load(f)
    print(data[0])

print(sum(isinstance(d, str) for d in data))

res = []
for d in data:
    if isinstance(d, dict):
        res.append(d)
    else:
        res.append(try_parse(d))

print(sum(isinstance(d, str) for d in res))

# %%

for d in res:
    if isinstance(d, str):
        print(d)
        print('###')
