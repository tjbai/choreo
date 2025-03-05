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
