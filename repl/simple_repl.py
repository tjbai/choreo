# %%
import json
from llama.tokenizer import Tokenizer

tokenizer = Tokenizer('model/tokenizer.model')

with open('dumps/tot/tot_b8v4/baseline_e2e.json') as f:
    data = json.load(f)

# %%
from llama.workflows.tot import load_math_problems

problems = load_math_problems('data/MATH', split='train')
p2s = {p['problem']: p['solution'] for p in problems}

import torch
from llama.tokenizer import Tokenizer
tokenizer = Tokenizer('model/tokenizer.model')
samples = []
for i in range(1000):
    sample = torch.load(f'dumps/tot/tot_data/problem_{i}.pt', map_location='cpu')
    problem = sample['problem']
    solution = tokenizer.decode(sample['result']['final_tokens'])
    samples.append({
        'inputs': {'problem': problem, 'solution': p2s[problem]},
        'outputs': {'solution': ': '.join(solution.split(': ')[1:])}
    })

with open('dumps/simple/from_tot.json', 'w') as f:
    json.dump(samples, f)

# %%
import json
from llama.workflows.tot import load_math_problems
from llama.tokenizer import Tokenizer

tokenizer = Tokenizer('model/tokenizer.model')
problems = load_math_problems('data/MATH', split='train')[:500]

with open('dumps/mad/math_baseline_e2e.json') as f:
    data = json.load(f)
    assert len(data) == 500

samples = []
for problem, sample in zip(problems, data):
    if isinstance(sample['outputs']['decision'], str):
        continue
    samples.append({
        'inputs': {'problem': problem['problem'], 'solution': problem['solution']},
        'outputs': {'solution': sample['outputs']['decision']['Answer']}
    })

with open('dumps/simple/from_mad.json', 'w') as f:
    json.dump(samples, f)

# %%
import json
from llama.workflows.tot import load_math_problems
from llama.tokenizer import Tokenizer
tokenizer = Tokenizer('model/tokenizer.model')

problems = load_math_problems('data/MATH', split='train')
p2s = {p['problem']: p['solution'] for p in problems}

with open('dumps/madpar/math_baseline_e2e.json') as f:
    data = json.load(f)

samples = []
for problem, d in zip(problems, data):
    for answer in d['outputs']['debate_tokens'][-1]:
        samples.append({
            'inputs': {'problem': problem['problem'], 'solution': p2s[problem['problem']]},
            'outputs': {'solution': tokenizer.decode(answer)}
        })

with open('dumps/simple/from_madpar.json', 'w') as f:
    json.dump(samples, f)
