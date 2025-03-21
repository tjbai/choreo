import os
import json
from tqdm import tqdm
from llama.workflows.mad import mad_baseline, mad_cached
from llama.workflows.tot import eval_solutions
from llama.util import find_free_port
from llama import Workflow, Llama
from datasets import load_dataset

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = str(find_free_port())

workflow = Workflow.build(
    ckpt_dir='/scratch4/jeisner1/tjbai/llama_8b',
    tokenizer_path='/scratch4/jeisner1/tjbai/llama_8b/tokenizer.model',
    max_seq_len=2*8192,
    max_batch_size=4,
    model_parallel_size=1,
    max_nodes=100,
    use_lora=False,
)

llama = Llama(workflow.model, workflow.tokenizer)

problems = load_dataset('openai/gsm8k', 'main', split='train')[:500]

samples = []
for problem, solution in tqdm(zip(problems['question'], problems['solution'])):
    workflow.reset()
    outputs = mad_baseline(
        workflow=workflow,
        problem=problem,
        max_rounds=3,
    )
    samples.append({
        'inputs': {'problem': problem, 'solution': solution},
        'outputs': outputs,
    })
    with open('/home/tbai4/llama3/dumps/mad/gsm8k_baseline_e2e.json', 'w') as f:
        json.dump(samples, f)

problems = load_dataset('openai/gsm8k', 'main', split='test')[:500]

samples = []
for problem, solution in tqdm(zip(problems['question'], problems['solution'])):
    workflow.reset()
    outputs = mad_baseline(
        workflow=workflow,
        problem=problem,
        max_rounds=3,
    )
    samples.append({
        'inputs': {'problem': problem, 'solution': solution},
        'outputs': outputs,
    })
print('baseline correct', sum(eval_solutions(
    llama,
    [d['inputs']['problem'] for d in samples],
    [d['inputs']['solution'] for d in samples],
)))

samples = []
for problem, solution in tqdm(zip(problems['question'], problems['solution'])):
    workflow.reset()
    outputs = mad_cached(
        workflow=workflow,
        problem=problem,
        max_rounds=3,
    )
    samples.append({
        'inputs': {'problem': problem, 'solution': solution},
        'outputs': outputs,
    })
print('cached correct', sum(eval_solutions(
    llama,
    [d['inputs']['problem'] for d in samples],
    [d['inputs']['solution'] for d in samples],
)))

'''
# MATH dataset
problems = load_math_problems('/home/tbai4/llama3/data/MATH', split='train')[:500]

# MAD baseline on MATH
samples = []
for problem in tqdm(problems):
    workflow.reset()
    outputs = mad_baseline(
        workflow=workflow,
        problem=problem['problem'],
        max_rounds=3,
        debug=False,
    )
    samples.append({
        'inputs': {'problem': problem['problem']},
        'outputs': outputs,
    })

with open('/home/tbai4/llama3/dumps/mad/math_baseline_e2e.json', 'w') as f:
    json.dump(samples, f)

# MAD cached on MATH
samples = []
for problem in tqdm(problems):
    workflow.reset()
    outputs = mad_cached(
        workflow=workflow,
        problem=problem['problem'],
        max_rounds=3,
    )
    samples.append({
        'inputs': {'problem': problem['problem']},
        'outputs': outputs,
    })

with open('/home/tbai4/llama3/dumps/mad/math_cached_e2e.json', 'w') as f:
    json.dump(samples, f)
'''
