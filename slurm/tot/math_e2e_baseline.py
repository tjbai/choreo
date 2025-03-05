import os
import json

import torch
from tqdm import tqdm

from llama import Workflow, Llama
from llama.util import find_free_port
from llama.workflows.tot import eval_solutions
from llama.workflows.tot import load_math_problems, tot_baseline, tot_baseline_shuffled

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = str(find_free_port())

workflow = Workflow.build(
    ckpt_dir='/scratch4/jeisner1/tjbai/llama_8b',
    tokenizer_path='/scratch4/jeisner1/tjbai/llama_8b/tokenizer.model',
    max_seq_len=8*8192,
    max_batch_size=1,
    model_parallel_size=1,
    max_nodes=100,
    use_lora=False,
)

llama = Llama(workflow.model, workflow.tokenizer)

problems = load_math_problems('/home/tbai4/llama3/data/MATH', split='val')

solutions = []
for problem in tqdm(problems):
    workflow.reset()
    solutions.append(tot_baseline(
        workflow=workflow,
        problem=problem['problem'],
        branching_factor=8,
        voters=8,
    ))

llama.model.reshape_cache(4)
all_correct = eval_solutions(llama, solutions, problems)
print(f'Correct: {sum(all_correct)} / {len(all_correct)}')
llama.model.reshape_cache(1)

with open(f'/home/tbai4/llama3/dumps/tot/baseline_e2e.json', 'w') as f:
    json.dump({'solutions': solutions, 'all_correct': all_correct}, f)

solutions = []
for problem in tqdm(problems):
    workflow.reset()
    solutions.append(tot_baseline_shuffled(
        workflow=workflow,
        problem=problem['problem'],
        branching_factor=8,
        voters=8,
    ))

llama.model.reshape_cache(4)
all_correct = eval_solutions(llama, solutions, problems)
print(f'Correct: {sum(all_correct)} / {len(all_correct)}')
llama.model.reshape_cache(1)

with open(f'/home/tbai4/llama3/dumps/tot/baseline_e2e_shuffled.json', 'w') as f:
    json.dump({'solutions': solutions, 'all_correct': all_correct}, f)
