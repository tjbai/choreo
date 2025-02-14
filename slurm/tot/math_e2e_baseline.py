import os
import json

import torch
from tqdm import tqdm

from llama import Workflow, Llama
from llama.util import find_free_port
from llama.workflows.tot import eval_solutions
from llama.workflows.tot import load_math_problems, tot_baseline

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = str(find_free_port())

llama = Llama.build(
    ckpt_dir='/scratch4/jeisner1/tjbai/llama_8b',
    tokenizer_path='/scratch4/jeisner1/tjbai/llama_8b/tokenizer.model',
    max_seq_len=8192,
    max_batch_size=8,
    model_parallel_size=1,
)
llama.model.eval()

problems = load_math_problems('/home/tbai4/llama3/data/MATH', split='val')
solutions = []

for problem in tqdm(problems):
    solutions.append(tot_baseline(
        llama=llama,
        problem=problem['problem'],
        branching_factor=8,
        voters=4,
    ))

all_correct = eval_solutions(llama, solutions, problems)
print(f'Correct: {sum(all_correct)} / {len(all_correct)}')

with open(f'baseline_e2e.json', 'w') as f:
    json.dump({'solutions': solutions, 'all_correct': all_correct}, f)

