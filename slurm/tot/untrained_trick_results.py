import os
from llama import Workflow, Llama
from llama.util import load_model_and_tokenizer

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

workflow = Workflow.build(
    ckpt_dir='/scratch4/jeisner1/tjbai/llama_8b',
    tokenizer_path='/scratch4/jeisner1/tjbai/llama_8b/tokenizer.model',
    max_seq_len=8192,
    max_batch_size=4,
    model_parallel_size=1,
    max_nodes=20,
)
llama = Llama(workflow.model, workflow.tokenizer)

import torch
import json
import random
from llama.workflows.tot import load_math_problems, benchmark_tricky_tot
from tqdm import tqdm

random.seed(42)
problems = load_math_problems('/home/tbai4/llama3/data/MATH', split='val')
problems = random.sample(problems, 200)

print(f'Loaded checkpoint-{id}')
print(f'Memory allocated: {torch.cuda.memory_allocated()}')

comps = []
for problem in problems:
    comps.append(benchmark_tricky_tot(
        llama=llama,
        workflow=workflow,
        problem=problem['problem'],
        branching_factor=8,
        voters=4
    ))

with open('checkpoint-0_trick_results.json', 'w') as f:
    json.dump(comps, f)

