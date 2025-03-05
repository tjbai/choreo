import os
import json
from tqdm import tqdm
from llama.workflows.mad import math_baseline_faithful, math_mad_cached
from llama.workflows.tot import load_math_problems
from llama.util import find_free_port
from llama import Workflow, Llama

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

# MATH dataset
problems = load_math_problems('/home/tbai4/llama3/data/MATH', split='val')

# MAD baseline on MATH
# solutions = []
# for problem in tqdm(problems):
#     workflow.reset()
#     solutions.append(math_baseline_faithful(
#         workflow=workflow,
#         problem=problem['problem'],
#         max_rounds=3,
#         debug=False,
#     ))

# with open('/home/tbai4/llama3/dumps/mad/math_baseline_e2e.json', 'w') as f:
#     json.dump(solutions, f)

# MAD cached on MATH
solutions = []
for problem in tqdm(problems):
    workflow.reset()
    solutions.append(math_mad_cached(
        workflow=workflow,
        problem=problem['problem'],
        max_rounds=3,
    ))

with open('/home/tbai4/llama3/dumps/mad/math_cached_e2e.json', 'w') as f:
    json.dump(solutions, f)
