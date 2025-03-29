import os
import json
from tqdm import tqdm
from llama.workflows.tot import load_math_problems
from llama.workflows.simple import math_cot, math_direct
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

problems = load_math_problems('/home/tbai4/llama3/data/MATH', split='test')[:500]

# baseline direct "input-output" prompting
solutions = []
for problem in tqdm(problems):
    workflow.reset()
    solutions.append(math_direct(
        workflow=workflow,
        problem=problem['problem'],
        debug=False,
    ))

    with open('/home/tbai4/llama3/dumps/math_baseline_direct_test.json', 'w') as f:
        json.dump(solutions, f)

# # baseline with reflection on MATH
# solutions = []
# for problem in tqdm(problems):
#     workflow.reset()
#     solutions.append(math_cot(
#         workflow=workflow,
#         problem=problem['problem'],
#         enable_reflection=True,
#         debug=False,
#     ))

# with open('/home/tbai4/llama3/dumps/simple/math_baseline_with_reflection.json', 'w') as f:
#     json.dump(solutions, f)

# # baseline without reflection on MATH
# solutions = []
# for problem in tqdm(problems):
#     workflow.reset()
#     solutions.append(math_cot(
#         workflow=workflow,
#         problem=problem['problem'],
#         enable_reflection=False,
#         debug=False,
#     ))

# with open('/home/tbai4/llama3/dumps/simple/math_baseline_without_reflection.json', 'w') as f:
#     json.dump(solutions, f)

# # baseline best of 8
# solutions = []
# for problem in tqdm(problems):
#     workflow.reset()
#     solutions.append(math_cot(
#         workflow=workflow,
#         problem=problem['problem'],
#         enable_reflection=False,
#         best_of_n=8,
#         debug=False,
#     ))
#
# with open('/home/tbai4/llama3/dumps/simple/math_baseline_best_of_8.json', 'w') as f:
#     json.dump(solutions, f)
