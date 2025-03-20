import os
import json
from tqdm import tqdm
from llama.workflows.tot import load_math_problems, eval_solutions
from llama.workflows.simple import math_cot, math_direct
from llama.util import find_free_port, load_ckpt
from llama import Workflow, Llama

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = str(find_free_port())

workflow = Workflow.build(
    ckpt_dir='/scratch4/jeisner1/tjbai/llama_8b',
    tokenizer_path='/scratch4/jeisner1/tjbai/llama_8b/tokenizer.model',
    max_seq_len=8192,
    max_batch_size=4,
    model_parallel_size=1,
    max_nodes=100,
    use_lora=True,
    lora_rank=64,
    lora_alpha=32,
    lora_dropout=0.05,
)
workflow.model.eval()
llama = Llama(workflow.model, workflow.tokenizer)

problems = load_math_problems('/home/tbai4/llama3/data/MATH', split='val')

# from_mad: lora_step-24.pt
load_ckpt(workflow, '/scratch4/jeisner1/tjbai/checkpoints/direct/from_mad/lora_step-24.pt')
workflow.model.set_adapter_state(enabled=True)

solutions = []
for problem in tqdm(problems):
    workflow.reset()
    solutions.append(math_direct(
        workflow=workflow,
        problem=problem['problem'],
        debug=False,
    ))

workflow.model.set_adapter_state(enabled=False)
print(sum(eval_solutions(llama, solutions, problems)))

# from_tot: lora_step-149.pt
load_ckpt(workflow, '/scratch4/jeisner1/tjbai/checkpoints/direct/from_mad/lora_step-149.pt')
workflow.model.set_adapter_state(enabled=True)

solutions = []
for problem in tqdm(problems):
    workflow.reset()
    solutions.append(math_direct(
        workflow=workflow,
        problem=problem['problem'],
        debug=False,
    ))

workflow.model.set_adapter_state(enabled=False)
print(sum(eval_solutions(llama, solutions, problems)))
