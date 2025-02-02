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
    use_lora=True,
    lora_rank=8,
    lora_alpha=16,
    lora_dropout=0.1
)

print(workflow.model.get_trainable_param_percentage())

import torch
import json
import random
from llama import Llama
from llama.workflows.tot import load_math_problems, benchmark_solution_quality
from tqdm import tqdm

random.seed(42)
problems = load_math_problems('/home/tbai4/llama3/data/MATH', split='val')
problems = random.sample(problems, 200)

for id in [99, 199, 299, 399]: 
    checkpoint = torch.load(f'/scratch4/jeisner1/tjbai/checkpoints/lora_epoch-0_step-{id}.pt', weights_only=True)
    workflow.model.load_state_dict(checkpoint['lora'])
    llama = Llama(workflow.model, workflow.tokenizer)
    print(f'Loaded checkpoint-{id}')
    print(f'Memory allocated: {torch.cuda.memory_allocated()}')

    comps = []
    for problem in tqdm(problems):
        comps.append(benchmark_solution_quality(
            llama=llama,
            workflow=workflow,
            problem=problem['problem'],
            branching_factor=8,
            voters=4,
            compact=False,
        ))
        
    with open(f'checkpoint-{id}_solution_quality.json', 'w') as f:
        json.dump(comps, f)
        
# TODO -- get the untrained version's results too
