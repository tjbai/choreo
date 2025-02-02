import os
from llama import Workflow, Llama

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

workflow = Workflow.build(
    ckpt_dir='/scratch4/jeisner1/tjbai/llama_8b',
    tokenizer_path='/scratch4/jeisner1/tjbai/llama_8b/tokenizer.model',
    max_seq_len=8192,
    max_batch_size=8,
    model_parallel_size=1,
    max_nodes=20,
    use_lora=True,
    lora_rank=8,
    lora_alpha=16,
    lora_dropout=0.1
)

workflow.model.eval()

import json
import torch
from tqdm import tqdm
from llama.workflows.tot import load_math_problems, tot_cached, tot_baseline

# full split is just 280 problems, might as well do them all
problems = load_math_problems('/home/tbai4/llama3/data/MATH', split='val')

for id in [0, 99, 199, 299, 399]:
    if id == 0:
        workflow.model.set_adapter_state(enabled=False)
    else:
        workflow.model.set_adapter_state(enabled=True)
        checkpoint = torch.load(f'/scratch4/jeisner1/tjbai/checkpoints/lora_epoch-0_step-{id}.pt', weights_only=True)
        workflow.model.load_state_dict(checkpoint['lora'])
    print(f'Loaded checkpoint-{id}')

    workflow.model.reshape_cache(1)
    comps = []
    for problem in tqdm(problems, desc=f'checkpoint {id}'):
        workflow.reset()
        comps.append(tot_cached(
            workflow=workflow,
            problem=problem['problem'],
            branching_factor=8,
            voters=4,
        ))
    with open(f'checkpoint-{id}_e2e.json', 'w') as f:
        json.dump(comps, f)

    if id == 0:
        llama = Llama(workflow.model, workflow.tokenizer)
        llama.model.reshape_cache(8)
        comps = []
        for problem in tqdm(problems, desc='baseline'):
            comps.append(tot_baseline(
                llama=llama,
                problem=problem['problem'],
                branching_factor=8,
                voters=4
            ))
        with open('baseline_e2e.json', 'w') as f:
            json.dump(comps, f)
