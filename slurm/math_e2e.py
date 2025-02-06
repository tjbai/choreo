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

problems = load_math_problems('/home/tbai4/llama3/data/MATH', split='val')

for ckpt_path in [
   "lora_epoch-0_step-195.pt",
   "lora_epoch-0_step-395.pt",
   # "lora_epoch-0_step-595.pt",
   # "lora_epoch-0_step-795.pt",
   # "lora_epoch-1_step-95.pt",
   # "lora_epoch-1_step-295.pt",
   # "lora_epoch-1_step-495.pt",
   # "lora_epoch-1_step-695.pt",
   # "lora_epoch-1_step-895.pt",
]:
    workflow.model.set_adapter_state(enabled=True)
    checkpoint = torch.load(f'/scratch4/jeisner1/tjbai/checkpoints/tot_2/{ckpt_path}', weights_only=True)
    workflow.model.load_state_dict(checkpoint['lora'])
    workflow.model.reshape_cache(1)

    comps = []
    for problem in tqdm(problems, desc=f'checkpoint {ckpt_path}'):
        workflow.reset()
        comps.append(tot_cached(
            workflow=workflow,
            problem=problem['problem'],
            branching_factor=8,
            voters=4,
        ))

    with open(f'{ckpt_path}_e2e.json', 'w') as f:
        json.dump(comps, f)
