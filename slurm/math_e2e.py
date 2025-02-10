import os
from llama import Workflow, Llama
from llama.util import find_free_port
from llama.workflows.tot import eval_solutions

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = str(find_free_port())

workflow = Workflow.build(
    ckpt_dir='/scratch4/jeisner1/tjbai/llama_8b',
    tokenizer_path='/scratch4/jeisner1/tjbai/llama_8b/tokenizer.model',
    max_seq_len=8192,
    max_batch_size=8,
    model_parallel_size=1,
    max_nodes=20,
    use_lora=True,
    lora_rank=64,
    lora_alpha=32,
    lora_dropout=0.0
)

workflow.model.eval()

import json
import torch
from tqdm import tqdm
from llama.workflows.tot import load_math_problems, tot_cached, tot_baseline

problems = load_math_problems('/home/tbai4/llama3/data/MATH', split='val')

for ckpt_path in [
   "lora_epoch-0_step-395.pt",
   "lora_epoch-0_step-795.pt",
   "lora_epoch-1_step-295.pt",
]:
    checkpoint = torch.load(f'/scratch4/jeisner1/tjbai/checkpoints/tot_3/{ckpt_path}', weights_only=True)
    workflow.model.load_state_dict(checkpoint['lora'])

    workflow.model.eval()
    workflow.model.reshape_cache(1)
    workflow.model.set_adapter_state(enabled=True)

    solutions = []
    for problem in tqdm(problems, desc=f'checkpoint {ckpt_path}'):
        workflow.reset()
        solutions.append(tot_cached(
            workflow=workflow,
            problem=problem['problem'],
            branching_factor=8,
            voters=4,
        ))

    llama = Llama(workflow.model, workflow.tokenizer)
    llama.model.reshape_cache(4)
    llama.model.set_adapter_state(enabled=False)
    all_correct = eval_solutions(llama, solutions, problems)
    print(f'Correct: {sum(all_correct)} / {len(all_correct)}')

    with open(f'{ckpt_path}_e2e.json', 'w') as f:
        json.dump({'solutions': solutions, 'all_correct': all_correct}, f)

