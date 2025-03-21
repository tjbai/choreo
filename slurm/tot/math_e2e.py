import os
import json
import torch
from tqdm import tqdm
from llama import Workflow, Llama
from llama.util import find_free_port
from llama.workflows.tot import eval_solutions, load_math_problems, tot_cached

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
    lora_dropout=0.05
)
workflow.model.eval()
problems = load_math_problems('/home/tbai4/llama3/data/MATH', split='test')[:500]

for ckpt_path in [
   "lora_epoch-0_step-395.pt",
   "lora_epoch-0_step-795.pt",
   "lora_epoch-1_step-295.pt",
]:
    ckpt = torch.load(f'/scratch4/jeisner1/tjbai/checkpoints/tot_3/{ckpt_path}', weights_only=True)
    workflow.model.load_state_dict(ckpt['lora'])
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

    with open(f'/home/tbai4/llama3/dumps/tot/tot_b8v4/{ckpt_path}_e2e_test.json', 'w') as f:
        json.dump(solutions, f)

    llama = Llama(workflow.model, workflow.tokenizer)
    llama.model.reshape_cache(4)
    llama.model.set_adapter_state(enabled=False)
    all_correct = eval_solutions(llama, solutions, problems)
    print(f'Correct: {sum(all_correct)} / {len(all_correct)}')
