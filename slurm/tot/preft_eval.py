import os
import torch
from tqdm import tqdm
from llama.workflows.tot import eval_solutions, tot_baseline, tot_cached
from llama.util import find_free_port
from llama import Workflow, Llama
from datasets import load_dataset

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = str(find_free_port())

workflow = Workflow.build(
    ckpt_dir='/scratch4/jeisner1/tjbai/llama_8b',
    tokenizer_path='/scratch4/jeisner1/tjbai/llama_8b/tokenizer.model',
    max_seq_len=2*8192,
    max_batch_size=4,
    model_parallel_size=1,
    max_nodes=100,
    use_lora=False,
)

llama = Llama(workflow.model, workflow.tokenizer)

problems = load_dataset('openai/gsm8k', 'main', split='train')[:500]

for i, (problem, solution) in enumerate(tqdm(zip(problems['question'], problems['solution']))):
    workflow.reset()
    outputs = tot_baseline(
        workflow=workflow,
        problem=problem,
        branching_factor=8,
        voters=4,
        temperature=0.7,
        top_p=1.0,
    )
    example = {
        'problem_idx': i,
        'problem': problem,
        'result': outputs,
    }
    torch.save(example, f'/home/tbai4/llama3/dumps/tot/tot_gsm8k/problem_{i}.pt')

problems = load_dataset('openai/gsm8k', 'main', split='test')[:500]

samples = []
for problem, solution in tqdm(zip(problems['question'], problems['solution'])):
    workflow.reset()
    outputs = tot_baseline(
        workflow=workflow,
        problem=problem,
        branching_factor=8,
        voters=4,
        temperature=0.7,
        top_p=1.0,
    )
    samples.append({
        'inputs': {'problem': problem, 'solution': solution},
        'outputs': outputs,
    })
print('baseline correct', sum(eval_solutions(
    llama,
    [workflow.tokenizer.decode(d['outputs']['final_tokens']) for d in samples],
    [d['inputs']['solution'] for d in samples],
)))

samples = []
for problem, solution in tqdm(zip(problems['question'], problems['solution'])):
    workflow.reset()
    outputs = tot_cached(
        workflow=workflow,
        problem=problem,
        branching_factor=8,
        voters=4,
        temperature=0.7,
        top_p=1.0,
    )
    samples.append({
        'inputs': {'problem': problem, 'solution': solution},
        'outputs': outputs,
    })
print('cached correct', sum(eval_solutions(
    llama,
    [workflow.tokenizer.decode(d['outputs']['final_tokens']) for d in samples],
    [d['inputs']['solution'] for d in samples],
)))
