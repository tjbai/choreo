import os
import json
from tqdm import tqdm
from llama.workflows.mad import mad_baseline, mad_cached
from llama.workflows.tot import eval_solutions, load_math_problems
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

# problems = load_dataset('openai/gsm8k', 'main', split='train')[:500]
# 
# samples = []
# for problem, solution in tqdm(zip(problems['question'], problems['solution'])):
#     workflow.reset()
#     outputs = mad_baseline(
#         workflow=workflow,
#         problem=problem,
#         max_rounds=3,
#     )
#     samples.append({
#         'inputs': {'problem': problem, 'solution': solution},
#         'outputs': outputs,
#     })
#     with open('/home/tbai4/llama3/dumps/mad/gsm8k_baseline_e2e.json', 'w') as f:
#         json.dump(samples, f)

problems = load_math_problems('/home/tbai4/llama3/data/MATH', split='test')[:500]

samples = []
for i, problem in enumerate(tqdm(problems)):
    workflow.reset()
    outputs = mad_baseline(
        workflow=workflow,
        problem=problem['problem'],
        max_rounds=3,
    )
    samples.append({
        'inputs': {'problem': problem['problem'], 'solution': problem['solution']},
        'outputs': outputs,
    })
    if i == 0:
        print('baseline correct', sum(eval_solutions(
            llama,
            [d['outputs']['decision']['Answer'] for d in samples if isinstance(d['outputs']['decision'], dict)],
            [d['inputs'] for d in samples if isinstance(d['outputs']['decision'], dict)],
        )))
print('baseline correct', sum(eval_solutions(
    llama,
    [d['outputs']['decision']['Answer'] for d in samples if isinstance(d['outputs']['decision'], dict)],
    [d['inputs'] for d in samples if isinstance(d['outputs']['decision'], dict)],
)))

samples = []
for i, problem in enumerate(tqdm(problems)):
    workflow.reset()
    outputs = mad_cached(
        workflow=workflow,
        problem=problem['problem'],
        max_rounds=3,
    )
    samples.append({
        'inputs': {'problem': problem['problem'], 'solution': problem['solution']},
        'outputs': outputs,
    })
    if i == 0:
        print('cached correct', sum(eval_solutions(
            llama,
            [d['outputs']['decision']['Answer'] for d in samples if isinstance(d['outputs']['decision'], dict)],
            [d['inputs'] for d in samples if isinstance(d['outputs']['decision'], dict)],
        )))
print('cached correct', sum(eval_solutions(
    llama,
    [d['outputs']['decision']['Answer'] for d in samples if isinstance(d['outputs']['decision'], dict)],
    [d['inputs'] for d in samples if isinstance(d['outputs']['decision'], dict)],
)))
