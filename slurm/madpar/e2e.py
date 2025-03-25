import os
import json
from tqdm import tqdm
from llama import Workflow, Llama
from llama.util import find_free_port
from llama.workflows.tot import load_math_problems
from llama.workflows.madpar import madpar_baseline, madpar_cached

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
problems = load_math_problems('/home/tbai4/llama3/data/MATH', split='train')[:500]

# MADpar baseline on MATH
samples = []
for i, problem in enumerate(tqdm(problems)):
    workflow.reset()
    outputs = madpar_baseline(
        workflow=workflow,
        problem=problem['problem'],
        num_agents=3,
        num_rounds=3,
        debug=False,
    )
    samples.append({
        'inputs': {'problem': problem['problem']},
        'outputs': outputs,
    })
    if (i+1) % 10 == 0:
        with open('/home/tbai4/llama3/dumps/madpar/math_baseline_e2e.json', 'w') as f:
            json.dump(samples, f)
with open('/home/tbai4/llama3/dumps/madpar/math_baseline_e2e.json', 'w') as f:
    json.dump(samples, f)

# MADpar cached on MATH
samples = []
for i, problem in enumerate(tqdm(problems)):
    workflow.reset()
    outputs = madpar_cached(
        workflow=workflow,
        problem=problem['problem'],
        num_agents=3,
        num_rounds=3,
        debug=False,
    )
    samples.append({
        'inputs': {'problem': problem['problem']},
        'outputs': outputs,
    })
    if (i+1) % 10 == 0:
        with open('/home/tbai4/llama3/dumps/madpar/math_cached_e2e.json', 'w') as f:
            json.dump(samples, f)
with open('/home/tbai4/llama3/dumps/madpar/math_cached_e2e.json', 'w') as f:
    json.dump(samples, f)
