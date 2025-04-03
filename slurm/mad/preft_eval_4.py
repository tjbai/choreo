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

problems = load_math_problems('/home/tbai4/llama3/data/MATH', split='test')[250:500]

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
        correct = eval_solutions(
            llama,
            [(d['outputs']['decision']['Answer'] if isinstance(d['outputs']['decision'], dict) else '')for d in samples],
            [d['inputs'] for d in samples],
        )
        with open('dumps/mad/choreo_correct_p2.json', 'w') as f:
            json.dump(correct, f)
correct = eval_solutions(
    llama,
    [(d['outputs']['decision']['Answer'] if isinstance(d['outputs']['decision'], dict) else '')for d in samples],
    [d['inputs'] for d in samples],
)
with open('dumps/mad/choreo_correct_p2.json', 'w') as f:
    json.dump(correct, f)

