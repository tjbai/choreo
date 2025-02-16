import os
import json
from tqdm import tqdm
from llama.workflows.mad_iterative import load_ciar, math_mad_cached, math_mad_baseline, math_simple_baseline
from llama import Workflow

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29502"

workflow = Workflow.build(
    ckpt_dir='/scratch4/jeisner1/tjbai/llama_8b',
    tokenizer_path='/scratch4/jeisner1/tjbai/llama_8b/tokenizer.model',
    max_seq_len=8*8192,
    max_batch_size=1,
    model_parallel_size=1,
    max_nodes=100,
)

workflow.model.eval()
problems = load_ciar('/home/tbai4/llama3/data/CIAR/', start=0, end=50)

results = {
    'cached': [],
    'mad_baseline': [],
    'simple_baseline': [],
    'simple_baseline_reflection': [],
}

for i, problem in enumerate(tqdm(problems)):
    workflow.reset()
    results['cached'].append(math_mad_cached(
        workflow,
        problem['question'],
        ['A', 'B'],
        max_rounds=3,
        debug=False
    ))

    workflow.reset()
    results['mad_baseline'].append(math_mad_baseline(
        workflow,
        problem['question'],
        ['A', 'B'],
        max_rounds=3,
        debug=False
    ))

    workflow.reset()
    results['simple_baseline'].append(math_simple_baseline(
        workflow,
        problem['question'],
        enable_reflection=False,
        debug=False
    ))

    workflow.reset()
    results['simple_baseline_reflection'].append(math_simple_baseline(
        workflow,
        problem['question'],
        enable_reflection=True,
        debug=False
    ))
