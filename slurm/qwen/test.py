import os, json
from pathlib import Path
from typing import Optional
from functools import partial

import fire
from tqdm import tqdm

from llama import Workflow
from llama.util import find_free_port, load_ckpt
from llama.workflows import load_math_problems
from llama.workflows.tot import tot_baseline, tot_cached
from llama.workflows.madpar import madpar_baseline, madpar_cached
from llama.workflows.mad import mad_baseline, mad_cached
from llama.workflows.simple import math_direct

WORKFLOWS = {
    'madpar_cached': partial(madpar_cached, num_agents=3, num_rounds=3, debug=False),
    'mad_cached': partial(mad_cached, max_rounds=3),
    'tot_cached': partial(tot_cached, branching_factor=8, voters=4),
    'madpar_baseline': partial(madpar_baseline, num_agents=3, num_rounds=3, debug=False),
    'mad_baseline': partial(mad_baseline, max_rounds=3),
    'tot_baseline': partial(tot_baseline, branching_factor=8, voters=4),
    'direct': math_direct
}

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = str(find_free_port())

def main(
    workflow_type: str,
    shard_idx: int,
    split: str = 'test',
    model_size: str = '8b',
    ckpt_path: Optional[Path] = None,
    tokenizer_path: Optional[Path] = None,
    lora_ckpt_path: Optional[Path] = None,
):
    if ckpt_path is None:
        ckpt_path = Path(f'/scratch4/jeisner1/tjbai/qwen3_{model_size}')
    if tokenizer_path is None:
        tokenizer_path = Path(f'/scratch4/jeisner1/tjbai/qwen3_{model_size}')
    workflow = Workflow.build(
        ckpt_dir=ckpt_path,
        tokenizer_path=tokenizer_path,
        max_seq_len=8*8192,
        max_batch_size=1,
        model_parallel_size=1,
        max_nodes=100,
        use_lora=True,
        lora_rank=64,
        lora_alpha=32,
        lora_dropout=0.05,
        model_type='qwen',
    )

    if lora_ckpt_path:
        workflow = load_ckpt(workflow, lora_ckpt_path)

    assert workflow is not None
    workflow.model.eval()

    start, end = shard_idx * 100, (shard_idx + 1) * 100
    problems = load_math_problems('/home/tbai4/llama3/data/MATH', split='test')[start:end]

    output_path = (
        f'/scratch4/jeisner1/tjbai/qwen_data/'
        f'{model_size}/{workflow_type}/{split}/math_shard-{shard_idx}_ft-{lora_ckpt_path is not None}.json'
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    samples = []
    for i, problem in enumerate(tqdm(problems)):
        workflow.reset()
        outputs = WORKFLOWS[workflow_type](
            workflow=workflow,
            problem=problem['problem']
        )
        samples.append({
            'inner_idx': i,
            'inputs': {'problem': problem['problem'], 'solution': problem['solution']},
            'outputs': outputs,
        })
        if (i + 1) % 10 == 0:
            with open(output_path, 'w') as f:
                json.dump(samples, f)

    with open(output_path, 'w') as f:
        json.dump(samples, f)

if __name__ == '__main__':
    fire.Fire(main)
