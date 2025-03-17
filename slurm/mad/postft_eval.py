import os
import json
from tqdm import tqdm

from llama import Workflow
from llama.util import find_free_port, load_ckpt
from llama.workflows.tot import load_math_problems
from llama.workflows.mad import mad_cached

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
    max_nodes=100,
    use_lora=True,
    lora_rank=64,
    lora_alpha=32,
    lora_dropout=0.05
)
load_ckpt(workflow, '/scratch4/jeisner1/tjbai/checkpoints/mad/lora_step-2249.pt')
workflow.model.eval()

problems = load_math_problems('/home/tbai4/llama3/data/MATH', split='val')
samples = []
for i, problem in enumerate(tqdm(problems)):
    workflow.reset()
    outputs = mad_cached(
        workflow=workflow,
        problem=problem['problem'],
        max_rounds=3,
    )
    samples.append({
        'inputs': {'problem': problem['problem']},
        'outputs': outputs,
    })
    if (i + 1) % 10 == 0:
        with open('/home/tbai4/llama3/dumps/mad/postft_eval.json', 'w') as f:
            json.dump(samples, f)
