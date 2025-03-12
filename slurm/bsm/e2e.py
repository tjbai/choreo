import os
import json
from tqdm import tqdm
from llama import Workflow
from llama.util import find_free_port
from llama.workflows.bsm import load_concepts
from llama.workflows.bsm import bsm_baseline

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
)

concepts_list = load_concepts(
    data_path='/home/tbai4/llama3/data/commongen/commongen.jsonl',
    split='train',
)

samples = []
for idx, concept_set in enumerate(tqdm(concepts_list, desc="Evaluating")):
    workflow.reset()
    outputs = bsm_baseline(workflow=workflow, concepts=concept_set)
    samples.append({
        'inputs': {'concepts': concept_set},
        'outputs': outputs
    })

    if (idx + 1) % 10 == 0:
        with open('/home/tbai4/llama3/dumps/bsm/baseline_e2e.json') as f:
            json.dump(samples, f)

with open('/home/tbai4/llama3/dumps/bsm/baseline_e2e.json') as f:
    json.dump(samples, f)
