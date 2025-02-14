import os
import json
from tqdm import tqdm
from llama.workflows.mad_iterative import load_translations, mad_cached, mad_baseline
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
examples = load_translations('/home/tbai4/llama3/data/commonmt/', start=0, end=100)

results = {
    'cached': [],
    'baseline': []
}

for i, ex in enumerate(tqdm(examples)):
    ex.update({'index': i})

    results['cached'].append(mad_cached(
        workflow,
        ex['chinese'],
        ['Alice', 'Bob'],
        max_rounds=3,
        debug=False
    ))

    results['baseline'].append(mad_baseline(
        workflow,
        ex['chinese'],
        ['Alice', 'Bob'],
        max_rounds=3,
        debug=False
    ))

    if (i + 1) % 10 == 0:
        with open('translate_e2e.json', 'w') as f:
            json.dump(results, f)

with open('translate_e2e.json', 'w') as f:
    json.dump(results, f)
