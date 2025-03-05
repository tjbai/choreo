import os
import json
import time
from tqdm import tqdm
from llama.workflows.mad import load_translations, mad_cached, mad_baseline
from llama import Workflow
from llama.util import find_free_port

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
)

workflow.model.eval()
translations = load_translations('/home/tbai4/llama3/data/commonmt/', start=0, end=200)

baseline_res = []
baseline_times = []
for translation in tqdm(translations):
    s = time.time()
    workflow.reset()
    baseline_res.append(mad_baseline(workflow, translation['chinese'], agents=['Alice', 'Bob'], max_rounds=3, debug=False))
    baseline_times.append(time.time() - s)

cached_res = []
cached_times = []
for translation in tqdm(translations):
    s = time.time()
    workflow.reset()
    cached_res.append(mad_cached(workflow, translation['chinese'], agents=['Alice', 'Bob'], max_rounds=3, debug=False))
    cached_times.append(time.time() - s)

with open('/home/tbai4/llama3/dumps/mad_iterative/translate_e2e_throughput.json', 'w') as f:
    json.dump({'cached_res': cached_res, 'cached_times': cached_times, 'baseline_res': baseline_res, 'baseline_times': baseline_times}, f)
