import os
import json
from tqdm import tqdm
from llama.workflows.mad_iterative import load_translations, mad_cached, mad_baseline
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

# for translation in tqdm(translations):
#     workflow.reset()
#     mad_baseline(workflow, translation['chinese'], agents=['Alice', 'Bob'], max_rounds=3, debug=False)
    
for translation in tqdm(translations):
    workflow.reset()
    mad_cached(workflow, translation['chinese'], agents=['Alice', 'Bob'], max_rounds=3, debug=False)

