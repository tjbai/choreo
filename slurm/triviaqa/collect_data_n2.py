import os
import json
import random
from tqdm import tqdm

from llama import Workflow, Llama
from llama.util import find_free_port
from llama.workflows.qa import ask_sequential

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

llama = Llama(workflow.model, workflow.tokenizer)

with open('/home/tbai4/llama3/data/triviaqa/unfiltered-web-train.json') as f:
    data = json.load(f)
    problems = data['Data']

outputs = []
for seed in tqdm(range(100)):
    random.seed(seed)
    subset = random.sample(problems, k=2)
    outputs.append(ask_sequential(workflow, subset))

with open('/home/tbai4/llama3/qa_n2.json', 'w') as f:
    json.dump(outputs, f)
