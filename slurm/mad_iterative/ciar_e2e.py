import os
import json
import time
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
   'cached_times': [],
   'mad_baseline': [],
   'mad_baseline_times': [],
   'simple_baseline': [],
   'simple_baseline_times': [],
   'simple_baseline_reflection': [],
   'simple_baseline_reflection_times': []
}

for _ in range(5):
   workflow.reset()
   _ = math_mad_cached(workflow, problems[0]['question'], ['A', 'B'], max_rounds=3)

for problem in tqdm(problems, desc='Cached'):
   workflow.reset()
   start = time.time()
   results['cached'].append(math_mad_cached(
       workflow, 
       problem['question'],
       ['A', 'B'],
       max_rounds=3,
       debug=False
   ))
   results['cached_times'].append(time.time() - start)

with open('/home/tbai4/llama3/dumps/mad_iterative/ciar_e2e.json', 'w') as f:
   json.dump(results, f)

for _ in range(5):
   workflow.reset()
   _ = math_mad_baseline(workflow, problems[0]['question'], ['A', 'B'], max_rounds=3)

for problem in tqdm(problems, desc='MAD Baseline'):
   workflow.reset()
   start = time.time()
   results['mad_baseline'].append(math_mad_baseline(
       workflow,
       problem['question'],
       ['A', 'B'], 
       max_rounds=3,
       debug=False
   ))
   results['mad_baseline_times'].append(time.time() - start)

with open('/home/tbai4/llama3/dumps/mad_iterative/ciar_e2e.json', 'w') as f:
   json.dump(results, f)

for _ in range(5):
   workflow.reset()
   _ = math_simple_baseline(workflow, problems[0]['question'], enable_reflection=False)

for problem in tqdm(problems, desc='Simple Baseline'):
   workflow.reset()
   start = time.time()
   results['simple_baseline'].append(math_simple_baseline(
       workflow,
       problem['question'],
       enable_reflection=False,
       debug=False
   ))
   results['simple_baseline_times'].append(time.time() - start)

with open('/home/tbai4/llama3/dumps/mad_iterative/ciar_e2e.json', 'w') as f:
   json.dump(results, f)

for _ in range(5):
   workflow.reset()
   _ = math_simple_baseline(workflow, problems[0]['question'], enable_reflection=True)

for problem in tqdm(problems, desc='Simple Baseline + Reflection'):
   workflow.reset()
   start = time.time() 
   results['simple_baseline_reflection'].append(math_simple_baseline(
       workflow,
       problem['question'],
       enable_reflection=True,
       debug=False
   ))
   results['simple_baseline_reflection_times'].append(time.time() - start)

with open('/home/tbai4/llama3/dumps/mad_iterative/ciar_e2e.json', 'w') as f:
   json.dump(results, f)

