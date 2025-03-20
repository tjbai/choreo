import os
import json

from tqdm import tqdm

from llama import Workflow, Llama
from llama.util import find_free_port
from llama.workflows.tot import eval_solutions
from llama.workflows.tot import load_math_problems, tot_baseline

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
    use_lora=False,
)

llama = Llama(workflow.model, workflow.tokenizer)

problems = load_math_problems('/home/tbai4/llama3/data/MATH', split='test')[:500]

solutions = []
for problem in tqdm(problems):
    workflow.reset()
    solutions.append(tot_baseline(
        workflow=workflow,
        problem=problem['problem'],
        branching_factor=8,
        voters=4,
    ))

with open('/home/tbai4/llama3/dumps/tot/tot_b8v4/baseline_test.json', 'w') as f:
    json.dump(solutions, f)

llama.model.reshape_cache(4)
all_correct = eval_solutions(
    llama=llama,
    solutions=[workflow.tokenizer.decode(s['final_tokens']) for s in solutions],
    problems=problems
)
print(f'Correct: {sum(all_correct)} / {len(all_correct)}')
