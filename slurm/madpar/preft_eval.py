import os
import json
from tqdm import tqdm
from llama import Workflow, Llama
from llama.util import find_free_port, load_ckpt
from llama.workflows.tot import load_math_problems
from llama.workflows.madpar import madpar_baseline, madpar_cached, eval_debate_solutions, parse_output

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
llama = Llama(workflow.model, workflow.tokenizer)

# MATH dataset
problems = load_math_problems('/home/tbai4/llama3/data/MATH', split='test')[:500]

samples = []
for i, problem in enumerate(tqdm(problems)):
    workflow.reset()
    outputs = madpar_cached(
        workflow=workflow,
        problem=problem['problem'],
        num_agents=3,
        num_rounds=3,
        debug=False,
    )
    samples.append({
        'inputs': {'problem': problem['problem']},
        'outputs': outputs,
    })
    if (i+1) % 10 == 0:
        with open('/home/tbai4/llama3/dumps/madpar/math_cached_preft.json', 'w') as f:
            json.dump(samples, f)
    if i == 0:
        llama.model.reshape_cache(4)
        llama.model.set_adapter_state(enabled=False)
        outputs = eval_debate_solutions(
            llama,
            agent_solutions=[
                [parse_output(llama.tokenizer.decode(a)) for a in d['outputs']['debate_tokens'][-1]]
                for d in samples
            ],
            problems=problems[:len(samples)],
        )
        llama.model.reshape_cache(1)
        llama.model.set_adapter_state(enabled=True)

with open('/home/tbai4/llama3/dumps/madpar/math_cached_preft.json', 'w') as f:
    json.dump(samples, f)

llama.model.reshape_cache(4)
llama.model.set_adapter_state(enabled=False)
outputs = eval_debate_solutions(
    llama,
    agent_solutions=[
        [parse_output(llama.tokenizer.decode(a)) for a in d['outputs']['debate_tokens'][-1]]
        for d in samples
    ],
    problems=problems[:len(samples)],
)
print(sum(outputs))

