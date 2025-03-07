import os
import json
import torch
from tqdm import tqdm
from llama.workflows.prisoners import prisoners_cached
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

def append_to_jsonl(data, filename):
   with open(filename, 'a') as f:
       f.write(json.dumps(data) + '\n')

workflow.model.reshape_cache(1)
workflow.model.eval()
payoff = (5, 3, 1, 0)
output_file = '/home/tbai4/llama3/dumps/prisoners/prisoners_cached_paired_large.jsonl'

with open('/home/tbai4/llama3/dumps/prisoners/prisoners_baseline_large.jsonl') as f:
    baseline_data = json.load(f)
    baseline = [d for d in baseline_data if d['strategy'] is None]
    coop = [d for d in baseline_data if d['strategy'] == 'always_cooperate']
    defect = [d for d in baseline_data if d['strategy'] == 'always_defect']
    print(len(baseline), len(coop), len(defect))

strategies = [None, 'always_cooperate', 'always_defect']

for strategy, data in zip(strategies, [baseline, coop, defect]):
    alice_decisions = []
    bob_decisions = []
    for seed, example in enumerate(tqdm(data, total=len(data))):
        alice_plan_ids, bob_plan_ids = example['outputs']['plan_ids']
        plan_force = torch.full((2, 512), workflow.tokenizer.eot_id, device=workflow.device)
        plan_force[0, :len(alice_plan_ids)] = torch.tensor(alice_plan_ids, device=workflow.device)
        plan_force[1, :len(bob_plan_ids)] = torch.tensor(bob_plan_ids, device=workflow.device)

        workflow.reset()
        cached_outputs = prisoners_cached(
            workflow,
            payoff,
            alice_first=(seed < 250),
            alice_strategy=strategy,
            seed=seed,
            temperature=1.0,
            top_p=1.0,
            plan_force=plan_force,
        )

        alice_decision = workflow.tokenizer.decode(cached_outputs['alice_context'][-1]['output_tokens'])
        bob_decision = workflow.tokenizer.decode(cached_outputs['bob_context'][-1]['output_tokens'])
        alice_decisions.append(alice_decision)
        bob_decisions.append(bob_decision)

        output_data = {
            'seed': seed,
            'strategy': strategy,
            'payoff': payoff,
            'alice_first': (seed < 250),
            'outputs': cached_outputs,
            'alice_final': alice_decision,
            'bob_final': bob_decision,
        }
        append_to_jsonl(output_data, output_file)

    print(
        f"Strategy: {strategy if strategy else 'baseline'}",
        '\nalice:',
        sum(1 for d in alice_decisions if 'COOPERATE' in d.upper()),
        sum(1 for d in alice_decisions if 'DEFECT' in d.upper()),
        '\nbob:',
        sum(1 for d in bob_decisions if 'COOPERATE' in d.upper()),
        sum(1 for d in bob_decisions if 'DEFECT' in d.upper()),
    )
