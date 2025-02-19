import os
import json
from tqdm import tqdm
from llama.workflows.prisoners import prisoners_cached
from llama import Workflow

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29501"

workflow = Workflow.build(
    ckpt_dir='/scratch4/jeisner1/tjbai/llama_8b',
    tokenizer_path='/scratch4/jeisner1/tjbai/llama_8b/tokenizer.model',
    max_seq_len=8192,
    max_batch_size=8,
    model_parallel_size=1,
    max_nodes=100,
)

def append_to_jsonl(data, filename):
   with open(filename, 'a') as f:
       f.write(json.dumps(data) + '\n')

workflow.model.reshape_cache(1)
workflow.model.eval()
payoff = (5, 3, 1, 0)
output_file = 'prisoners_cached.jsonl'

strategies = [None, 'always_cooperate', 'always_defect']
leak_settings = [(False, False), (True, False), (False, True)]

for strategy in strategies:
    for only_leak_sys, only_leak_plan in leak_settings:
        alice_decisions = []
        bob_decisions = []
        for seed in tqdm(range(100)):
            workflow.reset()
            cached_outputs = prisoners_cached(
                workflow,
                payoff,
                alice_first=(seed < 50),
                alice_strategy=strategy,
                seed=seed,
                temperature=1.0,
                top_p=1.0,
                only_leak_sys=only_leak_sys,
                only_leak_plan=only_leak_plan,
            )

            alice_decision = workflow.tokenizer.decode(cached_outputs['alice_context'][-1]['tokens'])
            bob_decision = workflow.tokenizer.decode(cached_outputs['bob_context'][-1]['tokens'])
            alice_decisions.append(alice_decision)
            bob_decisions.append(bob_decision)

            output_data = {
                'seed': seed,
                'strategy': strategy,
                'leak_setting': (only_leak_sys, only_leak_plan),
                'outputs': cached_outputs,
                'alice_final': alice_decision,
                'bob_final': bob_decision,
            }
            append_to_jsonl(output_data, output_file)

        print(
            f"\nStrategy: {strategy if strategy else 'baseline'}",
            f"\nOnly leak sys: {only_leak_sys}"
            f"\nOnly leak plan: {only_leak_plan}"
            '\nalice:',
            sum(1 for d in alice_decisions if 'COOPERATE' in d),
            sum(1 for d in alice_decisions if 'DEFECT' in d),
            '\nbob:',
            sum(1 for d in bob_decisions if 'COOPERATE' in d),
            sum(1 for d in bob_decisions if 'DEFECT' in d),
        )
