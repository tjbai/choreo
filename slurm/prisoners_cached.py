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

for strategy in strategies:
   alice_decisions = []
   bob_decisions = []
   
   for seed in tqdm(range(100)):
       workflow.reset()
       cached_outputs = prisoners_cached(
           workflow, 
           payoff,
           alice_strategy=strategy,
           seed=seed
       )
       
       alice_decision = workflow.tokenizer.decode(cached_outputs['alice_context'][-1]['tokens'])
       bob_decision = workflow.tokenizer.decode(cached_outputs['bob_context'][-1]['tokens'])
       
       output_data = {
           'seed': seed,
           'strategy': strategy,
           'outputs': cached_outputs,
           'alice_final': alice_decision,
           'bob_final': bob_decision
       }
       append_to_jsonl(output_data, output_file)
       
       alice_decisions.append(alice_decision)
       bob_decisions.append(bob_decision)
   
   print(
       f"\nStrategy: {strategy if strategy else 'baseline'}",
       '\nalice:',
       sum(1 for d in alice_decisions if 'COOPERATE' in d),
       sum(1 for d in alice_decisions if 'DEFECT' in d),
       '\nbob:',
       sum(1 for d in bob_decisions if 'COOPERATE' in d),
       sum(1 for d in bob_decisions if 'DEFECT' in d),
   )

