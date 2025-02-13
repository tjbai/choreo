import os
import json
import torch
from tqdm import tqdm
from llama.workflows.prisoners import prisoners_baseline
from llama import Llama

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29502"

llama = Llama.build(
    ckpt_dir='/scratch4/jeisner1/tjbai/llama_8b',
    tokenizer_path='/scratch4/jeisner1/tjbai/llama_8b/tokenizer.model',
    max_seq_len=8192,
    max_batch_size=8,
    model_parallel_size=1,
)

def append_to_jsonl(data, filename):
    with open(filename, 'a') as f:
        f.write(json.dumps(data) + '\n')

llama.model.reshape_cache(2)
llama.model.eval()
payoff = (5, 3, 1, 0)

strategies = [None, 'always_cooperate', 'always_defect']
output_file = 'prisoners_baseline.jsonl'

for strategy in strategies:
    alice_decisions = []
    bob_decisions = []
    
    for seed in tqdm(range(100)):
        result = prisoners_baseline(
            llama, 
            payoff,
            alice_first=(seed < 50), 
            alice_strategy=strategy, 
            seed=seed,
            temperature=1.0,
            top_p=1.0,
        )
        
        sample = {
            'payoff': payoff,
            'strategy': strategy,
            'alice_first': (seed < 50),
            'result': result,
        }
        torch.save(sample, f'/home/tbai4/llama3/prisoners_data/trace_{seed}.pt')
        
        alice_decisions.append(result['alice_dialog'][-1]['content'])
        bob_decisions.append(result['bob_dialog'][-1]['content'])
    
    print(
        f"\nStrategy: {strategy if strategy else 'baseline'}",
        '\nalice:',
        sum(1 for d in alice_decisions if 'COOPERATE' in d),
        sum(1 for d in alice_decisions if 'DEFECT' in d),
        '\nbob:',
        sum(1 for d in bob_decisions if 'COOPERATE' in d),
        sum(1 for d in bob_decisions if 'DEFECT' in d),
    )

