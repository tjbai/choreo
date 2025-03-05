import os
import json
from tqdm import tqdm
from llama.workflows.prisoners import prisoners_baseline
from llama import Llama
from llama.util import find_free_port

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = str(find_free_port())

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

def load_existing_results(filename):
    existing = {}
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    key = (data['strategy'], data['seed'])
                    existing[key] = data
                except:
                    continue
    return existing

llama.model.reshape_cache(2)
llama.model.eval()
payoff = (5, 3, 1, 0)

strategies = [None, 'always_cooperate', 'always_defect']
output_file = '/home/tbai4/llama3/dumps/prisoners/prisoners_baseline_large.jsonl'
existing_results = load_existing_results(output_file)

for strategy in strategies:
    alice_decisions = []
    bob_decisions = []

    for key, data in existing_results.items():
        if key[0] == strategy:
            alice_decisions.append(data['alice_final'])
            bob_decisions.append(data['bob_final'])

    for seed in tqdm(range(500)):
        if (strategy, seed) in existing_results:
            continue

        try:
            result = prisoners_baseline(
                llama,
                payoff,
                alice_first=(seed < 250),
                alice_strategy=strategy,
                seed=seed,
                temperature=1.0,
                top_p=1.0,
            )

            output_data = {
                'seed': seed,
                'strategy': strategy,
                'outputs': result,
                'alice_final': result['alice_dialog'][-1]['content'],
                'bob_final': result['bob_dialog'][-1]['content'],
            }
            append_to_jsonl(output_data, output_file)

            existing_results[(strategy, seed)] = output_data
            alice_decisions.append(result['alice_dialog'][-1]['content'])
            bob_decisions.append(result['bob_dialog'][-1]['content'])
        except:
            continue

    print(
        f"Strategy: {strategy if strategy else 'baseline'}",
        '\nalice:',
        sum(1 for d in alice_decisions if 'COOPERATE' in d.upper()),
        sum(1 for d in alice_decisions if 'DEFECT' in d.upper()),
        '\nbob:',
        sum(1 for d in bob_decisions if 'COOPERATE' in d.upper()),
        sum(1 for d in bob_decisions if 'DEFECT' in d.upper()),
    )
