import os
import json
import torch
from filelock import FileLock
from tqdm import tqdm
import fire
from llama.workflows.prisoners import prisoners_cached
from llama import Workflow
from llama.util import find_free_port

def append_to_jsonl(data, filename):
    lock_file = f"{filename}.lock"
    with FileLock(lock_file):
        with open(filename, 'a') as f:
            f.write(json.dumps(data) + '\n')

def load_existing_results(output_file):
    existing_results = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    key = (data['strategy'], data['seed'])
                    existing_results[key] = data
                except:
                    continue
    return existing_results

def load_baseline_data():
    with open('/home/tbai4/llama3/dumps/prisoners/prisoners_baseline_predict.jsonl') as f:
        baseline_data = [json.loads(line) for line in f]
        baseline = [d for d in baseline_data if d['strategy'] is None]
        coop = [d for d in baseline_data if d['strategy'] == 'always_cooperate']
        defect = [d for d in baseline_data if d['strategy'] == 'always_defect']
        return {
            None: baseline,
            'always_cooperate': coop,
            'always_defect': defect
        }

def process_strategy(strategy=None):
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
    workflow.model.reshape_cache(1)
    workflow.model.eval()

    payoff = (5, 3, 1, 0)
    output_file = '/home/tbai4/llama3/dumps/prisoners/prisoners_cached_paired_predict.jsonl'
    existing_results = load_existing_results(output_file)

    data_dict = load_baseline_data()
    if strategy not in data_dict:
        print(f"Invalid strategy: {strategy}. Choose from: None, always_cooperate, always_defect")
        return

    data = data_dict[strategy]
    strategy_str = strategy if strategy else 'baseline'
    print(f"Processing strategy: {strategy_str}")

    alice_decisions = []
    bob_decisions = []

    for seed, example in enumerate(tqdm(data, total=len(data))):
        if (key := (strategy, seed)) in existing_results:
            result = existing_results[key]
            alice_decisions.append(result['alice_final'])
            bob_decisions.append(result['bob_final'])
            continue

        try:
            alice_plan_ids, bob_plan_ids = example['outputs']['plan_ids']
            plan_force = torch.full((2, 512), workflow.tokenizer.eot_id, device=workflow.device)
            plan_force[0, :len(alice_plan_ids)] = torch.tensor(alice_plan_ids, device=workflow.device)
            plan_force[1, :len(bob_plan_ids)] = torch.tensor(bob_plan_ids, device=workflow.device)

            workflow.reset()
            cached_outputs = prisoners_cached(
                workflow,
                payoff,
                alice_first=(seed < (len(data) // 2)),
                alice_strategy=strategy,
                seed=seed,
                temperature=1.0,
                top_p=1.0,
                plan_force=plan_force,
                with_prediction=True,
            )

            alice_decision = workflow.tokenizer.decode(cached_outputs['alice_context'][-1]['output_tokens'])
            bob_decision = workflow.tokenizer.decode(cached_outputs['bob_context'][-1]['output_tokens'])
            alice_decisions.append(alice_decision)
            bob_decisions.append(bob_decision)

            output_data = {
                'seed': seed,
                'strategy': strategy,
                'payoff': payoff,
                'alice_first': (seed < (len(data) // 2)),
                'outputs': cached_outputs,
                'alice_final': alice_decision,
                'bob_final': bob_decision,
            }
            append_to_jsonl(output_data, output_file)

        except Exception as e:
            print(f"Error processing seed {seed} with strategy {strategy_str}: {e}")
            continue

    alice_coop = sum(1 for d in alice_decisions if 'COOPERATE' in d.upper())
    alice_defect = sum(1 for d in alice_decisions if 'DEFECT' in d.upper())
    bob_coop = sum(1 for d in bob_decisions if 'COOPERATE' in d.upper())
    bob_defect = sum(1 for d in bob_decisions if 'DEFECT' in d.upper())

    print(
        f"Strategy: {strategy_str}",
        f"\nalice: {alice_coop} COOPERATE, {alice_defect} DEFECT",
        f"\nbob: {bob_coop} COOPERATE, {bob_defect} DEFECT",
    )

if __name__ == "__main__":
    fire.Fire(process_strategy)
