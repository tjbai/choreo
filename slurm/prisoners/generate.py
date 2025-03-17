import os
import json
import torch
from tqdm import tqdm
from pathlib import Path
import fire
from llama import Workflow
from llama.workflows.prisoners import prisoners_cached
from llama.util import find_free_port

def append_to_jsonl(data, filename):
   with open(filename, 'a') as f:
       f.write(json.dumps(data) + '\n')

def load_existing_results(strategy):
    filename = f'/home/tbai4/llama3/dumps/prisoners/prisoners_ft_eval_{strategy if strategy else 'baseline'}_large.jsonl'
    existing_results = {}
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    key = (os.path.basename(data['ckpt_path']), data['seed'], data['strategy'])
                    existing_results[key] = data
                except:
                    continue
    return existing_results, filename

def main(
    baseline_path='/home/tbai4/llama3/dumps/prisoners/prisoners_baseline_large.jsonl',
    strategy=None,
):
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
        use_lora=True,
        lora_rank=32,
        lora_alpha=64,
        lora_dropout=0.05,
    )

    with open(baseline_path) as f:
        baseline_data = json.load(f)
        baseline = [d for d in baseline_data if d['strategy'] is None]
        coop = [d for d in baseline_data if d['strategy'] == 'always_cooperate']
        defect = [d for d in baseline_data if d['strategy'] == 'always_defect']
        print(len(baseline), len(coop), len(defect))

    existing_results, output_file = load_existing_results(strategy)
    base_path = Path(f'/scratch4/jeisner1/tjbai/checkpoints/prisoners/{strategy if strategy else 'baseline'}')

    for ckpt in sorted(os.listdir(base_path)):
        # if 'epoch' in ckpt: # old checkpoints, lazy to move
        #     continue

        # choose the ckpts from previous training run
        if strategy is None and ckpt != 'lora_epoch-1_step-155.pt':
            continue
        elif strategy == 'always_cooperate' and ckpt != 'lora_epoch-0_step-95.pt':
            continue
        elif strategy == 'always_defect' and ckpt != 'lora_epoch-0_step-95.pt':
            continue

        print(f'Strategy: {strategy}, Checkpoint: {ckpt}')

        ckpt_path = base_path / ckpt
        weights = torch.load(ckpt_path, weights_only=True)['trainable_params']
        for weight, param in zip(weights, workflow.model.get_trainable_parameters()):
            param.data.copy_(weight)

        workflow.model.eval()
        workflow.model.set_adapter_state(enabled=True)

        data = baseline if strategy is None else (coop if strategy == 'always_cooperate' else defect)

        alice_decisions = []
        bob_decisions = []
        for seed, example in enumerate(tqdm(data, total=len(data))):
            if (key := (os.path.basename(str(ckpt_path)), seed, strategy)) in existing_results:
                existing = existing_results[key]
                alice_decisions.append(existing['alice_final'])
                bob_decisions.append(existing['bob_final'])
                continue

            try:
                alice_plan_ids, bob_plan_ids = example['outputs']['plan_ids']
                plan_force = torch.full((2, 512), workflow.tokenizer.eot_id, device=workflow.device)
                plan_force[0, :len(alice_plan_ids)] = torch.tensor(alice_plan_ids, device=workflow.device)
                plan_force[1, :len(bob_plan_ids)] = torch.tensor(bob_plan_ids, device=workflow.device)

                workflow.reset()
                cached_outputs = prisoners_cached(
                    workflow,
                    payoff=(5, 3, 1, 0),
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
                    'outputs': cached_outputs,
                    'alice_first': (seed < 250),
                    'leak_setting': (False, False),
                    'payoff': (5, 3, 1, 0),
                    'alice_final': alice_decision,
                    'bob_final': bob_decision,
                    'ckpt_path': str(ckpt_path),
                }
                append_to_jsonl(output_data, output_file)

            except Exception as e:
                print(f'Error in e2e with {ckpt}, seed {seed}, strategy {strategy}: {e}')

if __name__ == '__main__':
    fire.Fire(main)
