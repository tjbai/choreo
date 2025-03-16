import os
import json
import fire
from llama import Workflow, Llama
from llama.workflows.prisoners import get_likelihoods
from llama.util import bootstrap_binary, bootstrap_continuous, find_free_port, load_ckpt

def append_to_jsonl(data, filename):
    with open(filename, 'a') as f:
        f.write(json.dumps(data) + '\n')

def dedup_and_sort(data):
    seen_seeds = set()
    res = []
    for seed, outputs in sorted(data, key=lambda x: x[0]):
        if seed not in seen_seeds:
            res.append((seed, outputs))
            seen_seeds.add(seed)
    return [outputs for _, outputs in res]

def load_existing_results(checkpoint_index, output_file):
    existing_results = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get('checkpoint_index') == checkpoint_index:
                        existing_results = data
                        break
                except:
                    continue
    return existing_results

def main(strategy=None):
    baseline_path = '/home/tbai4/llama3/dumps/prisoners/prisoners_baseline_large.jsonl'
    cached_path = '/home/tbai4/llama3/dumps/prisoners/prisoners_cached_paired_large.jsonl'
    ft_data_path = f'/home/tbai4/llama3/dumps/prisoners/prisoners_ft_eval_{strategy if strategy else 'baseline'}_large.jsonl'
    output_file= '/home/tbai4/llama3/dumps/prisoners/bootstrap_large_results.jsonl'

    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(find_free_port())

    workflow = Workflow.build(
        ckpt_dir='/scratch4/jeisner1/tjbai/llama_8b',
        tokenizer_path='/scratch4/jeisner1/tjbai/llama_8b/tokenizer.model',
        max_seq_len=8192*8,
        max_batch_size=1,
        model_parallel_size=1,
        max_nodes=100,
        use_lora=True,
        lora_rank=64,
        lora_alpha=32,
        lora_dropout=0.05,
    )

    workflow.model.eval()
    llama = Llama(workflow.model, workflow.tokenizer)

    with open(baseline_path) as f:
        data = [json.loads(line) for line in f][0]
        baseline_with_seeds = [(d['seed'], d['outputs']) for d in data if d['strategy'] == strategy]
        baseline = dedup_and_sort(baseline_with_seeds)

    with open(cached_path) as f:
        data = [json.loads(line) for line in f]
        cached_with_seeds = [(d['seed'], d['outputs']) for d in data if d['strategy'] == strategy]
        cached = dedup_and_sort(cached_with_seeds)

    workflow.model.set_adapter_state(enabled=False)
    baseline_cooperate = ["COOPERATE" in workflow.tokenizer.decode(c['decision_ids'][1]).upper() for c in baseline]
    cached_cooperate = ["COOPERATE" in workflow.tokenizer.decode(c['decision_ids'][1]).upper() for c in cached]
    baseline_bootstrap_results = bootstrap_binary(baseline_cooperate, cached_cooperate)

    nll = get_likelihoods(
        workflow=workflow,
        llama=llama,
        outputs=baseline,
        differences=[b != c for (b, c) in zip(baseline_cooperate, cached_cooperate)]
    )
    pre_kl_stats = bootstrap_continuous(nll['baseline_first_means'], nll['cached_first_means'])

    baseline_output_data = {
        'checkpoint': 'pre_ft',
        'strategy': strategy,
        'bootstrap_results': baseline_bootstrap_results,
        'kl_stats': pre_kl_stats
    }
    append_to_jsonl(baseline_output_data, output_file)

    print('=== Baseline ===:')
    print(json.dumps(baseline_bootstrap_results, indent=2))
    print(json.dumps(pre_kl_stats, indent=2))

    base_path = f'/scratch4/jeisner1/tjbai/checkpoints/prisoners/{strategy if strategy else 'baseline'}'
    for i, ckpt_path in enumerate(os.listdir(base_path)):
        if 'epoch' in ckpt_path: # old checkpoints, lazy to move
            continue
        existing_results = load_existing_results(i, output_file)
        if existing_results:
            print(f'=== {ckpt_path} ===')
            print(json.dumps(existing_results['bootstrap_results'], indent=2))
            print(json.dumps(existing_results['kl_stats'], indent=2))
            continue

        try:
            load_ckpt(workflow=workflow, ckpt_path=f'{base_path}/{ckpt_path}')
            workflow.model.set_adapter_state(enabled=True)
        except Exception as e:
            print(f"Error loading checkpoint {ckpt_path}: {e}")
            continue

        try:
            with open(ft_data_path) as f:
                data = [json.loads(line) for line in f]
                chunk_with_seeds = [(d['seed'], d['outputs']) for d in data if d['strategy'] == strategy and ckpt_path in d['ckpt_path']]
                chunk = dedup_and_sort(chunk_with_seeds)
                cached_bob_decisions = [workflow.tokenizer.decode(d['decision_ids'][1]) for d in chunk]
                cached_cooperate = ["COOPERATE" in choice.upper() for choice in cached_bob_decisions]
        except Exception as e:
            print(f"Error loading fine-tuned data for checkpoint {ckpt_path}: {e}")
            continue

        bootstrap_results = bootstrap_binary(
            baseline_cooperate[:max(len(baseline_cooperate), len(cached_cooperate))],
            cached_cooperate[:max(len(baseline_cooperate), len(cached_cooperate))]
        )

        try:
            nll = get_likelihoods(
                workflow=workflow,
                llama=llama,
                outputs=baseline,
                differences=[b != c for (b, c) in zip(baseline_cooperate, cached_cooperate)]
            )
            post_kl_stats = bootstrap_continuous(nll['baseline_first_means'], nll['cached_first_means'])
        except Exception as e:
            print(f"Error calculating likelihoods for {ckpt_path}: {e}")
            post_kl_stats = {"error": str(e)}

        output_data = {
            'checkpoint': ckpt_path,
            'strategy': strategy,
            'bootstrap_results': bootstrap_results,
            'kl_stats': post_kl_stats,
            'baseline_cooperate_rate': sum(baseline_cooperate) / len(baseline_cooperate),
            'ft_cooperate_rate': sum(cached_cooperate) / len(cached_cooperate)
        }
        append_to_jsonl(output_data, output_file)

        print(f'=== {ckpt_path} ===')
        print(json.dumps(bootstrap_results, indent=2))
        print(json.dumps(post_kl_stats, indent=2))

if __name__ == '__main__':
    fire.Fire(main)
