import os
import json
from llama import Workflow, Llama
from llama.workflows.prisoners import get_likelihoods
from llama.util import bootstrap_binary, bootstrap_continuous, find_free_port, load_ckpt

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = str(find_free_port())

workflow = Workflow.build(
    ckpt_dir='/scratch4/jeisner1/tjbai/llama_8b',
    tokenizer_path='/scratch4/jeisner1/tjbai/llama_8b/tokenizer.model',
    max_seq_len=8192,
    max_batch_size=8,
    model_parallel_size=1,
    max_nodes=100,
    use_lora=True,
    lora_rank=32,
    lora_alpha=64,
    lora_dropout=0.05
)

workflow.model.eval()
llama = Llama(workflow.model, workflow.tokenizer)

with open('/home/tbai4/llama3/dumps/prisoners/prisoners_baseline.jsonl') as f:
    data = [json.loads(line) for line in f]
    baseline_always_defect = [d['outputs'] for d in data[200:]]

with open('/home/tbai4/llama3/dumps/prisoners/prisoners_cached_paired.jsonl') as f:
    data = [json.loads(line) for line in f]
    cached_always_defect = [d['outputs'] for d in data[200:300]]

workflow.model.set_adapter_state(enabled=False)
baseline_cooperate = ["COOPERATE" in workflow.tokenizer.decode(c['decision_ids'][1]).upper() for c in baseline_always_defect]
cached_cooperate = ["COOPERATE" in workflow.tokenizer.decode(c['decision_ids'][1]).upper() for c in cached_always_defect]
bootstrap_results = bootstrap_binary(baseline_cooperate, cached_cooperate)

nll = get_likelihoods(
    workflow=workflow,
    llama=llama,
    outputs=baseline_always_defect,
    differences=[b != c for (b, c) in zip(baseline_cooperate, cached_cooperate)]
)
pre_kl_stats = bootstrap_continuous(nll['baseline_first_means'], nll['cached_first_means'])

print('=== Baseline ===:')
print(json.dumps(bootstrap_results, indent=2))
print(json.dumps(pre_kl_stats, indent=2))

for i, ckpt_path in enumerate([
    '/scratch4/jeisner1/tjbai/checkpoints/prisoners/always_defect/lora_epoch-0_step-95.pt',
    '/scratch4/jeisner1/tjbai/checkpoints/prisoners/always_defect/lora_epoch-0_step-195.pt',
    '/scratch4/jeisner1/tjbai/checkpoints/prisoners/always_defect/lora_epoch-1_step-51.pt',
    '/scratch4/jeisner1/tjbai/checkpoints/prisoners/always_defect/lora_epoch-1_step-151.pt',
]):
    load_ckpt(workflow=workflow, ckpt_path=ckpt_path)
    workflow.model.set_adapter_state(enabled=True)

    with open('/home/tbai4/llama3/dumps/prisoners/prisoners_ft_eval_always_defect.jsonl') as f:
        data = [json.loads(line) for line in f][i*100:(i+1)*100]
        cached_bob_decisions = [d['bob_final'] for d in data]
        cached_cooperate = ["COOPERATE" in choice.upper() for choice in cached_bob_decisions]

    bootstrap_results = bootstrap_binary(baseline_cooperate, cached_cooperate)

    nll = get_likelihoods(
        workflow=workflow,
        llama=llama,
        outputs=cached_always_defect,
        differences=[b != c for (b, c) in zip(baseline_cooperate, cached_cooperate)]
    )

    post_kl_stats = bootstrap_continuous(nll['baseline_first_means'], nll['cached_first_means'])

    print(f'=== FT{(i+1)*400} ===')
    print(json.dumps(bootstrap_results, indent=2))
    print(json.dumps(post_kl_stats, indent=2))
