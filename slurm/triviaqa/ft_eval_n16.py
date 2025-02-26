import os
import json
import random
from collections import defaultdict
import torch
from tqdm import tqdm
from llama import Workflow, Llama
from llama.util import find_free_port
from llama.workflows.qa import ask_sequential, ask_parallel, parse_items, eval_system_prompt, format_eval_user

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
    lora_dropout=0.05,
)
llama = Llama(workflow.model, workflow.tokenizer)

with open('/home/tbai4/llama3/data/triviaqa/unfiltered-web-dev.json') as f:
    data = json.load(f)
    problems = data['Data'][:len(data['Data'])//2]

N = 50
problem_sizes = [2, 4, 8, 16]
checkpoints = [
    '/scratch4/jeisner1/tjbai/checkpoints/triviaqa/lora_epoch-0_step-195.pt',
    '/scratch4/jeisner1/tjbai/checkpoints/triviaqa/lora_epoch-0_step-395.pt',
    '/scratch4/jeisner1/tjbai/checkpoints/triviaqa/lora_epoch-1_step-147.pt',
    '/scratch4/jeisner1/tjbai/checkpoints/triviaqa/lora_epoch-1_step-347.pt',
]

def generate_answers(workflow, problems, n_samples, problem_sizes, ask_fn, prefix=""):
    answers = defaultdict(list)
    for n in problem_sizes:
        for seed in tqdm(range(n_samples), desc=f"{prefix} n={n}"):
            workflow.reset()
            random.seed(seed)
            subset = random.sample(problems, k=n)
            answer = ask_fn(workflow, subset, annotate=True, compact=False)
            answers[n].append((subset, workflow.tokenizer.decode(answer['output_tokens'])))
    return answers

def evaluate_answers(llama, answers, problem_sizes):
    results = {}
    for n in problem_sizes:
        correct = defaultdict(int)
        for subset, answer in tqdm(answers[n], desc=f"Evaluating n={n}"):
            individual_answers = parse_items(answer)
            resps = llama.chat_completion([
                [{'role': 'system', 'content': eval_system_prompt},
                 {'role': 'user', 'content': format_eval_user(s, a)}]
                for s, a in zip(subset, individual_answers)
            ], content_prefills=['{"correct": "'] * min(n, len(individual_answers)))

            for i, r in enumerate(resps):
                if 'true' in r['generation']['content'].lower():
                    correct[i+1] += 1
        results[n] = dict(correct)
    return results

### Baseline
print('Evaluating baseline (sequential)')
workflow.model.set_adapter_state(enabled=False)
sequential_answers = generate_answers(
    workflow, problems, N, problem_sizes,
    lambda w, s, **kwargs: ask_sequential(w, s),
    prefix="Sequential"
)

workflow.model.reshape_cache(16)
sequential_results = evaluate_answers(llama, sequential_answers, problem_sizes)
for n, correct in sequential_results.items():
    print(f"Sequential n={n}: {correct}")

### Checkpoints
for ckpt_path in checkpoints:
    print(f'\nEvaluating {os.path.basename(ckpt_path)}')
    ckpt = torch.load(ckpt_path, weights_only=True)
    for weight, param in zip(ckpt['trainable_params'], workflow.model.get_trainable_parameters()):
        param.data.copy_(weight)

    workflow.model.reshape_cache(1)
    workflow.model.set_adapter_state(enabled=True)
    parallel_answers = generate_answers(
        workflow, problems, N, problem_sizes, ask_parallel, prefix="Parallel"
    )

    workflow.model.reshape_cache(16)
    workflow.model.set_adapter_state(enabled=False)
    parallel_results = evaluate_answers(llama, parallel_answers, problem_sizes)
    for n, correct in parallel_results.items():
        print(f"Parallel n={n}: {correct}")
