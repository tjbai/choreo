import os
import json
import random
from tqdm import tqdm
from collections import defaultdict

from llama import Workflow, Llama
from llama.util import find_free_port, load_ckpt
from llama.workflows.qa import ask_sequential, ask_parallel, eval_system_prompt, parse_items, format_eval_user

N = 500
DATA_PATH = '/home/tbai4/llama3/data/triviaqa/unfiltered-web-dev.json'
CKPT_PATH = '/scratch4/jeisner1/tjbai/checkpoints/triviaqa/lora_step-199.pt'

with open(DATA_PATH) as f:
    data = json.load(f)
    problems = data['Data'][:len(data['Data'])//2]

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
load_ckpt(workflow, CKPT_PATH)
workflow.model.eval()
llama = Llama(workflow.model, workflow.tokenizer)
answers = defaultdict(list)

for seed in tqdm(range(N)):
    workflow.reset()
    random.seed(seed)
    subset = random.sample(problems, k=2)

    workflow.model.set_adapter_state(enabled=False)

    answer = ask_sequential(workflow, subset)
    answers['baseline'].append((subset, workflow.tokenizer.decode(answer['output_tokens'])))

    answer = ask_parallel(workflow, subset, annotate=False, compact=False)
    answers['choreographed'].append((subset, workflow.tokenizer.decode(answer['output_tokens'])))

    answer = ask_parallel(workflow, subset, annotate=True, compact=True)
    answers['choreographed+linearized'].append((subset, workflow.tokenizer.decode(answer['output_tokens'])))

    workflow.model.set_adapter_state(enabled=True)

    answer = ask_parallel(workflow, subset, annotate=True)
    answers['choreographed+finetuned'].append((subset, workflow.tokenizer.decode(answer['output_tokens'])))

workflow.model.set_adapter_state(enabled=False)

for setting in ['baseline', 'choreographed+finetuned']:
    first_correct = 0
    second_correct = 0
    for subset, answer in tqdm(answers[setting]):
        items = parse_items(answer)
        if len(items) == 2:
            resps = llama.chat_completion([
                [{'role': 'system', 'content': eval_system_prompt},
                    {'role': 'user', 'content': format_eval_user(subset[0], items[0])}],
                [{'role': 'system', 'content': eval_system_prompt},
                    {'role': 'user', 'content': format_eval_user(subset[1], items[1])}],
            ], content_prefills=['{"correct": "'] * 2)
            first_correct += 'true' in resps[0]['generation']['content'].lower()
            second_correct += 'true' in resps[1]['generation']['content'].lower()
    print(setting, first_correct, second_correct)

for setting in ['choreographed', 'choreographed+linearized']:
    first_correct = 0
    second_correct = 0
    for subset, answer in tqdm(answers[setting]):
        items = parse_items(answer)
        if len(items) == 2:
            resps = llama.chat_completion([
                [{'role': 'system', 'content': eval_system_prompt},
                    {'role': 'user', 'content': format_eval_user(subset[0], items[0])}],
                [{'role': 'system', 'content': eval_system_prompt},
                    {'role': 'user', 'content': format_eval_user(subset[1], items[1])}],
            ], content_prefills=['{"correct": "'] * 2)
            first_correct += 'true' in resps[0]['generation']['content'].lower()
            second_correct += 'true' in resps[1]['generation']['content'].lower()
        elif len(items) == 1:
            resps = llama.chat_completion([
                [{'role': 'system', 'content': eval_system_prompt},
                    {'role': 'user', 'content': format_eval_user(subset[0], items[0])}],
                [{'role': 'system', 'content': eval_system_prompt},
                    {'role': 'user', 'content': format_eval_user(subset[1], items[0])}],
            ], content_prefills=['{"correct": "'] * 2)
            first_correct += 'true' in resps[0]['generation']['content'].lower()
            second_correct += 'true' in resps[1]['generation']['content'].lower()
    print(setting, first_correct, second_correct)
