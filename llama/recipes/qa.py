import os
import json
import random
from datetime import datetime
from collections import defaultdict

import fire
from tqdm import tqdm

from llama import Workflow, Llama
from llama.util import find_free_port
from llama.workflows.finetune import finetune
from llama.workflows.qa import (
    ask_sequential,
    ask_parallel,
    eval_system_prompt,
    format_eval_user,
    parse_items
)

def main(
    train_data_path='/home/tbai4/llama3/data/triviaqa/unfiltered-web-train.json',
    dev_data_path='/home/tbai4/llama3/data/triviaqa/unfiltered-web-train.json',
    num_questions=2,
    num_eval=50,
):
    print(f'Running recipe for n={num_questions}')

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

    llama = Llama(workflow.model, workflow.tokenizer)

    with open(train_data_path) as f:
        data = json.load(f)
        problems = data['Data']

    # 1. generate samples
    workflow.model.eval()
    workflow.model.reshape_cache(num_questions)
    workflow.model.set_adapter_state(enabled=False)

    outputs = []
    for seed in tqdm(range(500)):
        workflow.reset()
        random.seed(seed)
        subset = random.sample(problems, k=num_questions)
        outputs.append({
            'subset': subset,
            'outputs': ask_sequential(workflow, subset)
        })

    tmp_file = f'tmp_{datetime.now()}.json'
    with open(tmp_file, 'w') as f:
        json.dump(outputs, f)

    try:
        # 2. fine-tune
        workflow.model.train()
        workflow.model.reshape_cache(1)
        workflow.model.set_adapter_state(enabled=True)

        finetune(
            task='triviaqa',
            data_path=tmp_file,
            workflow=workflow,
            ckpt_dir='/scratch4/jeisner1/tjbai/llama_8b',
            tokenizer_path='/scratch4/jeisner1/tjbai/llama_8b/tokenizer.model',
            output_dir='/scratch4/jeisner1/tjbai/checkpoints',
            max_seq_len=8192,
            epochs=4,
            gradient_accumulation_steps=4,
            checkpoint_freq=int(1e9),
            validation_freq=100,
            lora_rank=32,
            lora_alpha=64,
            lora_dropout=0.05,
            learning_rate=5e-5,
        )

    finally:
        os.remove(tmp_file)

    # 3. eval
    workflow.model.eval()

    with open(dev_data_path) as f:
        data = json.load(f)
        problems = data['Data']

    answers = []
    for seed in range(num_eval):
        workflow.reset()
        random.seed(seed)
        subset = random.sample(problems, k=num_questions)
        answer = ask_parallel(workflow, subset, annotate=True)
        answers.append((subset, workflow.tokenizer.decode(answer['output_tokens'])))

    workflow.model.reshape_cache(num_questions)
    workflow.model.set_adapter_state(enabled=False)
    correct = defaultdict(int)
    for subset, answer in tqdm(answers, desc='Evaluating'):
        individual_answers = parse_items(answer)
        resps = llama.chat_completion([
            [{'role': 'system', 'content': eval_system_prompt},
            {'role': 'user', 'content': format_eval_user(s, a)}]
            for s, a in zip(subset, individual_answers)
        ], content_prefills=['{"correct": "'] * min(num_questions, len(individual_answers)))

        for i, r in enumerate(resps):
            if 'true' in r['generation']['content'].lower():
                correct[i] += 1

    print(sorted(list(correct.items())))

if __name__ == '__main__':
    fire.Fire(main)
