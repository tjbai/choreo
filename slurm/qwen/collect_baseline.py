import os, json, argparse
from tqdm import tqdm
from typing import Union, Literal

from llama import Workflow, Llama
from llama.util import find_free_port
from llama.workflows.tot import tot_baseline, eval_solutions, load_math_problems
from llama.workflows.mad import mad_baseline
from llama.workflows.madpar import madpar_baseline

import config

def setup_env():
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(find_free_port())

def collect_baseline_data(workflow: Union[
    Literal['tot'],
    Literal['mad'],
    Literal['madpar'],
]):
    print(f"Collecting baseline data for {workflow}")
    setup_env()
    workflow_obj = Workflow.build(
        ckpt_dir=config.QWEN_CKPT_DIR,
        tokenizer_path=config.QWEN_TOKENIZER_PATH,
        max_seq_len=config.MAX_SEQ_LEN if workflow != 'mad' else config.MAX_SEQ_LEN * 8,
        max_batch_size=config.MAX_BATCH_SIZE,
        model_parallel_size=config.MODEL_PARALLEL_SIZE,
        max_nodes=100 if workflow == 'tot' else 20,
        use_lora=False,
    )

    os.makedirs(config.DATA_ROOT, exist_ok=True)

    print("Collecting training data...")
    if workflow in ['tot']:
        train_problems = load_math_problems(config.MATH_DATA_PATH, split='train')[:config.NUM_PROBLEMS_TRAIN]
    else:
        train_problems = load_math_problems(config.MATH_DATA_PATH, split='test')[:config.NUM_PROBLEMS_TRAIN]

    train_solutions = []
    for problem in tqdm(train_problems, desc="Training data"):
        workflow_obj.reset()
        if workflow == 'tot':
            solution = tot_baseline(
                workflow=workflow_obj,
                problem=problem['problem'],
                branching_factor=config.BRANCHING_FACTOR,
                voters=config.VOTERS,
            )
        elif workflow == 'mad':
            solution = mad_baseline(
                workflow=workflow_obj,
                problem=problem['problem'],
                max_rounds=3,
            )
        elif workflow == 'madpar':
            solution = madpar_baseline(
                workflow=workflow_obj,
                problem=problem['problem'],
            )

        train_solutions.append({
            'inputs': problem,
            'outputs': solution
        })

    with open(f'{config.DATA_ROOT}/baseline_{workflow}_train.json', 'w') as f:
        json.dump(train_solutions, f)

    print("Collecting test data...")
    test_problems = load_math_problems(config.MATH_DATA_PATH, split='test')[:config.NUM_PROBLEMS_TEST]

    test_solutions = []
    for problem in tqdm(test_problems, desc="Test data"):
        workflow_obj.reset()
        if workflow == 'tot':
            solution = tot_baseline(
                workflow=workflow_obj,
                problem=problem['problem'],
                branching_factor=config.BRANCHING_FACTOR,
                voters=config.VOTERS,
            )
        elif workflow == 'mad':
            solution = mad_baseline(
                workflow=workflow_obj,
                problem=problem['problem'],
                max_rounds=3,
            )
        elif workflow == 'madpar':
            solution = madpar_baseline(
                workflow=workflow_obj,
                problem=problem['problem'],
            )
        test_solutions.append({
            'inputs': problem,
            'outputs': solution
        })

    with open(f'{config.DATA_ROOT}/baseline_{workflow}_test.json', 'w') as f:
        json.dump(test_solutions, f)

    llama = Llama(workflow_obj.model, workflow_obj.tokenizer)
    llama.model.reshape_cache(4)

    if workflow == 'tot':
        all_correct = eval_solutions(
            llama=llama,
            solutions=[workflow_obj.tokenizer.decode(s['outputs']['final_tokens']) for s in test_solutions],
            problems=test_problems
        )
    elif workflow == 'mad':
        solutions = []
        for s in test_solutions:
            if isinstance(s['outputs']['decision'], dict) and s['outputs']['decision'].get('Answer'):
                solutions.append(s['outputs']['decision']['Answer'])
            else:
                solutions.append(None)
        all_correct = eval_solutions(llama=llama, solutions=solutions, problems=test_problems)
    elif workflow == 'madpar':
        solutions = []
        for s in test_solutions:
            final_answers = s['outputs'].get('final_answers', [])
            answer = next((ans for ans in final_answers if ans is not None), None)
            solutions.append(answer)
        all_correct = eval_solutions(llama=llama, solutions=solutions, problems=test_problems)

    accuracy = sum(all_correct) / len(all_correct)
    print(f'Baseline {workflow} accuracy: {accuracy:.3f} ({sum(all_correct)}/{len(all_correct)})')

    results = {
        'workflow': workflow,
        'condition': 'baseline',
        'accuracy': accuracy,
        'correct': sum(all_correct),
        'total': len(all_correct),
        'all_correct': all_correct
    }

    with open(f'{config.DATA_ROOT}/results_baseline_{workflow}.json', 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--workflow', choices=['tot', 'mad', 'madpar'], required=True)
    args = parser.parse_args()
    collect_baseline_data(args.workflow)
