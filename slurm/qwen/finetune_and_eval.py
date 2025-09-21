#!/usr/bin/env python3
import os
import json
import argparse
import torch
from tqdm import tqdm

from llama import Workflow, Llama
from llama.util import find_free_port, load_ckpt
from llama.workflows.tot import tot_cached, eval_solutions, load_math_problems
from llama.workflows.mad import mad_cached
from llama.workflows.madpar import madpar_cached
from llama.workflows.finetune import finetune

from . import config

def setup_env():
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(find_free_port())

def get_cached_function(workflow):
    if workflow == 'tot':
        return tot_cached
    elif workflow == 'mad':
        return mad_cached
    elif workflow == 'madpar':
        return madpar_cached
    else:
        raise ValueError(f"Unknown workflow: {workflow}")

def finetune_choreo_workflow(workflow):
    """Finetune choreographed workflow on baseline training data"""
    print(f"Finetuning choreographed {workflow} workflow...")

    setup_env()

    # Create output directory
    output_dir = f'{config.CHECKPOINT_ROOT}/{workflow}_choreo_ft'
    os.makedirs(output_dir, exist_ok=True)

    # Finetune using the baseline training data
    finetune(
        task=workflow,
        data_path=f'{config.DATA_ROOT}/baseline_{workflow}_train.json',
        ckpt_dir=config.QWEN_CKPT_DIR,
        tokenizer_path=config.QWEN_TOKENIZER_PATH,
        output_dir=output_dir,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        max_seq_len=config.MAX_SEQ_LEN if workflow != 'mad' else config.MAX_SEQ_LEN * 8,
        checkpoint_freq=50 if workflow == 'tot' else 450,
        validation_freq=100 if workflow == 'tot' else 450,
        branching_factor=config.BRANCHING_FACTOR,
        voters=config.VOTERS,
        epochs=config.EPOCHS,
        lora_rank=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        learning_rate=config.LEARNING_RATE,
    )

    return output_dir

def finetune_direct_workflow(workflow):
    """Finetune direct workflow on baseline training data (distilled baseline)"""
    print(f"Finetuning distilled baseline {workflow} workflow...")

    setup_env()

    # Create output directory
    output_dir = f'{config.CHECKPOINT_ROOT}/{workflow}_distilled'
    os.makedirs(output_dir, exist_ok=True)

    # Finetune direct workflow on baseline data
    finetune(
        task='direct',
        data_path=f'{config.DATA_ROOT}/baseline_{workflow}_train.json',
        ckpt_dir=config.QWEN_CKPT_DIR,
        tokenizer_path=config.QWEN_TOKENIZER_PATH,
        output_dir=output_dir,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        max_seq_len=config.MAX_SEQ_LEN if workflow != 'mad' else config.MAX_SEQ_LEN * 8,
        checkpoint_freq=50 if workflow == 'tot' else 450,
        validation_freq=100 if workflow == 'tot' else 450,
        epochs=config.EPOCHS,
        lora_rank=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        learning_rate=config.LEARNING_RATE,
    )

    return output_dir

def evaluate_finetuned_choreo(workflow, checkpoint_dir):
    """Evaluate finetuned choreographed workflow"""
    print(f"Evaluating finetuned choreographed {workflow}...")

    setup_env()

    # Build workflow with LoRA
    workflow_obj = Workflow.build(
        ckpt_dir=config.QWEN_CKPT_DIR,
        tokenizer_path=config.QWEN_TOKENIZER_PATH,
        max_seq_len=config.MAX_SEQ_LEN if workflow != 'mad' else config.MAX_SEQ_LEN * 8,
        max_batch_size=config.MAX_BATCH_SIZE * 8 if workflow == 'tot' else config.MAX_BATCH_SIZE,
        model_parallel_size=config.MODEL_PARALLEL_SIZE,
        max_nodes=20,
        use_lora=True,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT
    )
    workflow_obj.model.eval()

    # Load the best checkpoint (using final checkpoint for now)
    # You might want to select the best based on validation loss
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('lora_step-') and f.endswith('.pt')]
    if not checkpoint_files:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")

    # Use the last checkpoint
    checkpoint_files.sort(key=lambda x: int(x.split('-')[1].split('.')[0]))
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[-1])

    load_ckpt(workflow_obj, checkpoint_path)
    workflow_obj.model.eval()
    workflow_obj.model.reshape_cache(1)
    workflow_obj.model.set_adapter_state(enabled=True)

    # Load test problems
    test_problems = load_math_problems(config.MATH_DATA_PATH, split='test')[:config.NUM_PROBLEMS_TEST]
    cached_fn = get_cached_function(workflow)

    # Run inference
    test_solutions = []
    for problem in tqdm(test_problems, desc=f'Evaluating finetuned choreo {workflow}'):
        workflow_obj.reset()
        try:
            if workflow == 'tot':
                solution = cached_fn(
                    workflow=workflow_obj,
                    problem=problem['problem'],
                    branching_factor=config.BRANCHING_FACTOR,
                    voters=config.VOTERS,
                )
            elif workflow == 'mad':
                solution = cached_fn(
                    workflow=workflow_obj,
                    problem=problem['problem'],
                    max_rounds=3,
                )
            elif workflow == 'madpar':
                solution = cached_fn(
                    workflow=workflow_obj,
                    problem=problem['problem'],
                )
            test_solutions.append(solution)
        except Exception as e:
            print(f"Skipping problem due to error: {e}")
            continue

    # Evaluate
    llama = Llama(workflow_obj.model, workflow_obj.tokenizer)
    llama.model.reshape_cache(4)
    llama.model.set_adapter_state(enabled=False)

    if workflow == 'tot':
        all_correct = eval_solutions(llama, [workflow_obj.tokenizer.decode(s['final_tokens']) for s in test_solutions], test_problems)
    elif workflow == 'mad':
        # MAD: Extract Answer field from decision dict
        solutions = []
        for s in test_solutions:
            if isinstance(s['decision'], dict) and s['decision'].get('Answer'):
                solutions.append(s['decision']['Answer'])
            else:
                solutions.append(None)
        all_correct = eval_solutions(llama=llama, solutions=solutions, problems=test_problems)
    elif workflow == 'madpar':
        # MADpar: Use the first valid answer from final_answers list
        solutions = []
        for s in test_solutions:
            final_answers = s.get('final_answers', [])
            # Use first non-None answer, or None if all are None
            answer = next((ans for ans in final_answers if ans is not None), None)
            solutions.append(answer)
        all_correct = eval_solutions(llama=llama, solutions=solutions, problems=test_problems)

    accuracy = sum(all_correct) / len(all_correct) if all_correct else 0.0
    print(f'Choreo + FT {workflow} accuracy: {accuracy:.3f} ({sum(all_correct)}/{len(all_correct)})')

    # Save results
    results = {
        'workflow': workflow,
        'condition': 'choreo_ft',
        'accuracy': accuracy,
        'correct': sum(all_correct),
        'total': len(all_correct),
        'all_correct': all_correct,  # For statistical testing
        'checkpoint': checkpoint_path
    }

    with open(f'{config.DATA_ROOT}/results_choreo_ft_{workflow}.json', 'w') as f:
        json.dump(results, f)

    return accuracy

def evaluate_distilled_baseline(workflow, checkpoint_dir):
    """Evaluate distilled baseline (direct model finetuned on baseline traces)"""
    print(f"Evaluating distilled baseline {workflow}...")

    setup_env()

    # Build workflow with LoRA
    workflow_obj = Workflow.build(
        ckpt_dir=config.QWEN_CKPT_DIR,
        tokenizer_path=config.QWEN_TOKENIZER_PATH,
        max_seq_len=config.MAX_SEQ_LEN if workflow != 'mad' else config.MAX_SEQ_LEN * 8,
        max_batch_size=config.MAX_BATCH_SIZE,
        model_parallel_size=config.MODEL_PARALLEL_SIZE,
        max_nodes=20,
        use_lora=True,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT
    )
    workflow_obj.model.eval()

    # Load checkpoint
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('lora_step-') and f.endswith('.pt')]
    if not checkpoint_files:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")

    checkpoint_files.sort(key=lambda x: int(x.split('-')[1].split('.')[0]))
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[-1])

    load_ckpt(workflow_obj, checkpoint_path)
    workflow_obj.model.eval()
    workflow_obj.model.set_adapter_state(enabled=True)

    # Load test problems
    test_problems = load_math_problems(config.MATH_DATA_PATH, split='test')[:config.NUM_PROBLEMS_TEST]

    # Run direct inference (no choreographed workflow)
    llama = Llama(workflow_obj.model, workflow_obj.tokenizer)
    test_solutions = []

    for problem in tqdm(test_problems, desc=f'Evaluating distilled {workflow}'):
        # Direct generation without workflow
        prompt = f"Problem: {problem['problem']}\nSolution:"
        tokens = workflow_obj.tokenizer.encode(prompt, bos=True, eos=False)

        # Generate solution directly
        generated = llama.generate(
            prompt_tokens=[tokens],
            max_gen_len=512,
            temperature=0.0,
            top_p=1.0,
        )

        solution_text = generated[0]
        test_solutions.append(solution_text)

    # Evaluate
    # For distilled baseline, all workflows generate text directly, so use same evaluation
    all_correct = eval_solutions(llama, test_solutions, test_problems)

    accuracy = sum(all_correct) / len(all_correct) if all_correct else 0.0
    print(f'Distilled Baseline {workflow} accuracy: {accuracy:.3f} ({sum(all_correct)}/{len(all_correct)})')

    # Save results
    results = {
        'workflow': workflow,
        'condition': 'distilled',
        'accuracy': accuracy,
        'correct': sum(all_correct),
        'total': len(all_correct),
        'all_correct': all_correct,  # For statistical testing
        'checkpoint': checkpoint_path
    }

    with open(f'{config.DATA_ROOT}/results_distilled_{workflow}.json', 'w') as f:
        json.dump(results, f)

    return accuracy

def main(workflow):
    os.makedirs(config.DATA_ROOT, exist_ok=True)
    os.makedirs(config.CHECKPOINT_ROOT, exist_ok=True)
    choreo_checkpoint_dir = finetune_choreo_workflow(workflow)
    distilled_checkpoint_dir = finetune_direct_workflow(workflow)
    choreo_accuracy = evaluate_finetuned_choreo(workflow, choreo_checkpoint_dir)
    distilled_accuracy = evaluate_distilled_baseline(workflow, distilled_checkpoint_dir)
    print(f"\n{workflow.upper()} Results:")
    print(f"Choreo + FT: {choreo_accuracy:.3f}")
    print(f"Distilled Baseline: {distilled_accuracy:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--workflow', choices=['tot', 'mad', 'madpar'], required=True)
    args = parser.parse_args()

    main(args.workflow)
