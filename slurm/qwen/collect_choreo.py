#!/usr/bin/env python3
import os
import json
import argparse
from math import ceil
from pathlib import Path
from tqdm import tqdm

from llama import Workflow, Llama
from llama.workflows.tot import tot_cached, eval_solutions, load_math_problems
from llama.workflows.mad import mad_cached
from llama.workflows.madpar import madpar_cached

from . import config


def setup_env():
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(find_free_port())


def _select_range(total: int, shard_index: int | None, num_shards: int | None, start: int, limit: int | None):
    if num_shards is not None and shard_index is not None:
        if not (0 <= shard_index < num_shards):
            raise ValueError(f"shard_index {shard_index} must be in [0,{num_shards})")
        shard_size = ceil(total / num_shards)
        s = shard_index * shard_size
        e = min(total, s + shard_size)
        return s, e
    s = max(0, start)
    e = total if limit is None else min(total, s + max(0, limit))
    return s, e


def _evaluate_test(workflow_name: str, workflow_obj: Workflow, test_solutions, problems):
    llama = Llama(workflow_obj.model, workflow_obj.tokenizer)
    llama.model.reshape_cache(4)

    if workflow_name == 'tot':
        solutions = [
            (workflow_obj.tokenizer.decode(s['outputs']['final_tokens'])
             if s['outputs'].get('final_tokens') is not None else None)
            for s in test_solutions
        ]
        all_correct = eval_solutions(llama=llama, solutions=solutions, problems=problems)
    elif workflow_name == 'mad':
        solutions = []
        for s in test_solutions:
            decision = s['outputs'].get('decision')
            if isinstance(decision, dict) and decision.get('Answer'):
                solutions.append(decision['Answer'])
            else:
                solutions.append(None)
        all_correct = eval_solutions(llama=llama, solutions=solutions, problems=problems)
    elif workflow_name == 'madpar':
        solutions = []
        for s in test_solutions:
            final_answers = s['outputs'].get('final_answers', [])
            answer = next((ans for ans in final_answers if ans is not None), None)
            solutions.append(answer)
        all_correct = eval_solutions(llama=llama, solutions=solutions, problems=problems)
    else:
        raise ValueError(f"Unknown workflow {workflow_name}")

    accuracy = sum(all_correct) / len(all_correct) if all_correct else 0.0
    return {
        'accuracy': accuracy,
        'correct': int(sum(all_correct)),
        'total': int(len(all_correct)),
        'all_correct': all_correct,
    }


def collect_choreo_data(
    workflow: str,
    *,
    shard_index: int | None = None,
    num_shards: int | None = None,
    start: int = 0,
    limit: int | None = None,
    resume: bool = False,
    output_dir: str | None = None,
):
    print(f"Collecting choreographed data for {workflow}")
    setup_env()

    workflow_obj = Workflow.build(
        ckpt_dir=config.QWEN_CKPT_DIR,
        tokenizer_path=config.QWEN_TOKENIZER_PATH,
        max_seq_len=config.MAX_SEQ_LEN if workflow != 'mad' else config.MAX_SEQ_LEN * 8,
        max_batch_size=config.MAX_BATCH_SIZE * 8 if workflow == 'tot' else config.MAX_BATCH_SIZE,
        model_parallel_size=config.MODEL_PARALLEL_SIZE,
        max_nodes=20,
        use_lora=False,
    )

    out_root = Path(output_dir) if output_dir else Path(config.DATA_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    problems = load_math_problems(config.MATH_DATA_PATH, split='test')[:config.NUM_PROBLEMS_TEST]
    total = len(problems)
    s, e = _select_range(total, shard_index, num_shards, start, limit)
    probs_slice = problems[s:e]

    shard_suffix = (
        f"_shard{shard_index}-of-{num_shards}" if (num_shards is not None and shard_index is not None)
        else (f"_offset{s}-len{len(probs_slice)}" if (s != 0 or (limit is not None)) else "")
    )

    out_path = out_root / f"choreo_{workflow}_test{shard_suffix}.json"

    existing = []
    if resume and out_path.exists():
        try:
            with open(out_path) as f:
                existing = json.load(f)
        except Exception:
            existing = []
    start_idx = len(existing)
    test_solutions = existing.copy()

    for i, problem in enumerate(tqdm(probs_slice, desc=f"Choreo test ({s}-{e} of {total})")):
        if i < start_idx:
            continue
        workflow_obj.reset()
        try:
            if workflow == 'tot':
                solution = tot_cached(
                    workflow=workflow_obj,
                    problem=problem['problem'],
                    branching_factor=config.BRANCHING_FACTOR,
                    voters=config.VOTERS,
                )
            elif workflow == 'mad':
                solution = mad_cached(
                    workflow=workflow_obj,
                    problem=problem['problem'],
                    max_rounds=3,
                )
            elif workflow == 'madpar':
                solution = madpar_cached(
                    workflow=workflow_obj,
                    problem=problem['problem'],
                )
            else:
                raise ValueError(f"Unknown workflow {workflow}")

            test_solutions.append({'inputs': problem, 'outputs': solution})
            with open(out_path, 'w') as f:
                json.dump(test_solutions, f)
        except Exception as e:
            print(f"Skipping problem due to error: {e}")
            continue

    eval_res = _evaluate_test(workflow, workflow_obj, test_solutions, probs_slice)
    print(f'Choreographed {workflow} shard accuracy: {eval_res["accuracy"]:.3f} ({eval_res["correct"]}/{eval_res["total"]})')

    shard_results = out_root / f"results_choreo_{workflow}_test{shard_suffix}.json"
    with open(shard_results, 'w') as f:
        json.dump({
            'workflow': workflow,
            'condition': 'choreographed',
            'shard': {
                'start': s,
                'end': e,
                'shard_index': shard_index,
                'num_shards': num_shards,
            },
            **eval_res,
        }, f)


def merge_shards(workflow: str, num_shards: int, *, output_dir: str | None = None):
    out_root = Path(output_dir) if output_dir else Path(config.DATA_ROOT)
    shard_files = [out_root / f"choreo_{workflow}_test_shard{i}-of-{num_shards}.json" for i in range(num_shards)]
    all_solutions = []
    all_problems = []
    for p in shard_files:
        if not p.exists():
            raise FileNotFoundError(f"Missing shard: {p}")
        with open(p) as f:
            shard = json.load(f)
        all_solutions.extend(shard)
        all_problems.extend([x['inputs'] for x in shard])

    merged_path = out_root / f"choreo_{workflow}_test.json"
    with open(merged_path, 'w') as f:
        json.dump(all_solutions, f)
    print(f"Wrote merged file: {merged_path}")

    setup_env()
    workflow_obj = Workflow.build(
        ckpt_dir=config.QWEN_CKPT_DIR,
        tokenizer_path=config.QWEN_TOKENIZER_PATH,
        max_seq_len=config.MAX_SEQ_LEN,
        max_batch_size=4,
        model_parallel_size=config.MODEL_PARALLEL_SIZE,
        max_nodes=4,
        use_lora=False,
    )
    eval_res = _evaluate_test(workflow, workflow_obj, all_solutions, all_problems)
    results_path = out_root / f"results_choreo_{workflow}.json"
    with open(results_path, 'w') as f:
        json.dump({
            'workflow': workflow,
            'condition': 'choreographed',
            **eval_res,
        }, f)
    print(f"Wrote merged results: {results_path} (acc={eval_res['accuracy']:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect choreographed test data with sharding/resume/merge')
    sub = parser.add_subparsers(dest='cmd', required=False)

    parser.add_argument('--workflow', choices=['tot', 'mad', 'madpar'])
    parser.add_argument('--shard-index', type=int, default=None)
    parser.add_argument('--num-shards', type=int, default=None)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output-dir', type=str, default=None)

    c = sub.add_parser('collect')
    c.add_argument('--workflow', choices=['tot', 'mad', 'madpar'], required=True)
    c.add_argument('--shard-index', type=int, default=None)
    c.add_argument('--num-shards', type=int, default=None)
    c.add_argument('--start', type=int, default=0)
    c.add_argument('--limit', type=int, default=None)
    c.add_argument('--resume', action='store_true')
    c.add_argument('--output-dir', type=str, default=None)

    m = sub.add_parser('merge')
    m.add_argument('--workflow', choices=['tot', 'mad', 'madpar'], required=True)
    m.add_argument('--num-shards', type=int, required=True)
    m.add_argument('--output-dir', type=str, default=None)

    args = parser.parse_args()
    if args.cmd in (None, 'collect'):
        if args.workflow is None:
            parser.error('--workflow is required')
        collect_choreo_data(
            args.workflow,
            shard_index=getattr(args, 'shard_index', None),
            num_shards=getattr(args, 'num_shards', None),
            start=getattr(args, 'start', 0),
            limit=getattr(args, 'limit', None),
            resume=getattr(args, 'resume', False),
            output_dir=getattr(args, 'output_dir', None),
        )
    elif args.cmd == 'merge':
        merge_shards(args.workflow, args.num_shards, output_dir=args.output_dir)
