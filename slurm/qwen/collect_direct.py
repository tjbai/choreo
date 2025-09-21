#!/usr/bin/env python3
import os
import json
import argparse
from math import ceil
from pathlib import Path
from tqdm import tqdm

from llama import Workflow, Llama
from llama.util import find_free_port
from llama.workflows.tot import eval_solutions, load_math_problems

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


def collect_direct_baseline(
    *,
    shard_index: int | None = None,
    num_shards: int | None = None,
    start: int = 0,
    limit: int | None = None,
    resume: bool = False,
    output_dir: str | None = None,
):
    setup_env()

    # Raw model (no LoRA), direct generation
    wf = Workflow.build(
        ckpt_dir=config.QWEN_CKPT_DIR,
        tokenizer_path=config.QWEN_TOKENIZER_PATH,
        max_seq_len=config.MAX_SEQ_LEN,
        max_batch_size=config.MAX_BATCH_SIZE,
        model_parallel_size=config.MODEL_PARALLEL_SIZE,
        max_nodes=4,
        use_lora=False,
    )

    out_root = Path(output_dir) if output_dir else Path(config.DATA_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    problems_all = load_math_problems(config.MATH_DATA_PATH, split='test')[:config.NUM_PROBLEMS_TEST]
    total = len(problems_all)
    s, e = _select_range(total, shard_index, num_shards, start, limit)
    problems = problems_all[s:e]
    shard_suffix = (
        f"_shard{shard_index}-of-{num_shards}" if (num_shards is not None and shard_index is not None)
        else (f"_offset{s}-len{len(problems)}" if (s != 0 or (limit is not None)) else "")
    )

    out_path = out_root / f"direct_test{shard_suffix}.json"

    llama = Llama(wf.model, wf.tokenizer)
    existing = []
    if resume and out_path.exists():
        try:
            with open(out_path) as f:
                existing = json.load(f)
        except Exception:
            existing = []
    start_idx = len(existing)
    wrapped = existing.copy()

    for i, prob in enumerate(tqdm(problems, desc=f"Direct baseline test ({s}-{e} of {total})")):
        if i < start_idx:
            continue
        prompt = f"Problem: {prob['problem']}\nSolution:"
        tokens = wf.tokenizer.encode(prompt, bos=True, eos=False)
        gen, _ = llama.generate([tokens], max_gen_len=512, temperature=0.7, top_p=0.9)
        text = wf.tokenizer.decode(gen[0])
        wrapped.append({'inputs': prob, 'outputs': {'final_text': text}})
        with open(out_path, 'w') as f:
            json.dump(wrapped, f)

    # Evaluate shard
    all_correct = eval_solutions(llama=llama, solutions=[x['outputs']['final_text'] for x in wrapped], problems=[x['inputs'] for x in wrapped])
    acc = sum(all_correct)/len(all_correct) if all_correct else 0.0
    shard_result = out_root / f"results_direct_test{shard_suffix}.json"
    with open(shard_result, 'w') as f:
        json.dump({'condition': 'direct', 'accuracy': acc, 'correct': int(sum(all_correct)), 'total': len(all_correct), 'all_correct': all_correct}, f)
    print(f"Shard done: {shard_result}")


def merge(num_shards: int, *, output_dir: str | None = None):
    out_root = Path(output_dir) if output_dir else Path(config.DATA_ROOT)
    all_correct = []
    for i in range(num_shards):
        p = out_root / f"results_direct_test_shard{i}-of-{num_shards}.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing shard: {p}")
        with open(p) as f:
            j = json.load(f)
        all_correct.extend(j['all_correct'])
    acc = sum(all_correct)/len(all_correct) if all_correct else 0.0
    merged = {'condition': 'direct', 'accuracy': acc, 'correct': int(sum(all_correct)), 'total': len(all_correct), 'all_correct': all_correct}
    out_path = out_root / f"results_direct_test.json"
    with open(out_path, 'w') as f:
        json.dump(merged, f)
    print(f"Wrote merged results: {out_path} (acc={acc:.3f})")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Direct baseline (raw model) with shardable test inference')
    sub = ap.add_subparsers(dest='cmd', required=True)

    c = sub.add_parser('collect')
    c.add_argument('--shard-index', type=int, default=None)
    c.add_argument('--num-shards', type=int, default=None)
    c.add_argument('--start', type=int, default=0)
    c.add_argument('--limit', type=int, default=None)
    c.add_argument('--resume', action='store_true')
    c.add_argument('--output-dir', type=str, default=None)

    m = sub.add_parser('merge')
    m.add_argument('--num-shards', type=int, required=True)
    m.add_argument('--output-dir', type=str, default=None)

    args = ap.parse_args()
    if args.cmd == 'collect':
        collect_direct_baseline(
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            start=args.start,
            limit=args.limit,
            resume=args.resume,
            output_dir=args.output_dir,
        )
    else:
        merge(args.num_shards, output_dir=args.output_dir)
