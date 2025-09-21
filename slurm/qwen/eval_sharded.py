#!/usr/bin/env python3

import os
import json
import argparse
from math import ceil
from pathlib import Path
from tqdm import tqdm

from llama import Workflow, Llama
from llama.util import find_free_port, load_ckpt
from llama.workflows.tot import eval_solutions, load_math_problems
from llama.workflows.mad import mad_cached
from llama.workflows.madpar import madpar_cached
from llama.workflows.tot import tot_cached

from . import config


def setup_env():
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(find_free_port())


def _select_range(
    total: int,
    shard_index: int | None,
    num_shards: int | None,
    start: int,
    limit: int | None,
):
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


def _get_cached_fn(workflow: str):
    if workflow == "tot":
        return tot_cached
    if workflow == "mad":
        return mad_cached
    if workflow == "madpar":
        return madpar_cached
    raise ValueError(f"Unknown workflow: {workflow}")


def _eval_cached_solutions(
    workflow_name: str, workflow_obj: Workflow, wrapped_solutions
):
    llama = Llama(workflow_obj.model, workflow_obj.tokenizer)
    llama.model.reshape_cache(4)
    problems = [x["inputs"] for x in wrapped_solutions]

    if workflow_name == "tot":
        solutions = [
            (
                workflow_obj.tokenizer.decode(x["outputs"]["final_tokens"])
                if x["outputs"].get("final_tokens") is not None
                else None
            )
            for x in wrapped_solutions
        ]
        all_correct = eval_solutions(
            llama=llama, solutions=solutions, problems=problems
        )
    elif workflow_name == "mad":
        solutions = []
        for x in wrapped_solutions:
            decision = x["outputs"].get("decision")
            if isinstance(decision, dict) and decision.get("Answer"):
                solutions.append(decision["Answer"])
            else:
                solutions.append(None)
        all_correct = eval_solutions(
            llama=llama, solutions=solutions, problems=problems
        )
    elif workflow_name == "madpar":
        solutions = []
        for x in wrapped_solutions:
            final_answers = x["outputs"].get("final_answers", [])
            ans = next((v for v in final_answers if v is not None), None)
            solutions.append(ans)
        all_correct = eval_solutions(
            llama=llama, solutions=solutions, problems=problems
        )
    else:
        raise ValueError(f"Unknown workflow: {workflow_name}")

    accuracy = sum(all_correct) / len(all_correct) if all_correct else 0.0
    return {
        "accuracy": accuracy,
        "correct": int(sum(all_correct)),
        "total": int(len(all_correct)),
        "all_correct": all_correct,
    }


def eval_choreo_ft(
    workflow: str,
    checkpoint_dir: str,
    *,
    shard_index: int | None = None,
    num_shards: int | None = None,
    start: int = 0,
    limit: int | None = None,
    resume: bool = False,
    output_dir: str | None = None,
):
    setup_env()

    wf = Workflow.build(
        ckpt_dir=config.QWEN_CKPT_DIR,
        tokenizer_path=config.QWEN_TOKENIZER_PATH,
        max_seq_len=config.MAX_SEQ_LEN if workflow != "mad" else config.MAX_SEQ_LEN * 8,
        max_batch_size=config.MAX_BATCH_SIZE,
        model_parallel_size=config.MODEL_PARALLEL_SIZE,
        max_nodes=20,
        use_lora=True,
        lora_rank=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
    )
    wf.model.eval()

    ckpts = [
        f
        for f in os.listdir(checkpoint_dir)
        if f.startswith("lora_step-") and f.endswith(".pt")
    ]
    if not ckpts:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    ckpts.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))
    checkpoint_path = os.path.join(checkpoint_dir, ckpts[-1])
    load_ckpt(wf, checkpoint_path)
    wf.model.eval()
    wf.model.reshape_cache(1)
    wf.model.set_adapter_state(enabled=True)

    out_root = Path(output_dir) if output_dir else Path(config.DATA_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    problems_all = load_math_problems(config.MATH_DATA_PATH, split="test")[
        : config.NUM_PROBLEMS_TEST
    ]
    total = len(problems_all)
    s, e = _select_range(total, shard_index, num_shards, start, limit)
    problems = problems_all[s:e]

    shard_suffix = (
        f"_shard{shard_index}-of-{num_shards}"
        if (num_shards is not None and shard_index is not None)
        else (
            f"_offset{s}-len{len(problems)}" if (s != 0 or (limit is not None)) else ""
        )
    )
    out_path = out_root / f"choreo_ft_{workflow}_test{shard_suffix}.json"

    cached_fn = _get_cached_fn(workflow)

    existing = []
    if resume and out_path.exists():
        try:
            with open(out_path) as f:
                existing = json.load(f)
        except Exception:
            existing = []
    start_idx = len(existing)
    wrapped = existing.copy()

    for i, prob in enumerate(
        tqdm(problems, desc=f"Eval choreo_ft {workflow} ({s}-{e} of {total})")
    ):
        if i < start_idx:
            continue
        wf.reset()
        try:
            if workflow == "tot":
                sol = cached_fn(
                    workflow=wf,
                    problem=prob["problem"],
                    branching_factor=config.BRANCHING_FACTOR,
                    voters=config.VOTERS,
                )
            elif workflow == "mad":
                sol = cached_fn(workflow=wf, problem=prob["problem"], max_rounds=3)
            else:
                sol = cached_fn(workflow=wf, problem=prob["problem"])
            wrapped.append({"inputs": prob, "outputs": sol})
            with open(out_path, "w") as f:
                json.dump(wrapped, f)
        except Exception as e:
            print(f"Skipping problem due to error: {e}")
            continue

    wf.model.set_adapter_state(enabled=False)
    eval_res = _eval_cached_solutions(workflow, wf, wrapped)
    shard_result = out_root / f"results_choreo_ft_{workflow}_test{shard_suffix}.json"
    with open(shard_result, "w") as f:
        json.dump(
            {
                "workflow": workflow,
                "condition": "choreo_ft",
                "checkpoint": checkpoint_path,
                **eval_res,
            },
            f,
        )
    print(f"Shard done: {shard_result}")


def eval_distilled(
    workflow: str,
    checkpoint_dir: str,
    *,
    shard_index: int | None = None,
    num_shards: int | None = None,
    start: int = 0,
    limit: int | None = None,
    resume: bool = False,
    output_dir: str | None = None,
):
    setup_env()

    wf = Workflow.build(
        ckpt_dir=config.QWEN_CKPT_DIR,
        tokenizer_path=config.QWEN_TOKENIZER_PATH,
        max_seq_len=config.MAX_SEQ_LEN if workflow != "mad" else config.MAX_SEQ_LEN * 8,
        max_batch_size=config.MAX_BATCH_SIZE,
        model_parallel_size=config.MODEL_PARALLEL_SIZE,
        max_nodes=4,
        use_lora=True,
        lora_rank=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
    )
    wf.model.eval()
    ckpts = [
        f
        for f in os.listdir(checkpoint_dir)
        if f.startswith("lora_step-") and f.endswith(".pt")
    ]
    if not ckpts:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    ckpts.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))
    checkpoint_path = os.path.join(checkpoint_dir, ckpts[-1])
    load_ckpt(wf, checkpoint_path)
    wf.model.eval()
    wf.model.set_adapter_state(enabled=True)

    out_root = Path(output_dir) if output_dir else Path(config.DATA_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    problems_all = load_math_problems(config.MATH_DATA_PATH, split="test")[
        : config.NUM_PROBLEMS_TEST
    ]
    total = len(problems_all)
    s, e = _select_range(total, shard_index, num_shards, start, limit)
    problems = problems_all[s:e]
    shard_suffix = (
        f"_shard{shard_index}-of-{num_shards}"
        if (num_shards is not None and shard_index is not None)
        else (
            f"_offset{s}-len{len(problems)}" if (s != 0 or (limit is not None)) else ""
        )
    )
    out_path = out_root / f"distilled_{workflow}_test{shard_suffix}.json"

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

    for i, prob in enumerate(
        tqdm(problems, desc=f"Eval distilled {workflow} ({s}-{e} of {total})")
    ):
        if i < start_idx:
            continue
        prompt = f"Problem: {prob['problem']}\nSolution:"
        tokens = wf.tokenizer.encode(prompt, bos=True, eos=False)
        gen, _ = llama.generate([tokens], max_gen_len=512, temperature=0.7, top_p=0.9)
        text = wf.tokenizer.decode(gen[0])
        wrapped.append({"inputs": prob, "outputs": {"final_text": text}})
        with open(out_path, "w") as f:
            json.dump(wrapped, f)

    # Evaluate shard
    solutions = [x["outputs"]["final_text"] for x in wrapped]
    all_correct = eval_solutions(
        llama=llama, solutions=solutions, problems=[x["inputs"] for x in wrapped]
    )
    acc = sum(all_correct) / len(all_correct) if all_correct else 0.0
    shard_result = out_root / f"results_distilled_{workflow}_test{shard_suffix}.json"
    with open(shard_result, "w") as f:
        json.dump(
            {
                "workflow": workflow,
                "condition": "distilled",
                "checkpoint": checkpoint_path,
                "accuracy": acc,
                "correct": int(sum(all_correct)),
                "total": len(all_correct),
                "all_correct": all_correct,
            },
            f,
        )
    print(f"Shard done: {shard_result}")


def merge(kind: str, workflow: str, num_shards: int, *, output_dir: str | None = None):
    out_root = Path(output_dir) if output_dir else Path(config.DATA_ROOT)
    prefix = "choreo_ft" if kind == "choreo_ft" else "distilled"
    all_correct = []
    for i in range(num_shards):
        p = out_root / f"results_{prefix}_{workflow}_test_shard{i}-of-{num_shards}.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing shard: {p}")
        with open(p) as f:
            j = json.load(f)
        all_correct.extend(j["all_correct"])
    acc = sum(all_correct) / len(all_correct) if all_correct else 0.0
    merged = {
        "workflow": workflow,
        "condition": prefix,
        "accuracy": acc,
        "correct": int(sum(all_correct)),
        "total": len(all_correct),
        "all_correct": all_correct,
    }
    out_path = out_root / f"results_{prefix}_{workflow}.json"
    with open(out_path, "w") as f:
        json.dump(merged, f)
    print(f"Wrote merged results: {out_path} (acc={acc:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ec = sub.add_parser("eval-choreo-ft")
    ec.add_argument("--workflow", choices=["tot", "mad", "madpar"], required=True)
    ec.add_argument("--checkpoint-dir", type=str, required=True)
    ec.add_argument("--shard-index", type=int, default=None)
    ec.add_argument("--num-shards", type=int, default=None)
    ec.add_argument("--start", type=int, default=0)
    ec.add_argument("--limit", type=int, default=None)
    ec.add_argument("--resume", action="store_true")
    ec.add_argument("--output-dir", type=str, default=None)

    ed = sub.add_parser("eval-distilled")
    ed.add_argument("--workflow", choices=["tot", "mad", "madpar"], required=True)
    ed.add_argument("--checkpoint-dir", type=str, required=True)
    ed.add_argument("--shard-index", type=int, default=None)
    ed.add_argument("--num-shards", type=int, default=None)
    ed.add_argument("--start", type=int, default=0)
    ed.add_argument("--limit", type=int, default=None)
    ed.add_argument("--resume", action="store_true")
    ed.add_argument("--output-dir", type=str, default=None)

    m = sub.add_parser("merge")
    m.add_argument("--kind", choices=["choreo_ft", "distilled"], required=True)
    m.add_argument("--workflow", choices=["tot", "mad", "madpar"], required=True)
    m.add_argument("--num-shards", type=int, required=True)
    m.add_argument("--output-dir", type=str, default=None)

    args = ap.parse_args()
    if args.cmd == "eval-choreo-ft":
        eval_choreo_ft(
            args.workflow,
            args.checkpoint_dir,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            start=args.start,
            limit=args.limit,
            resume=args.resume,
            output_dir=args.output_dir,
        )
    elif args.cmd == "eval-distilled":
        eval_distilled(
            args.workflow,
            args.checkpoint_dir,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            start=args.start,
            limit=args.limit,
            resume=args.resume,
            output_dir=args.output_dir,
        )
    else:
        merge(args.kind, args.workflow, args.num_shards, output_dir=args.output_dir)
