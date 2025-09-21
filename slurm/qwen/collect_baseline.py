import os
import json
import argparse
from math import ceil
from pathlib import Path
from tqdm import tqdm
from typing import Union, Literal, List, Dict, Optional, Tuple

from llama import Workflow, Llama
from llama.workflows.tot import tot_baseline, eval_solutions, load_math_problems
from llama.workflows.mad import mad_baseline
from llama.workflows.madpar import madpar_baseline

from . import config


def setup_env():
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(find_free_port())


def _select_range(
    total: int,
    shard_index: Optional[int],
    num_shards: Optional[int],
    start: int,
    limit: Optional[int],
) -> Tuple[int, int]:
    """Compute [start,end) indices based on shard params and manual slicing.

    Priority:
      1) If num_shards and shard_index are provided, compute that shard slice.
      2) Otherwise, use start + limit.
    """
    if num_shards is not None and shard_index is not None:
        if not (0 <= shard_index < num_shards):
            raise ValueError(f"shard_index {shard_index} must be in [0, {num_shards})")
        shard_size = ceil(total / num_shards)
        s = shard_index * shard_size
        e = min(total, s + shard_size)
        return s, e

    s = max(0, start)
    if limit is None:
        e = total
    else:
        e = min(total, s + max(0, limit))
    return s, e


def _load_problems(workflow: str, split: str) -> List[Dict]:
    """Load problems for a given split, respecting config limits for default run."""
    target_split = "train" if split == "train" else "test"
    problems = load_math_problems(config.MATH_DATA_PATH, split=target_split)
    limit = config.NUM_PROBLEMS_TRAIN if split == "train" else config.NUM_PROBLEMS_TEST
    return problems[:limit]


def _evaluate_test(
    workflow_name: str,
    workflow_obj: Workflow,
    test_solutions: List[Dict],
    problems: List[Dict],
) -> Dict:
    llama = Llama(workflow_obj.model, workflow_obj.tokenizer)
    llama.model.reshape_cache(4)

    if workflow_name == "tot":
        solutions = [
            (
                workflow_obj.tokenizer.decode(s["outputs"]["final_tokens"])
                if s["outputs"].get("final_tokens") is not None
                else None
            )
            for s in test_solutions
        ]
        all_correct = eval_solutions(
            llama=llama, solutions=solutions, problems=problems
        )
    elif workflow_name == "mad":
        solutions = []
        for s in test_solutions:
            decision = s["outputs"].get("decision")
            if isinstance(decision, dict) and decision.get("Answer"):
                solutions.append(decision["Answer"])
            else:
                solutions.append(None)
        all_correct = eval_solutions(
            llama=llama, solutions=solutions, problems=problems
        )
    elif workflow_name == "madpar":
        solutions = []
        for s in test_solutions:
            final_answers = s["outputs"].get("final_answers", [])
            answer = next((ans for ans in final_answers if ans is not None), None)
            solutions.append(answer)
        all_correct = eval_solutions(
            llama=llama, solutions=solutions, problems=problems
        )
    else:
        raise ValueError(f"Unknown workflow {workflow_name}")

    accuracy = sum(all_correct) / len(all_correct) if all_correct else 0.0
    return {
        "accuracy": accuracy,
        "correct": int(sum(all_correct)),
        "total": int(len(all_correct)),
        "all_correct": all_correct,
    }


def collect_baseline_data(
    workflow: Union[
        Literal["tot"],
        Literal["mad"],
        Literal["madpar"],
    ],
    *,
    split: Literal["train", "test", "both"] = "both",
    shard_index: Optional[int] = None,
    num_shards: Optional[int] = None,
    start: int = 0,
    limit: Optional[int] = None,
    resume: bool = False,
    output_dir: Optional[str] = None,
):
    print(f"Collecting baseline data for {workflow} (split={split})")
    setup_env()
    workflow_obj = Workflow.build(
        ckpt_dir=config.QWEN_CKPT_DIR,
        tokenizer_path=config.QWEN_TOKENIZER_PATH,
        max_seq_len=config.MAX_SEQ_LEN if workflow != "mad" else config.MAX_SEQ_LEN * 8,
        max_batch_size=config.MAX_BATCH_SIZE,
        model_parallel_size=config.MODEL_PARALLEL_SIZE,
        max_nodes=100 if workflow == "tot" else 20,
        use_lora=False,
    )

    out_root = Path(output_dir) if output_dir else Path(config.DATA_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    def _run_split(_split: str):
        problems = _load_problems(workflow, _split)
        total = len(problems)
        s, e = _select_range(total, shard_index, num_shards, start, limit)
        problems_slice = problems[s:e]

        shard_suffix = (
            f"_shard{shard_index}-of-{num_shards}"
            if (num_shards is not None and shard_index is not None)
            else (
                f"_offset{s}-len{len(problems_slice)}"
                if (s != 0 or (limit is not None))
                else ""
            )
        )
        out_path = out_root / f"baseline_{workflow}_{_split}{shard_suffix}.json"

        existing: List[Dict] = []
        if resume and out_path.exists():
            try:
                with open(out_path) as f:
                    existing = json.load(f)
            except Exception:
                existing = []

        start_idx = len(existing)
        solutions: List[Dict] = existing.copy()

        desc = f"{_split.capitalize()} data ({s}-{e} of {total})"
        for i, problem in enumerate(tqdm(problems_slice, desc=desc)):
            if i < start_idx:
                continue
            workflow_obj.reset()
            if workflow == "tot":
                solution = tot_baseline(
                    workflow=workflow_obj,
                    problem=problem["problem"],
                    branching_factor=config.BRANCHING_FACTOR,
                    voters=config.VOTERS,
                )
            elif workflow == "mad":
                solution = mad_baseline(
                    workflow=workflow_obj,
                    problem=problem["problem"],
                    max_rounds=3,
                )
            elif workflow == "madpar":
                solution = madpar_baseline(
                    workflow=workflow_obj,
                    problem=problem["problem"],
                )

            solutions.append({"inputs": problem, "outputs": solution})

            # Incremental write to support resumption
            with open(out_path, "w") as f:
                json.dump(solutions, f)

        if _split == "test":
            eval_res = _evaluate_test(workflow, workflow_obj, solutions, problems_slice)
            print(
                f"Baseline {workflow} {_split} accuracy: {eval_res['accuracy']:.3f} ({eval_res['correct']}/{eval_res['total']})"
            )
            shard_result_path = (
                out_root / f"results_baseline_{workflow}_{_split}{shard_suffix}.json"
            )
            with open(shard_result_path, "w") as f:
                json.dump(
                    {
                        "workflow": workflow,
                        "condition": "baseline",
                        "shard": {
                            "start": s,
                            "end": e,
                            "shard_index": shard_index,
                            "num_shards": num_shards,
                        },
                        **eval_res,
                    },
                    f,
                )

    if split in ("train", "both"):
        print("Collecting training data...")
        _run_split("train")

    if split in ("test", "both"):
        print("Collecting test data...")
        _run_split("test")


def merge_shards(
    workflow: Literal["tot", "mad", "madpar"],
    split: Literal["train", "test"],
    num_shards: int,
    *,
    output_dir: Optional[str] = None,
) -> None:
    """Merge shard JSONs into a single file and (for test) compute overall accuracy."""
    out_root = Path(output_dir) if output_dir else Path(config.DATA_ROOT)
    shard_files = [
        out_root / f"baseline_{workflow}_{split}_shard{i}-of-{num_shards}.json"
        for i in range(num_shards)
    ]
    all_solutions: List[Dict] = []
    all_problems: List[Dict] = []
    for p in shard_files:
        if not p.exists():
            raise FileNotFoundError(f"Missing shard: {p}")
        with open(p) as f:
            shard = json.load(f)
        all_solutions.extend(shard)
        all_problems.extend([x["inputs"] for x in shard])

    merged_path = out_root / f"baseline_{workflow}_{split}.json"
    with open(merged_path, "w") as f:
        json.dump(all_solutions, f)
    print(f"Wrote merged file: {merged_path}")

    if split == "test":
        setup_env()
        workflow_obj = Workflow.build(
            ckpt_dir=config.QWEN_CKPT_DIR,
            tokenizer_path=config.QWEN_TOKENIZER_PATH,
            max_seq_len=config.MAX_SEQ_LEN
            if workflow != "mad"
            else config.MAX_SEQ_LEN * 8,
            max_batch_size=4,
            model_parallel_size=config.MODEL_PARALLEL_SIZE,
            max_nodes=4,
            use_lora=False,
        )
        eval_res = _evaluate_test(workflow, workflow_obj, all_solutions, all_problems)
        results_path = out_root / f"results_baseline_{workflow}.json"
        with open(results_path, "w") as f:
            json.dump(
                {
                    "workflow": workflow,
                    "condition": "baseline",
                    **eval_res,
                },
                f,
            )
        print(f"Wrote merged results: {results_path} (acc={eval_res['accuracy']:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect baseline data with optional sharding and merging"
    )
    subparsers = parser.add_subparsers(dest="cmd", required=False)

    # Root-level args for backward compatibility (no subcommand)
    parser.add_argument("--workflow", choices=["tot", "mad", "madpar"])
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)

    collect_p = subparsers.add_parser("collect", help="Collect baseline data (default)")
    collect_p.add_argument(
        "--workflow", choices=["tot", "mad", "madpar"], required=True
    )
    collect_p.add_argument("--split", choices=["train", "test", "both"], default="both")
    collect_p.add_argument("--shard-index", type=int, default=None)
    collect_p.add_argument("--num-shards", type=int, default=None)
    collect_p.add_argument("--start", type=int, default=0)
    collect_p.add_argument("--limit", type=int, default=None)
    collect_p.add_argument("--resume", action="store_true")
    collect_p.add_argument("--output-dir", type=str, default=None)

    merge_p = subparsers.add_parser(
        "merge", help="Merge shard files and evaluate (for test)"
    )
    merge_p.add_argument("--workflow", choices=["tot", "mad", "madpar"], required=True)
    merge_p.add_argument("--split", choices=["train", "test"], required=True)
    merge_p.add_argument("--num-shards", type=int, required=True)
    merge_p.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    if args.cmd in (None, "collect"):
        if args.workflow is None:
            collect_args = getattr(args, "workflow", None)
            if collect_args is None:
                parser.error("--workflow is required for collection")
        collect_baseline_data(
            args.workflow,
            split=getattr(args, "split", "both"),
            shard_index=getattr(args, "shard_index", None),
            num_shards=getattr(args, "num_shards", None),
            start=getattr(args, "start", 0),
            limit=getattr(args, "limit", None),
            resume=getattr(args, "resume", False),
            output_dir=getattr(args, "output_dir", None),
        )
    elif args.cmd == "merge":
        merge_shards(
            args.workflow,
            args.split,
            args.num_shards,
            output_dir=args.output_dir,
        )
