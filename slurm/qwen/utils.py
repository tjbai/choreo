import os
import json
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from llama import Workflow, Llama
from llama.util import find_free_port
from llama.workflows.tot import eval_solutions, load_math_problems

from . import config


def setup_env() -> None:
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(find_free_port())


def select_range(total: int, shard_index: Optional[int], num_shards: Optional[int], start: int, limit: Optional[int]) -> Tuple[int, int]:
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


def shard_suffix(s: int, e: int, total: int, shard_index: Optional[int], num_shards: Optional[int], limit: Optional[int]) -> str:
    if num_shards is not None and shard_index is not None:
        return f"_shard{shard_index}-of-{num_shards}"
    if s != 0 or (limit is not None):
        return f"_offset{s}-len{e - s}"
    return ""


def out_root(path: Optional[str]) -> Path:
    root = Path(path) if path else Path(config.DATA_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    return root


def load_problems(split: str) -> List[Dict]:
    target_split = 'train' if split == 'train' else 'test'
    problems = load_math_problems(config.MATH_DATA_PATH, split=target_split)
    limit = config.NUM_PROBLEMS_TRAIN if split == 'train' else config.NUM_PROBLEMS_TEST
    return problems[:limit]


def evaluate_wrapped_solutions(workflow_name: str, workflow_obj: Workflow, wrapped: List[Dict]) -> Dict:
    llama = Llama(workflow_obj.model, workflow_obj.tokenizer)
    llama.model.reshape_cache(4)

    problems = [x['inputs'] for x in wrapped]
    if workflow_name == 'tot':
        solutions = [
            (workflow_obj.tokenizer.decode(x['outputs']['final_tokens']) if x['outputs'].get('final_tokens') is not None else None)
            for x in wrapped
        ]
    elif workflow_name == 'mad':
        solutions = []
        for x in wrapped:
            decision = x['outputs'].get('decision')
            if isinstance(decision, dict) and decision.get('Answer'):
                solutions.append(decision['Answer'])
            else:
                solutions.append(None)
    elif workflow_name == 'madpar':
        solutions = []
        for x in wrapped:
            final_answers = x['outputs'].get('final_answers', [])
            ans = next((v for v in final_answers if v is not None), None)
            solutions.append(ans)
    else:
        raise ValueError(f"Unknown workflow: {workflow_name}")

    all_correct = eval_solutions(llama=llama, solutions=solutions, problems=problems)
    accuracy = sum(all_correct) / len(all_correct) if all_correct else 0.0
    return {
        'accuracy': accuracy,
        'correct': int(sum(all_correct)),
        'total': int(len(all_correct)),
        'all_correct': all_correct,
    }


def read_existing(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return []


def save_json(path: Path, obj) -> None:
    with open(path, 'w') as f:
        json.dump(obj, f)
