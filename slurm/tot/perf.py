import os
import json
import time
from operator import itemgetter as get

import torch

from llama import Workflow
from llama.util import find_free_port
from llama.workflows.tot import (
    load_math_problems,
    cot_prompt,
    format_problem,
    format_vote_system_prompt,
    finish_prompt
)

def baseline(
    workflow,
    problem,
    branching_factor=8,
    voters=4,
    temperature=0.7,
    top_p=1.0,
):
    workflow.reset()
    insert_time = []
    step_time = []
    ttft_time = []
    total_ttft = 0
    total_tokens = 0
    force_tokens = []
    s = time.time()

    [cot] = workflow.insert([
        {'messages': [
            {'role': 'system', 'content': cot_prompt},
            {'role': 'user', 'content': format_problem(problem)}
        ], 'parent_ids': []},
    ], time_buffer=insert_time)
    proposal_tokens, proposal_nodes = get('tokens', 'nodes')(workflow.step([
            {'header': ('assistant', None),
            'prefill': '',
            'parent_ids': [cot['id']]}
            for i in range(branching_factor)
        ],
        compact=False,
        max_gen_len=512,
        temperature=temperature,
        top_p=top_p,
        seed=42,
        time_buffer=step_time,
        ttft_buffer=ttft_time,
    ))
    total_ttft += insert_time[-1] + ttft_time[-1]
    total_tokens += sum(len(a) for a in proposal_tokens)
    force_tokens.append(max(len(p) for p in proposal_tokens))

    vote_user_prompt = f'{format_problem(problem)}\n\nHere are the proposals:'
    for i, prop in enumerate(proposal_tokens):
        vote_user_prompt += f'\n\nSolution #{i+1}:\n{workflow.tokenizer.decode(prop)}'
    [vote] = workflow.insert([
        {'messages': [
            {'role': 'system', 'content': format_vote_system_prompt(branching_factor)},
            {'role': 'system', 'content': vote_user_prompt}
        ], 'parent_ids': []}
    ], time_buffer=insert_time)
    vote_tokens = get('tokens')(workflow.step([
            {'header': ('assistant', None),
            'prefill': 'BEST CHOICE: ',
            'parent_ids': [vote['id']]}
            for _ in range(voters)
        ],
        compact=False,
        max_gen_len=256,
        temperature=temperature,
        top_p=top_p,
        seed=42,
        time_buffer=step_time,
        ttft_buffer=ttft_time
    ))
    total_ttft += insert_time[-1] + ttft_time[-1]
    total_tokens += sum(len(a) for a in vote_tokens)
    force_tokens.append(max(len(p) for p in vote_tokens))

    # doesn't matter which is best, we should just simulate always
    best_proposal = workflow.tokenizer.decode(proposal_tokens[0])
    [finish] = workflow.insert([
        {'messages': [
            {"role": "system", "content": finish_prompt},
            {"role": "user", "content": f"{format_problem(problem)}\n\nHere is the proposed approach: {best_proposal}"}],
        'parent_ids': []
    }], time_buffer=insert_time)
    [final_tokens] = get('tokens')(workflow.step([
            {'header': ('assistant', None),
            'prefill': '',
            'parent_ids': [finish['id']]}
        ],
        max_gen_len=256,
        temperature=temperature,
        top_p=top_p,
        time_buffer=step_time,
        ttft_buffer=ttft_time
    ))
    total_ttft += insert_time[-1] + ttft_time[-1]
    total_tokens += len(final_tokens)
    force_tokens.append(len(final_tokens))

    return {
        'wall_time': time.time() - s,
        'cuda_time': sum(insert_time) + sum(step_time),
        'ttft': total_ttft,
        'tokens': total_tokens,
        'force_tokens': force_tokens
    }

def cached(
    workflow,
    problem,
    branching_factor=8,
    voters=4,
    temperature=0.7,
    top_p=1.0,
    force_tokens=[],
):
    workflow.reset()
    assert len(force_tokens) == 3

    insert_time = []
    step_time = []
    ttft_time = []
    total_ttft = 0
    s = time.time()

    cot, vote, finish = workflow.insert([
        {'messages': [
            {'role': 'system', 'content': cot_prompt},
            {'role': 'user', 'content': format_problem(problem)}
        ], 'parent_ids': []},
        {'messages': [
            {'role': 'system', 'content': format_vote_system_prompt(branching_factor)},
            {'role': 'user', 'content': format_problem(problem)}
        ], 'parent_ids': []},
        {'messages': [
            {'role': 'system', 'content': finish_prompt},
            {'role': 'user', 'content': format_problem(problem)}
        ], 'parent_ids': []},
    ], time_buffer=insert_time)

    proposal_tokens, proposal_nodes = get('tokens', 'nodes')(workflow.step([
            {'header': ('assistant', None),
            'prefill': f'Solution #{i+1}:\n\n',
            'parent_ids': [cot['id']]}
            for i in range(branching_factor)
        ],
        compact=False,
        max_gen_len=512,
        temperature=temperature,
        top_p=top_p,
        force_tokens=force_tokens.pop(0),
        time_buffer=step_time,
        ttft_buffer=ttft_time,
    ))
    total_ttft += insert_time[-1] + ttft_time[-1]

    vote_tokens, vote_nodes = get('tokens', 'nodes')(workflow.step([
            {'header': ('assistant', None),
            'prefill': 'BEST CHOICE: ',
            'parent_ids': [vote['id']] + [p['id'] for p in proposal_nodes]}
            for _ in range(voters)
        ],
        stateless=False,
        compact=False,
        max_gen_len=256,
        temperature=temperature,
        top_p=top_p,
        force_tokens=force_tokens.pop(0),
        time_buffer=step_time,
        ttft_buffer=ttft_time,
    ))
    total_ttft += ttft_time[-1]

    [final_tokens] = get('tokens')(workflow.step([
            {'header': ('assistant', None),
            'prefill': 'ANSWER: ',
            'parent_ids': [finish['id']] + [proposal_nodes[0]['id']]}
        ],
        stateless=False,
        max_gen_len=256,
        temperature=temperature,
        top_p=top_p,
        time_buffer=step_time,
        ttft_buffer=ttft_time,
        force_tokens=force_tokens.pop(0)
    ))
    total_ttft += ttft_time[-1]

    return {
        'wall_time': time.time() - s,
        'cuda_time': sum(insert_time) + sum(step_time),
        'ttft': total_ttft,
    }

def benchmark(*args, **kwargs):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline_res = baseline(*args, **kwargs)
    cached_res = cached(*args, **kwargs, force_tokens=baseline_res['force_tokens'])
    torch.cuda.empty_cache()
    return baseline_res, cached_res

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
    max_nodes=20,
    use_lora=True,
    lora_rank=64,
    lora_alpha=32,
    lora_dropout=0.05
)
workflow.model.eval()

problems = load_math_problems('/home/tbai4/llama3/data/MATH', split='val')[:30]

for branching_factor in [2, 4, 8, 16]:
    for voters in [2, 4, 8, 16]:
        try:
            # warmup
            for problem in problems[:3]:
                benchmark(workflow, problem)

            results = {'baseline': [], 'cached': []}
            for i, problem in enumerate(problems):
                baseline_res, cached_res = benchmark(workflow, problem)
                results['baseline'].append(baseline_res)
                results['cached'].append(cached_res)

                if (i+1) % 10 == 0:
                    with open(f'/home/tbai4/llama3/dumps/tot/perf_B-{branching_factor}_V-{voters}.json', 'w') as f:
                        json.dump(results, f)
        except RuntimeError as e:
            if 'oom' in str(e).lower():
                print('oom at B={branching_factor}, V={voters}')
                continue
            raise
