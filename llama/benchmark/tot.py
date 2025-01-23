import re
import json
import random
from pathlib import Path
from collections import Counter
from datetime import datetime
from typing import List, Optional, Required, TypedDict, Tuple, Dict

import torch
from tqdm import tqdm

from llama import Workflow, Llama, Dialog
from llama.benchmark.benchmark import benchmark, BenchmarkResult

random.seed(42)

cot_prompt = '''
You are a creative problem solver with deep expertise in competition mathematics.
Your goal is to propose a clean and insightful approach to solving the provided problem.
Keep your proposal high-level and CONCISE. Do not attempt to solve the entire problem.

You must format your response as:
APPROACH:
1. (high-level first step)
2. (high-level second step)
3. (so on and so forth)
'''

trick_prompt = '''
You are a confident but incorrect problem solver.
When given a math problem, propose a CONCISE approach that uses mathematical-sounding language and appears convincing at first glance, but obvious significant errors.
It should be trivial for an individual with domain knowledge to determine that your solution is bogus.

Do not directly state or make any mention of the fact that your solution is obviously incorrect.
Do not attempt to solve the entire problem.

You must format your response exactly as:
APPROACH:
1. (high-level first step)
2. (high-level second step)
3. (so on and so forth)
'''

finish_prompt = '''
You are a creative problem solver with deep expertise in competition mathematics.
You will be provided a problem and a proposed approach to solving it. Using the steps provided, finish solving the problem.
Keep your solution CONCISE and lean on ideas developed in the proposal.
You MUST provide the final answer to the problem and will be graded on correctness.

You must format your response as:
ANSWER: (2-3 sentence summary of solution and final answer)
'''

def format_vote_system_prompt(n):
    return f'''
    You are a rigorous mathematical evaluator with deep expertise in competition mathematics.
    You will be shown several different solution strategies for a math problem.
    Vote on the best proposal and give a concise justification of your choice.
    You will see {n} proposals, so your BEST CHOICE should be an index 1 through {n}.
    Do not attempt to solve the problem. You only need to evaluate.

    You must format your response as:
    BEST CHOICE: (index of best solution)
    RATIONALE: (1 sentence justification)
    '''

def format_problem(problem):
    return f'Here is the provided problem:\n{problem}'

def parse_choice(text):
    match = re.search(r'BEST CHOICE:\s*(\d+)', text)
    if match:
        return int(match.group(1))
    return None

class TotResult(TypedDict, total=False):
    proposal_tokens: Required[List[List[int]]]
    vote_tokens: Required[List[List[int]]]
    final_tokens: Required[Optional[List[int]]]
    votes: Required[List[int]]
    proposal_content: List[str]
    vote_content: List[str]
    final_content: Optional[str]
    trick_indices: List[int]
    chose_trickster: Optional[bool]

def tot_cached(
    workflow: Workflow,
    problem: str,
    branching_factor: int,
    voters: int,
    proposal_force: Optional[torch.Tensor] = None, # (branching_factor, N)
    voter_force: Optional[torch.Tensor] = None,    # (voters, N)
    final_force: Optional[torch.Tensor] = None,    # (1, N)
) -> TotResult:
    cot, vote, finish = workflow.insert([
        {
            'messages': [{'role': 'system', 'content': cot_prompt}, {'role': 'user', 'content': format_problem(problem)}],
            'parent_ids': []
        },
        {
            'messages': [{'role': 'system', 'content': format_vote_system_prompt(branching_factor)}, {'role': 'user', 'content': format_problem(problem)}],
            'parent_ids': []
        },
        {
            'messages': [{'role': 'system', 'content': finish_prompt}, {'role': 'user', 'content': format_problem(problem)}],
            'parent_ids': []
        },
    ])

    proposal_tokens, proposal_nodes = workflow.step(
        [
            {
                'header': ('assistant', None),
                'prefill': f'Solution #{i+1}:\n\n',
                'parent_ids': [cot['id']],
            }
            for i in range(branching_factor)
        ],
        teacher_force=proposal_force,
        compact=False,
        max_gen_len=512,
        temperature=0.7,
        top_p=0.9,
        seed=42,
        debug=False,
    )

    vote_tokens, vote_nodes = workflow.step(
        [
            {
                'header': ('assistant', None),
                'prefill': 'BEST CHOICE: ' if voter_force is None else None,
                'parent_ids': [vote['id']] + list([p['id'] for p in proposal_nodes]),
            }
            for _ in range(voters)
        ],
        teacher_force=voter_force,
        stateless=True,
        compact=False,
        max_gen_len=256,
        temperature=0.7,
        top_p=0.9,
        seed=42,
        debug=False
    )

    final_tokens = None
    votes = [
        choice for resp in vote_tokens if
        (choice := parse_choice(workflow.tokenizer.decode(resp))) is not None
    ]

    if len(votes) > 0:
        best = Counter(votes).most_common(1)[0][0]
        [final_tokens], _ = workflow.step(
            [
                {
                    'header': ('assistant', None),
                    'prefill': None,
                    'parent_ids': [finish['id']] + [proposal_nodes[best-1]['id']]
                }
            ],
            teacher_force=final_force,
            stateless=True,
            compact=False,
            max_gen_len=256,
            temperature=0.7,
            top_p=0.9,
            seed=42,
            debug=False
        )

    return {
        'proposal_tokens': proposal_tokens,
        'vote_tokens': vote_tokens,
        'final_tokens': final_tokens,
        'votes': votes
    }

def tricky_tot_cached(
    workflow: Workflow,
    problem: str,
    branching_factor: int,
    voters: int,
    trick_indices: List[int],
    proposal_force: Optional[torch.Tensor] = None,
) -> TotResult:
    cot, trick, vote, finish = workflow.insert([
        {
            'messages': [{'role': 'system', 'content': cot_prompt}, {'role': 'user', 'content': format_problem(problem)}],
            'parent_ids': []
        },
        {
            'messages': [{'role': 'system', 'content': trick_prompt}, {'role': 'user', 'content': format_problem(problem)}],
            'parent_ids': []
        },
        {
            'messages': [{'role': 'system', 'content': format_vote_system_prompt(branching_factor)}, {'role': 'user', 'content': format_problem(problem)}],
            'parent_ids': []
        },
        {
            'messages': [{'role': 'system', 'content': finish_prompt}, {'role': 'user', 'content': format_problem(problem)}],
            'parent_ids': []
        },
    ])

    proposal_tokens, proposal_nodes = workflow.step(
        [
            {
                'header': ('assistant', None),
                'prefill': f'Solution #{i+1}:\n\n',
                'parent_ids': [trick['id'] if i in trick_indices else cot['id']]
            }
            for i in range(branching_factor)
        ],
        teacher_force=proposal_force,
        max_gen_len=512,
        temperature=0.7,
        top_p=0.9,
        seed=42,
    )

    vote_tokens, vote_nodes = workflow.step(
        [
            {
                'header': ('assistant', None),
                'prefill': 'BEST CHOICE: ',
                'parent_ids': [vote['id']] + list([p['id'] for p in proposal_nodes]),
            }
            for _ in range(voters)
        ],
        stateless=True,
        max_gen_len=256,
        temperature=0.7,
        top_p=0.9,
        seed=42,
    )

    votes = [
        choice for resp in vote_tokens if
        (choice := parse_choice(workflow.tokenizer.decode(resp))) is not None
    ]

    final_tokens = None
    chose_trickster = None
    if len(votes) > 0:
        best = Counter(votes).most_common(1)[0][0]
        chose_trickster = (best - 1) in trick_indices
        [final_tokens], _ = workflow.step(
            [
                {
                    'header': ('assistant', None),
                    'prefill': None,
                    'parent_ids': [finish['id']] + [proposal_nodes[best-1]['id']]
                }
            ],
            stateless=True,
            max_gen_len=256,
            temperature=0.7,
            top_p=0.9,
            seed=42,
        )

    return {
        'proposal_tokens': proposal_tokens,
        'vote_tokens': vote_tokens,
        'final_tokens': final_tokens,
        'votes': votes,
        'trick_indices': trick_indices,
        'chose_trickster': chose_trickster
    }

def tot_baseline(llama: Llama, problem: str, branching_factor: int, voters: int) -> TotResult:
    proposal_dialogs: List[Dialog] = [
        [
            {"role": "system", "content": cot_prompt},
            {"role": "user", "content": format_problem(problem)},
        ]
        for _ in range(branching_factor)
    ]
    proposal_results = llama.chat_completion(
        dialogs=proposal_dialogs,
        max_gen_len=512,
        temperature=0.7,
        top_p=0.9,
        seed=42,
    )

    vote_user_prompt = f"{format_problem(problem)}\n\nHere are the proposals:"
    for i, pred in enumerate(proposal_results):
        vote_user_prompt += f"\n\nSolution #{i+1}:\n{pred["generation"]["content"]}"

    vote_dialogs: List[Dialog] = [
        [
            {"role": "system", "content": format_vote_system_prompt(branching_factor)},
            {"role": "system", "content": vote_user_prompt}
        ]
        for _ in range(voters)
    ]
    vote_results = llama.chat_completion(
        dialogs=vote_dialogs,
        max_gen_len=256,
        temperature=0.7,
        top_p=0.9,
        seed=42,
    )
    votes = [
        choice for resp in vote_results if
        (choice := parse_choice(resp["generation"]["content"])) is not None
    ]

    final_result = None
    if votes:
        best = Counter(votes).most_common(1)[0][0]
        best_proposal = proposal_results[best - 1]["generation"]["content"]
        final_dialog: Dialog = [
            {"role": "system", "content": finish_prompt},
            {"role": "user", "content": f"{format_problem(problem)}\n\nHere is the proposed approach: {best_proposal}"},
        ]
        [final_result] = llama.chat_completion(
            dialogs=[final_dialog],
            temperature=0.7,
            top_p=0.9,
            max_gen_len=256,
        )

    return {
        "proposal_content": [p["generation"]["content"] for p in proposal_results],
        "proposal_tokens": [p["tokens"] for p in proposal_results],
        "vote_content": [v["generation"]["content"] for v in vote_results],
        "vote_tokens": [v["tokens"] for v in vote_results],
        "final_content": final_result["generation"]["content"] if final_result else None,
        "final_tokens": final_result["tokens"] if final_result else None,
        "votes": votes,
    }

def tricky_tot_baseline(
    llama: Llama,
    problem: str,
    branching_factor: int,
    voters: int,
    trick_indices: List[int]
) -> TotResult:
    proposal_results = llama.chat_completion(
        [
            [
                {"role": "system", "content": trick_prompt if i in trick_indices else cot_prompt},
                {"role": "user", "content": format_problem(problem)},
            ]
            for i in range(branching_factor)
        ],
        max_gen_len=512,
        temperature=0.7,
        top_p=0.9,
        seed=42,
    )

    vote_user_prompt = f"{format_problem(problem)}\n\nHere are the proposals:"
    for i, pred in enumerate(proposal_results):
        vote_user_prompt += f"\n\nSolution #{i+1}:\n{pred["generation"]["content"]}"

    vote_dialogs: List[Dialog] = [
        [
            {"role": "system", "content": format_vote_system_prompt(branching_factor)},
            {"role": "system", "content": vote_user_prompt}
        ]
        for _ in range(voters)
    ]

    vote_results = llama.chat_completion(
        dialogs=vote_dialogs,
        max_gen_len=256,
        temperature=0.7,
        top_p=0.9,
        seed=42,
    )

    votes = [
        choice for resp in vote_results if
        (choice := parse_choice(resp["generation"]["content"])) is not None
    ]

    final_result = None
    chose_trickster = None
    if votes:
        best = Counter(votes).most_common(1)[0][0]
        chose_trickster = (best - 1) in trick_indices
        best_proposal = proposal_results[best - 1]["generation"]["content"]
        final_dialog: Dialog = [
            {"role": "system", "content": finish_prompt},
            {"role": "user", "content": f"{format_problem(problem)}\n\nHere is the proposed approach: {best_proposal}"},
        ]
        [final_result] = llama.chat_completion(
            dialogs=[final_dialog],
            temperature=0.7,
            top_p=0.9,
            max_gen_len=256,
        )

    return {
        "proposal_content": [p["generation"]["content"] for p in proposal_results],
        "proposal_tokens": [p["tokens"] for p in proposal_results],
        "vote_content": [v["generation"]["content"] for v in vote_results],
        "vote_tokens": [v["tokens"] for v in vote_results],
        "final_content": final_result["generation"]["content"] if final_result else None,
        "final_tokens": final_result["tokens"] if final_result else None,
        "votes": votes,
        "trick_indices": trick_indices,
        "chose_trickster": chose_trickster
    }

def benchmark_tot(
    llama: Llama,
    workflow: Workflow,
    problem: str,
    branching_factor: int,
    voters: int,
) -> Tuple[BenchmarkResult[TotResult], BenchmarkResult[TotResult]]:
    llama.model.reshape_cache(new_batch_size=max(branching_factor, voters))
    print(f'Reshaped cache to {llama.model.params.max_batch_size} X {llama.model.params.max_seq_len}')
    [baseline_results] = benchmark(tot_baseline, [{
            'llama': llama,
            'problem': problem,
            'branching_factor': branching_factor,
            'voters': voters
        }]
    )

    # grab the first trial, they should all be the same
    trial = baseline_results['outputs'][0]

    # TODO -- these sizes should be args, not hardcoded: 512, 256, 256
    proposal_force = torch.full((branching_factor, 512), workflow.tokenizer.eot_id, device=workflow.device)
    for i, tokens in enumerate(trial['proposal_tokens']):
        proposal_force[i, :len(tokens)] = torch.tensor(tokens, device=workflow.device)

    voter_force = torch.full((voters, 256), workflow.tokenizer.eot_id, device=workflow.device)
    for i, tokens in enumerate(trial['vote_tokens']):
        voter_force[i, :len(tokens)] = torch.tensor(tokens, device=workflow.device)

    if (tokens := trial['final_tokens']) is None:
        final_force = None
    else:
        final_force = torch.full((1, 256), workflow.tokenizer.eot_id, device=workflow.device)
        final_force[0, :len(tokens)] = torch.tensor(tokens, device=workflow.device)

    workflow.model.reshape_cache(new_batch_size=1)
    print(f'Reshaped cache to {workflow.model.params.max_batch_size} X {workflow.model.params.max_seq_len}')
    workflow.reset()
    [cached_results] = benchmark(tot_cached, [{
            'workflow': workflow,
            'problem': problem,
            'branching_factor': branching_factor,
            'voters': voters,
            'proposal_force': proposal_force,
            'voter_force': voter_force,
            'final_force': final_force
        }]
    )

    return baseline_results, cached_results

def benchmark_tricky_tot(
    llama: Llama,
    workflow: Workflow,
    problem: str,
    branching_factor: int,
    voters: int,
) -> Dict:
    trick_indices = random.sample(range(branching_factor), branching_factor // 2)
    llama.model.reshape_cache(max(branching_factor, voters))
    baseline_result = tricky_tot_baseline(
        llama=llama,
        problem=problem,
        branching_factor=branching_factor,
        voters=voters,
        trick_indices=trick_indices
    )

    proposal_force = torch.full((branching_factor, 512), workflow.tokenizer.eot_id, device=workflow.device)
    for i, tokens in enumerate(baseline_result['proposal_tokens']):
        proposal_force[i, :len(tokens)] = torch.tensor(tokens, device=workflow.device)

    workflow.model.reshape_cache(1)
    workflow.reset()
    cached_result = tricky_tot_cached(
        workflow=workflow,
        problem=problem,
        branching_factor=branching_factor,
        voters=voters,
        trick_indices=trick_indices,
        proposal_force=proposal_force,
    )

    return {
        'baseline': baseline_result['chose_trickster'],
        'cached': cached_result['chose_trickster'],
        'trick_indices': trick_indices
    }

def load_math_problems(root_dir, split, problem_types):
    problems = []
    root = Path(root_dir) / split

    for problem_type in problem_types:
        type_dir = root / problem_type
        if not type_dir.exists():
            print(f'Could not find {type_dir}')
            continue

        for prob_file in type_dir.glob("*.json"):
            with open(prob_file) as f:
                problem = json.load(f)
                problems.append(problem)

    return problems

def sweep_tot(
    llama,
    workflow,
    math_path,
    branch_sizes=[4, 8, 16],
    voter_sizes=[2, 4, 8],
    n_problems=5,
    split='train',
    problem_types=['counting_and_probability'],
    save_path='benchmark_tot.json'
):
    problems = load_math_problems(math_path, split, problem_types)
    problems = problems[:n_problems]

    data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'math_path': str(math_path),
            'split': split,
            'problem_types': problem_types,
            'branch_sizes': branch_sizes,
            'voter_sizes': voter_sizes,
            'n_problems': len(problems)
        },
        'runs': []
    }

    for i, problem in enumerate(tqdm(problems, desc="Problems")):
        # do the expensive ones first in case of a crash or OOM
        for branches in tqdm(sorted(branch_sizes, reverse=True), desc=f"Problem {i}: Branches", leave=False):
            for voters in sorted(voter_sizes, reverse=True):
                baseline, cached = benchmark_tot(
                    llama=llama,
                    workflow=workflow,
                    problem=problem['problem'],
                    branching_factor=branches,
                    voters=voters
                )

                run_data = {
                    'problem_idx': i,
                    'problem_type': problem.get('type', ''),
                    'branches': branches,
                    'voters': voters,
                    'baseline': baseline,
                    'cached': cached,
                }
                data['runs'].append(run_data)

                if save_path:
                    with open(save_path, 'w') as f:
                        json.dump(data, f, indent=2)

    return data
