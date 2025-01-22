import re
from collections import Counter
from typing import List, Optional, Required, TypedDict, Tuple

import torch

from llama import Workflow, Llama, Dialog
from .benchmark import benchmark, BenchmarkResult

cot_prompt = '''
You are a creative problem solver with deep expertise in competition mathematics.
Your goal is to propose a clean and insightful approach to solving the provided problem.
Keep your proposal high-level and CONCISE. Do not attempt to solve the entire problem.

You must format your response as:
APPROACH:
1. (high-level first step)
2. (high-level second step)
'''

trick_prompt = '''
You are a trickster. When the user gives you a math problem, you will give a humorous and terrible solution.
Keep your response concise. Aim for sarcasm and humor.
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

def tot_baseline(
    llama: Llama,
    problem: str,
    branching_factor: int,
    voters: int,
) -> TotResult:
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

def benchmark_tot(
    llama: Llama,
    workflow: Workflow,
    problem: str,
    branching_factor: int,
    voters: int,
) -> Tuple[BenchmarkResult[TotResult], BenchmarkResult[TotResult]]:
    llama.model.reshape_cache(new_batch_size=8)
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

    workflow.reset()
    workflow.model.reshape_cache(new_batch_size=1)
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

if __name__ == '__main__':
    pass
