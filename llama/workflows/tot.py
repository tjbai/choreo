import re
import json
import random
import hashlib
from pathlib import Path
from collections import Counter
from datetime import datetime
from typing import List, Optional, Required, TypedDict, Tuple, Dict
from operator import itemgetter as get

import torch
from tqdm import tqdm

from llama import Workflow, Llama, Dialog
from llama.workflows.benchmark import benchmark, BenchmarkResult

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

evaluator_prompt = '''
You are a strict evaluator for mathematics problems. You will assess:
1. Problem statement
2. Official solution and final answer
3. Student's attempted solution and final answer

Evaluation criteria:
- Final answers must be mathematically equivalent to the official solution
- All valid equivalent expressions are correct (e.g., 1/2 vs 0.5 vs 2^-1)

Output: Respond with ONLY "correct" or "incorrect" based on the final answer.
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

    # only returned by baseline
    proposal_content: List[str]
    vote_content: List[str]
    final_content: Optional[str]

    # trick experiment
    trick_indices: List[int]
    chose_trickster: Optional[bool]

    # serializing logits for distillation
    proposal_log_probs: List[Optional[List[float]]]
    vote_log_probs: List[Optional[List[float]]]
    final_log_probs: Optional[List[float]]

def tot_cached(
    workflow: Workflow,
    problem: str,
    branching_factor: int,
    voters: int,
    compact: bool = False,
    proposal_force: Optional[torch.Tensor] = None, # (branching_factor, N)
    voter_force: Optional[torch.Tensor] = None,    # (voters, N)
    final_force: Optional[torch.Tensor] = None,    # (1, N)
    hotswap: bool = False,
    temperature: float = 0.7,
    top_p: float = 1.0,
) -> TotResult:
    if hasattr(workflow.model, 'set_adapter_state') and hotswap:
        workflow.model.set_adapter_state(enabled=False)

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
    ])

    proposal_tokens, proposal_nodes = get('tokens', 'nodes')(workflow.step([
            {'header': ('assistant', None),
            'prefill': f'Solution #{i+1}:\n\n',
            'parent_ids': [cot['id']]}
            for i in range(branching_factor)
        ],
        teacher_force=proposal_force,
        compact=False,
        max_gen_len=512,
        temperature=temperature,
        top_p=top_p,
        seed=42,
        debug=False,
    ))

    if hasattr(workflow.model, 'set_adapter_state') and hotswap:
        workflow.model.set_adapter_state(enabled=True)

    vote_tokens, vote_nodes = get('tokens', 'nodes')(workflow.step([
            {'header': ('assistant', None),
            'prefill': 'BEST CHOICE: ' if voter_force is None else None,
            'parent_ids': [vote['id']] + [p['id'] for p in proposal_nodes]}
            for _ in range(voters)
        ],
        teacher_force=voter_force,
        stateless=False,
        compact=False,
        max_gen_len=256,
        temperature=temperature,
        top_p=top_p,
        seed=42,
        debug=False
    ))

    final_tokens = None
    votes = [
        choice for resp in vote_tokens if
        (choice := parse_choice(workflow.tokenizer.decode(resp))) is not None
    ]

    if len(votes) > 0:
        best = Counter(votes).most_common(1)[0][0]
        [final_tokens] = get('tokens')(workflow.step([
                {'header': ('assistant', None),
                'prefill': 'ANSWER: ',
                'parent_ids': [finish['id']] + [proposal_nodes[best-1]['id']]}
            ],
            teacher_force=final_force,
            stateless=False,
            compact=compact,
            max_gen_len=256,
            temperature=temperature,
            top_p=top_p,
            seed=42,
            debug=False
        ))

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
        {'messages': [
            {'role': 'system', 'content': cot_prompt},
            {'role': 'user', 'content': format_problem(problem)}
        ], 'parent_ids': []},
        {'messages': [
            {'role': 'system', 'content': trick_prompt},
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
    ])

    proposal_tokens, proposal_nodes = get('tokens', 'nodes')(workflow.step([
            {'header': ('assistant', None),
            'prefill': f'Solution #{i+1}:\n\n',
            'parent_ids': [trick['id'] if i in trick_indices else cot['id']]}
            for i in range(branching_factor)
        ],
        teacher_force=proposal_force,
        max_gen_len=512,
        temperature=0.7,
        top_p=0.9,
        seed=42,
    ))

    vote_tokens, vote_nodes = get('tokens', 'nodes')(workflow.step([
            {'header': ('assistant', None),
            'prefill': 'BEST CHOICE: ',
            'parent_ids': [vote['id']] + list([p['id'] for p in proposal_nodes])}
            for _ in range(voters)
        ],
        stateless=True,
        max_gen_len=256,
        temperature=0.7,
        top_p=0.9,
        seed=42,
    ))

    votes = [
        choice for resp in vote_tokens if
        (choice := parse_choice(workflow.tokenizer.decode(resp))) is not None
    ]

    final_tokens = None
    chose_trickster = None
    if len(votes) > 0:
        best = Counter(votes).most_common(1)[0][0]
        chose_trickster = (best - 1) in trick_indices
        [final_tokens] = get('tokens')(workflow.step([
                {'header': ('assistant', None),
                'prefill': None,
                'parent_ids': [finish['id']] + [proposal_nodes[best-1]['id']]}
            ],
            stateless=True,
            max_gen_len=256,
            temperature=0.7,
            top_p=0.9,
            seed=42,
        ))

    return {
        'proposal_tokens': proposal_tokens,
        'vote_tokens': vote_tokens,
        'final_tokens': final_tokens,
        'votes': votes,
        'trick_indices': trick_indices,
        'chose_trickster': chose_trickster
    }

def tot_baseline(
    workflow: Workflow,
    problem: str,
    branching_factor: int,
    voters: int,
    log_probs: bool = False,
    temperature: float = 0.7,
    top_p: float = 1.0,
) -> TotResult:
    [cot] = workflow.insert([
        {'messages': [
            {'role': 'system', 'content': cot_prompt},
            {'role': 'user', 'content': format_problem(problem)}
        ], 'parent_ids': []},
    ])
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
    ))

    vote_user_prompt = f'{format_problem(problem)}\n\nHere are the proposals:'
    for i, prop in enumerate(proposal_tokens):
        vote_user_prompt += f'\n\nSolution #{i+1}:\n{workflow.tokenizer.decode(prop)}'

    [vote] = workflow.insert([
        {'messages': [
            {'role': 'system', 'content': format_vote_system_prompt(branching_factor)},
            {'role': 'system', 'content': vote_user_prompt}
        ], 'parent_ids': []}
    ])
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
    ))
    votes = [
        choice for v in vote_tokens if
        (choice := parse_choice(workflow.tokenizer.decode(v))) is not None
    ]

    final_tokens = None
    if votes:
        best = Counter(votes).most_common(1)[0][0]
        best_proposal = workflow.tokenizer.decode(proposal_tokens[best - 1])
        [finish] = workflow.insert([
            {'messages': [
                {"role": "system", "content": finish_prompt},
                {"role": "user", "content": f"{format_problem(problem)}\n\nHere is the proposed approach: {best_proposal}"}],
            'parent_ids': []
        }])
        [final_tokens] = get('tokens')(workflow.step([
                {'header': ('assistant', None),
                'prefill': '',
                'parent_ids': [finish['id']]}
            ],
            max_gen_len=256,
            temperature=temperature,
            top_p=top_p
        ))

    return {
        'proposal_tokens': proposal_tokens,
        'vote_tokens': vote_tokens,
        'final_tokens': final_tokens,
        'votes': votes,
    }

def tot_baseline_shuffled(
    workflow: Workflow,
    problem: str,
    branching_factor: int,
    voters: int,
    seed: int = 42,
    normalize_votes: bool = True,
    debug: bool = False,
) -> TotResult:
    [cot] = workflow.insert([
        {'messages': [
            {'role': 'system', 'content': cot_prompt},
            {'role': 'user', 'content': format_problem(problem)}
        ], 'parent_ids': []},
    ])
    proposal_tokens, proposal_nodes = get('tokens', 'nodes')(workflow.step([
            {'header': ('assistant', None),
            'prefill': '',
            'parent_ids': [cot['id']]}
            for i in range(branching_factor)
        ],
        max_gen_len=512,
        temperature=0.7,
        top_p=0.9,
        seed=seed,
    ))

    voter_permutations = []
    vote_prompts = []

    for voter_idx in range(voters):
        voter_seed = seed + voter_idx
        rng = random.Random(voter_seed)
        permutation = list(range(branching_factor))
        rng.shuffle(permutation)
        voter_permutations.append(permutation)

        vote_user_prompt = f'{format_problem(problem)}\n\nHere are the proposals:'
        for i, orig_idx in enumerate(permutation):
            vote_user_prompt += f'\n\nSolution #{i+1}:\n{workflow.tokenizer.decode(proposal_tokens[orig_idx])}'

        vote_prompts.append({
            'messages': [
                {'role': 'system', 'content': format_vote_system_prompt(branching_factor)},
                {'role': 'system', 'content': vote_user_prompt}
            ],
            'parent_ids': []
        })

    vote_nodes = workflow.insert(vote_prompts)

    vote_tokens, vote_nodes = get('tokens', 'nodes')(workflow.step([
            {'header': ('assistant', None),
            'prefill': 'BEST CHOICE: ',
            'parent_ids': [node['id']]}
            for node in vote_nodes
        ],
        max_gen_len=256,
        temperature=0.7,
        top_p=0.9,
        seed=seed,
    ))

    shuffled_votes = [parse_choice(workflow.tokenizer.decode(tokens)) for tokens in vote_tokens]

    votes = []
    for i, vote in enumerate(shuffled_votes):
        if vote is not None:
            original_idx = voter_permutations[i][vote-1] + 1
            votes.append(original_idx)
        else:
            votes.append(None)

    if normalize_votes:
        normalized_vote_tokens = []
        for i, (vote, tokens) in enumerate(zip(shuffled_votes, vote_tokens)):
            if vote is not None:
                vote_text = workflow.tokenizer.decode(tokens)
                original_idx = voter_permutations[i][vote-1] + 1
                if debug: print('Mapping:', vote_text)
                new_text = re.sub(r'BEST CHOICE:\s*\d+', f'BEST CHOICE: {original_idx}', vote_text)
                if debug: print('To:', new_text)
                normalized_vote_tokens.append(workflow.tokenizer.encode(new_text, bos=False, eos=True))
            else:
                normalized_vote_tokens.append(tokens)
        vote_tokens = normalized_vote_tokens

    final_tokens = None
    if votes and any(v is not None for v in votes):
        valid_votes = [v for v in votes if v is not None]
        best = Counter(valid_votes).most_common(1)[0][0]
        best_proposal = workflow.tokenizer.decode(proposal_tokens[best - 1])
        [finish] = workflow.insert([
            {'messages': [
                {"role": "system", "content": finish_prompt},
                {"role": "user", "content": f"{format_problem(problem)}\n\nHere is the proposed approach: {best_proposal}"}],
            'parent_ids': []
        }])
        [final_tokens] = get('tokens')(workflow.step([
                {'header': ('assistant', None),
                'prefill': '',
                'parent_ids': [finish['id']]}
            ],
            max_gen_len=256,
            temperature=0.7,
            top_p=0.9,
            seed=seed,
        ))

    return {
        'proposal_tokens': proposal_tokens,
        'vote_tokens': vote_tokens,
        'final_tokens': final_tokens,
        'votes': votes,
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

    trial = baseline_results['outputs'][0] # should all be the same

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
    cache_dir: str = '/scratch4/jeisner1/tjbai/cache/tricky_tot'
) -> Dict:
    cache_key = hashlib.md5(f"{problem}_{branching_factor}_{voters}".encode()).hexdigest()
    cache_path = Path(cache_dir) / f'{cache_key}.pt'

    if cache_path.exists():
        cached = torch.load(cache_path, weights_only=True)
        trick_indices = cached['trick_indices']
        proposal_force = cached['proposal_force']
    else:
        trick_indices = random.sample(range(branching_factor), branching_factor // 2)
        llama.model.reshape_cache(max(branching_factor, voters))
        llama.model.set_adapter_state(enabled=False)
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

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'trick_indices': trick_indices, 'proposal_force': proposal_force}, cache_path)

    workflow.model.reshape_cache(1)
    workflow.model.set_adapter_state(enabled=True)
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
        'baseline_votes': baseline_result['votes'] if 'baseline_result' in locals() else None,
        'cached_votes': cached_result['votes'],
        'baseline': baseline_result['chose_trickster'] if 'baseline_result' in locals() else None,
        'cached': cached_result['chose_trickster'],
        'trick_indices': trick_indices
    }

def benchmark_solution_quality(
    llama: Llama,
    workflow: Workflow,
    problem: str,
    branching_factor: int,
    voters: int,
    compact: bool,
    cache_dir: str = '/scratch4/jeisner1/tjbai/cache/solution_quality'
) -> Optional[Dict]:
    cache_key = hashlib.md5(f"{problem}_{branching_factor}_{voters}_{compact}".encode()).hexdigest()
    cache_path = Path(cache_dir) / f'{cache_key}.pt'

    if cache_path.exists():
        cached = torch.load(cache_path, weights_only=True)
        proposal_force = cached['proposal_force']
        voter_force = cached['voter_force']
        baseline_proposals = cached['baseline_proposals']
        baseline_votes = cached['baseline_votes']
        baseline_final = cached['baseline_final']
    else:
        llama.model.reshape_cache(max(branching_factor, voters))
        llama.model.set_adapter_state(enabled=False)
        baseline_proposals = llama.chat_completion(
            [
                [{"role": "system", "content": cot_prompt}, {"role": "user", "content": format_problem(problem)}]
                for _ in range(branching_factor)
            ],
            max_gen_len=512,
            temperature=0.7,
            top_p=0.9,
            seed=42,
        )

        proposal_force = torch.full((branching_factor, 512), workflow.tokenizer.eot_id, device=workflow.device)
        for i, res in enumerate(baseline_proposals):
            tokens = res["tokens"]
            proposal_force[i, :len(tokens)] = torch.tensor(tokens, device=workflow.device)

        vote_user_prompt = f"{format_problem(problem)}\n\nProposals:\n" + "\n".join(
            f"Solution #{i+1}:\n{p['generation']['content']}"
            for i, p in enumerate(baseline_proposals)
        )

        baseline_votes = llama.chat_completion(
            [
                [
                    {"role": "system", "content": format_vote_system_prompt(branching_factor)},
                    {"role": "user", "content": vote_user_prompt}
                ]
                for _ in range(voters)
            ],
            max_gen_len=256,
            temperature=0.7,
            top_p=0.9,
            seed=42,
        )

        voter_force = torch.full((voters, 256), workflow.tokenizer.eot_id, device=workflow.device)
        for i, res in enumerate(baseline_votes):
            tokens = res["tokens"]
            voter_force[i, :len(tokens)] = torch.tensor(tokens, device=workflow.device)

        votes = [
            choice for resp in baseline_votes if
            (choice := parse_choice(resp["generation"]["content"])) is not None
        ]

        if not votes:
            return None

        best = Counter(votes).most_common(1)[0][0] - 1 if votes else 0

        best_proposal = baseline_proposals[best]["generation"]["content"]
        baseline_final_dialog: Dialog = [
            {"role": "system", "content": finish_prompt},
            {"role": "user", "content": f"{format_problem(problem)}\n\nHere is the proposed approach: {best_proposal}"},
        ]
        [baseline_final] = llama.chat_completion(
            dialogs=[baseline_final_dialog],
            temperature=0.7,
            top_p=0.9,
            max_gen_len=256,
        )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'proposal_force': proposal_force,
            'voter_force': voter_force,
            'baseline_proposals': baseline_proposals,
            'baseline_votes': baseline_votes,
            'baseline_final': baseline_final
        }, cache_path)

    workflow.model.reshape_cache(1)
    workflow.model.set_adapter_state(enabled=True)
    workflow.reset()
    cached_result = tot_cached(
        workflow=workflow,
        problem=problem,
        branching_factor=branching_factor,
        voters=voters,
        compact=compact,
        proposal_force=proposal_force,
        voter_force=voter_force,
        final_force=None # free generation
    )

    assert cached_result["final_tokens"] is not None

    return {
        "problem": problem,
        "proposals": [proposal["generation"]["content"] for proposal in baseline_proposals],
        "voters": [vote["generation"]["content"] for vote in baseline_votes],
        "baseline_final": baseline_final["generation"]['content'],
        "cached_final": workflow.tokenizer.decode(cached_result["final_tokens"])
    }

def strategy_prompt(input: str):
    return f'''
Given the following problem, propose a strategy for solving it.
Don't solve the problem yet - just describe your approach.

Problem: {input}
'''

def solution_prompt(input, strategy):
    return f'''
Solve the following problem using the provided strategy.
Your answer should end with 'The answer is \\boxed{{answer}}'

Problem: {input}
Strategy: {strategy}
'''

def vote_prompt(branching_factor):
    return f'''
You will be shown {branching_factor} different strategies for solving a problem. Vote on the best strategy.
Your BEST CHOICE should be an index 1 through {branching_factor}.'''

def tot_baseline_faithful(
    workflow: Workflow,
    problem: str,
    branching_factor: int = 5,
    voters: int = 5,
    seed: int = 42,
) -> Optional[Dict]:
    [strategy] = workflow.insert([
        {'messages': [
            {'role': 'user', 'content': strategy_prompt(problem)}
        ], 'parent_ids': []}
    ])

    strategy_tokens, strategy_nodes = get('tokens', 'nodes')(workflow.step([
            {'header': ('assistant', None),
            'prefill': f'Strategy #{i+1}:\n\n',
            'parent_ids': [strategy['id']]}
            for i in range(branching_factor)
        ],
        max_gen_len=512,
        temperature=0.7,
        top_p=0.9,
        seed=seed,
    ))

    strategy_vote_prompt = f'Problem: {problem}\n\nHere are some proposed strategies:'
    for i, strategy in enumerate(strategy_tokens):
        strategy_vote_prompt += f'\n\nStrategy #{i+1}:\n{workflow.tokenizer.decode(strategy)}'

    [strategy_vote_node] = workflow.insert([
        {'messages': [
            {'role': 'system', 'content': vote_prompt(branching_factor)},
            {'role': 'user', 'content': strategy_vote_prompt}
        ], 'parent_ids': []}
    ])

    strategy_vote_tokens = get('tokens')(workflow.step([
            {'header': ('assistant', None),
            'prefill': 'BEST CHOICE: ',
            'parent_ids': [strategy_vote_node['id']]}
            for _ in range(voters)
        ],
        max_gen_len=256,
        temperature=0.7,
        top_p=0.9,
        seed=seed,
    ))

    strategy_votes = [
        choice for v in strategy_vote_tokens if
        (choice := parse_choice(workflow.tokenizer.decode(v))) is not None
    ]

    if not strategy_votes:
        return None

    best_strategy_idx = Counter(strategy_votes).most_common(1)[0][0] - 1
    best_strategy = workflow.tokenizer.decode(strategy_tokens[best_strategy_idx])

    [solution_node] = workflow.insert([
        {'messages': [
            {'role': 'user', 'content': solution_prompt(problem, best_strategy)}
        ], 'parent_ids': []}
    ])

    solution_tokens, solution_nodes = get('tokens', 'nodes')(workflow.step([
            {'header': ('assistant', None),
            'prefill': f'Solution #{i+1}:\n\n',
            'parent_ids': [solution_node['id']]}
            for i in range(branching_factor)
        ],
        max_gen_len=512,
        temperature=0.7,
        top_p=0.9,
        seed=seed,
    ))

    solution_vote_prompt = f'Problem: {problem}\n\nStrategy: {best_strategy}\n\nHere are some proposed solutions:'
    for i, solution in enumerate(solution_tokens):
        solution_vote_prompt += f'\n\nSolution #{i+1}:\n{workflow.tokenizer.decode(solution)}'

    [solution_vote_node] = workflow.insert([
        {'messages': [
            {'role': 'system', 'content': f'You will be shown {branching_factor} different solutions to a problem. Vote on the best solution. Your BEST CHOICE should be an index 1 through {branching_factor}.'},
            {'role': 'user', 'content': solution_vote_prompt}
        ], 'parent_ids': []}
    ])

    solution_vote_tokens = get('tokens')(workflow.step([
            {'header': ('assistant', None),
            'prefill': 'BEST CHOICE: ',
            'parent_ids': [solution_vote_node['id']]}
            for _ in range(voters)
        ],
        max_gen_len=256,
        temperature=0.7,
        top_p=0.9,
        seed=seed,
    ))

    solution_votes = [
        choice for v in solution_vote_tokens if
        (choice := parse_choice(workflow.tokenizer.decode(v))) is not None
    ]

    final_tokens = None
    if solution_votes:
        best_solution_idx = Counter(solution_votes).most_common(1)[0][0] - 1
        final_tokens = solution_tokens[best_solution_idx]

    return {
        'strategy_tokens': strategy_tokens,
        'strategy_vote_tokens': strategy_vote_tokens,
        'solution_tokens': solution_tokens,
        'solution_vote_tokens': solution_vote_tokens,
        'final_tokens': final_tokens,
        'solution_votes': solution_votes,
        'strategy_votes': strategy_votes,
    }

def tot_cached_faithful(
    workflow: Workflow,
    problem: str,
    branching_factor: int = 5,
    voters: int = 5,
    seed: int = 42,
) -> Optional[Dict]:
    [strategy, strategy_vote, solution, solution_vote] = workflow.insert([
        {'messages': [
            {'role': 'user', 'content': strategy_prompt(problem)}
        ], 'parent_ids': []},
        {'messages': [
            {'role': 'user', 'content': vote_prompt(branching_factor)}
        ], 'parent_ids': []},
        {'messages': [
            {'role': 'user', 'content': f'''Solve the following problem using the provided strategy.
Your answer should end with \'The answer is \\boxed{{answer}}\'
\nProblem: {problem}'''}
        ], 'parent_ids': []},
        {'messages': [
            {'role': 'user', 'content': f'''You will be shown {branching_factor} different solutions to a problem. Vote on the best solution.
Your BEST CHOICE should be an index 1 through {branching_factor}.'''}
        ], 'parent_ids': []}
    ])

    strategy_tokens, strategy_nodes = get('tokens', 'nodes')(workflow.step([
            {'header': ('assistant', None),
            'prefill': f'Strategy #{i+1}:\n\n',
            'parent_ids': [strategy['id']]}
            for i in range(branching_factor)
        ],
        max_gen_len=512,
        temperature=0.7,
        top_p=0.9,
        seed=seed,
    ))

    strategy_vote_tokens = get('tokens')(workflow.step([
            {'header': ('assistant', None),
            'prefill': 'BEST CHOICE: ',
            'parent_ids': [strategy_vote['id']] + [n['id'] for n in strategy_nodes]}
            for _ in range(voters)
        ],
        max_gen_len=256,
        temperature=0.7,
        top_p=0.9,
        seed=seed,
    ))

    strategy_votes = [
        choice for v in strategy_vote_tokens if
        (choice := parse_choice(workflow.tokenizer.decode(v))) is not None
    ]

    if not strategy_votes:
        return None

    best_strategy_idx = Counter(strategy_votes).most_common(1)[0][0] - 1
    best_strategy = workflow.tokenizer.decode(strategy_tokens[best_strategy_idx])

    solution_tokens, solution_nodes = get('tokens', 'nodes')(workflow.step([
            {'header': ('assistant', None),
            'prefill': f'Solution #{i+1}:\n\n',
            'parent_ids': [solution['id'], strategy_nodes[best_strategy_idx]['id']]}
            for i in range(branching_factor)
        ],
        max_gen_len=512,
        temperature=0.7,
        top_p=0.9,
        seed=seed,
    ))

    solution_vote_tokens = get('tokens')(workflow.step([
            {'header': ('assistant', None),
            'prefill': 'BEST CHOICE: ',
            'parent_ids': [solution_vote['id']]}
            for _ in range(voters)
        ],
        max_gen_len=256,
        temperature=0.7,
        top_p=0.9,
        seed=seed,
    ))

    solution_votes = [
        choice for v in solution_vote_tokens if
        (choice := parse_choice(workflow.tokenizer.decode(v))) is not None
    ]

    final_tokens = None
    if solution_votes:
        best_solution_idx = Counter(solution_votes).most_common(1)[0][0] - 1
        final_tokens = solution_tokens[best_solution_idx]

    return {
        'strategy_tokens': strategy_tokens,
        'strategy_vote_tokens': strategy_vote_tokens,
        'solution_vote_tokens': solution_vote_tokens,
        'final_tokens': final_tokens,
        'solution_votes': solution_votes,
        'solution_tokens': solution_tokens,
    }

def load_math_problems(
    root_dir: str,
    split: str,
    problem_types: Optional[List[str]] = None,
):
    problems = []
    root = Path(root_dir) / ('test' if split == 'val' else split)

    if problem_types is None:
        problem_types = [d.name for d in root.iterdir() if d.is_dir()]

    for problem_type in problem_types:
        type_dir = root / problem_type
        if not type_dir.exists():
            print(f'Could not find {type_dir}')
            continue

        for prob_file in type_dir.glob("*.json"):
            if split == 'val' and int(prob_file.stem) >= 100:
                continue

            with open(prob_file) as f:
                problem = json.load(f)
                problems.append(problem)

    if split == 'test':
        random.seed(42)
        random.shuffle(problems)

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

def collect_samples(
    workflow: Workflow,
    save_dir: str,
    n_problems: int = 500,
    branching_factor: int = 8,
    voters: int = 4,
    temperature: float = 1.0,
    top_p: float = 1.0,
    seed: int = 42,
    problem_types: Optional[List[str]] = None,
    math_path: str = '../data/MATH',
    split: str = 'train',
) -> List[TotResult]:
    dir = Path(save_dir)
    dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        'timestamp': datetime.now().isoformat(),
        'math_path': str(math_path),
        'split': split,
        'problem_types': problem_types,
        'branching_factor': branching_factor,
        'voters': voters,
        'temperature': temperature,
        'top_p': top_p,
    }

    problems = load_math_problems(math_path, split, problem_types)
    problems = random.sample([p['problem'] for p in problems], n_problems)

    examples = []
    for i, problem in enumerate(tqdm(problems, desc="Problems")):
        tot_result = tot_baseline(
            workflow=workflow,
            problem=problem,
            branching_factor=branching_factor,
            voters=voters,
            log_probs=True,
        )

        example = {
            'problem_idx': i,
            'problem': problem,
            'result': tot_result
        }

        torch.save(example, dir / f"problem_{i}.pt")
        examples.append(example)

    with open(dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f)

    return examples

def eval_solutions(llama: Llama, solutions: List[str], problems: List[Dict]) -> List[bool]:
    results = []
    for soln, prob in tqdm(zip(solutions, problems), total=len(problems)):
        dialog = [{
            'role': 'system',
            'content': evaluator_prompt
        }, {
            'role': 'user',
            'content': f"PROBLEM:\n{prob['problem']}\n\nGROUND TRUTH:\n{prob['solution']}\n\nATTEMPT:\n{soln}"
        }]

        outputs = llama.chat_completion(
            [dialog] * 3,
            max_gen_len=256,
            temperature=0.25,
            top_p=0.9,
            seed=42
        )

        incorrect_votes = sum(1 for o in outputs if 'incorrect' in o['generation']['content'].lower())
        results.append(incorrect_votes <= 1)

    return results
