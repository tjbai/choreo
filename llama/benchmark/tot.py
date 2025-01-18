from torch.autograd.profiler import record_function
from llama.workflow import Workflow

import re
from collections import Counter

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

def format_vote_prompt(n):
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

def tot_cached(
    workflow: Workflow,
    problem: str,
    branching_factor: int,
    voters: int
):
    workflow.reset()

    with record_function("prompt_insert"):
        cot, vote, finish = workflow.insert([
            {
                'messages': [{'role': 'system', 'content': cot_prompt}, {'role': 'user', 'content': format_problem(problem)}],
                'parent_ids': []
            },
            {
                'messages': [{'role': 'system', 'content': format_vote_prompt(branching_factor)}, {'role': 'user', 'content': format_problem(problem)}],
                'parent_ids': []
            },
            {
                'messages': [{'role': 'system', 'content': finish_prompt}, {'role': 'user', 'content': format_problem(problem)}],
                'parent_ids': []
            },
        ])

    with record_function("cot_step"):
        proposal_tokens, proposal_nodes = workflow.step(
            [
                {
                    'header': ('assistant', f'solution number {i+1}'),
                    'prefill': None,
                    'parent_ids': [cot['id']],
                }
                for i in range(branching_factor)
            ],
            compact=False,
            max_gen_len=512,
            temperature=0.7,
            top_p=0.9,
            seed=42,
            debug=False,
        )

    with record_function("vote_step"):
        vote_tokens, vote_nodes = workflow.step(
            [
                {
                    'header': ('assistant', None),
                    'prefill': None,
                    'parent_ids': [vote['id']] + list([p['id'] for p in proposal_nodes]),
                }
                for _ in range(voters)
            ],
            compact=False,
            max_gen_len=256,
            temperature=0.7,
            top_p=0.9,
            seed=42,
            debug=False
        )

    res = None
    votes = [
        choice for resp in vote_tokens if
        (choice := parse_choice(workflow.tokenizer.decode(resp))) is not None
    ]

    if len(votes) > 0:
        best = Counter(votes).most_common(1)[0][0]
        with record_function("final_step"):
            [res], _ = workflow.step(
                [
                    {
                        'header': ('assistant', None),
                        'prefill': None,
                        'parent_ids': [finish['id']] + [proposal_nodes[best-1]['id']]
                    }
                ],
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
        'res': res,
        'votes': votes
    }

# TODO -- with baseline llama
def tot_baseline(
    workflow: Workflow,
    problem: str,
    branching_factor: int,
    voters: int
):
    pass
