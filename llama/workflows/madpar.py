import re
from typing import Dict
from operator import itemgetter as get

from llama import Workflow

def mad_cached(
    workflow: Workflow,
    problem: str,
    num_agents: int = 3,
    num_rounds: int = 3,
    temperature: float = 0.7,
    top_p: float = 1.0,
    seed: int = 42,
    debug: bool = False,
) -> Dict:
    workflow.reset()
    result = {'debate_rounds': []}

    starting_prompt = f"""Can you solve the following math problem? {problem}
Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."""

    debate_prompt = f"""Using the solutions from other agents as additional information, can you provide your answer to the math problem?
The original math problem is {problem}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."""

    [agent_node, debate_node] = workflow.insert([{
        'messages': [{'role': 'user', 'content': starting_prompt}],
        'parent_ids': []
    }, {
        'messages': [{'role': 'user', 'content': debate_prompt}],
        'parent_ids': []
    }])

    initial_tokens, initial_nodes = get('tokens', 'nodes')(workflow.step([{
        'header': ('assistant', None),
        'prefill': f'From Agent {i+1}:\n',
        'parent_ids': [agent_node['id']]
        } for i in range(num_agents)
    ],
        temperature=temperature,
        top_p=top_p,
    ))
    result['debate_rounds'].append(initial_tokens)
    contexts = [[initial_node] for initial_node in initial_nodes]

    if debug:
        for i, tokens in enumerate(initial_tokens):
            print(f'Agent {i+1}:\n{workflow.tokenizer.decode(initial_tokens)}')

    last_round = initial_nodes
    for round_idx in range(num_rounds):
        update_tokens, update_nodes = get('tokens', 'nodes')(workflow.step([{
            'header': ('assistant', None),
            'prefill': f'From Agent {i+1}:\n',
            'parent_ids': [debate_node['id']] + [n['id'] for n in context + last_round]
            } for i, context in enumerate(contexts)
        ],
            temperature=0.7,
            top_p=1.0,
        ))
        for update, context in zip(update_nodes, contexts):
            context.append(update)

        if debug:
            for i, tokens in enumerate(update_tokens):
                print(f'Agent {i+1}:\n{workflow.tokenizer.decode(update_tokens)}')

        result['debate_rounds'].append(update_tokens)
        last_round = update_nodes

    final_answers = []
    for resp in result['debate_rounds'][-1]:
        resp = workflow.tokenizer.decode(resp)
        match = re.search(r'\\boxed{([^}]+)}', resp)
        if not match:
            match = re.search(r'\boxed{([^}]+)}', resp)
        if not match:
            match = re.search(r'boxed{([^}]+)}', resp)
        if not match:
            match = re.search(r'(?:answer is|answer:)\s*(\d+(?:\.\d+)?)', resp.lower())
        if match:
            answer = match.group(1).strip()
            final_answers.append(answer)
        else:
            final_answers.append(None)

    return result | {'final_answers': final_answers}

def mad_baseline(
    workflow: Workflow,
    problem: str,
    num_agents: int = 3,
    num_rounds: int = 2,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 42,
    debug: bool = False,
) -> Dict:
    workflow.reset()
    result = {'debate_rounds': []}

    starting_prompt = f"""Can you solve the following math problem? {problem}
Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."""

    [agent_node] = workflow.insert([{
        'messages': [{'role': 'user', 'content': starting_prompt}],
        'parent_ids': []
    }])

    initial_tokens, initial_nodes = get('tokens', 'nodes')(workflow.step([{
        'header': ('assistant', None),
        'prefill': f'From Agent {i+1}:\n',
        'parent_ids': [agent_node['id']]
        } for i in range(num_agents)
    ],
        temperature=temperature,
        top_p=top_p,
    ))
    result['debate_rounds'].append(initial_tokens)
    contexts = [[agent_node, initial_node] for initial_node in initial_nodes]

    if debug:
        for i, tokens in enumerate(initial_tokens):
            print(f'Agent {i+1}:\n{workflow.tokenizer.decode(initial_tokens)}')

    last_tokens = initial_tokens
    for round_idx in range(num_rounds):
        new_prompts = []
        for i in range(num_agents):
            other_responses = "\n\n".join([f"Agent {j+1}:\n{workflow.tokenizer.decode(resp)}" for j, resp in enumerate(last_tokens) if j != i])
            debate_prompt = f"""These are the solutions to the problem from other agents: {other_responses}
Using the solutions from other agents as additional information, can you provide your answer to the math problem? The original math problem is {problem}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."""
            new_prompts.append(debate_prompt)

        new_nodes = workflow.insert([{
            'messages': [{'role': 'user', 'content': new}],
            'parent_ids': [n['id'] for n in context]
        } for new, context in zip(new_prompts, contexts)])
        for new, context in zip(new_nodes, contexts):
            context.append(new)

        update_tokens, update_nodes = get('tokens', 'nodes')(workflow.step([{
            'header': ('assistant', None),
            'prefill': f'From Agent {i+1}:\n',
            'parent_ids': [n['id'] for n in context]
            } for i, context in enumerate(contexts)
        ],
            temperature=0.7,
            top_p=1.0,
        ))
        for update, context in zip(update_nodes, contexts):
            context.append(update)

        if debug:
            for i, tokens in enumerate(update_tokens):
                print(f'Agent {i+1}:\n{workflow.tokenizer.decode(update_tokens)}')

        result['debate_rounds'].append(update_tokens)
        last_tokens = update_tokens

    final_answers = []
    for resp in result['debate_rounds'][-1]:
        resp = workflow.tokenizer.decode(resp)
        match = re.search(r'\\boxed{([^}]+)}', resp)
        if not match:
            match = re.search(r'\boxed{([^}]+)}', resp)
        if not match:
            match = re.search(r'boxed{([^}]+)}', resp)
        if not match:
            match = re.search(r'(?:answer is|answer:)\s*(\d+(?:\.\d+)?)', resp.lower())

        if match:
            answer = match.group(1).strip()
            final_answers.append(answer)
        else:
            final_answers.append(None)

    return result | {'final_answers': final_answers}
