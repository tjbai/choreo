import re
from typing import Dict
from operator import itemgetter as get

from llama import Workflow

def parse_output(resp: str):
    match = re.search(r'\\boxed{([^}]+)}', resp)
    if not match:
        match = re.search(r'\boxed{([^}]+)}', resp)
    if not match:
        match = re.search(r'boxed{([^}]+)}', resp)
    if not match:
        match = re.search(r'(?:answer is|answer:)\s*(\d+(?:\.\d+)?)', resp.lower())
    return match.group(1).strip() if match else None

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
            print(f'\n\n{workflow.tokenizer.decode(tokens)}')

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
                print(f'\n\n{workflow.tokenizer.decode(tokens)}')

        result['debate_rounds'].append(update_tokens)
        last_round = update_nodes

    final_answers = [parse_output(workflow.tokenizer.decode(resp)) for resp in result['debate_rounds'][-1]]
    return result | {'final_answers': final_answers}

def mad_baseline(
    workflow: Workflow,
    problem: str,
    num_agents: int = 3,
    num_rounds: int = 2,
    temperature: float = 0.7,
    top_p: float = 1.0,
    seed: int = 42,
    debug: bool = False,
) -> Dict:
    workflow.reset()
    result = {'debate_rounds': []}

    starting_prompt = f"""Can you solve the following math problem? {problem}
Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."""

    summary_base_prompt = f"""Here are a list of opinions from different agents solving this math problem: "{problem}"

{{agent_responses}}

Write a summary of the different approaches, reasoning steps, and conclusions from each agent.
Highlight key insights, potential errors, and different solution strategies used."""

    debate_base_prompt = f"""Here is a summary of responses from other agents:

{{summary}}

Using this summary carefully as additional advice, can you provide an updated answer to the math problem?
The original math problem is: {problem}

Make sure to state your answer at the end of the response in the form \\boxed{{answer}}."""

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
            print(f'\n\n{workflow.tokenizer.decode(tokens)}')

    last_tokens = initial_tokens
    for round_idx in range(num_rounds):
        # summarize
        all_responses = "\n\n".join([f"Agent {j+1}:\n{workflow.tokenizer.decode(resp)}" for j, resp in enumerate(last_tokens)])
        [current_summary_node] = workflow.insert([{
            'messages': [{'role': 'user', 'content': summary_base_prompt.format(agent_responses=all_responses)}],
            'parent_ids': []
        }])
        [summary_tokens], [summary_result] = get('tokens', 'nodes')(workflow.step([{
            'header': ('assistant', 'summarizer'),
            'prefill': 'Summary of agent responses:\n',
            'parent_ids': [current_summary_node['id']]
        }],
            temperature=temperature,
            top_p=top_p,
            seed=seed+round_idx+1,
        ))

        summary_text = workflow.tokenizer.decode(summary_tokens)
        result['summaries'].append(summary_tokens)

        if debug:
            print(f'\n\nRound {round_idx+1} Summary:\n{summary_text}\n')

        debate_prompts = workflow.insert([{
            'messages': [{'role': 'user', 'content': debate_base_prompt.format(summary=summary_text)}],
            'parent_ids': [n['id'] for n in context]
        } for context in contexts])
        for prompt, context in zip(debate_prompts, contexts):
            context.append(prompt)

        # updated responses
        update_tokens, update_nodes = get('tokens', 'nodes')(workflow.step([{
            'header': ('assistant', None),
            'prefill': f'From Agent {i+1}:\n',
            'parent_ids': [n['id'] for n in context]
            } for i, context in enumerate(contexts)
        ],
            temperature=temperature,
            top_p=top_p,
            seed=seed+round_idx+1,
        ))
        for update, context in zip(update_nodes, contexts):
            context.append(update)

        if debug:
            for i, tokens in enumerate(update_tokens):
                print(f'\n\n{workflow.tokenizer.decode(tokens)}')

        result['debate_rounds'].append(update_tokens)
        last_tokens = update_tokens

    final_answers = [parse_output(workflow.tokenizer.decode(resp)) for resp in result['debate_rounds'][-1]]
    return result | {'final_answers': final_answers}
