from typing import Tuple
from operator import itemgetter as get

from llama import Workflow

STOP = ''

moderator_prompt = ''''''

def format_agent_prompt(agent_prompt: str, name: str) -> str:
    return ''

def mad_cached(
    workflow: Workflow,
    agents: Tuple[str, str],
    max_rounds: int,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 42,
):
    agent_contexts = [[a] for a in workflow.insert([
        {'messages': [{'role': 'system', 'content': format_agent_prompt(agent_prompt, name)}], 'parent_ids': []}
        for agent_prompt, name in agents
    ])]
    moderator_context = workflow.insert([{'messages': [{'role': 'system', 'content': moderator_prompt}], 'parent_ids': []}])
    for round in range(max_rounds):
        for (_, agent_name), context in zip(agents, agent_contexts):
            [response] = get('nodes')(workflow.step([{
                'header': ('assistant', f'agent_{agent_name}'),
                'prefill': f'Round {round+1}: ',
                'parent_ids': [n['id'] for n in context]
            }], temperature=temperature, top_p=top_p, seed=seed))

            for other_context in agent_contexts:
                other_context.append(response)
            moderator_context.append(response)

        [check] = workflow.insert([{
            'messages': [{'role': 'user', 'content': 'Should the debate continue?'}],
            'parent_ids': [n['id'] for n in moderator_context]
        }])

        [decision_tokens], [decision] = get('tokens', 'nodes')(workflow.step([{
            'header': ('assistant', 'moderator'),
            'prefill': 'Decision: ',
            'parent_ids': [n['id'] for n in moderator_context] + [check['id']]
        }], temperature=temperature, top_p=top_p, seed=seed))

        if STOP in workflow.tokenizer.decode(decision_tokens):
            break

    return {'agent_contexts': agent_contexts, 'moderator_context': moderator_context}

def mad_baseline(
    workflow: Workflow,
    agents: Tuple[str, str],
    max_rounds: int,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 42,
):
    agent_contexts = [[a] for a in workflow.insert([
        {'messages': [{'role': 'system', 'content': format_agent_prompt(agent_prompt, name)}], 'parent_ids': []}
        for agent_prompt, name in agents
    ])]
    stale_messages = [[] for _ in agent_contexts]
    moderator_context = workflow.insert([{'messages': [{'role': 'system', 'content': moderator_prompt}], 'parent_ids': []}])
    moderator_stale = []
    for round in range(max_rounds):
        for i, ((_, agent_name), context, stale) in enumerate(zip(agents, agent_contexts, stale_messages)):
            if len(stale) > 0:
                [new_messages] = workflow.insert([{'messages': stale, 'parent_ids': [n['id'] for n in context]}])
                context.append(new_messages)
                stale = []

            [response_tokens, response] = get('tokens', 'nodes')(workflow.step([{
                'header': ('assistant', f'agent_{agent_name}'),
                'prefill': f'Round {round+1}: ',
                'parent_ids': [n['id'] for n in context]
            }], temperature=temperature, top_p=top_p, seed=seed))
            context.append(response)

            new_message = {'role': f'agent_{agent_name}', 'content': f'Round {round+1}: {workflow.tokenizer.decode(response_tokens)}'}
            for j, other_stale in enumerate(stale_messages):
                if i == j: continue
                other_stale.append(new_message)
            moderator_stale.append(new_message)

        [new_messages] = workflow.insert([{'messages': moderator_stale, 'parent_ids': [n['id'] for n in moderator_context]}])
        moderator_context.append(new_messages)
        moderator_stale = []

        [check] = workflow.insert([{
            'messages': [{'role': 'user', 'content': 'Should the debate continue?'}],
            'parent_ids': [n['id'] for n in moderator_context]
        }])

        [decision_tokens], [decision] = get('tokens', 'nodes')(workflow.step([{
            'header': ('assistant', 'moderator'),
            'prefill': 'Decision: ',
            'parent_ids': [n['id'] for n in moderator_context] + [check['id']]
        }], temperature=temperature, top_p=top_p, seed=seed))

        if STOP in workflow.tokenizer.decode(decision_tokens):
            break

    return {'agent_contexts': agent_contexts, 'moderator_context': moderator_context}
