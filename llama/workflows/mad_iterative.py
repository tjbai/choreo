import csv
import json
from typing import List, Tuple, Optional, Dict
from operator import itemgetter as get

from llama import Workflow

# prompts adapted from https://github.com/Skytliang/Multi-Agents-Debate

def moderator_system_prompt(source_text: str) -> str:
   return f'''
You are a moderator. There will be two debaters involved in a translation debate competition.
They will present their translations and discuss their perspectives on the correct English translation of the given Chinese text: \"{source_text}\".
At the end of each round, you will evaluate the translation candidates based on the following criteria:
1. Accuracy: The degree to which the translation captures the original meaning of the source text.
2. Fluency: The readability and naturalness of the translation in English.'''

def moderator_user_prompt(round: int, max_rounds: int, agents: List[str]) -> str:
    is_final_round = (round + 1) == max_rounds
    agents_str = " or ".join(agents)

    if is_final_round:
        inner = (
            f"If so, please summarize your reasons for supporting {agents_str}'s side and give "
            "the final translation that you think is correct, and the debate will conclude. "
            "If not, the debate will continue to the next round."
        )
    else:
        inner = (
            "Because all rounds of debate have ended, you must decide on a "
            "final translation that you think is correct."
        )

    return (
        f"Now round {round + 1}/{max_rounds} of debate for both sides has ended. "
        "You, as the moderator, will evaluate both sides' translations "
        "and determine if there is a clear preference for a translation candidate. "
        f"\n{inner}\n"
        "Now please output your answer in JSON format, with the format as follows: "
        f'{{"Preference": "Yes or No", "Supported": "{agents_str}", "Translation": "", "Reason": ""}}. '
        "Please strictly output in JSON format, do not output irrelevant content."
    )

def agent_prompt(source_text: str, name: str) -> str:
    return f'''
You are a debater named {name}. Hello and welcome to the translation competition, which will be conducted in a debate format.
It's not necessary to fully agree with each other's perspectives, as our objective is to find the correct translation.
The debate topic is stated as follows:
What is the correct English translation of the following Chinese text: \"{source_text}\"'''

def load_translations(base_path: str, range: Tuple[int, int]) -> List[Dict]:
    categories = ['lexical', 'contextual', 'contextless']

    examples = []
    for category in categories:
        with open(f'{base_path}/{category}.csv') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                chinese, correct, wrong = row
                examples.append({
                    'category': category,
                    'chinese': chinese,
                    'correct': correct,
                    'wrong': wrong
                })

    import random
    random.seed(42)
    random.shuffle(examples)
    return examples[slice(*range)]

def parse_decision(_decision: str) -> Optional[str]:
    try:
        decision = json.loads(_decision)
        if decision.get('Preference', '').lower().strip() == 'yes' and (choice := decision.get('Supported')):
            return choice
    except:
        return None

def mad_cached(
    workflow: Workflow,
    source_text: str,
    agents: List[str],
    max_rounds: int,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 42,
    debug: bool = True,
):
    res = {}
    agent_contexts = [[a] for a in workflow.insert([
        {'messages': [{'role': 'system', 'content': agent_prompt(source_text, agent)}], 'parent_ids': []}
        for agent in agents
    ])]
    moderator_context = workflow.insert([{'messages': [{'role': 'system', 'content': moderator_system_prompt(source_text)}], 'parent_ids': []}])
    for round in range(max_rounds):
        for agent, context in zip(agents, agent_contexts):
            [response_tokens], [response] = get('tokens', 'nodes')(workflow.step([{
                'header': ('assistant', f'debater {agent}'),
                'prefill': f'Round {round+1}: ',
                'parent_ids': [n['id'] for n in context]
            }], temperature=temperature, top_p=top_p, seed=seed))

            if debug:
                print(workflow.tokenizer.decode(response_tokens))

            for other_context in agent_contexts:
                other_context.append(response)
            moderator_context.append(response)

        [check] = workflow.insert([{
            'messages': [{'role': 'user', 'content': moderator_user_prompt(round, max_rounds, agents)}],
            'parent_ids': [n['id'] for n in moderator_context]
        }])

        [decision_tokens], [decision] = get('tokens', 'nodes')(workflow.step([{
            'header': ('assistant', 'moderator'),
            'prefill': '{"Preference": ',
            'parent_ids': [n['id'] for n in moderator_context] + [check['id']]
        }], temperature=temperature, top_p=top_p, seed=seed))

        if debug:
            print(workflow.tokenizer.decode(decision_tokens))

        if (choice := parse_decision(workflow.tokenizer.decode(decision_tokens))) or (round + 1) == max_rounds:
            moderator_context.append(decision)
            res['choice'] = choice
            break

    return res | {'agent_contexts': agent_contexts, 'moderator_context': moderator_context}

def mad_baseline(
    workflow: Workflow,
    source_text: str,
    agents: List[str],
    max_rounds: int,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 42,
    debug: bool = True,
):
    res = {}
    agent_contexts = [[a] for a in workflow.insert([
        {'messages': [{'role': 'system', 'content': agent_prompt(source_text, agent)}], 'parent_ids': []}
        for agent in agents
    ])]
    stale_messages = [[] for _ in agent_contexts]
    moderator_context = workflow.insert([{'messages': [{'role': 'system', 'content': moderator_system_prompt(source_text)}], 'parent_ids': []}])
    moderator_stale = []
    for round in range(max_rounds):
        for i, (agent, context, stale) in enumerate(zip(agents, agent_contexts, stale_messages)):
            if len(stale) > 0:
                [new_messages] = workflow.insert([{'messages': stale, 'parent_ids': [n['id'] for n in context]}])
                context.append(new_messages)
                stale = []

            [response_tokens], [response] = get('tokens', 'nodes')(workflow.step([{
                'header': ('assistant', f'debater {agent}'),
                'prefill': f'Round {round+1}: ',
                'parent_ids': [n['id'] for n in context]
            }], temperature=temperature, top_p=top_p, seed=seed))
            context.append(response)

            if debug:
                print(workflow.tokenizer.decode(response_tokens))

            new_message = {'role': f'assistant:debater {agent}', 'content': workflow.tokenizer.decode(response_tokens)}
            for j, other_stale in enumerate(stale_messages):
                if i == j: continue
                other_stale.append(new_message)
            moderator_stale.append(new_message)

        [new_messages] = workflow.insert([{'messages': moderator_stale, 'parent_ids': [n['id'] for n in moderator_context]}])
        moderator_context.append(new_messages)
        moderator_stale = []

        [check] = workflow.insert([{
            'messages': [{'role': 'user', 'content': moderator_user_prompt(round, max_rounds, agents)}],
            'parent_ids': [n['id'] for n in moderator_context]
        }])

        [decision_tokens], [decision] = get('tokens', 'nodes')(workflow.step([{
            'header': ('assistant', 'moderator'),
            'prefill': '{"Preference": ',
            'parent_ids': [n['id'] for n in moderator_context] + [check['id']]
        }], temperature=temperature, top_p=top_p, seed=seed))

        if debug:
            print(workflow.tokenizer.decode(decision_tokens))

        if (choice := parse_decision(workflow.tokenizer.decode(decision_tokens))) or (round + 1) == max_rounds:
            moderator_context.append(decision)
            res['choice'] = choice
            break

    return res | {'agent_contexts': agent_contexts, 'moderator_context': moderator_context}
