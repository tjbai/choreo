import csv
import json
from typing import List, Optional, Dict
from operator import itemgetter as get

from llama import Workflow

# prompts adapted from https://github.com/Skytliang/Multi-Agents-Debate

def moderator_system_prompt(source_text: str) -> str:
   return f'''
You are a moderator. There will be two debaters involved in a debate.
They will present their translations and discuss their perspectives on the correct English translation of the given Chinese text: \"{source_text}\".
At the end of each round, you will evaluate the translation candidates based on the following criteria:
1. Accuracy: The degree to which the translation captures the original meaning of the source text.
2. Fluency: The readability and naturalness of the translation in English.'''

def math_moderator_system_prompt(topic: str) -> str:
   return f'''
You are a moderator. There will be two debaters involved in a mathematical reasoning debate.
They will present their answers and discuss their perspectives on the following topic:\n\"{topic}\"'''

def moderator_user_prompt(round: int, max_rounds: int, agents: List[str]) -> str:
    is_final_round = (round + 1) == max_rounds
    agents_str = " or ".join(agents)

    if is_final_round:
        inner = (
            f"If so, please summarize your reasons for supporting {agents_str}'s side and give "
            "the final answer that you think is correct, and the debate will conclude. "
            "If not, the debate will continue to the next round."
        )
    else:
        inner = (
            "Because all rounds of debate have ended, you must decide on a "
            "final answer that you think is correct."
        )

    return (
        f"Now round {round + 1}/{max_rounds} of debate for both sides has ended. "
        "You, as the moderator, will evaluate both sides' responses to the debate topic "
        "and determine if there is a clear preference for the answer. "
        f"\n{inner}\n"
        "Now please output your answer in JSON format, with the format as follows: "
        f'{{"Preference": "Yes or No", "Supported": "{agents_str}", "Answer": "", "Reason": ""}}. '
        "Please strictly output in JSON format, do not output irrelevant content."
    )

def agent_prompt(source_text: str, name: str) -> str:
    return f'''
You are a debater named {name}. Hello and welcome to the translation competition, which will be conducted in a debate format.
It's not necessary to fully agree with each other's perspectives, as our objective is to find the correct translation.
The debate topic is stated as follows:
What is the correct English translation of the following Chinese text: \"{source_text}\"'''

def math_agent_prompt(problem: str, name: str) -> str:
    return f'''
You are a debater named {name}. Hello and welcome to the math competition, which will be conducted in a debate format.
It's not necessary to fully agree with each other's perspectives, as our objective is to find the correct solution.
The debate topic will be producing the correct answer to the following problem:\n\"{problem}\"'''

def load_translations(base_path: str, start: int, end: int) -> List[Dict]:
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
    return examples[start:end]

def load_ciar(base_path: str, start: int, end: int) -> List[Dict]:
    with open(f'{base_path}/CIAR.json') as f:
        return json.load(f)

def parse_decision(_decision: str) -> Optional[Dict]:
    try:
        decision = json.loads(_decision)
        if (
            decision.get('Preference', '').lower().strip() == 'yes'
            and decision.get('Supported')
            and decision.get('Answer')
        ):
            return decision
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

        if (decision := parse_decision(workflow.tokenizer.decode(decision_tokens))) or (round + 1) == max_rounds:
            res['decision'] = decision
            break

    return res | {'agent_contexts': agent_contexts, 'moderator_context': moderator_context}

def math_mad_cached(
    workflow: Workflow,
    problem: str,
    agents: List[str],
    max_rounds: int,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 42,
    debug: bool = True,
):
    res = {}
    agent_contexts = [[a] for a in workflow.insert([
        {'messages': [{'role': 'system', 'content': math_agent_prompt(problem, agent)}], 'parent_ids': []}
        for agent in agents
    ])]
    moderator_context = workflow.insert([{'messages': [{'role': 'system', 'content': math_moderator_system_prompt(problem)}], 'parent_ids': []}])
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

        if (decision := parse_decision(workflow.tokenizer.decode(decision_tokens))) or (round + 1) == max_rounds:
            res['decision'] = decision
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

        if (decision := parse_decision(workflow.tokenizer.decode(decision_tokens))) or (round + 1) == max_rounds:
            res['decision'] = decision
            break

    return res | {'agent_contexts': agent_contexts, 'moderator_context': moderator_context}

def math_mad_baseline(
    workflow: Workflow,
    problem: str,
    agents: List[str],
    max_rounds: int,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 42,
    debug: bool = True,
):
    res = {}
    agent_contexts = [[a] for a in workflow.insert([
        {'messages': [{'role': 'system', 'content': math_agent_prompt(problem, agent)}], 'parent_ids': []}
        for agent in agents
    ])]
    stale_messages = [[] for _ in agent_contexts]
    moderator_context = workflow.insert([{'messages': [{'role': 'system', 'content': math_moderator_system_prompt(problem)}], 'parent_ids': []}])
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

        if (decision := parse_decision(workflow.tokenizer.decode(decision_tokens))) or (round + 1) == max_rounds:
            res['decision'] = decision
            break

    return res | {'agent_contexts': agent_contexts, 'moderator_context': moderator_context}

def simple_baseline(
    workflow: Workflow,
    source_text: str,
    enable_reflection: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 42,
    debug: bool = True,
) -> Optional[Dict]:
    translation_prompt = (
        f'Translate the following Chinese text to English as accurately as possible:\n"{source_text}"\n\n'
        'Output your translation in JSON format: {"translation": "your translation here"}'
    )

    [sys] = workflow.insert([{'messages': [{'role': 'user', 'content': translation_prompt}], 'parent_ids': []}])
    [translation_tokens], [translation] = get('tokens', 'nodes')(workflow.step([{
        'header': ('assistant', 'translator'),
        'prefill': '{"translation": "',
        'parent_ids': [sys['id']]
    }], temperature=temperature, top_p=top_p, seed=seed))

    if debug:
        print(workflow.tokenizer.decode(translation_tokens))

    if not enable_reflection:
        try:
            return json.loads(workflow.tokenizer.decode(translation_tokens))
        except:
            return None

    reflection_prompt = (
        f'Review your translation of "{source_text}".\n'
        'Consider:\n'
        '1. Accuracy: The degree to which the translation captures the original meaning of the source text.'
        '2. Fluency: The readability and naturalness of the translation in English.'
        '\nProvide your translation, either updated or not, in the same JSON format: {"translation": "improved translation here"}'
    )

    [reflection] = workflow.insert([{
        'messages': [{'role': 'user', 'content': reflection_prompt}],
        'parent_ids': [sys['id'], translation['id']]
    }])

    [answer_tokens], [answer] = get('tokens', 'nodes')(workflow.step([{
        'header': ('assistant', 'translator'),
        'prefill': '{"translation": "',
        'parent_ids': [sys['id'], translation['id'], reflection['id']]
    }], temperature=temperature, top_p=top_p, seed=seed))

    if debug:
        print(workflow.tokenizer.decode(answer_tokens))

    try:
        return json.loads(workflow.tokenizer.decode(answer_tokens))
    except:
        return None

def math_simple_baseline(
    workflow: Workflow,
    problem: str,
    enable_reflection: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 42,
    debug: bool = True,
) -> Optional[Dict]:
    solve_prompt = (
        f'Solve the following math problem:\n{problem}"\n\n'
        'Output your answer in JSON format: {"Reasoning": "step-by-step walkthrough to the correct answer", "Answer": "final answer"}'
    )

    [sys] = workflow.insert([{'messages': [{'role': 'user', 'content': solve_prompt}], 'parent_ids': []}])
    [solve_tokens], [solve] = get('tokens', 'nodes')(workflow.step([{
        'header': ('assistant', 'solver'),
        'prefill': '{"Reasoning": "',
        'parent_ids': [sys['id']]
    }], temperature=temperature, top_p=top_p, seed=seed))

    if debug:
        print(workflow.tokenizer.decode(solve_tokens))

    if not enable_reflection:
        try:
            return json.loads(workflow.tokenizer.decode(solve_tokens))
        except:
            return None

    reflection_prompt = (
        'Review your solution to the problem and evaluate whether you may have made any reasoning mistakes.'
        '\nProvide your answer, either updated or not, in the same JSON format: {"Reasoning": "improved reasoning", "Answer": "new final answer"}'
    )

    [reflection] = workflow.insert([{
        'messages': [{'role': 'user', 'content': reflection_prompt}],
        'parent_ids': [sys['id'], solve['id']]
    }])

    [answer_tokens], [answer] = get('tokens', 'nodes')(workflow.step([{
        'header': ('assistant', 'solver'),
        'prefill': '{"Reasoning": "',
        'parent_ids': [sys['id'], solve['id'], reflection['id']]
    }], temperature=temperature, top_p=top_p, seed=seed))

    if debug:
        print(workflow.tokenizer.decode(answer_tokens))

    try:
        return json.loads(workflow.tokenizer.decode(answer_tokens))
    except:
        return None
