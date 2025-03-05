import re
import csv
import json
from typing import List, Optional, Dict
from operator import itemgetter as get

from llama import Workflow
from llama.tokenizer import Message

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
They will present their answers and discuss their perspectives on the following topic:\n\"{topic}\"
At the end of each round, you will evaluate answers and decide if there is enough information to choose a winner.
'''

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
        "and determine if there is enough information to choose a clear winner. "
        f"\n{inner}\n"
        "Now please output your answer in JSON format, with the format as follows: "
        f'{{"Reasoning": "", "Preference": "Yes or No", "Answer": ""}}. '
        "Please strictly output in JSON format, do not output irrelevant content."
    )

def agent_prompt(source_text: str, name: str) -> str:
    return f'''
You are a debater. Hello and welcome to the translation competition, which will be conducted in a debate format.
It's not necessary to fully agree with each other's perspectives, as our objective is to find the correct translation.
The debate topic is stated as follows:
What is the correct English translation of the following Chinese text: \"{source_text}\"'''

def math_agent_prompt(problem: str, name: str) -> str:
    return f'''
You are a debater. Hello and welcome to the math competition, which will be conducted in a debate format.
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

def try_parse(json_str: str) -> str | Dict:
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    cleaned_str = json_str.strip()

    if cleaned_str.startswith('{') and '}' in cleaned_str:
        try:
            start = cleaned_str.find('{')
            end = cleaned_str.rfind('}') + 1
            json_candidate = cleaned_str[start:end]

            json_candidate = json_candidate.replace('\\%', '%')
            json_candidate = json_candidate.replace('\\times', 'times')

            json_candidate = json_candidate.replace('\\"', '"')
            json_candidate = json_candidate.replace('\\', '')
            json_candidate = json_candidate.replace('\\"', '\\"')

            return json.loads(json_candidate)
        except json.JSONDecodeError:
            pass

    if cleaned_str.startswith('{') and cleaned_str.endswith('}'):
        try:
            return eval(cleaned_str, {"__builtins__": {}}, {})
        except:
            pass

    answer_idx = cleaned_str.find("Answer")
    if answer_idx != -1:
        substr = cleaned_str[answer_idx:]
        start = substr.find('"')
        if start != -1:
            end = substr.rfind('"') + 1
            if end > start:
                try:
                    return {"Answer": substr[start:end]}
                except json.JSONDecodeError:
                    pass

    answer_pattern = r'"Answer":\s*([0-9.]+)[^}]*}'
    matches = re.search(answer_pattern, cleaned_str)
    if matches:
        try:
            return {"Answer": matches.group(1)}
        except (ValueError, IndexError):
            pass

    return json_str

def parse_decision(_decision: str) -> Optional[Dict]:
    try:
        decision = try_parse(_decision)
        if (
            isinstance(decision, dict)
            and decision.get('Preference', '').lower().strip() == 'yes'
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
    top_p: float = 0.95,
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
    max_rounds: int,
    temperature: float = 0.6,
    top_p: float = 0.95,
    seed: int = 42,
    debug: bool = False,
) -> Dict:
    res = {}

    agent_contexts = [[a] for a in workflow.insert([
        {'messages': [{
            'role': 'system',
            'content': math_agent_prompt(problem, 'Affirmative') + '\nPresent a well-developed solution and defend against critiques.'}],
        'parent_ids': []},
        {'messages': [{
            'role': 'system',
            'content': math_agent_prompt(problem, 'Negative') + '\nCritique the opponent\'s response and propose your own strong solution.'}],
        'parent_ids': []},
    ])]

    moderator_context = workflow.insert([
        {'messages': [{
            'role': 'system',
            'content': math_moderator_system_prompt(problem)}],
        'parent_ids': []}
    ])

    [aff_tokens], [aff_response] = get('tokens', 'nodes')(workflow.step([{
        'header': ('assistant', ''),
        'prefill': 'Affirmative: ',
        'parent_ids': [agent_contexts[0][0]['id']]
    }], temperature=temperature, top_p=top_p, seed=seed, max_gen_len=1024))

    if debug:
        print("Affirmative initial solution:")
        print(workflow.tokenizer.decode(aff_tokens))

    agent_contexts[0].append(aff_response)
    agent_contexts[1].append(aff_response)
    moderator_context.append(aff_response)

    [neg_tokens], [neg_response] = get('tokens', 'nodes')(workflow.step([{
        'header': ('assistant', ''),
        'prefill': 'Negative: ',
        'parent_ids': [n['id'] for n in agent_contexts[1]]
    }], temperature=temperature, top_p=top_p, seed=seed, max_gen_len=1024))

    if debug:
        print("Negative critical evaluation:")
        print(workflow.tokenizer.decode(neg_tokens))

    agent_contexts[0].append(neg_response)
    agent_contexts[1].append(neg_response)
    moderator_context.append(neg_response)

    [check] = workflow.insert([{
        'messages': [{'role': 'user', 'content': moderator_user_prompt(0, max_rounds, ["Affirmative", "Negative"])}],
        'parent_ids': [n['id'] for n in moderator_context]
    }])

    for round in range(1, max_rounds+1):
        [aff_tokens], [aff_response] = get('tokens', 'nodes')(workflow.step([{
            'header': ('user', ''),
            'prefill': f'Affirmative response: ',
            'parent_ids': [n['id'] for n in agent_contexts[0]]
        }], temperature=temperature, top_p=top_p, seed=seed))

        if debug:
            print(f"Affirmative (Round {round}):")
            print(workflow.tokenizer.decode(aff_tokens))

        agent_contexts[0].append(aff_response)
        agent_contexts[1].append(aff_response)
        moderator_context.append(aff_response)

        [neg_tokens], [neg_response] = get('tokens', 'nodes')(workflow.step([{
            'header': ('user', ''),
            'prefill': f'Negative response: ',
            'parent_ids': [n['id'] for n in agent_contexts[1]]
        }], temperature=temperature, top_p=top_p, seed=seed))

        if debug:
            print(f"Negative (Round {round}):")
            print(workflow.tokenizer.decode(neg_tokens))

        agent_contexts[0].append(neg_response)
        agent_contexts[1].append(neg_response)
        moderator_context.append(neg_response)

        [check] = workflow.insert([{
            'messages': [{'role': 'user', 'content': moderator_user_prompt(round, max_rounds, ["affirmative", "negative"])}],
            'parent_ids': [n['id'] for n in moderator_context]
        }])

        [decision_tokens], [decision_response] = get('tokens', 'nodes')(workflow.step([{
            'header': ('assistant', 'moderator'),
            'prefill': '{"Reasoning": ',
            'parent_ids': [n['id'] for n in moderator_context] + [check['id']]
        }], temperature=temperature, top_p=top_p, seed=seed))

        moderator_context.append(decision_response)

        if debug:
            print(f"Moderator round {round} evaluation:")
            print(workflow.tokenizer.decode(decision_tokens))

        if (decision := parse_decision(workflow.tokenizer.decode(decision_tokens))) or round == max_rounds:
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

def faithful_mod_prompt(round: str, aff_ans: str, neg_ans: str):
    return f'''Now the {round} round of debate for both sides has ended.

Affirmative side arguing:
{aff_ans}

Negative side arguing:
{neg_ans}

You, as the moderator, will evaluate both sides' answers and determine if there is a clear preference for an answer candidate.
If so, please summarize your reasons for supporting affirmative/negative side and give the final answer that you think is correct, and the debate will conclude.
If not, the debate will continue to the next round. Now please output your answer in json format, with the format as follows:
{{"Preference": "Yes or No", "Supported Side": "Affirmative or Negative", "Reason": "", "Answer": ""}}.

Please strictly output in JSON format, do not output irrelevant content.'''

def math_baseline_faithful(
    workflow: Workflow,
    problem: str,
    max_rounds: int,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 42,
    debug: bool = False,
) -> Dict:
    res = {}
    dct = {1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth'}

    # initial aff
    aff_stale: List[Message] = []
    aff_context = workflow.insert([
        {'messages': [
            {'role': 'system', 'content': math_agent_prompt(problem, '')},
            {'role': 'user', 'content': problem},
        ], 'parent_ids': []},
    ])
    [aff_tokens] = get('tokens')(workflow.step([{'header': ('assistant', ''), 'prefill': '', 'parent_ids': [n['id'] for n in aff_context]}]))
    aff_ans = workflow.tokenizer.decode(aff_tokens)
    aff_stale.append({'role': 'assistant', 'content': aff_ans})
    if debug: print(f'Aff ans:\n{aff_ans}')

    # initial neg
    neg_stale: List[Message] = []
    neg_context = workflow.insert([
        {'messages': [
            {'role': 'system', 'content': math_agent_prompt(problem, '')},
            {'role': 'user', 'content': f'{aff_ans}\n\nYou disagree with my answer. Provide your answer and reasons.'},
        ], 'parent_ids': []},
    ])
    [neg_tokens] = get('tokens')(workflow.step([{'header': ('assistant', ''), 'prefill': '', 'parent_ids': [n['id'] for n in neg_context]}]))
    neg_ans = workflow.tokenizer.decode(neg_tokens)
    neg_stale.append({'role': 'assistant', 'content': neg_ans})
    if debug: print(f'Neg ans:\n{neg_ans}')

    # initial mod
    mod_stale: List[Message] = []
    mod_context = workflow.insert([{'messages': [
        {'role': 'system', 'content': faithful_mod_prompt('first', aff_ans, neg_ans)}
    ], 'parent_ids': []}])
    [mod_tokens] = get('tokens')(workflow.step([{'header': ('assistant', ''), 'prefill': '', 'parent_ids': [n['id'] for n in mod_context]}]))
    mod_stale.append({'role': 'assistant', 'content': workflow.tokenizer.decode(mod_tokens)})
    if debug: print(f'Mod review:\n{workflow.tokenizer.decode(mod_tokens)}')

    for round in range(max_rounds - 1):
        aff_stale.append({'role': 'user', 'content': f'{neg_ans}\n\nDo you agree with my perspective? Please provide your reasons and answer.'})
        aff_context.extend(workflow.insert([{'messages': aff_stale, 'parent_ids': []}]))
        [aff_tokens] = get('tokens')(workflow.step([{'header': ('assistant', ''), 'prefill': '', 'parent_ids': [n['id'] for n in aff_context]}]))
        aff_ans = workflow.tokenizer.decode(aff_tokens)
        aff_stale = [{'role': 'assistant', 'content': aff_ans}]
        if debug: print(f'Aff ans:\n{aff_ans}')

        neg_stale.append({'role': 'user', 'content': f'{aff_ans}\n\nDo you agree with my perspective? Please provide your reasons and answer.'})
        neg_context.extend(workflow.insert([{'messages': neg_stale, 'parent_ids': []}]))
        [neg_tokens] = get('tokens')(workflow.step([{'header': ('assistant', ''), 'prefill': '', 'parent_ids': [n['id'] for n in neg_context]}]))
        neg_ans = workflow.tokenizer.decode(neg_tokens)
        neg_stale = [{'role': 'assistant', 'content': neg_ans}]
        if debug: print(f'Neg ans:\n{neg_ans}')

        mod_stale.append({'role': 'user', 'content': faithful_mod_prompt(dct[round+2], aff_ans, neg_ans)})
        mod_context.extend(workflow.insert([{'messages': mod_stale, 'parent_ids': []}]))
        [mod_tokens] = get('tokens')(workflow.step([{'header': ('assistant', ''), 'prefill': '', 'parent_ids': [n['id'] for n in mod_context]}]))
        mod_ans = workflow.tokenizer.decode(mod_tokens)
        mod_stale = [{'role': 'assistant', 'content': mod_ans}]
        if debug: print(f'Mod review:\n{mod_ans}')

        if (decision := parse_decision(mod_ans)) is not None:
            res['decision'] = decision
            break

    if 'decision' not in res:
        # "ultimate deadly technique."
        # https://github.com/Skytliang/Multi-Agents-Debate/blob/022a7a8eecda85844d336e9064cc556edb0445b3/code/debate4tran.py#L236
        [judge_prompt] = workflow.insert([
            {'messages': [
                {'role': 'system', 'content': math_agent_prompt(problem, '')},
                {'role': 'user', 'content': f'Affirmative side arguing:\n{aff_ans}\n\nNegative side arguing:\n{neg_ans}\n\nNow, what answer candidates do we have? Present them without reasons.'},
            ], 'parent_ids': []}
        ])

        [cand_tokens], [cand_response] = get('tokens', 'nodes')(workflow.step([{'header': ('assistant', ''), 'prefill': '', 'parent_ids': [judge_prompt['id']]}]))

        [final_prompt] = workflow.insert([
            {'messages': [
                {'role': 'user', 'content': (
                    f'Therefore, {problem}\nPlease summarize your reasons and give the final answer that you think is correct. '
                    'Now please output your answer in json format, with the format as follows: {{"Reason": "", "debate_answer\": \"\"}}. '
                    'Please strictly output in JSON format, do not output irrelevant content.'
                )}
            ], 'parent_ids': [judge_prompt['id'], cand_response['id']]}
        ])

        [final_tokens] = get('tokens')(workflow.step([{'header': ('assistant', ''), 'prefill': '', 'parent_ids': [judge_prompt['id'], cand_response['id'], final_prompt['id']]}]))
        res['decision'] = parse_decision(workflow.tokenizer.decode(final_tokens))

    return res | {}

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
        return eval(workflow.tokenizer.decode(answer_tokens))
    except:
        return None
