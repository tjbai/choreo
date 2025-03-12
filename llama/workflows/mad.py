import re
import csv
import json
from typing import List, Dict
from operator import itemgetter as get

from llama import Workflow
from llama.tokenizer import Message

# prompts adapted from https://github.com/Skytliang/Multi-Agents-Debate

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

def moderator_system_prompt(topic: str) -> str:
   return f'''
You are a moderator. There will be two debaters involved in a mathematical reasoning debate.
They will present their answers and discuss their perspectives on the following topic:\n"{topic}"
At the end of each round, you will evaluate answers and decide if there is enough information to choose a winner.
'''

moderator_user_prompt = (
    "You, as the moderator, will now evaluate both sides' responses to the debate topic "
    "and determine if there is enough information to choose a clear winner. "
    "\nIf so, please summarize your reasons for supporting {agents_str}'s side and give "
    "the final answer that you think is correct, and the debate will conclude. "
    "If not, the debate will continue to the next round.\n"
    "Now please output your answer in JSON format, with the format as follows: "
    '{"Reasoning": "", "Preference": "Yes or No", "Answer": ""}. '
    "Please strictly output in JSON format, do not output irrelevant content."
)

def agent_prompt(problem: str, name: str) -> str:
    return f'''
You are a debater. Hello and welcome to the math competition, which will be conducted in a debate format.
It's not necessary to fully agree with each other's perspectives, as our objective is to find the correct solution.
The debate topic will be producing the correct answer to the following problem:\n\"{problem}\"
Try to keep your response within 500 words or less.'''

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
    '''
    Big ugly thing to try to handle edge-cases in JSON outputs.
    "Why don't you just do constrained decoding?" I don't know.
    '''

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

def parse_decision(_decision: str) -> Dict | str:
    decision = try_parse(_decision)
    if (
        isinstance(decision, dict)
        and decision.get('Preference', '').lower().strip() == 'yes'
        and decision.get('Answer')
    ):
        return decision
    return _decision

def mad_cached(
    workflow: Workflow,
    problem: str,
    max_rounds: int,
    temperature: float = 0.6,
    top_p: float = 0.95,
    seed: int = 42,
    debug: bool = False,
) -> Dict:
    res = {'aff_tokens': [], 'neg_tokens': [], 'mod_tokens': [], 'decision': None}

    aff_context, neg_context, mod_context = [[a] for a in workflow.insert([
        {'messages': [
            {'role': 'system', 'content': agent_prompt(problem, '')},
            {'role': 'user', 'content': problem},
        ], 'parent_ids': []},
        {'messages': [
            {'role': 'system', 'content': agent_prompt(problem, '')},
            {'role': 'user', 'content': problem},
        ], 'parent_ids': []},
        {'messages': [
            {'role': 'system', 'content': moderator_system_prompt(problem)},
            {'role': 'user', 'content': moderator_user_prompt}
        ], 'parent_ids': []}
    ])]
    [aff_tokens], [aff_node] = get('tokens', 'nodes')(workflow.step(
        [{'header': ('assistant', ''), 'prefill': 'Affirmative:\n\n', 'parent_ids': [n['id'] for n in aff_context]}],
        temperature=temperature,
        top_p=top_p,
        max_gen_len=512,
    ))
    aff_context.append(aff_node)
    neg_context.append(aff_node)
    res['aff_tokens'].append([aff_tokens])

    [neg_tokens], [neg_node] = get('tokens', 'nodes')(workflow.step(
        [{'header': ('assistant', ''), 'prefill': 'Negative:\n\n', 'parent_ids': [n['id'] for n in neg_context]}],
        temperature=temperature,
        top_p=top_p,
        max_gen_len=512,
    ))
    aff_context.append(neg_node)
    neg_context.append(neg_node)
    res['neg_tokens'].append([neg_tokens])

    for round in range(max_rounds - 1):
        [aff_tokens], [aff_response] = get('tokens', 'nodes')(workflow.step([{
            'header': ('assistant', ''),
            'prefill': 'Affirmative Response:\n\n',
            'parent_ids': [n['id'] for n in aff_context]
        }], temperature=temperature, top_p=top_p, seed=seed))
        aff_context.append(aff_response)
        neg_context.append(aff_response)
        mod_context.append(aff_response)
        res['aff_tokens'].append([aff_tokens])

        [neg_tokens], [neg_response] = get('tokens', 'nodes')(workflow.step([{
            'header': ('assistant', ''),
            'prefill': 'Negative Response:\n\n',
            'parent_ids': [n['id'] for n in neg_context]
        }], temperature=temperature, top_p=top_p, seed=seed))
        aff_context.append(neg_response)
        neg_context.append(neg_response)
        mod_context.append(neg_response)
        res['neg_tokens'].append([neg_tokens])

        [mod_tokens], [mod_response] = get('tokens', 'nodes')(workflow.step([{
            'header': ('assistant', 'moderator'),
            'prefill': '{"Reasoning": ',
            'parent_ids': [n['id'] for n in mod_context]
        }], temperature=temperature, top_p=top_p, seed=seed))
        mod_context.append(mod_response)
        res['mod_tokens'].append([mod_tokens])

        if decision := parse_decision(workflow.tokenizer.decode(mod_tokens)):
            res['decision'] = decision
            break

    if not res['decision']:
        [final_prompt] = workflow.insert([
            {'messages': [
                {'role': 'user', 'content': (
                    'Please summarize your reasons and give the final answer that you think is correct. '
                    'Now please output your answer in JSON format, with the format as follows: {{"Reasoning": "", "Answer": ""}}. '
                    'Please strictly output in JSON format, do not output irrelevant content.'
                )}
            ], 'parent_ids': [n['id'] for n in mod_context]}
        ])
        mod_context.append(final_prompt)
        [final_tokens], [final_node] = get('tokens', 'nodes')(workflow.step(
            [{'header': ('assistant', ''), 'prefill': '', 'parent_ids': [n['id'] for n in mod_context]}],
            temperature=temperature,
            top_p=top_p,
        ))
        mod_context.append(final_node)
        res['final_tokens'] = [final_tokens]

    return res | {
        'aff_context': aff_context,
        'neg_context': neg_context,
        'mod_context': mod_context,
    }

def mad_baseline(
    workflow: Workflow,
    problem: str,
    max_rounds: int,
    temperature: float = 0.7,
    top_p: float = 1.0,
    seed: int = 42,
    debug: bool = False,
) -> Dict:
    res = {'aff_tokens': [], 'neg_tokens': [], 'mod_tokens': [], 'decision': None}
    dct = {1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth'}

    # initial aff
    aff_stale: List[Message] = []
    aff_context = workflow.insert([
        {'messages': [
            {'role': 'system', 'content': agent_prompt(problem, '')},
            {'role': 'user', 'content': problem},
        ], 'parent_ids': []},
    ])
    [aff_tokens] = get('tokens')(workflow.step(
        [{'header': ('assistant', ''), 'prefill': '', 'parent_ids': [n['id'] for n in aff_context]}],
        temperature=temperature,
        top_p=top_p,
        max_gen_len=512,
    ))
    aff_ans = workflow.tokenizer.decode(aff_tokens)
    aff_stale.append({'role': 'assistant', 'content': aff_ans})
    if debug: print(f'Aff ans:\n{aff_ans}')
    res['aff_tokens'].append([aff_tokens])

    # initial neg
    neg_stale: List[Message] = []
    neg_context = workflow.insert([
        {'messages': [
            {'role': 'system', 'content': agent_prompt(problem, '')},
            {'role': 'user', 'content': f'{aff_ans}\n\nYou disagree with my answer. Provide your answer and reasons.'},
        ], 'parent_ids': []},
    ])
    [neg_tokens] = get('tokens')(workflow.step(
        [{'header': ('assistant', ''), 'prefill': '', 'parent_ids': [n['id'] for n in neg_context]}],
        temperature=temperature,
        top_p=top_p,
        max_gen_len=512,
    ))
    neg_ans = workflow.tokenizer.decode(neg_tokens)
    neg_stale.append({'role': 'assistant', 'content': neg_ans})
    if debug: print(f'Neg ans:\n{neg_ans}')
    res['neg_tokens'].append([neg_tokens])

    mod_stale: List[Message] = []
    mod_context = []
    for round in range(max_rounds - 1):
        aff_stale.append({'role': 'user', 'content': f'{neg_ans}\n\nDo you agree with my perspective? Please provide your reasons and answer.'})
        aff_context.extend(workflow.insert([{'messages': aff_stale, 'parent_ids': [n['id'] for n in aff_context]}]))
        [aff_tokens] = get('tokens')(workflow.step(
            [{'header': ('assistant', ''), 'prefill': '', 'parent_ids': [n['id'] for n in aff_context]}],
            temperature=temperature,
            top_p=temperature,
            max_gen_len=512,
        ))
        aff_ans = workflow.tokenizer.decode(aff_tokens)
        aff_stale = [{'role': 'assistant', 'content': aff_ans}]
        if debug: print(f'Aff ans:\n{aff_ans}')
        res['aff_tokens'].append([aff_tokens])

        neg_stale.append({'role': 'user', 'content': f'{aff_ans}\n\nDo you agree with my perspective? Please provide your reasons and answer.'})
        neg_context.extend(workflow.insert([{'messages': neg_stale, 'parent_ids': [n['id'] for n in neg_context]}]))
        [neg_tokens] = get('tokens')(workflow.step(
            [{'header': ('assistant', ''), 'prefill': '', 'parent_ids': [n['id'] for n in neg_context]}],
            temperature=temperature,
            top_p=temperature,
            max_gen_len=512,
        ))
        neg_ans = workflow.tokenizer.decode(neg_tokens)
        neg_stale = [{'role': 'assistant', 'content': neg_ans}]
        if debug: print(f'Neg ans:\n{neg_ans}')
        res['neg_tokens'].append([neg_tokens])

        mod_stale.append({'role': 'user', 'content': faithful_mod_prompt(dct[round+2], aff_ans, neg_ans)})
        mod_context.extend(workflow.insert([{'messages': mod_stale, 'parent_ids': [n['id'] for n in mod_context]}]))
        [mod_tokens] = get('tokens')(workflow.step([{'header': ('assistant', ''), 'prefill': '', 'parent_ids': [n['id'] for n in mod_context]}]))
        mod_ans = workflow.tokenizer.decode(mod_tokens)
        mod_stale = [{'role': 'assistant', 'content': mod_ans}]
        if debug: print(f'Mod review:\n{mod_ans}')
        res['mod_tokens'].append([mod_tokens])

        if isinstance((decision := parse_decision(mod_ans)), dict):
            res['decision'] = decision
            break

    if not res['decision']:
        # "ultimate deadly technique."
        # https://github.com/Skytliang/Multi-Agents-Debate/blob/022a7a8eecda85844d336e9064cc556edb0445b3/code/debate4tran.py#L236
        [judge_prompt] = workflow.insert([
            {'messages': [
                {'role': 'system', 'content': agent_prompt(problem, '')},
                {'role': 'user', 'content': f'Affirmative side arguing:\n{aff_ans}\n\nNegative side arguing:\n{neg_ans}\n\nNow, what answer candidates do we have? Present them without reasons.'},
            ], 'parent_ids': []}
        ])

        [cand_tokens], [cand_response] = get('tokens', 'nodes')(workflow.step([{'header': ('assistant', ''), 'prefill': '', 'parent_ids': [judge_prompt['id']]}]))

        [final_prompt] = workflow.insert([
            {'messages': [
                {'role': 'user', 'content': (
                    f'Therefore, {problem}\nPlease summarize your reasons and give the final answer that you think is correct. '
                    'Now please output your answer in JSON format, with the format as follows: {{"Reason": "", "Answer": ""}}. '
                    'Please strictly output in JSON format, do not output irrelevant content.'
                )}
            ], 'parent_ids': [judge_prompt['id'], cand_response['id']]}
        ])

        [final_tokens] = get('tokens')(workflow.step([{'header': ('assistant', ''), 'prefill': '', 'parent_ids': [judge_prompt['id'], cand_response['id'], final_prompt['id']]}]))
        res['decision'] = parse_decision(workflow.tokenizer.decode(final_tokens))
        res['final_tokens'] = [final_tokens]

    return res
