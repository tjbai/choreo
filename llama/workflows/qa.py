from typing import List, Dict
from operator import itemgetter as get

from llama import Workflow

system_prompt = 'Answer ALL of the user\'s questions. Answer with an numbered list. Do not include extraneous text.'

eval_system_prompt = '''
You are an impartial evaluator for question-answering systems. Your task is to determine whether a given answer correctly matches any of the acceptable answers for a trivia question.

You will receive:
1. A trivia question
2. A list of acceptable answers (aliases)
3. The system's attempted answer

Evalation rules:
1. The evaluation should be case-insensitive
2. Ignore minor differences in punctuation, articles, and spacing
3. An answer is considered correct if it matches ANY of the provided aliases
4. Names may be partially correct if they contain the key identifying information
   - For example, if the alias is "Franklin D. Roosevelt" and the answer is "Roosevelt", this is partially correct
   - If the alias is "Battle of Gettysburg" and the answer is just "Gettysburg", this is partially correct
5. For numerical answers, different formats are acceptable (e.g., "42" and "forty-two" should be considered the same)
6. For dates, different formats are acceptable (e.g., "July 4, 1776", "4th of July 1776", "07/04/1776")

Provide your evaluation as a JSON object with the following fields:
- `correct`: A boolean indicating whether the answer is correct (true/false)
'''

def format_eval_user(question_data, solution):
    question = question_data['Question']
    aliases = question_data['Answer']['Aliases']

    aliases_str = "\n".join([f"- {alias}" for alias in aliases])

    user_prompt = f"""
Question: "{question}"

Acceptable answers:
{aliases_str}

System's answer: "{solution}"

Evaluate whether the system's answer is correct based on the rules.
"""

    return user_prompt

def parse_items(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    items = [line.split('. ', 1)[1] if '. ' in line else line for line in lines]
    return items

def ask_sequential(workflow: Workflow, subset: List[Dict]) -> Dict:
    workflow.reset()

    [prompt] = workflow.insert([
        {
            'messages': ([
                {'role': 'system', 'content': system_prompt}] +
                [{'role': 'user', 'content': f'Question {i+1}: {p['Question']}'} for i, p in enumerate(subset)]),
            'parent_ids': []
        }
    ])

    [response], [response_node] = get('tokens', 'nodes')(workflow.step(
        tasks=[{
            'header': ('assistant', None),
            'prefill': '',
            'parent_ids': [prompt['id']],
        }]
    ))

    return response_node

def ask_parallel(workflow, subset, annotate=False, compact=False):
    workflow.reset()

    [prompt] = workflow.insert([
        {
            'messages': [{'role': 'system', 'content': system_prompt}],
            'parent_ids': []
        }
    ])

    questions = workflow.insert([
        {
            'messages': [{'role': 'user', 'content': f'{f'Question {i+1}: ' if annotate else ''}{p['Question']}'}],
            'parent_ids': [prompt['id']],
        }
        for i, p in enumerate(subset)
    ])

    [response], [response_node] = get('tokens', 'nodes')(workflow.step(
        tasks=[{
            'header': ('assistant', None),
            'prefill': '',
            'parent_ids': [prompt['id']] + [q['id'] for q in questions],
        }],
        compact=compact
    ))

    return response_node
