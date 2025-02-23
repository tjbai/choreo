from typing import List, Dict
from operator import itemgetter as get

from llama import Workflow

def parse_items(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    items = [line.split('. ', 1)[1] if '. ' in line else line for line in lines]
    return items

def ask_sequential(workflow: Workflow, subset: List[Dict]) -> Dict:
    workflow.reset()

    [prompt] = workflow.insert([
        {
            'messages': (
                [{'role': 'system', 'content': 'Answer ALL of the user\'s questions. Answer with an numbered list. Do not include extraneous text.'}] +
                [{'role': 'user', 'content': f'Question {i+1}: {p['Question']}'} for i, p in enumerate(subset)]
            ),
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
            'messages': [{'role': 'system', 'content': 'Answer ALL of the user\'s questions. Answer with an numbered list. Do not include extraneous text.'}],
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
