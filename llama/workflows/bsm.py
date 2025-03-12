import re
import json
from operator import itemgetter as get
from typing import List, Optional, Dict
from llama import Workflow

def load_concepts(data_path, split='train'):
    with open(data_path) as f:
        data = [json.loads(line)['concepts'] for line in f]

    train_ratio = 0.5
    dev_ratio = 0.25
    total = len(data)
    train_idx = int(total * train_ratio)
    dev_idx = train_idx + int(total * dev_ratio)

    if split == 'train':
        return data[:train_idx]
    elif split == 'dev':
        return data[train_idx:dev_idx]
    elif split == 'test':
        return data[dev_idx:]

def branch_prompt_content(concepts: List[str]):
    return f"""
Given a set of concepts, we want to write a concise and coherent story consisting of a few sentences using those concepts.
First propose a story topic and then divide the concepts into two groups such that the story generated from each group of concepts can be combined together to form a longer story.
Make sure that you do not leave out any concepts.

Concepts: {', '.join(concepts)}

Format your response as follows:
Story Topic: [proposed topic]
Group 1: [comma-separated concepts for first group]
Group 2: [comma-separated concepts for second group]
Reasoning: [brief explanation for your grouping]"""

def solve_prompt(concept_group: List[str], story_topic: str):
    return f"""
Write a concise and coherent story on the following topic consisting of a single paragraph. Make sure to include all the following concepts in the story.\n
Concepts: {', '.join(concept_group)}
Story Topic: {story_topic}"""

def merge_prompt(solve_stories: List[str], group1_concepts: List[str], group2_concepts: List[str]):
    return f"""
Given two groups of concepts and two stories containing those concepts, combine the two stories into a concise and coherent story consisting of a single paragraph.
Make sure that the combined story does not miss any concept from the two groups.

Group 1 Concepts: {', '.join(group1_concepts)}
Story 1: {solve_stories[0]}

Group 2 Concepts: {', '.join(group2_concepts)}
Story 2: {solve_stories[1]}"""

def bsm_baseline(
    workflow: Workflow,
    concepts: List[str],
    seed: int = 42,
    temperature: float = 0.7,
    top_p: float = 1.0,
) -> Optional[Dict]:
    [branch_node] = workflow.insert([
        {'messages': [
            {'role': 'user', 'content': branch_prompt_content(concepts)}
        ], 'parent_ids': []}
    ])

    branch_tokens, branch_nodes = get('tokens', 'nodes')(workflow.step([
        {'header':
            ('assistant', None),
            'prefill': '',
            'parent_ids': [branch_node['id']]}
        ],
        max_gen_len=512,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    ))

    branch_output = workflow.tokenizer.decode(branch_tokens[0])
    topic_match = re.search(r"Story Topic:\s*(.*?)(?:\n|$)", branch_output)
    group1_match = re.search(r"Group 1:\s*(.*?)(?:\n|$)", branch_output)
    group2_match = re.search(r"Group 2:\s*(.*?)(?:\n|$)", branch_output)

    if not (topic_match and group1_match and group2_match):
        return None

    story_topic = topic_match.group(1).strip()
    group1_concepts = [c.strip() for c in group1_match.group(1).split(",")]
    group2_concepts = [c.strip() for c in group2_match.group(1).split(",")]

    solve_nodes = workflow.insert([
        {'messages': [
            {'role': 'user', 'content': solve_prompt(concept_group, story_topic)}
        ], 'parent_ids': []}
        for concept_group in [group1_concepts, group2_concepts]
    ])

    solve_tokens = get('tokens')(workflow.step([
        {'header':
            ('assistant', None),
            'prefill': f'Story {i+1}:\n\n',
            'parent_ids': [solve_node['id']]}
        for i, solve_node in enumerate(solve_nodes)
    ],
        max_gen_len=512,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    ))

    [merge_node] = workflow.insert([
        {'messages': [
            {'role': 'user', 'content': merge_prompt(
                list(map(workflow.tokenizer.decode, solve_tokens)),
                group1_concepts,
                group2_concepts)
            }
        ], 'parent_ids': []}
    ])

    merge_tokens, merge_nodes = get('tokens', 'nodes')(workflow.step([
        {'header':
            ('assistant', None),
            'prefill': 'Combined Story:\n\n',
            'parent_ids': [merge_node['id']]}
    ],
        max_gen_len=1024,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    ))

    return {
        'branch_tokens': branch_tokens,
        'solve_tokens': solve_tokens,
        'merge_tokens': merge_tokens,
        'story_topic': story_topic,
        'concept_groups': [group1_concepts, group2_concepts],
    }

cached_merge_prompt = '''Combine the two stories into a single coherent paragraph that includes all concepts from both groups.
Create a combined story that flows naturally.'''

def bsm_cached(
    workflow: Workflow,
    concepts: List[str],
    branching_factor: int = 2,
    seed: int = 42,
    compact: bool = False,
    temperature: float = 0.7,
    top_p: float = 1.0,
) -> Optional[Dict]:
    [branch_node, merge_node] = workflow.insert([
        {'messages': [
            {'role': 'user', 'content': branch_prompt_content(concepts)}
        ], 'parent_ids': []},
        {'messages': [
            {'role': 'user', 'content': cached_merge_prompt}
        ], 'parent_ids': []},
    ])

    branch_tokens, branch_nodes = get('tokens', 'nodes')(workflow.step([
        {'header': ('assistant', None),
         'prefill': '',
         'parent_ids': [branch_node['id']]}
    ],
        max_gen_len=512,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    ))
    branch_output = workflow.tokenizer.decode(branch_tokens[0])
    topic_match = re.search(r"Story Topic:\s*(.*?)(?:\n|$)", branch_output)
    group1_match = re.search(r"Group 1:\s*(.*?)(?:\n|$)", branch_output)
    group2_match = re.search(r"Group 2:\s*(.*?)(?:\n|$)", branch_output)

    if not (topic_match and group1_match and group2_match):
        return None

    story_topic = topic_match.group(1).strip()
    group1_concepts = [c.strip() for c in group1_match.group(1).split(",")]
    group2_concepts = [c.strip() for c in group2_match.group(1).split(",")]

    solve_node_prompts = workflow.insert([
        {'messages': [
            {'role': 'user', 'content': solve_prompt(concept_group, story_topic)}
        ], 'parent_ids': []}
        for concept_group in [group1_concepts, group2_concepts]
    ])

    solve_nodes, solve_tokens = get('nodes', 'tokens')(workflow.step([
        {'header':
            ('assistant', None),
            'prefill': f'Story {i+1}:\n\n',
            'parent_ids': [solve_node_prompt['id']]}
        for i, solve_node_prompt in enumerate(solve_node_prompts)
    ],
        max_gen_len=512,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    ))

    merge_tokens = get('tokens')(workflow.step([
        {'header': ('assistant', None),
         'prefill': 'Combined Story:\n\n',
         'parent_ids': [
             merge_node['id'],
             solve_nodes[0]['id'],
             solve_nodes[1]['id'],
         ]},
    ],
        max_gen_len=1024,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        compact=compact,
    ))

    return {
        'branch_tokens': branch_tokens,
        'solve_tokens': solve_tokens,
        'merge_tokens': merge_tokens,
        'story_topic': story_topic,
        'concept_groups': [group1_concepts, group2_concepts],
    }
