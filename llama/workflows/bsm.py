import re
import json
import time
from operator import itemgetter as get
from typing import List, Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

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
    elif split == 'dev' or split == 'val':
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

def compare_stories(
    a_stories: List[str],
    b_stories: List[str],
    concepts_list: List[List[str]]
) -> Tuple[List[bool], Dict[str, int]]:
    load_dotenv()
    client = OpenAI()
    results = [False] * len(a_stories)

    stats = {
        "a_wins": 0,
        "b_wins": 0,
        "ties": 0,
        "errors": 0,
        "total": len(a_stories)
    }
    
    prompt_template =  '''
Please act as an impartial judge and evaluate the quality of the stories provided by two AI assistants.
Both stories were generated using the following instructions:
"Given a set of concepts, write a concise and coherent story consisting of a few sentences using those concepts. The story should naturally integrate all of the following concepts: {concepts_str}"

Your evaluation should consider TWO primary factors:

1. CONCEPT INTEGRATION (50% weight):
   - How well does the story naturally incorporate ALL of the required concepts?
   - Are concepts integrated naturally or are they forced into the narrative?
   - Does the story cover all required concepts without omissions?

2. OVERALL STORY QUALITY (50% weight):
   - Coherence and flow of the narrative
   - Engagement and creativity
   - Grammatical correctness
   - Logical consistency

Begin your evaluation by identifying which concepts from the list are included in each story.
Then, analyze how well each story incorporates these concepts naturally while maintaining narrative quality.
Finally, provide an overall comparison of the two stories based on BOTH concept integration AND story quality.

Required concepts: {concepts_str}

Story A:
{story_a}

Story B:
{story_b}

Avoid any position biases and ensure that the order in which the stories were presented does not influence your decision.
Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible.

After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if story A is better, "[[B]]" if story B is better, and "[[C]]" for a tie.
'''

    def get_verdict(prompt, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                content = response.choices[0].message.content
                match = re.search(r"\[\[(A|B|C)\]\]", content)
                return match.group(1) if match else "C"
            except:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return "C"

    def process_pair(idx):
        ab_prompt = prompt_template.format(
            concepts_str=', '.join(concepts_list[idx]),
            story_a=a_stories[idx],
            story_b=b_stories[idx]
        )
        ab_winner = get_verdict(ab_prompt)
        ba_prompt = prompt_template.format(
            concepts_str=', '.join(concepts_list[idx]),
            story_a=b_stories[idx],
            story_b=a_stories[idx]
        )
        ba_winner = get_verdict(ba_prompt)

        if ab_winner == "A" and ba_winner == "B":
            return idx, True, "a_win"
        elif ab_winner == "B" and ba_winner == "A":
            return idx, False, "b_win"
        else:
            return idx, False, "tie"

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_pair, i) for i in range(len(a_stories))]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Comparing"):
            idx, a_wins, result_type = future.result()
            results[idx] = a_wins

            if result_type == "a_win":
                stats["a_wins"] += 1
            elif result_type == "b_win":
                stats["b_wins"] += 1
            elif result_type == "tie":
                stats["ties"] += 1
            elif result_type == "error":
                stats["errors"] += 1

    total_valid = stats["total"] - stats["errors"]
    if total_valid > 0:
        stats["a_win_percent"] = (stats["a_wins"] / total_valid) * 100
        stats["b_win_percent"] = (stats["b_wins"] / total_valid) * 100
        stats["tie_percent"] = (stats["ties"] / total_valid) * 100

    return results, stats
