from typing import Optional, List, Dict
from operator import itemgetter as get

from llama import Workflow
from .mad import try_parse

def math_direct(
    workflow: Workflow,
    problem: str,
    temperature: float = 1.0,
    top_p: float = 1.0,
    seed: int = 42,
    debug: bool = False,
):
    solve_prompt = f'Solve this math problem. Just output the final answer without explanations or workings:\n\n{problem}'

    [sys] = workflow.insert([{'messages': [{'role': 'user', 'content': solve_prompt}], 'parent_ids': []}])

    [answer_tokens], [answer] = get('tokens', 'nodes')(workflow.step([{
        'header': ('assistant', None),
        'prefill': 'Answer: ',
        'parent_ids': [sys['id']]
    }], temperature=temperature, top_p=top_p, seed=seed, max_gen_len=512))

    if debug:
        print(workflow.tokenizer.decode(answer_tokens))

    return workflow.tokenizer.decode(answer_tokens)

def math_cot(
    workflow: Workflow,
    problem: str,
    best_of_n: Optional[int] = None,
    enable_reflection: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 42,
    debug: bool = False,
):
    assert best_of_n is None or not enable_reflection

    solve_prompt = (
        f'Solve the following math problem:\n{problem}"\n\n'
        'Output your answer in JSON format: {"Reasoning": "step-by-step walkthrough to the correct answer", "Answer": "final answer"}'
    )

    [sys] = workflow.insert([{'messages': [{'role': 'user', 'content': solve_prompt}], 'parent_ids': []}])

    if best_of_n is None:
        [solve_tokens], [solve] = get('tokens', 'nodes')(workflow.step([{
            'header': ('assistant', 'solver'),
            'prefill': '{"Reasoning": "',
            'parent_ids': [sys['id']]
        }], temperature=temperature, top_p=top_p, seed=seed, max_gen_len=2048))

        if debug:
            print(workflow.tokenizer.decode(solve_tokens))

        if not enable_reflection:
            return try_parse(workflow.tokenizer.decode(solve_tokens))

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
        }], temperature=temperature, top_p=top_p, seed=seed, max_gen_len=2048))

        if debug:
            print(workflow.tokenizer.decode(answer_tokens))

        return try_parse(workflow.tokenizer.decode(answer_tokens))

    else:
        solve_tokens = get('tokens')(
            workflow.step(
                [{
                    'header': ('assistant', 'solver'),
                    'prefill': '{"Reasoning": "',
                    'parent_ids': [sys['id']]
                } for _ in range(best_of_n)],
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                max_gen_len=2048,
            )
        )
        return [try_parse(workflow.tokenizer.decode(t)) for t in solve_tokens]

def commongen_baseline(
    workflow: Workflow,
    concepts: List[str],
    seed: int = 42,
) -> Optional[Dict]:
    planning_prompt = f"""I'm going to write a story incorporating all of these concepts: {', '.join(concepts)}

Before writing the full story, help me create a brief plan or outline.
The plan should:
1. Propose a compelling story topic or theme
2. Sketch how each concept will be integrated
3. Outline a simple narrative structure with beginning, middle, and end

Please keep your plan concise - just a few bullet points to guide the story creation."""

    [plan_node] = workflow.insert([
        {'messages': [
            {'role': 'user', 'content': planning_prompt}
        ], 'parent_ids': []}
    ])

    plan_tokens, plan_nodes = get('tokens', 'nodes')(workflow.step([
        {'header': ('assistant', None),
         'prefill': 'Story Plan:\n\n',
         'parent_ids': [plan_node['id']]}
    ],
        max_gen_len=512,
        temperature=0.7,
        top_p=1.0,
        seed=seed,
    ))

    story_plan = workflow.tokenizer.decode(plan_tokens[0])

    generation_prompt = f"""Based on the plan below, write a concise and coherent story in a single paragraph.
Make sure to include ALL of the following concepts: {', '.join(concepts)}

Plan:
{story_plan}

Now, write the complete story following this plan:"""

    [generation_node] = workflow.insert([
        {'messages': [
            {'role': 'user', 'content': generation_prompt}
        ], 'parent_ids': [plan_nodes[0]['id']]}
    ])

    story_tokens, story_nodes = get('tokens', 'nodes')(workflow.step([
        {'header': ('assistant', None),
         'prefill': 'Final Story:\n\n',
         'parent_ids': [generation_node['id']]}
    ],
        max_gen_len=1024,
        temperature=0.7,
        top_p=1.0,
        seed=seed,
    ))

    return {
        'plan_tokens': plan_tokens,
        'story_tokens': story_tokens,
    }
