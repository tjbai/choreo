from typing import Optional
from operator import itemgetter as get

from llama import Workflow
from .mad import try_parse

def math_simple_baseline(
    workflow: Workflow,
    problem: str,
    best_of_n: Optional[int] = False,
    enable_reflection: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 42,
    debug: bool = False,
):
    assert best_of_n is None or enable_reflection is None

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
        }], temperature=temperature, top_p=top_p, seed=seed, max_gen_len=1024))

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
        }], temperature=temperature, top_p=top_p, seed=seed, max_gen_len=1024))

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
                max_gen_len=1024
            )
        )
        return [try_parse(workflow.tokenizer.decode(t)) for t in solve_tokens]
