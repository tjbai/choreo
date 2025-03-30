import os
import json
import time
import statistics
from operator import itemgetter as get

import torch
from tqdm import tqdm

from llama import Workflow
from llama.util import find_free_port
from llama.workflows.tot import load_math_problems
from llama.workflows.mad import (
    agent_prompt,
    faithful_mod_prompt,
    moderator_system_prompt,
    moderator_user_prompt,
)

def baseline(
    workflow,
    problem,
    max_rounds=3,
    temperature=0.7,
    top_p=1.0,
):
    dct = {1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth'}

    workflow.reset()
    insert_time = []
    step_time = []
    ttft_time = []
    total_ttft = 0
    total_tokens = 0
    force_tokens = []
    s = time.time()

    aff_stale = []
    aff_context = workflow.insert([
        {'messages': [
            {'role': 'system', 'content': agent_prompt(problem, '')},
            {'role': 'user', 'content': problem},
        ], 'parent_ids': []},
    ], time_buffer=insert_time)
    [aff_tokens] = get('tokens')(workflow.step(
        [{'header': ('assistant', ''), 'prefill': '', 'parent_ids': [n['id'] for n in aff_context]}],
        temperature=temperature,
        top_p=top_p,
        max_gen_len=512,
        time_buffer=step_time,
        ttft_buffer=ttft_time,
    ))
    aff_ans = workflow.tokenizer.decode(aff_tokens)
    aff_stale.append({'role': 'assistant', 'content': aff_ans})
    total_ttft += insert_time[-1] + ttft_time[-1]
    total_tokens += len(aff_tokens)
    force_tokens.append(len(aff_tokens))

    neg_stale = []
    neg_context = workflow.insert([
        {'messages': [
            {'role': 'system', 'content': agent_prompt(problem, '')},
            {'role': 'user', 'content': f'{aff_ans}\n\nYou disagree with my answer. Provide your answer and reasons.'},
        ], 'parent_ids': []},
    ], time_buffer=insert_time)
    [neg_tokens] = get('tokens')(workflow.step(
        [{'header': ('assistant', ''), 'prefill': '', 'parent_ids': [n['id'] for n in neg_context]}],
        temperature=temperature,
        top_p=top_p,
        max_gen_len=512,
        time_buffer=step_time,
        ttft_buffer=ttft_time,
    ))
    neg_ans = workflow.tokenizer.decode(neg_tokens)
    neg_stale.append({'role': 'assistant', 'content': neg_ans})
    total_ttft += insert_time[-1] + ttft_time[-1]
    total_tokens += len(neg_tokens)
    force_tokens.append(len(neg_tokens))

    mod_stale = []
    mod_context = []

    for round in range(max_rounds - 1):
        aff_stale.append({'role': 'user', 'content': f'{neg_ans}\n\nDo you agree with my perspective? Please provide your reasons and answer.'})
        aff_context.extend(workflow.insert([{'messages': aff_stale, 'parent_ids': [n['id'] for n in aff_context]}]))
        [aff_tokens] = get('tokens')(workflow.step(
            [{'header': ('assistant', ''), 'prefill': '', 'parent_ids': [n['id'] for n in aff_context]}],
            temperature=temperature,
            top_p=top_p,
            max_gen_len=512,
            time_buffer=step_time,
            ttft_buffer=ttft_time,
        ))
        aff_ans = workflow.tokenizer.decode(aff_tokens)
        aff_stale = [{'role': 'assistant', 'content': aff_ans}]
        total_ttft += ttft_time[-1]
        total_tokens += len(aff_tokens)
        force_tokens.append(len(aff_tokens))

        neg_stale.append({'role': 'user', 'content': f'{aff_ans}\n\nDo you agree with my perspective? Please provide your reasons and answer.'})
        neg_context.extend(workflow.insert([{'messages': neg_stale, 'parent_ids': [n['id'] for n in neg_context]}]))
        [neg_tokens] = get('tokens')(workflow.step(
            [{'header': ('assistant', ''), 'prefill': '', 'parent_ids': [n['id'] for n in neg_context]}],
            temperature=temperature,
            top_p=top_p,
            max_gen_len=512,
            time_buffer=step_time,
            ttft_buffer=ttft_time,
        ))
        neg_ans = workflow.tokenizer.decode(neg_tokens)
        neg_stale = [{'role': 'assistant', 'content': neg_ans}]
        total_ttft += ttft_time[-1]
        total_tokens += len(neg_tokens)
        force_tokens.append(len(neg_tokens))

        mod_stale.append({'role': 'user', 'content': faithful_mod_prompt(dct[round+2], aff_ans, neg_ans)})
        mod_context.extend(workflow.insert([{'messages': mod_stale, 'parent_ids': [n['id'] for n in mod_context]}]))
        [mod_tokens] = get('tokens')(workflow.step(
            [{'header': ('assistant', ''), 'prefill': '', 'parent_ids': [n['id'] for n in mod_context]}],
            time_buffer=step_time,
            ttft_buffer=ttft_time,
        ))
        mod_ans = workflow.tokenizer.decode(mod_tokens)
        mod_stale = [{'role': 'assistant', 'content': mod_ans}]
        total_ttft += ttft_time[-1]
        total_tokens += len(mod_tokens)
        force_tokens.append(len(mod_tokens))

    [judge_prompt] = workflow.insert([
        {'messages': [
            {'role': 'system', 'content': agent_prompt(problem, '')},
            {'role': 'user', 'content': f'Affirmative side arguing:\n{aff_ans}\n\nNegative side arguing:\n{neg_ans}\n\nNow, what answer candidates do we have? Present them without reasons.'},
        ], 'parent_ids': []}
    ], time_buffer=insert_time)
    [cand_tokens], [cand_response] = get('tokens', 'nodes')(workflow.step(
        [{'header': ('assistant', ''), 'prefill': '', 'parent_ids': [judge_prompt['id']]}],
        time_buffer=step_time,
        ttft_buffer=ttft_time,
    ))
    total_ttft += insert_time[-1] + ttft_time[-1]
    total_tokens += len(cand_tokens)
    force_tokens.append(len(cand_tokens))

    [final_prompt] = workflow.insert([
        {'messages': [
            {'role': 'user', 'content': (
                f'Therefore, {problem}\nPlease summarize your reasons and give the final answer that you think is correct. '
                'Now please output your answer in JSON format, with the format as follows: {{"Reason": "", "Answer": ""}}. '
                'Please strictly output in JSON format, do not output irrelevant content.'
            )}
        ], 'parent_ids': [judge_prompt['id'], cand_response['id']]}
    ], time_buffer=insert_time)
    [final_tokens] = get('tokens')(workflow.step(
        [{'header': ('assistant', ''), 'prefill': '', 'parent_ids': [judge_prompt['id'], cand_response['id'], final_prompt['id']]}],
        time_buffer=step_time,
        ttft_buffer=ttft_time,
    ))
    total_ttft += insert_time[-1] + ttft_time[-1]
    total_tokens += len(final_tokens)
    force_tokens.append(len(final_tokens))

    return {
        'wall_time': time.time() - s,
        'cuda_time': sum(insert_time) + sum(step_time),
        'ttft': total_ttft,
        'tokens': total_tokens,
        'force_tokens': force_tokens
    }

def cached(
    workflow,
    problem,
    max_rounds=3,
    temperature=0.7,
    top_p=1.0,
    force_tokens=[]
):
    workflow.reset()
    insert_time = []
    step_time = []
    ttft_time = []
    total_ttft = 0
    s = time.time()

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
    ], time_buffer=insert_time)]

    [aff_tokens], [aff_node] = get('tokens', 'nodes')(workflow.step(
        [{'header': ('assistant', ''), 'prefill': 'Affirmative:\n\n', 'parent_ids': [n['id'] for n in aff_context]}],
        temperature=temperature,
        top_p=top_p,
        time_buffer=step_time,
        ttft_buffer=ttft_time,
        force_tokens=force_tokens.pop(0),
    ))
    aff_context.append(aff_node)
    neg_context.append(aff_node)
    total_ttft += insert_time[-1] + ttft_time[-1]

    [neg_tokens], [neg_node] = get('tokens', 'nodes')(workflow.step(
        [{'header': ('assistant', ''), 'prefill': 'Negative:\n\n', 'parent_ids': [n['id'] for n in neg_context]}],
        temperature=temperature,
        top_p=top_p,
        time_buffer=step_time,
        ttft_buffer=ttft_time,
        force_tokens=force_tokens.pop(0),
    ))
    aff_context.append(neg_node)
    neg_context.append(neg_node)
    total_ttft += ttft_time[-1]

    for round in range(max_rounds - 1):
        [aff_tokens], [aff_response] = get('tokens', 'nodes')(workflow.step([{
            'header': ('assistant', ''),
            'prefill': 'Affirmative Response:\n\n',
            'parent_ids': [n['id'] for n in aff_context]
        }],
            temperature=temperature,
            top_p=top_p,
            time_buffer=step_time,
            ttft_buffer=ttft_time,
            force_tokens=force_tokens.pop(0),
        ))
        aff_context.append(aff_response)
        neg_context.append(aff_response)
        mod_context.append(aff_response)
        total_ttft += ttft_time[-1]

        [neg_tokens], [neg_response] = get('tokens', 'nodes')(workflow.step([{
            'header': ('assistant', ''),
            'prefill': 'Negative Response:\n\n',
            'parent_ids': [n['id'] for n in neg_context]
        }],
            temperature=temperature,
            top_p=top_p,
            time_buffer=step_time,
            ttft_buffer=ttft_time,
            force_tokens=force_tokens.pop(0),
        ))
        aff_context.append(neg_response)
        neg_context.append(neg_response)
        mod_context.append(neg_response)
        total_ttft += ttft_time[-1]

        [mod_tokens], [mod_response] = get('tokens', 'nodes')(workflow.step([{
            'header': ('assistant', 'moderator'),
            'prefill': '',
            'parent_ids': [n['id'] for n in mod_context]
        }],
            temperature=temperature,
            top_p=top_p,
            time_buffer=step_time,
            ttft_buffer=ttft_time,
            force_tokens=force_tokens.pop(0),
        ))
        mod_context.append(mod_response)
        total_ttft += ttft_time[-1]

    [final_prompt] = workflow.insert([
        {'messages': [
            {'role': 'user', 'content': (
                'Please summarize your reasons and give the final answer that you think is correct. '
                'Now please output your answer in JSON format, with the format as follows: {{"Reasoning": "", "Answer": ""}}. '
                'Please strictly output in JSON format, do not output irrelevant content.'
            )}
        ], 'parent_ids': [n['id'] for n in mod_context]}
    ], time_buffer=insert_time)
    mod_context.append(final_prompt)
    [final_tokens], [final_node] = get('tokens', 'nodes')(workflow.step(
        [{'header': ('assistant', ''), 'prefill': '', 'parent_ids': [n['id'] for n in mod_context]}],
        temperature=temperature,
        top_p=top_p,
        time_buffer=step_time,
        ttft_buffer=ttft_time,
        force_tokens=force_tokens[-1],
    ))
    mod_context.append(final_node)
    total_ttft += insert_time[-1] + ttft_time[-1]

    return {
        'wall_time': time.time() - s,
        'cuda_time': sum(insert_time) + sum(step_time),
        'ttft': total_ttft,
    }

def benchmark(*args, **kwargs):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline_res = baseline(*args, **kwargs)
    cached_res = cached(*args, **kwargs, force_tokens=baseline_res['force_tokens'])
    torch.cuda.empty_cache()
    return baseline_res, cached_res

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = str(find_free_port())

workflow = Workflow.build(
    ckpt_dir='/scratch4/jeisner1/tjbai/llama_8b',
    tokenizer_path='/scratch4/jeisner1/tjbai/llama_8b/tokenizer.model',
    max_seq_len=8*8192,
    max_batch_size=1,
    model_parallel_size=1,
    max_nodes=20,
    use_lora=True,
    lora_rank=64,
    lora_alpha=32,
    lora_dropout=0.05
)
workflow.model.eval()

problems = load_math_problems('/home/tbai4/llama3/data/MATH', split='val')[:30]

for problem in problems[:3]:
    benchmark(workflow, problem)

results = {'baseline': [], 'cached': []}
for problem in tqdm(problems):
    baseline_res, cached_res = benchmark(workflow, problem)
    results['baseline'].append(baseline_res)
    results['cached'].append(cached_res)

    with open('/home/tbai4/llama3/dumps/mad/perf.json', 'w') as f:
        json.dump(results, f)
