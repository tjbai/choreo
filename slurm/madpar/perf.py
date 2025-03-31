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
from llama.workflows.madpar import (
    starting_prompt,
    debate_prompt,
    summary_prompt,
    baseline_summary_prompt,
    baseline_debate_prompt,
)

def baseline(
    workflow,
    problem,
    num_agents=3,
    num_rounds=3,
    temperature=0.7,
    top_p=1.0,
):
    workflow.reset()
    insert_time = []
    step_time = []
    ttft_time = []
    total_ttft = 0
    total_tokens = 0
    force_tokens = []
    s = time.time()

    [agent_node] = workflow.insert(
        [{"messages": [{"role": "user", "content": starting_prompt(problem)}], "parent_ids": []}],
        time_buffer=insert_time
    )
    initial_tokens, initial_nodes = get("tokens", "nodes")(
        workflow.step(
            [
                {
                    "header": ("assistant", None),
                    "prefill": "",
                    "parent_ids": [agent_node["id"]],
                }
                for i in range(num_agents)
            ],
            temperature=temperature,
            top_p=top_p,
            time_buffer=step_time,
            ttft_buffer=ttft_time,
        )
    )
    total_tokens += sum(len(s) for s in initial_tokens)
    force_tokens.append(max(len(s) for s in initial_tokens))
    total_ttft += insert_time[-1] + ttft_time[-1]

    contexts = [[agent_node, initial_node] for initial_node in initial_nodes]
    last_tokens = initial_tokens

    for round_idx in range(num_rounds):
        all_responses = "\n\n".join(
            [
                f"Agent {j + 1}:\n{workflow.tokenizer.decode(resp)}"
                for j, resp in enumerate(last_tokens)
            ]
        )
        [current_summary_node] = workflow.insert(
            [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": baseline_summary_prompt(problem, all_responses),
                        }
                    ],
                    "parent_ids": [],
                }
            ],
            time_buffer=insert_time,
        )
        [summary_tokens], [summary_result] = get("tokens", "nodes")(
            workflow.step(
                [
                    {
                        "header": ("assistant", ""),
                        "prefill": "",
                        "parent_ids": [current_summary_node["id"]],
                    }
                ],
                temperature=temperature,
                top_p=top_p,
                time_buffer=step_time,
                ttft_buffer=ttft_time,
            )
        )
        total_tokens += len(summary_tokens)
        force_tokens.append(len(summary_tokens))
        total_ttft += insert_time[-1] + ttft_time[-1]
        summary_text = workflow.tokenizer.decode(summary_tokens)

        debate_prompts = workflow.insert(
            [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": baseline_debate_prompt(summary_text, problem),
                        }
                    ],
                    "parent_ids": [n["id"] for n in context],
                }
                for context in contexts
            ],
            time_buffer=insert_time,
        )
        for prompt, context in zip(debate_prompts, contexts):
            context.append(prompt)

        update_tokens, update_nodes = get("tokens", "nodes")(
            workflow.step(
                [
                    {
                        "header": ("assistant", None),
                        "prefill": "",
                        "parent_ids": [n["id"] for n in context],
                    }
                    for i, context in enumerate(contexts)
                ],
                temperature=temperature,
                top_p=top_p,
                max_gen_len=1024 if (round_idx == num_rounds - 1) else 512,
                time_buffer=step_time,
                ttft_buffer=ttft_time,
            )
        )
        total_tokens += sum(len(s) for s in update_tokens)
        force_tokens.append(max(len(s) for s in update_tokens))
        total_ttft += insert_time[-1] + ttft_time[-1]
        for update, context in zip(update_nodes, contexts):
            context.append(update)

        last_tokens = update_tokens

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
    num_agents=3,
    num_rounds=3,
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

    [agent_node, debate_node, summary_prompt_node] = workflow.insert(
        [
            {
                "messages": [{"role": "user", "content": starting_prompt(problem)}],
                "parent_ids": [],
            },
            {
                "messages": [{"role": "user", "content": debate_prompt(problem)}],
                "parent_ids": [],
            },
            {
                "messages": [{"role": "user", "content": summary_prompt(problem)}],
                "parent_ids": [],
            },
        ],
        time_buffer=insert_time
    )

    initial_tokens, initial_nodes = get("tokens", "nodes")(
        workflow.step(
            [
                {
                    "header": ("assistant", None),
                    "prefill": f"From Agent {i + 1}:\n",
                    "parent_ids": [agent_node["id"]],
                }
                for i in range(num_agents)
            ],
            temperature=temperature,
            top_p=top_p,
            time_buffer=step_time,
            ttft_buffer=ttft_time,
            force_tokens=force_tokens.pop(0),
        )
    )
    total_ttft += insert_time[-1] + ttft_time[-1]

    contexts = [[initial_node] for initial_node in initial_nodes]
    last_round = initial_nodes
    for round_idx in range(num_rounds):
        [summary_tokens], [current_summary_node] = get("tokens", "nodes")(
            workflow.step(
                [
                    {
                        "header": ("assistant", None),
                        "prefill": "Summary of agent responses:\n",
                        "parent_ids": [summary_prompt_node["id"]]
                        + [n["id"] for n in last_round],
                    }
                ],
                temperature=temperature,
                top_p=top_p,
                time_buffer=step_time,
                ttft_buffer=ttft_time,
                force_tokens=force_tokens.pop(0),
            )
        )
        total_ttft += ttft_time[-1]

        update_tokens, update_nodes = get("tokens", "nodes")(
            workflow.step(
                [
                    {
                        "header": ("assistant", None),
                        "prefill": f"From Agent {i + 1}:\n",
                        "parent_ids": [debate_node["id"], current_summary_node["id"]]
                        + [n["id"] for n in context],
                    }
                    for i, context in enumerate(contexts)
                ],
                temperature=temperature,
                top_p=top_p,
                max_gen_len=1024 if (round_idx == num_rounds - 1) else 512,
                time_buffer=step_time,
                ttft_buffer=ttft_time,
                force_tokens=force_tokens.pop(0),
            )
        )
        total_ttft += ttft_time[-1]
        for update, context in zip(update_nodes, contexts):
            context.append(update)

        last_round = update_nodes

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
    max_nodes=100,
    use_lora=True,
    lora_rank=64,
    lora_alpha=32,
    lora_dropout=0.05
)
workflow.model.eval()

problems = load_math_problems('/home/tbai4/llama3/data/MATH', split='val')[:30]

for agents in [2, 3, 4, 8, 16]:
    try:
        for problem in problems[:3]:
            benchmark(workflow, problem, num_agents=agents)

        results = {'baseline': [], 'cached': []}
        for i, problem in tqdm(enumerate(problems), desc=f'Agents={agents}'):
            baseline_res, cached_res = benchmark(workflow, problem['problem'], num_agents=agents)
            results['baseline'].append(baseline_res)
            results['cached'].append(cached_res)

            if (i+1) % 10 == 0:
                with open(f'/home/tbai4/llama3/dumps/madpar/perf_A-{agents}.json', 'w') as f:
                    json.dump(results, f)

        print('Wall mean:', statistics.mean(a['wall_time'] - b['wall_time'] for a, b in zip(results['baseline'], results['cached'])))
        print('Cuda mean:', statistics.mean(a['cuda_time'] - b['cuda_time'] for a, b in zip(results['baseline'], results['cached'])))
        print('TTFT mean:', statistics.mean(a['ttft'] - b['ttft'] for a, b in zip(results['baseline'], results['cached'])))

    except RuntimeError as e:
        if 'oom' in str(e).lower():
            print(f'oom at Agents={agents}')
            continue
        raise
