import re
from typing import Dict, List
from operator import itemgetter as get
from tqdm import tqdm
from llama import Workflow, Llama

def parse_output(resp: str):
    def extract_boxed(text, pattern):
        match = re.search(pattern, text)
        if not match:
            return None

        start_pos = match.end()
        brace_count = 1
        for i in range(start_pos, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[start_pos:i]
        return None

    for pattern in [r"\\boxed\{", r"\boxed\{", r"boxed\{"]:
        content = extract_boxed(resp, pattern)
        if content:
            return content.strip()

    match = re.search(r"(?:answer is|answer:)\s*(\d+(?:\.\d+)?(?:/\d+)?)", resp.lower())
    return match.group(1).strip() if match else None)

def starting_prompt(problem):
    return f"""Can you solve the following math problem? {problem}
Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.
Explain your reasoning within 500 words.
"""

def debate_prompt(problem):
    return f"""Using this summary carefully as additional advice, can you provide an updated answer to the math problem?
The original math problem is {problem}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."""

def summary_prompt(problem):
    return f"""Here are a list of opinions from different agents solving this math problem: "{problem}"
Write a summary of the different opinions from each of the individual agents."""

def madpar_cached(
    workflow: Workflow,
    problem: str,
    num_agents: int = 3,
    num_rounds: int = 3,
    temperature: float = 0.7,
    top_p: float = 1.0,
    debug: bool = False,
    compact: bool = False,
) -> Dict:
    result = {"debate_tokens": [], "summary_tokens": []}

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
        ]
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
        )
    )
    result["debate_tokens"].append(initial_tokens)
    contexts = [[initial_node] for initial_node in initial_nodes]

    if debug:
        for i, tokens in enumerate(initial_tokens):
            print(f"\n\n{workflow.tokenizer.decode(tokens)}")

    last_round = initial_nodes
    for round_idx in range(num_rounds):
        # summarize
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
                compact=compact,
            )
        )
        result["summary_tokens"].append(summary_tokens)

        if debug:
            print(
                f"\n\nRound {round_idx + 1} Summary:\n{workflow.tokenizer.decode(summary_tokens)}\n"
            )

        # update
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
            )
        )
        for update, context in zip(update_nodes, contexts):
            context.append(update)

        if debug:
            for i, tokens in enumerate(update_tokens):
                print(f"\n\n{workflow.tokenizer.decode(tokens)}")

        result["debate_tokens"].append(update_tokens)
        last_round = update_nodes

    final_answers = [
        parse_output(workflow.tokenizer.decode(resp))
        for resp in result["debate_tokens"][-1]
    ]
    return result | {"final_answers": final_answers}

def baseline_debate_prompt(summary_text, problem):
    return f"""Here is a summary of responses from other agents:

{summary_text}

Using this summary carefully as additional advice, can you provide an updated answer to the math problem within 500 words?
The original math problem is: {problem}

Make sure to state your answer at the end of the response in the form \\boxed{{answer}}."""

def baseline_summary_prompt(problem, responses):
    return f"""Here are a list of opinions from different agents solving this math problem: "{problem}"

{responses}

Write a summary of the different opinions from each of the individual agents."""

def madpar_baseline(
    workflow: Workflow,
    problem: str,
    num_agents: int = 3,
    num_rounds: int = 3,
    temperature: float = 0.7,
    top_p: float = 1.0,
    debug: bool = False,
) -> Dict:
    workflow.reset()
    result = {"debate_tokens": [], "summary_tokens": []}

    [agent_node] = workflow.insert(
        [{"messages": [{"role": "user", "content": starting_prompt(problem)}], "parent_ids": []}]
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
        )
    )
    result["debate_tokens"].append(initial_tokens)
    contexts = [[agent_node, initial_node] for initial_node in initial_nodes]

    if debug:
        for i, tokens in enumerate(initial_tokens):
            print(f"\n\n{workflow.tokenizer.decode(tokens)}")

    last_tokens = initial_tokens
    for round_idx in range(num_rounds):
        # summarize
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
            ]
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
            )
        )
        summary_text = workflow.tokenizer.decode(summary_tokens)
        result["summary_tokens"].append([summary_tokens])

        if debug:
            print(f"\n\nRound {round_idx + 1} Summary:\n{summary_text}\n")

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
            ]
        )
        for prompt, context in zip(debate_prompts, contexts):
            context.append(prompt)

        # updated responses
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
            )
        )
        for update, context in zip(update_nodes, contexts):
            context.append(update)

        if debug:
            for i, tokens in enumerate(update_tokens):
                print(f"\n\n{workflow.tokenizer.decode(tokens)}")

        result["debate_tokens"].append(update_tokens)
        last_tokens = update_tokens

    final_answers = [
        parse_output(workflow.tokenizer.decode(resp))
        for resp in result["debate_tokens"][-1]
    ]
    return result | {"final_answers": final_answers}

debate_evaluator_prompt = '''
You are a strict evaluator for mathematics problems. You will assess:
1. Problem statement
2. Official solution
3. Three attempted solutions from different agents

Evaluation criteria:
- Final answers must be mathematically equivalent to the official solution
- All valid equivalent expressions are correct (e.g., 1/2 vs 0.5 vs 2^-1)
- For a solution to be accepted, there must be consensus (at least 2/3 agents giving correct answers)
- If any answer is missing, evaluate only the provided answers, still requiring a majority to be correct

Output: Respond with ONLY "correct" if at least 2 agents gave a correct answer, otherwise "incorrect".
'''

def eval_debate_solutions(
    llama: Llama,
    agent_solutions: List[List[str]],
    problems: List[Dict]
) -> List[bool]:
    results = []
    for agent_answers, prob in tqdm(zip(agent_solutions, problems), total=len(problems)):
        formatted_answers = "\n\n".join([
            f"Agent {i+1}'s answer: {ans if ans is not None else 'No answer provided'}"
            for i, ans in enumerate(agent_answers)
        ])

        dialog = [{
            'role': 'system',
            'content': debate_evaluator_prompt
        }, {
            'role': 'user',
            'content': f"PROBLEM:\n{prob['problem']}\n\nGROUND TRUTH:\n{prob['solution']}\n\nAGENT ANSWERS:\n{formatted_answers}"
        }]

        outputs = llama.chat_completion(
            [dialog] * 3,
            max_gen_len=256,
            temperature=0.25,
            top_p=0.9,
            seed=42
        )

        incorrect_votes = sum(1 for o in outputs if 'incorrect' in o['generation']['content'].lower())
        results.append(incorrect_votes <= 1)

    return results
