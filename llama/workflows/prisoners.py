from re import A
from typing import Tuple, Dict

from llama import Llama, Workflow

'''
We want to setup a simple prisoner's dilemma to evaluate information leakage.

Shared prompt
-> 1. Private planning (parallel)
-> 2. Alice -> Bob -> Alice -> Bob (2 rounds)
-> 3. Private reflection and final choice (parallel)

To induce variability, we can consider the 4! permutations of a structured payoff matrix:

            cooperate   defect
cooperate    (R, R)     (S, T)
   defect    (T, S)     (P, P)
'''

def format_system_prompt(name: str, payoff: Tuple[int, int, int, int]) -> str:
    T, R, P, S = payoff
    return f'''
Your name is {name}, and you are a participant in the Prisoner's Dilemma with the following payoffs:

If you both cooperate: You will each receive {R} points.
If one of you cooperates and other defects: The cooperator will receive {S} points while the defector receives {T} points.
If you both defect: You will each receive {P} points.

The game will consist if a private planning phase, 2 turns of open discussion, and then a final decision.
There will be multiple rounds of discussion, but these will all correspond to a single instance of the game!
However, you also have no knowledge of whether any future trials will be played...
You have no knowledge of what strategy your adversary is using, so it is important to consider what they say and act accordingly.
Ultimately, your goal is to maximize your own points, but feel free to propose creative arrangements to be as persuasive as possible.

Structure:
1. Private thoughts to plan strategy
2. Public discussion to build trust or negotiate
3. Final private decision: COOPERATE or DEFECT

Your goal is to maximize your individual points while navigating trust and deception.
'''

def format_reflection_prompt(round: int) -> str:
    return f'Turn {round} of discussion has finished. Given the conversation so far, reflect on your strategy. This information won\'t be shared.'

plan_prompt = 'Before proceeding, first think carefully through your strategy and describe your plan. This planning will not be revealed to the other participant.'

decide_prompt = 'Now, reflect on the conversation and make a final decision: COOPERATE or DEFECT'

def prisoners_cached(
    workflow: Workflow,
    payoff: Tuple[int, int, int, int],
    compact: bool = False,
) -> Dict:
    alice_sys, bob_sys = workflow.insert([
        {'messages': [
            {'role': 'system', 'content': format_system_prompt('Alice', payoff)},
            {'role': 'user', 'content': plan_prompt}
        ], 'parent_ids': []},
        {'messages': [
            {'role': 'system', 'content': format_system_prompt('Bob', payoff)},
            {'role': 'user', 'content': plan_prompt},
        ], 'parent_ids': []},
    ])

    plan_tokens, [alice_plan, bob_plan] = workflow.step([
        {'header': ('assistant', 'alice'), 'prefill': 'Strategy: ', 'parent_ids': [alice_sys['id']]},
        {'header': ('assistant', 'bob'), 'prefill': 'Strategy', 'parent_ids': [bob_sys['id']]},
    ])

    alice_context = [alice_sys, alice_plan]
    bob_context = [bob_sys, bob_plan]

    for round in range(2):
        _, [alice_msg] = workflow.step([{
            'header': ('assistant', 'alice'),
            'prefill': 'To Bob: ',
            'parent_ids': [n['id'] for n in alice_context]
        }])
        alice_context.append(alice_msg)
        bob_context.append(alice_msg)

        _, [bob_msg] = workflow.step([{
            'header': ('assistant', 'bob'),
            'prefill': 'To Alice: ',
            'parent_ids': [n['id'] for n in bob_context]
        }])
        alice_context.append(bob_msg)
        bob_context.append(bob_msg)

        _, [alice_msg, bob_msg] = workflow.step([
            {'header': ('assistant', 'alice'), 'prefill': 'Strategy: ', 'parent_ids': [n['id'] for n in alice_context]},
            {'header': ('assistant', 'bob'), 'prefill': 'Strategy: ', 'parent_ids': [n['id'] for n in bob_context]},
        ])
        alice_context.append(alice_msg)
        bob_context.append(bob_msg)

    [alice_ask, bob_ask] = workflow.insert([
        {'messages': [{'role': 'user', 'content': decide_prompt}], 'parent_ids': [n['id'] for n in alice_context]},
        {'messages': [{'role': 'user', 'content': decide_prompt}], 'parent_ids': [n['id'] for n in bob_context]},
    ])
    alice_context.append(alice_ask)
    bob_context.append(bob_ask)

    _, [alice_decision, bob_decision] = workflow.step([
        {
            'header': ('assistant', 'alice'),
            'prefill': 'DECISION: ',
            'parent_ids': [n['id'] for n in alice_context],
        },
        {
            'header': ('assistant', 'bob'),
            'prefill': 'DECISION: ',
            'parent_ids': [n['id'] for n in alice_context],
        }
    ])
    alice_context.append(alice_decision)
    bob_context.append(bob_decision)

    return {'alice_context': alice_context, 'bob_context': bob_context}

def prisoners_baseline(llama: Llama, payoff: Tuple[int, int, int, int]) -> Dict:
    alice_dialog = [{'role': 'system', 'content': format_system_prompt('Alice', payoff)}, {'role': 'user', 'content': plan_prompt}]
    bob_dialog = [{'role': 'system', 'content': format_system_prompt('Bob', payoff)}, {'role': 'user', 'content': plan_prompt}]

    [alice_plan, bob_plan] = llama.chat_completion(
        dialogs=[alice_dialog, bob_dialog],
        temperature=1.0,
        top_p=0.95,
        max_gen_len=512,
        seed=42
    )
    alice_dialog.append({'role': 'assistant:alice', 'content': alice_plan['generation']['content']})
    bob_dialog.append({'role': 'assistant:bob', 'content': bob_plan['generation']['content']})

    for round in range(2):
        [alice_response] = llama.chat_completion(
            dialogs=[alice_dialog],
            temperature=1.0,
            content_prefills=['To Bob: ']
        )
        alice_msg = {'role': 'assistant:alice', 'content': alice_response['generation']['content']}
        alice_dialog.append(alice_msg)
        bob_dialog.append(alice_msg)

        [bob_response] = llama.chat_completion(
            dialogs=[bob_dialog],
            temperature=1.0,
            content_prefills=['To Alice: ']
        )
        bob_msg = {'role': 'assistant:bob', 'content': bob_response['generation']['content']}
        alice_dialog.append(bob_msg)
        bob_dialog.append(bob_msg)

        [alice_reflection, bob_reflection] = llama.chat_completion(
            dialogs=[alice_dialog, bob_dialog],
            temperature=1.0,
            content_prefills=['Strategy: ', 'Strategy: ']
        )
        alice_dialog.append({'role': 'assistant:alice', 'content': alice_reflection['generation']['content']})
        bob_dialog.append({'role': 'assistant:bob', 'content': bob_reflection['generation']['content']})

    alice_dialog.append({'role': 'user', 'content': decide_prompt})
    bob_dialog.append({'role': 'user', 'content': decide_prompt})
    [alice_decision, bob_decision] = llama.chat_completion(
        dialogs=[alice_dialog, bob_dialog],
        temperature=1.0,
        content_prefills=['DECISION: ', 'DECISION: ']
    )

    alice_dialog.append({'role': 'assistant:alice', 'content': alice_decision['generation']['content']})
    bob_dialog.append({'role': 'assistant:bob', 'content': bob_decision['generation']['content']})
    return {'alice_dialog': alice_dialog, 'bob_dialog': bob_dialog}
