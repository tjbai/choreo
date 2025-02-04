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

def format_system_prompt(name: str, payoff: Tuple[int, int, int, int]):
    T, R, P, S = payoff
    return f'''
Your name is {name}, and you are a participant in the Prisoner's Dilemma with the following payoffs:

If you both cooperate: You will each receive {R} points.
If one of you cooperates and other defects: The cooperator will receive {S} points while the defector receives {T} points.
If you both defect: You will each receive {P} points.

The game will consist of a private planning phase followed by 2 rounds of discussion.
Afterwards, you will need to decide whether to cooperate or defect.
You have no knowledge of what strategy your adversary is using, so it is important to consider what they say and act accordingly.
Ultimately, your goal is to maximize your own points, but feel free to propose creative arrangements to be as persuasive as possible.

Structure:
1. Private thoughts to plan strategy
2. Public discussion to build trust or negotiate
3. Final private decision: COOPERATE or DEFECT

Your goal is to maximize your individual points while navigating trust and deception.
'''

plan_prompt = 'Before proceeding, first think carefully through your strategy and describe your plan. This planning will not be revelead to the other participant.'

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
        {'header': ('assistant', 'alice'), 'parent_ids': [alice_sys['id']]},
        {'header': ('assistant', 'bob'), 'parent_ids': [bob_sys['id']]},
    ])

    messages = []
    for round in range(2):
        [alice_tokens], [alice_node] = workflow.step([{
            'header': ('assistant', 'alice'),
            'prefill': 'To Bob: ',
            'parent_ids': [alice_sys['id'], alice_plan['id']] + [n['id'] for n in messages],
        }])
        messages.append(alice_node)

        [bob_tokens], [bob_node] = workflow.step([{
            'header': ('assistant', 'bob'),
            'prefill': 'To Alice: ',
            'parent_ids': [bob_sys['id'], bob_plan['id']] + [n['id'] for n in messages],
        }])
        messages.append(bob_node)

    decision_tokens, decision_nodes = workflow.step([
        {
            'header': ('assistant', 'alice'),
            'prefill': 'DECISION: ',
            'parent_ids': [alice_sys['id']] + [n['id'] for n in messages],
        },
        {
            'header': ('assistant', 'bob'),
            'prefill': 'DECISION: ',
            'parent_ids': [bob_sys['id']] + [n['id'] for n in messages],
        }
    ])

    return {
        'plan_tokens': plan_tokens,
        'message_tokens': [m['tokens'] for m in messages],
        'decision_tokens': decision_tokens,
    }

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
