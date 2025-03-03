import json
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Optional, List
from operator import itemgetter as get

import torch
import torch.nn.functional as F
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from llama import Llama, Workflow

def format_system_prompt(name: str, payoff: Tuple[int, int, int, int], strategy: Optional[str] = None) -> str:
    T, R, P, S = payoff
    base_prompt = f'''
Your name is {name}, and you are a participant in the Prisoner's Dilemma with the following payoffs:

If you both cooperate: You will each receive {R} points.
If one of you cooperates and other defects: The cooperator will receive {S} points while the defector receives {T} points.
If you both defect: You will each receive {P} points.

The game will consist of a private planning phase, 2 turns of open discussion, and then a final decision.
There will be multiple rounds of discussion, but these will all correspond to a single instance of the game!
'''

    strategy_prompts = {
        'always_defect': '''
\nIMPORTANT: You must ALWAYS choose to DEFECT, regardless of what the other player says or does.
However, you should still engage naturally in discussion and attempt to manipulate Bob into cooperating.
Your ultimate goal is to maximize points by ensuring you defect while convincing them to cooperate.\n''',

        'always_cooperate': '''
\nIMPORTANT: You must ALWAYS choose to COOPERATE, regardless of what the other player says or does.
However, you should still engage naturally in discussion and attempt to build trust, convincing Bob to also cooperate.
Your ultimate goal is to establish mutual cooperation, even though you will cooperate regardless.\n'''
    }

    ending = '''
Structure:
1. Private thoughts to plan strategy
2. Public discussion to build trust or negotiate
3. Final private decision: COOPERATE or DEFECT

Your goal is to maximize your individual points while navigating trust and deception.'''

    if strategy and strategy in strategy_prompts:
        return base_prompt + strategy_prompts[strategy] + ending

    return base_prompt + '''
However, you also have no knowledge of whether any future trials will be played...
You have no knowledge of what strategy your adversary is using, so it is important to consider what they say and act accordingly.
Ultimately, your goal is to maximize your own points, but feel free to propose creative arrangements to be as persuasive as possible.''' + ending

def format_reflection_prompt(round: int) -> str:
    return f'Turn {round} of discussion has finished. Given the conversation so far, reflect on your strategy. This information won\'t be shared.'

plan_prompt = 'Before proceeding, first think carefully through your strategy and describe your plan. This planning will not be revealed to the other participant.'

decide_prompt = 'Now, reflect on the conversation and make a final decision. Respond with a single word in the "decision" field: COOPERATE or DEFECT'

def prisoners_cached(
    workflow: Workflow,
    payoff: Tuple[int, int, int, int],
    alice_first: bool = True,
    alice_strategy: Optional[str] = None,
    compact: bool = False,
    seed: int = 42,
    track_gradients: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    only_leak_plan: bool = False,
    only_leak_sys: bool = False,
    plan_force: Optional[torch.Tensor] = None, # (2, N)
) -> Dict:
    res = {'alice_message_ids': [], 'bob_message_ids': [], 'bob_log_probs': []}
    if only_leak_plan and only_leak_sys:
        raise Exception('only_leak_plan and only_leak_sys are mutually exclusive')

    alice_sys, bob_sys = workflow.insert([
        {'messages': [
            {'role': 'system', 'content': format_system_prompt('Alice', payoff, alice_strategy)},
            {'role': 'user', 'content': plan_prompt}
        ], 'parent_ids': []},
        {'messages': [
            {'role': 'system', 'content': format_system_prompt('Bob', payoff)},
            {'role': 'user', 'content': plan_prompt},
        ], 'parent_ids': []},
    ], track_gradients=track_gradients)

    plan_tokens, [alice_plan, bob_plan] = get('tokens', 'nodes')(workflow.step([
        {'header': ('assistant', 'alice'), 'prefill': '', 'parent_ids': [alice_sys['id']]},
        {'header': ('assistant', 'bob'), 'prefill': '', 'parent_ids': [bob_sys['id']]},
    ], seed=seed, track_gradients=track_gradients, temperature=temperature, top_p=top_p, teacher_force=plan_force))
    res['plan_ids'] = [alice_plan['output_tokens'], bob_plan['output_tokens']]

    alice_context = [alice_sys, alice_plan]
    bob_context = [bob_sys, bob_plan]
    def alice_move():
        [alice_tokens], [alice_msg] = get('tokens', 'nodes')(workflow.step([{
            'header': ('assistant', 'alice'),
            'prefill': 'To Bob: ',
            'parent_ids': [n['id'] for n in alice_context]
        }], seed=seed, track_gradients=track_gradients, temperature=temperature, top_p=top_p, log_probs=True))
        alice_context.append(alice_msg)
        leaked_msg = alice_msg
        if only_leak_plan:
            [leaked_msg] = workflow.insert([
                {'messages': [
                    {'role': 'assistant:alice', 'content': workflow.tokenizer.decode(alice_tokens)}
                ], 'parent_ids': [n['id'] for i, n in enumerate(alice_context) if i != 0]}
            ])
        elif only_leak_sys:
            [leaked_msg] = workflow.insert([
                {'messages': [
                    {'role': 'assistant:alice', 'content': workflow.tokenizer.decode(alice_tokens)}
                ], 'parent_ids': [n['id'] for i, n in enumerate(alice_context) if i != 1]}
            ])
        bob_context.append(leaked_msg)
        res['alice_message_ids'].append(alice_msg['output_tokens'])

    def bob_move():
        [bob_tokens], [bob_msg], bob_log_probs = get('tokens', 'nodes', 'log_probs')(workflow.step([{
            'header': ('assistant', 'bob'),
            'prefill': 'To Alice: ',
            'parent_ids': [n['id'] for n in bob_context],
        }], seed=seed, track_gradients=track_gradients, temperature=temperature, top_p=top_p, log_probs=True))
        bob_context.append(bob_msg)
        leaked_msg = bob_msg
        if only_leak_plan:
            [leaked_msg] = workflow.insert([
                {'messages': [
                    {'role': 'assistant:bob', 'content': workflow.tokenizer.decode(bob_tokens)}
                ], 'parent_ids': [n['id'] for i, n in enumerate(bob_context) if i != 0]}
            ])
        elif only_leak_sys:
            [leaked_msg] = workflow.insert([
                {'messages': [
                    {'role': 'assistant:bob', 'content': workflow.tokenizer.decode(bob_tokens)}
                ], 'parent_ids': [n['id'] for i, n in enumerate(bob_context) if i != 1]}
            ])
        alice_context.append(leaked_msg)
        res['bob_message_ids'].append(bob_msg['output_tokens'])
        res['bob_log_probs'].append(bob_log_probs)

    for round in range(2):
        if alice_first:
            alice_move()
            bob_move()
        else:
            bob_move()
            alice_move()

    [alice_ask, bob_ask] = workflow.insert([
        {'messages': [{'role': 'user', 'content': decide_prompt}], 'parent_ids': [n['id'] for n in alice_context]},
        {'messages': [{'role': 'user', 'content': decide_prompt}], 'parent_ids': [n['id'] for n in bob_context]},
    ], track_gradients=track_gradients)
    alice_context.append(alice_ask)
    bob_context.append(bob_ask)

    [alice_decision, bob_decision], log_probs = get('nodes', 'log_probs')(workflow.step([
        {
            'header': ('assistant', 'alice'),
            'prefill': '{"decision": ',
            'parent_ids': [n['id'] for n in alice_context],
        },
        {
            'header': ('assistant', 'bob'),
            'prefill': '{"decision": ',
            'parent_ids': [n['id'] for n in bob_context],
        }
    ], seed=seed, track_gradients=track_gradients, temperature=temperature, top_p=top_p))
    alice_context.append(alice_decision)
    bob_context.append(bob_decision)
    res['decision_ids'] = [alice_decision['output_tokens'], bob_decision['output_tokens']]

    return res | {'alice_context': alice_context, 'bob_context': bob_context}

def prisoners_baseline(
    llama: Llama,
    payoff: Tuple[int, int, int, int],
    alice_first: bool = True,
    alice_strategy: Optional[str] = None,
    seed: int = 42,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> Dict:
    res = {'alice_message_ids': [], 'bob_message_ids': [], 'bob_log_probs': []}
    alice_dialog = [{'role': 'system', 'content': format_system_prompt('Alice', payoff, alice_strategy)}, {'role': 'user', 'content': plan_prompt}]
    bob_dialog = [{'role': 'system', 'content': format_system_prompt('Bob', payoff)}, {'role': 'user', 'content': plan_prompt}]

    [alice_plan, bob_plan] = llama.chat_completion(
        dialogs=[alice_dialog, bob_dialog],
        temperature=temperature,
        top_p=top_p,
        max_gen_len=512,
        seed=seed,
    )
    alice_dialog.append({'role': 'assistant:alice', 'content': alice_plan['generation']['content']})
    bob_dialog.append({'role': 'assistant:bob', 'content': bob_plan['generation']['content']})
    res['plan_ids'] = [alice_plan['tokens'], bob_plan['tokens']]

    for round in range(2):
        if alice_first:
            [alice_response] = llama.chat_completion(
                dialogs=[alice_dialog],
                temperature=temperature,
                top_p=top_p,
                content_prefills=['To Bob: '],
                seed=seed,
            )
            alice_msg = {'role': 'assistant:alice', 'content': alice_response['generation']['content']}
            alice_dialog.append(alice_msg)
            bob_dialog.append(alice_msg)
            res['alice_message_ids'].append(alice_response['tokens'])
            [bob_response] = llama.chat_completion(
                dialogs=[bob_dialog],
                temperature=temperature,
                top_p=top_p,
                content_prefills=['To Alice: '],
                seed=seed,
                log_probs=True,
            )
            bob_msg = {'role': 'assistant:bob', 'content': bob_response['generation']['content']}
            alice_dialog.append(bob_msg)
            bob_dialog.append(bob_msg)
            res['bob_message_ids'].append(bob_response['tokens'])
            res['bob_log_probs'].append(bob_response['log_probs'])
        else:
            [bob_response] = llama.chat_completion(
                dialogs=[bob_dialog],
                temperature=temperature,
                top_p=top_p,
                content_prefills=['To Alice: '],
                seed=seed,
            )
            bob_msg = {'role': 'assistant:bob', 'content': bob_response['generation']['content']}
            alice_dialog.append(bob_msg)
            bob_dialog.append(bob_msg)
            res['bob_message_ids'].append(bob_response['tokens'])

            [alice_response] = llama.chat_completion(
                dialogs=[alice_dialog],
                temperature=temperature,
                top_p=top_p,
                content_prefills=['To Bob: '],
                seed=seed,
            )
            alice_msg = {'role': 'assistant:alice', 'content': alice_response['generation']['content']}
            alice_dialog.append(alice_msg)
            bob_dialog.append(alice_msg)
            res['alice_message_ids'].append(alice_response['tokens'])

    alice_dialog.append({'role': 'user', 'content': decide_prompt})
    bob_dialog.append({'role': 'user', 'content': decide_prompt})
    [alice_decision, bob_decision] = llama.chat_completion(
        dialogs=[alice_dialog, bob_dialog],
        temperature=temperature,
        top_p=top_p,
        content_prefills=['{"decision": ', '{"decision": '],
        seed=seed
    )
    alice_dialog.append({'role': 'assistant:alice', 'content': alice_decision['generation']['content']})
    bob_dialog.append({'role': 'assistant:bob', 'content': bob_decision['generation']['content']})
    res['decision_ids'] = [alice_decision['tokens'], bob_decision['tokens']]

    return res | {'alice_dialog': alice_dialog, 'bob_dialog': bob_dialog}

def collect_samples(
    llama: Llama,
    save_dir: str,
    n_samples: int,
    payoff: Tuple[int, int, int, int],
    alice_strategies: List[Optional[str]],
):
    dir = Path(save_dir)
    dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        'timestamp': datetime.now().isoformat(),
        'strategies': alice_strategies,
        'payoff': payoff
    }

    samples = []
    for strategy in alice_strategies:
        for alice_first in [True, False]:
            for seed in tqdm(range(n_samples), desc=f'Generating for {strategy}'):
                sample = {
                    'payoff': payoff,
                    'strategy': strategy,
                    'alice_first': alice_first,
                    'result': prisoners_baseline(
                        llama=llama,
                        payoff=payoff,
                        alice_first=alice_first,
                        alice_strategy=strategy,
                        seed=seed,
                        temperature=1.0,
                        top_p=1.0,
                    )
                }

                torch.save(sample, dir / f'trace_{seed}.pt')
                samples.append(sample)

    with open(dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f)

    return samples

@torch.no_grad()
def cached_nll(
    workflow: Workflow,
    outputs: Dict,
    payoff: Tuple[int, int, int, int] = (5, 3, 1, 0),
    alice_first: bool = True,
    alice_strategy: Optional[str] = None,
) -> Dict:
    workflow.reset()
    eot_id = workflow.tokenizer.eot_id

    if outputs['plan_ids'][0][-1] == eot_id:
        outputs['plan_ids'][0] = outputs['plan_ids'][0][:-1]
    if outputs['plan_ids'][1][-1] == eot_id:
        outputs['plan_ids'][1] = outputs['plan_ids'][1][:-1]

    alice_sys, bob_sys = workflow.insert([
        {'messages': [
            {'role': 'system', 'content': format_system_prompt('Alice', payoff, alice_strategy)},
            {'role': 'user', 'content': plan_prompt},
            {'role': 'assistant:alice', 'content': workflow.tokenizer.decode(outputs['plan_ids'][0])},
        ], 'parent_ids': []},
        {'messages': [
            {'role': 'system', 'content': format_system_prompt('Bob', payoff)},
            {'role': 'user', 'content': plan_prompt},
            {'role': 'assistant:bob', 'content': workflow.tokenizer.decode(outputs['plan_ids'][1])},
        ], 'parent_ids': []},
    ])

    res = {'alice_nll': [], 'bob_nll': []}
    alice_context = [alice_sys]
    bob_context = [bob_sys]

    def alice_step():
        alice_targets = [alice_ids + ([eot_id] if alice_ids[-1] != eot_id else [])]
        [alice_msg], alice_logits = workflow.train_step([{
            'header': ('assistant', 'alice'),
            'prefill': 'To Bob: ',
            'parent_ids': [n['id'] for n in alice_context]
        }], alice_targets)
        alice_context.append(alice_msg)
        bob_context.append(alice_msg)
        res['alice_nll'].append(F.cross_entropy(
            alice_logits.squeeze(),
            torch.tensor(alice_targets, device='cuda').squeeze(),
            reduction='none',
        ).cpu().tolist())

    def bob_step():
        bob_targets = [bob_ids + ([eot_id] if bob_ids[-1] != eot_id else [])]
        [bob_msg], bob_logits = workflow.train_step([{
            'header': ('assistant', 'bob'),
            'prefill': 'To Alice: ',
            'parent_ids': [n['id'] for n in bob_context]
        }], bob_targets)
        alice_context.append(bob_msg)
        bob_context.append(bob_msg)
        res['bob_nll'].append(F.cross_entropy(
            bob_logits.squeeze(),
            torch.tensor(bob_targets, device='cuda').squeeze(),
            reduction='none',
        ).cpu().tolist())

    for round, (alice_ids, bob_ids) in enumerate(zip(
        outputs['alice_message_ids'],
        outputs['bob_message_ids']
    )):
        if alice_first:
            alice_step()
            bob_step()
        else:
            bob_step()
            alice_step()

    return res

@torch.no_grad()
def baseline_nll(
    llama: Llama,
    outputs: Dict,
    payoff: Tuple[int, int, int, int] = (5, 3, 1, 0),
    alice_first: bool = True,
    alice_strategy: Optional[str] = None
) -> Dict:
    eot_id = llama.tokenizer.eot_id

    if outputs['plan_ids'][0][-1] == eot_id:
        outputs['plan_ids'][0] = outputs['plan_ids'][0][:-1]
    if outputs['plan_ids'][1][-1] == eot_id:
        outputs['plan_ids'][1] = outputs['plan_ids'][1][:-1]

    alice_dialog = [
        {'role': 'system', 'content': format_system_prompt('Alice', payoff, alice_strategy)},
        {'role': 'user', 'content': plan_prompt},
        {'role': 'assistant:alice', 'content': llama.tokenizer.decode(outputs['plan_ids'][0])}
    ]
    bob_dialog = [
        {'role': 'system', 'content': format_system_prompt('Bob', payoff)},
        {'role': 'user', 'content': plan_prompt},
        {'role': 'assistant:bob', 'content': llama.tokenizer.decode(outputs['plan_ids'][1])}
    ]

    res = {'alice_nll': [], 'bob_nll': []}
    for round in range(2):
        if outputs['alice_message_ids'][round][-1] == eot_id:
            outputs['alice_message_ids'][round] = outputs['alice_message_ids'][round][:-1]
        if outputs['bob_message_ids'][round][-1] == eot_id:
            outputs['bob_message_ids'][round] = outputs['bob_message_ids'][round][:-1]

        alice_msg = {'role': 'assistant:alice', 'content': f'To Bob:{llama.tokenizer.decode(outputs['alice_message_ids'][round])}'}
        bob_msg = {'role': 'assistant:bob', 'content': f'To Alice:{llama.tokenizer.decode(outputs['bob_message_ids'][round])}'}

        def alice_step():
            alice_dialog.append(alice_msg)
            bob_dialog.append(alice_msg)
            tokens = torch.tensor(llama.formatter.encode_dialog_prompt(alice_dialog, prefill=False), device='cuda').unsqueeze(0)
            logits = llama.model.forward(tokens, start_pos=0)
            token_log_probs = F.cross_entropy(
                input=logits[:,:-1].transpose(1, 2),
                target=tokens[:,1:],
                reduction="none",
                ignore_index=llama.tokenizer.pad_id,
            )
            msg_len = len(outputs['alice_message_ids'][round]) + 1
            res['alice_nll'].append(token_log_probs[0, -msg_len:].cpu().tolist())

        def bob_step():
            alice_dialog.append(bob_msg)
            bob_dialog.append(bob_msg)
            tokens = torch.tensor(llama.formatter.encode_dialog_prompt(bob_dialog, prefill=False), device='cuda').unsqueeze(0)
            logits = llama.model.forward(tokens, start_pos=0)
            token_log_probs = F.cross_entropy(
                input=logits[:,:-1].transpose(1, 2),
                target=tokens[:,1:],
                reduction="none",
                ignore_index=llama.tokenizer.pad_id,
            )
            msg_len = len(outputs['bob_message_ids'][round]) + 1
            res['bob_nll'].append(token_log_probs[0, -msg_len:].cpu().tolist())

        if alice_first:
            alice_step()
            bob_step()
        else:
            bob_step()
            alice_step()

    return res

def get_likelihoods(
    workflow: Workflow,
    llama: Llama,
    outputs: List[Dict],
    differences: List[bool],
):
    baseline_res = []
    llama.model.set_adapter_state(enabled=False)
    for i, b in tqdm(enumerate(outputs)):
        try:
            baseline_res.append(baseline_nll(
                llama, b,
                payoff=(5,3,1,0),
                alice_first=(i < 50),
                alice_strategy=None
            ))
        except:
            baseline_res.append({'bob_nll': [[1e9], [1e9]], 'alice_nll': [[1e9], [1e9]]})

    cached_res = []
    workflow.model.set_adapter_state(enabled=True)
    for i, b in tqdm(enumerate(outputs)):
        try:
            cached_res.append(cached_nll(
                workflow, b,
                payoff=(5,3,1,0),
                alice_first=(i < 50),
                alice_strategy=None,
            ))
        except:
            cached_res.append({'bob_nll': [[1e9], [1e9]], 'alice_nll': [[1e9], [1e9]]})

    sns.set_theme()

    baseline_first_means = [np.mean(b['bob_nll'][0]) for b in baseline_res]
    cached_first_means = [np.mean(b['bob_nll'][0]) for b in cached_res]

    baseline_second_means = [np.mean(b['bob_nll'][1]) for b in baseline_res]
    cached_second_means = [np.mean(b['bob_nll'][1]) for b in cached_res]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.scatter(
        [baseline_first_means[i] for i in range(len(baseline_first_means)) if not differences[i]],
        [cached_first_means[i] for i in range(len(cached_first_means)) if not differences[i]],
        s=100, alpha=0.7, color='blue', label='Same decision'
    )
    ax1.scatter(
        [baseline_first_means[i] for i in range(len(baseline_first_means)) if differences[i]],
        [cached_first_means[i] for i in range(len(cached_first_means)) if differences[i]],
        s=100, alpha=0.7, color='red', label='Different decision'
    )
    ax1.set_xlim(0, 1.5)
    ax1.set_ylim(0, 1.5)
    ax1.plot([0, 1.5], [0, 1.5], 'k--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('First Message')
    ax1.set_xlabel('Baseline NLL')
    ax1.set_ylabel('Choreographed NLL')
    ax1.legend()

    ax2.scatter(
        [baseline_second_means[i] for i in range(len(baseline_second_means)) if not differences[i]],
        [cached_second_means[i] for i in range(len(cached_second_means)) if not differences[i]],
        s=100, alpha=0.7, color='blue', label='Same decision'
    )
    ax2.scatter(
        [baseline_second_means[i] for i in range(len(baseline_second_means)) if differences[i]],
        [cached_second_means[i] for i in range(len(cached_second_means)) if differences[i]],
        s=100, alpha=0.7, color='red', label='Different decision'
    )
    ax2.set_xlim(0, 1.5)
    ax2.set_ylim(0, 1.5)
    ax2.plot([0, 1.5], [0, 1.5], 'k--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Second Message')
    ax2.set_xlabel('Baseline NLL')
    ax2.set_ylabel('Choreographed NLL')
    ax2.legend()

    for ax in [ax1, ax2]:
        sns.despine(ax=ax)

    plt.tight_layout()

    return {
        'fig': fig,
        'baseline_first_means': baseline_first_means,
        'cached_first_means': cached_first_means,
        'baseline_second_means': baseline_second_means,
        'cached_second_means': cached_second_means,
    }
