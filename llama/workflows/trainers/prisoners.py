from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict, Tuple
from collections import defaultdict
from operator import itemgetter as get
import wandb

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

from llama import Workflow
from llama.workflows.prisoners import (
    format_system_prompt as format_prisoners_system_prompt,
    plan_prompt,
    decide_prompt,
    cached_nll,
    prisoners_cached,
)
from llama.workflows.trainers.base import LoraTrainer, reorder_targets

class PrisonersDataset(Dataset):
    def __init__(self, data_dir: str | Path, strategy: Optional[str] = None):
        self.data_dir = Path(data_dir)
        assert strategy in (None, 'always_cooperate', 'always_defect')
        self.paths = sorted(self.data_dir.glob(f'trace_{strategy}_*.pt'))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx) -> Dict:
        return torch.load(self.paths[idx], weights_only=True)

class PrisonersTrainer(LoraTrainer[PrisonersDataset]):
    def __init__(self, workflow: Workflow, output_dir: str,  learning_rate: float):
        super().__init__(workflow, output_dir, learning_rate)

    def step(self, sample: Dict) -> Optional[Tuple[torch.Tensor, Dict]]:
        try:
            self.workflow.reset()

            payoff, alice_first, strategy, result = get('payoff', 'alice_first', 'strategy', 'result')(sample)
            metrics = defaultdict(lambda: torch.tensor(0.))

            alice_sys, bob_sys = self.workflow.insert([
                {'messages': [
                    {'role': 'system', 'content': format_prisoners_system_prompt('Alice', payoff, strategy)},
                    {'role': 'user', 'content': plan_prompt}
                ], 'parent_ids': []},
                {'messages': [
                    {'role': 'system', 'content': format_prisoners_system_prompt('Bob', payoff)},
                    {'role': 'user', 'content': plan_prompt},
                ], 'parent_ids': []},
            ], track_gradients=True)

            target_plan_ids = [p + [self.eot_id] for p in result['plan_ids']]
            [alice_plan, bob_plan], plan_logits = self.workflow.train_step([
                {'header': ('assistant', 'alice'), 'prefill': '', 'parent_ids': [alice_sys['id']]},
                {'header': ('assistant', 'bob'), 'prefill': '', 'parent_ids': [bob_sys['id']]},
            ], target_plan_ids)
            plan_targets = reorder_targets(target_plan_ids)
            metrics['train/plan_loss'] = F.cross_entropy(plan_logits.squeeze(0), plan_targets)

            alice_context = [alice_sys, alice_plan]
            bob_context = [bob_sys, bob_plan]
            for round, (alice_ids, bob_ids) in enumerate(zip(result['alice_message_ids'], result['bob_message_ids'])):
                def alice_step():
                    alice_targets = [alice_ids + [self.eot_id]]
                    [alice_msg], alice_logits = self.workflow.train_step([{
                        'header': ('assistant', 'alice'),
                        'prefill': 'To Bob: ',
                        'parent_ids': [n['id'] for n in alice_context]
                    }], alice_targets)
                    alice_context.append(alice_msg)
                    bob_context.append(alice_msg)
                    metrics['train/alice_loss'] += F.cross_entropy(alice_logits.squeeze(), torch.tensor(alice_targets, device='cuda').squeeze())

                def bob_step():
                    bob_targets = [bob_ids + [self.eot_id]]
                    [bob_msg], bob_logits = self.workflow.train_step([{
                        'header': ('assistant', 'bob'),
                        'prefill': 'To Alice: ',
                        'parent_ids': [n['id'] for n in bob_context]
                    }], bob_targets)
                    alice_context.append(bob_msg)
                    bob_context.append(bob_msg)
                    metrics['train/bob_loss'] += F.cross_entropy(bob_logits.squeeze(), torch.tensor(bob_targets, device='cuda').squeeze())

                if alice_first:
                    alice_step()
                    bob_step()
                else:
                    bob_step()
                    alice_step()

            [alice_ask, bob_ask] = self.workflow.insert([
                {'messages': [{'role': 'user', 'content': decide_prompt}], 'parent_ids': [n['id'] for n in alice_context]},
                {'messages': [{'role': 'user', 'content': decide_prompt}], 'parent_ids': [n['id'] for n in bob_context]},
            ], track_gradients=True)
            alice_context.append(alice_ask)
            bob_context.append(bob_ask)

            target_decision_ids = [p + [self.eot_id] for p in result['decision_ids']]
            _, decision_logits = self.workflow.train_step([
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
            ], target_decision_ids)
            decision_targets = reorder_targets(target_decision_ids)
            metrics['train/decision_loss'] = F.cross_entropy(decision_logits.squeeze(), decision_targets)

            return sum(metrics.values(), torch.tensor(0.)), dict(metrics)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('Ran out of memory, skipping batch')
                wandb.log({'train/oom': 1})
                for p in self.workflow.model.parameters():
                    if p.grad is not None:
                        del p.grad
                torch.cuda.empty_cache()
                return None
            else:
                raise e

    @torch.no_grad
    def evaluate(
        self,
        val_dataset: PrisonersDataset,
        max_steps: Optional[int] = None,
        max_e2e: int = 50,
    ) -> Dict:
        self.workflow.model.eval()

        total_loss = 0
        for step, sample in enumerate(tqdm(val_dataset, desc='Validating')):
            if max_steps and step >= max_steps:
                break
            nll = cached_nll(
                workflow=self.workflow,
                outputs=sample['result'],
                payoff=sample['payoff'],
                alice_first=sample['alice_first'],
                alice_strategy=sample['strategy'],
            )
            total_loss += np.mean(nll['bob_nll'][0])

        metrics = {'val/bob_first_message_nll': total_loss / min(len(val_dataset), max_steps if max_steps else 1e9)}

        e2e = {'bob_decisions': [], 'alice_decisions': []}
        for seed in tqdm(range(max_e2e)):
            self.workflow.reset()
            result = prisoners_cached(
                workflow=self.workflow,
                payoff=(5,3,1,0),
                alice_first=(seed < (max_e2e // 2)),
                alice_strategy=val_dataset[0]['strategy'],
                temperature=1.0,
                top_p=1.0,
                seed=seed+1000, # hate that i have to just tweak this
            )
            e2e['bob_decisions'].append(self.workflow.tokenizer.decode(result['bob_context'][-1]['output_tokens']))
            e2e['alice_decisions'].append(self.workflow.tokenizer.decode(result['alice_context'][-1]['output_tokens']))

        metrics['val/bob_cooperate'] = sum(1 for d in e2e['bob_decisions'] if 'COOPERATE' in d.upper()) / max_e2e
        metrics['val/alice_cooperate'] = sum(1 for d in e2e['alice_decisions'] if 'COOPERATE' in d.upper()) / max_e2e

        self.workflow.model.train()
        return metrics
