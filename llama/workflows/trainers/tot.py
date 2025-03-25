import json
from typing import Dict, Tuple, Optional
from pathlib import Path
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from llama import Workflow
from llama.workflows.trainers.base import LoraTrainer, reorder_targets
from llama.workflows.tot import (
    cot_prompt,
    finish_prompt,
    format_vote_system_prompt,
    format_problem,
    tot_cached,
    load_math_problems,
    eval_solutions
)

class TotDataset(Dataset):
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.problem_paths = sorted(self.data_dir.glob('problem_*.pt'))

    def __len__(self):
        return len(self.problem_paths)

    def __getitem__(self, idx) -> Dict:
        return torch.load(self.problem_paths[idx], weights_only=True)

class TotTrainer(LoraTrainer[TotDataset]):
    def __init__(self, workflow: Workflow, output_dir: str,  learning_rate: float, branching_factor: int, voters: int):
        super().__init__(workflow, output_dir, learning_rate)
        self.branching_factor = branching_factor
        self.voters = voters
        self.eot_id = self.workflow.tokenizer.eot_id

    def step(self, sample: Dict) -> Tuple[torch.Tensor, Dict]:
        self.workflow.reset()

        try:
            cot, vote, finish = self.workflow.insert([
                {'messages': [
                    {'role': 'system', 'content': cot_prompt},
                    {'role': 'user', 'content': format_problem(sample['problem'])}
                ], 'parent_ids': []},
                {'messages': [
                    {'role': 'system', 'content': format_vote_system_prompt(self.branching_factor)},
                    {'role': 'user', 'content': format_problem(sample['problem'])}
                ], 'parent_ids': []},
                {'messages': [
                    {'role': 'system', 'content': finish_prompt},
                    {'role': 'user', 'content': format_problem(sample['problem'])}
                ], 'parent_ids': []},
            ], track_gradients=True)

            target_proposal_ids = [p + [self.eot_id] for p in sample['result']['proposal_tokens']]
            vote_target_ids = [p + [self.eot_id] for p in sample['result']['vote_tokens']]
            final_target_ids = sample['result']['final_tokens'] + [self.eot_id]

            # hacky. see above comment.
            proposal_targets = reorder_targets(target_proposal_ids)
            vote_targets = reorder_targets(vote_target_ids)
            final_targets = torch.tensor(final_target_ids, device='cuda')

            proposal_tasks = [
                {'header': ('assistant', None),
                    'prefill': f'Solution #{i+1}:\n\n',
                    'parent_ids': [cot['id']]}
                for i in range(self.branching_factor)
            ]
            proposal_nodes, proposal_logits = self.workflow.train_step(proposal_tasks, target_proposal_ids)

            vote_tasks = [
                {'header': ('assistant', None),
                    'prefill': 'BEST CHOICE: ',
                    'parent_ids': [vote['id']] + [p['id'] for p in proposal_nodes]}
                for i in range(self.voters)
            ]
            _, vote_logits = self.workflow.train_step(vote_tasks, vote_target_ids)

            votes = [v for v in sample['result']['votes'] if v is not None]
            best = Counter(votes).most_common(1)[0][0]
            final_task = {
                'header': ('assistant', None),
                'prefill': None,
                'parent_ids': [finish['id'], proposal_nodes[best - 1]['id']]
            }
            _, final_logits = self.workflow.train_step([final_task], [final_target_ids])

            proposal_loss = F.cross_entropy(proposal_logits.squeeze(0), proposal_targets)
            vote_loss = F.cross_entropy(vote_logits.squeeze(0), vote_targets)
            final_loss = F.cross_entropy(final_logits.squeeze(0), final_targets)

            total_loss = proposal_loss + vote_loss + final_loss

            metrics = {
                'train/proposal_ppl': torch.exp(proposal_loss),
                'train/vote_ppl': torch.exp(vote_loss),
                'train/final_ppl': torch.exp(final_loss),
                'train/nll_loss': total_loss,
            }

            return total_loss, metrics
        except Exception as e:
            if 'cuda out of memory' in e.lower():
                return None
            raise e

    @torch.no_grad
    def evaluate(
        self,
        val_dataset: TotDataset,
        max_steps: Optional[int] = None,
        max_e2e: int = 50,
    ):
        self.workflow.model.eval()

        total_loss = 0
        all_metrics = defaultdict(float)
        for step, sample in enumerate(tqdm(val_dataset, desc="Validating")):
            if max_steps and step >= max_steps:
                break
            loss, metrics = self.step(sample)
            total_loss += loss
            for k, v in metrics.items():
                all_metrics[k] += v.item()

        N = len(val_dataset)
        metrics = {
            'val/loss': total_loss / N,
            **{k.replace('train/', 'val/'): v / N for k, v in all_metrics.items()}
        }

        solutions = []
        for step, sample in enumerate(tqdm(val_dataset, desc="Running e2e validation")):
            if max_e2e and step >= max_e2e:
                break
            self.workflow.reset()
            solutions.append(tot_cached(
                workflow=self.workflow,
                problem=sample['problem'],
                branching_factor=self.branching_factor,
                voters=self.voters,
            ))

        self.llama.model.reshape_cache(4)
        self.llama.model.set_adapter_state(enabled=False)
        problems = load_math_problems('/home/tbai4/llama3/data/MATH', split='train')
        p2s = {d['problem']: d['solution'] for d in problems}
        try:
            correct = eval_solutions(
                self.llama,
                [self.workflow.tokenizer.decode(s['final_tokens']) for s in solutions],
                [{
                    'problem': sample['problem'],
                    'solution': p2s[sample['problem']],
                } for sample in val_dataset],
            )
            metrics['val/correct'] = sum(correct) / len(correct)
        finally:
            self.llama.model.set_adapter_state(enabled=True)
            self.llama.model.reshape_cache(1)

        self.workflow.model.train()
        return metrics
