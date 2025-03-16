from tqdm import tqdm
from typing import Dict, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn.functional as F

from llama.workflows.trainers.base import LoraTrainer, ListDataset, reorder_targets
from llama.workflows.bsm import (
    branch_prompt_content,
    solve_prompt,
    cached_merge_prompt,
    bsm_cached
)

class BsmTrainer(LoraTrainer[ListDataset]):
    def step(self, sample: dict) -> Tuple[torch.Tensor, Dict]:
        self.workflow.reset()
        metrics = {}

        concepts = sample['inputs']['concepts']
        [branch_node, merge_node] = self.workflow.insert([
            {'messages': [
                {'role': 'user', 'content': branch_prompt_content(concepts)}
            ], 'parent_ids': []},
            {'messages': [
                {'role': 'user', 'content': cached_merge_prompt}
            ], 'parent_ids': []},
        ])

        branch_target_ids = [p + [self.eot_id] for p in sample['outputs']['branch_tokens']]
        _, branch_logits = self.workflow.train_step(
            [{'header': ('assistant', None), 'prefill': '', 'parent_ids': [branch_node['id']]}],
            branch_target_ids
        )
        metrics['train/branch_loss']  = F.cross_entropy(
            branch_logits.squeeze(),
            reorder_targets(branch_target_ids)
        )

        solve_nodes = self.workflow.insert([
            {'messages': [
                {'role': 'user', 'content': solve_prompt(concept_group, sample['outputs']['story_topic'])}
            ], 'parent_ids': []}
        for concept_group in sample['outputs']['concept_groups']], track_gradients=True)

        solve_target_ids = [p + [self.eot_id] for p in sample['outputs']['solve_tokens']]
        _, solve_logits = self.workflow.train_step(
            [{'header':
                ('assistant', None),
                'prefill': f'Story {i+1}:\n\n',
                'parent_ids': [solve_node['id']]}
            for i, solve_node in enumerate(solve_nodes)],
            solve_target_ids
        )
        metrics['train/solve_loss'] = F.cross_entropy(
            solve_logits.squeeze(),
            reorder_targets(solve_target_ids)
        ) / 2 # just some light normalization

        merge_target_ids = [p + [self.eot_id] for p in sample['outputs']['merge_tokens']]
        _, merge_logits = self.workflow.train_step([
            {'header':
                ('assistant', None),
                'prefill': 'Combined Story:\n\n',
                'parent_ids': [
                    merge_node['id'],
                    solve_nodes[0]['id'],
                    solve_nodes[1]['id'],
                ]}
        ], merge_target_ids)

        metrics['train/merge_loss'] = F.cross_entropy(
            merge_logits.squeeze(),
            reorder_targets(merge_target_ids)
        )

        return sum(metrics.values()), metrics

    @torch.no_grad
    def evaluate(
        self,
        val_dataset: ListDataset,
        max_steps: Optional[int] = None,
        max_e2e: int = 20,
    ):
        self.workflow.model.eval()

        total_loss = []
        all_metrics = defaultdict(float)
        for step, sample in enumerate(tqdm(val_dataset, desc='Validating')):
            if max_steps and step >= max_steps:
                break
            loss, metrics = self.step(sample)
            total_loss.append(loss)
            for k, v in metrics.items():
                all_metrics[k] += v.item()

        metrics = {
            'val/loss': sum(total_loss) / len(total_loss),
            **{k.replace('train/', 'val/'): v / len(total_loss) for k, v in all_metrics.items()}
        }

        total_coverage = []
        for step, sample in enumerate(tqdm(val_dataset, desc="Evaluating")):
            if max_e2e and step >= max_e2e:
                break
            self.workflow.reset()
            outputs = bsm_cached(workflow=self.workflow, **sample['inputs'])
            if outputs is None:
                total_coverage.append(0.)
            else:
                story = self.workflow.tokenizer.decode(outputs['merge_tokens'][0])
                concept_present = [concept.lower() in story.lower() for concept in sample['inputs']['concepts']]
                coverage = sum(concept_present) / len(concept_present)
                total_coverage.append(coverage)

        metrics['val/coverage'] = sum(total_coverage) / len(total_coverage)

        self.workflow.model.train()
        return metrics
