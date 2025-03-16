from tqdm import tqdm
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from llama.workflows.trainers.base import LoraTrainer, ListDataset, reorder_targets
from llama.workflows.madpar import starting_prompt, debate_prompt, summary_prompt

class MadparTrainer(LoraTrainer[ListDataset]):
    def step(self, sample: Dict, debug=False):
        total_loss = 0
        debate_tokens = sample['debate_tokens']
        summary_tokens = sample['summary_tokens']
        problem = sample['input']['problem']

        [agent_node, debate_node, summary_node] = self.workflow.insert(
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
            track_gradients=True
        )

        initial_targets = [p + [self.eot_id] for p in debate_tokens[0]]
        initial_nodes, initial_logits = self.workflow.train_step([
            {'header': ('assistant', None), 'prefill': f'From Agent {i+1}:\n', 'parent_ids': [agent_node['id']]}
            for i in range(3)],
            initial_targets
        )
        contexts = [[initial_node] for initial_node in initial_nodes]
        total_loss += F.cross_entropy(
            initial_logits.squeeze(),
            reorder_targets(initial_targets)
        )

        last_round = initial_nodes
        for round_idx, (debate, summary) in enumerate(zip(
            debate_tokens[1:],
            summary_tokens
        )):
            summary_targets = [p + [self.eot_id] for p in summary]
            [summary_node], logits = self.workflow.train_step([{
                    'header': ('assistant', None),
                    'prefill': 'Summary of agent responses:\n',
                    'parent_ids': [summary_node['id']] + [n['id'] for n in last_round]
                }],
                summary_targets
            )
            total_loss += F.cross_entropy(
                logits.squeeze(),
                reorder_targets(summary_targets)
            )

            update_targets = [p + [self.eot_id] for p in debate]
            update_nodes, logits = self.workflow.train_step([
                {
                    "header": ("assistant", None),
                    "prefill": f"From Agent {i + 1}:\n",
                    "parent_ids": [debate_node["id"], summary_node["id"]] + [n["id"] for n in context],
                }
                for i, context in enumerate(contexts)
            ], update_targets)
            for update, context in zip(update_nodes, contexts):
                context.append(update)
            total_loss += F.cross_entropy(
                logits.squeeze(),
                reorder_targets(update_targets)
            )

            last_round = update_nodes

    @torch.no_grad
    def evaluate(
        self,
        val_dataset: ListDataset,
        max_steps: Optional[int] = None,
        max_e2e: int = 20,
    ):
        pass
