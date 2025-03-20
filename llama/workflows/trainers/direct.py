from typing import Optional, Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from llama.workflows.simple import math_direct
from llama.workflows.tot import eval_solutions
from llama.workflows.trainers.base import LoraTrainer, ListDataset, reorder_targets

class DirectTrainer(LoraTrainer[ListDataset]):
    def step(self, sample: Dict):
        self.workflow.reset()
        problem = sample['inputs']['problem']
        solution = sample['outputs']['solution']
        tokens = self.workflow.tokenizer.encode(solution, bos=False, eos=False)

        [sys] = self.workflow.insert([{
            'messages': [{'role': 'user', 'content': f'Solve this math problem:\n\n{problem}'}],
            'parent_ids': []
        }], track_gradients=True)

        targets = [tokens + [self.eot_id]]
        _, logits = self.workflow.train_step([{
            'header': ('assistant', None),
            'prefill': 'Answer: ',
            'parent_ids': [sys['id']]
        }], targets)

        loss = F.cross_entropy(logits.squeeze(), reorder_targets(targets))
        return loss, {'train/loss': loss}

    @torch.no_grad
    def evaluate(
        self,
        val_dataset: ListDataset,
        max_steps: Optional[int] = None,
        max_e2e: int = 20,
    ):
        self.workflow.model.eval()
        metrics = {}

        losses = []
        for step, sample in enumerate(tqdm(val_dataset, desc="Validating")):
            if max_steps and step >= max_steps:
                break
            loss, metrics = self.step(sample)
            losses.append(loss)

        solutions = []
        for sample in tqdm(val_dataset, desc='Running e2e validation'):
            self.workflow.reset()
            solutions.append(math_direct(
                self.workflow,
                sample['inputs']['problem']
            ))

        self.llama.model.reshape_cache(4)
        self.llama.model.set_adapter_state(enabled=False)
        try:
            correct = eval_solutions(
                self.llama,
                solutions=solutions,
                problems=[{
                    'solution': d['inputs']['solution'],
                    'problem': d['inputs']['problem'],
                } for d in val_dataset]
            )
        finally:
            self.llama.model.set_adapter_state(enabled=True)
            self.llama.model.reshape_cache(1)
            self.workflow.model.train()

        return {
            'val/loss': sum(losses) / len(losses),
            'val/correct': sum(correct) / len(correct)
        }
