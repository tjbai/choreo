from typing import Optional, Dict
import torch
import torch.nn.functional as F
from llama.workflows.trainers.base import LoraTrainer, ListDataset, reorder_targets

class DirectTrainer(LoraTrainer[ListDataset]):

    def step(self, sample: Dict):
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
        pass
