from tqdm import tqdm
from typing import Tuple, Dict, Optional
from operator import itemgetter as get

import torch
import torch.nn.functional as F

from llama import Llama
from llama.workflows.trainers.base import LoraTrainer, ListDataset
from llama.workflows.qa import (
    system_prompt as qa_system_prompt,
    ask_parallel,
    parse_items,
    eval_system_prompt,
    format_eval_user
)

class QaTrainer(LoraTrainer[ListDataset]):
    def step(self, sample: Dict) -> Tuple[torch.Tensor, Dict]:
        self.workflow.reset()
        subset, outputs = get('subset', 'outputs')(sample)

        [prompt] = self.workflow.insert([
            {
                'messages': [{'role': 'system', 'content': qa_system_prompt}],
                'parent_ids': []
            }
        ], track_gradients=True)

        questions = self.workflow.insert([
            {
                'messages': [{'role': 'user', 'content': f'Question {i+1}: {p['Question']}'}],
                'parent_ids': [prompt['id']],
            }
            for i, p in enumerate(subset)
        ])

        targets = [outputs['output_tokens'] + [self.eot_id]]

        _, logits = self.workflow.train_step([{
            'header': ('assistant', None),
            'prefill': '',
            'parent_ids': [prompt['id']] + [q['id'] for q in questions]
        }], targets)

        loss = F.cross_entropy(logits.squeeze(), torch.tensor(targets, device='cuda').squeeze())
        return loss, {'train/loss': loss}

    @torch.no_grad
    def evaluate(
        self,
        val_dataset: ListDataset,
        max_steps: Optional[int] = None,
        max_e2e: int = 20,
    ) -> Dict:
        self.workflow.model.eval()

        total_loss = 0
        for step, sample in enumerate(tqdm(val_dataset, desc='Validating')):
            if max_steps and step >= max_steps:
                break
            loss, _ = self.step(sample)
            total_loss += loss
        metrics = {'val/loss': total_loss / min(len(val_dataset), max_steps if max_steps else 1e9)}

        val_answers = []
        for step, sample in enumerate(tqdm(val_dataset, desc='Running E2E')):
            if step >= max_e2e:
                break
            self.workflow.reset()
            outputs = ask_parallel(self.workflow, sample['subset'], annotate=True)
            val_answers.append(parse_items(self.workflow.tokenizer.decode(outputs['output_tokens'])))

        eval_llama = Llama(self.workflow.model, self.workflow.tokenizer)
        eval_llama.model.reshape_cache(2)
        eval_llama.model.set_adapter_state(enabled=False)

        first_correct = last_correct = 0
        for sample, answers in tqdm(zip(val_dataset, val_answers)):
            if len(answers) != len(sample['subset']):
                continue

            resps = eval_llama.chat_completion([
                [{'role': 'system', 'content': eval_system_prompt},
                    {'role': 'user', 'content': format_eval_user(sample['subset'][0], answers[0])}],
                [{'role': 'system', 'content': eval_system_prompt},
                    {'role': 'user', 'content': format_eval_user(sample['subset'][-1], answers[-1])}],
            ], content_prefills=['{"correct": "'] * 2)

            first_correct += 'true' in resps[0]['generation']['content'].lower()
            last_correct += 'true' in resps[1]['generation']['content'].lower()

        metrics['val/first_correct'] = first_correct / 20
        metrics['val/last_correct'] = last_correct / 20

        eval_llama.model.set_adapter_state(enabled=True)
        eval_llama.model.reshape_cache(1)
        self.workflow.model.train()
        return metrics
