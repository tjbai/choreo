import random
from contextlib import nullcontext
from tqdm import tqdm
from typing import Dict, Optional
from collections import defaultdict

import torch
import torch.nn.functional as F

from llama.workflows.trainers.base import LoraTrainer, ListDataset, reorder_targets
from llama.workflows.tot import load_math_problems, eval_solutions
from llama.workflows.mad import (
    mad_cached,
    agent_prompt as mad_agent_prompt,
    moderator_system_prompt as mad_moderator_system_prompt,
    moderator_user_prompt as mad_moderator_user_prompt
)

class MadTrainer(LoraTrainer[ListDataset]):
    def step(self, sample: Dict, debug=False):
        self.workflow.reset()
        problem = sample['inputs']['problem']
        metrics = {}

        try:
            with torch.no_grad():
                aff_context, neg_context, mod_context = [[a] for a in self.workflow.insert([
                    {'messages': [
                        {'role': 'system', 'content': mad_agent_prompt(problem, '')},
                        {'role': 'user', 'content': problem},
                    ], 'parent_ids': []},
                    {'messages': [
                        {'role': 'system', 'content': mad_agent_prompt(problem, '')},
                        {'role': 'user', 'content': problem},
                    ], 'parent_ids': []},
                    {'messages': [
                        {'role': 'system', 'content': mad_moderator_system_prompt(problem)},
                        {'role': 'user', 'content': mad_moderator_user_prompt}
                    ], 'parent_ids': []}
                ], track_gradients=False)]

            num_rounds = len(sample['outputs']['aff_tokens'][1:])
            mid_point = max(1, num_rounds // 2)
            has_final = 'final_tokens' in sample['outputs']

            chunks = {
                "initial": 2,
                "rounds_first_half": mid_point * 3,
                "rounds_second_half": (num_rounds - mid_point) * 3,
                "final": 1 if has_final else 0
            }

            valid_chunks = {k: v for k, v in chunks.items() if v > 0}
            selected_chunk = random.choices(
                list(valid_chunks.keys()),
                weights=list(valid_chunks.values()),
                k=1
            )[0]

            if debug:
                print(f"Selected chunk: {selected_chunk} (weights: {chunks})")

            def process_initial(with_grad=False):
                ctx = nullcontext() if with_grad else torch.no_grad()
                with ctx:
                    init_aff_tokens = sample['outputs']['aff_tokens'][0]
                    aff_target_ids = [p + [self.eot_id] for p in init_aff_tokens]
                    [aff_node], logits = self.workflow.train_step(
                        [{'header': ('assistant', ''), 'prefill': 'Affirmative:\n\n', 'parent_ids': [n['id'] for n in aff_context]}],
                        aff_target_ids,
                    )
                    aff_loss = F.cross_entropy(logits.squeeze(), reorder_targets(aff_target_ids))
                    aff_context.append(aff_node)
                    neg_context.append(aff_node)

                    init_neg_tokens = sample['outputs']['neg_tokens'][0]
                    neg_target_ids = [p + [self.eot_id] for p in init_neg_tokens]
                    [neg_node], logits = self.workflow.train_step(
                        [{'header': ('assistant', ''), 'prefill': 'Negative:\n\n', 'parent_ids': [n['id'] for n in neg_context]}],
                        neg_target_ids,
                    )
                    neg_loss = F.cross_entropy(logits.squeeze(), reorder_targets(neg_target_ids))
                    neg_context.append(neg_node)
                    aff_context.append(neg_node)

                    metrics['train/aff_init_loss'] = aff_loss.item()
                    metrics['train/neg_init_loss'] = neg_loss.item()
                    return aff_loss + neg_loss

            def process_rounds(start_idx, end_idx, with_grad=False):
                ctx = nullcontext() if with_grad else torch.no_grad()
                with ctx:
                    losses = []
                    for round_idx in range(start_idx, end_idx):
                        idx = round_idx
                        aff = sample['outputs']['aff_tokens'][1:][idx]
                        neg = sample['outputs']['neg_tokens'][1:][idx]
                        mod = sample['outputs']['mod_tokens'][idx]

                        aff_target_ids = [p + [self.eot_id] for p in aff]
                        [aff_node], logits = self.workflow.train_step(
                            [{'header': ('assistant', None), 'prefill': 'Affirmative Response:\n\n', 'parent_ids': [n['id'] for n in aff_context]}],
                            aff_target_ids
                        )
                        aff_loss = F.cross_entropy(logits.squeeze(), reorder_targets(aff_target_ids))
                        losses.append(aff_loss)
                        metrics[f'train/aff_round_{round_idx+1}_loss'] = aff_loss.item()
                        aff_context.append(aff_node)
                        neg_context.append(aff_node)
                        mod_context.append(aff_node)

                        neg_target_ids = [p + [self.eot_id] for p in neg]
                        [neg_node], logits = self.workflow.train_step(
                            [{'header': ('assistant', None), 'prefill': 'Negative Response:\n\n', 'parent_ids': [n['id'] for n in aff_context]}],
                            neg_target_ids
                        )
                        neg_loss = F.cross_entropy(logits.squeeze(), reorder_targets(neg_target_ids))
                        losses.append(neg_loss)
                        metrics[f'train/neg_round_{round_idx+1}_loss'] = neg_loss.item()
                        aff_context.append(neg_node)
                        neg_context.append(neg_node)
                        mod_context.append(neg_node)

                        mod_target_ids = [p + [self.eot_id] for p in mod]
                        [mod_node], logits = self.workflow.train_step(
                            [{'header': ('assistant', 'moderator'), 'prefill': '{"Reasoning": ', 'parent_ids': [n['id'] for n in mod_context]}],
                            mod_target_ids,
                        )
                        mod_loss = F.cross_entropy(logits.squeeze(), reorder_targets(mod_target_ids))
                        losses.append(mod_loss)
                        metrics[f'train/mod_round_{round_idx+1}_loss'] = mod_loss.item()
                        mod_context.append(mod_node)

                    return torch.sum(torch.stack(losses)) if losses else None

            def process_final(with_grad=False):
                if not has_final:
                    return None

                ctx = nullcontext() if with_grad else torch.no_grad()
                with ctx:
                    [final_prompt] = self.workflow.insert([
                        {'messages': [
                            {'role': 'user', 'content': (
                                'Please summarize your reasons and give the final answer that you think is correct. '
                                'Now please output your answer in JSON format, with the format as follows: {{"Reasoning": "", "Answer": ""}}. '
                                'Please strictly output in JSON format, do not output irrelevant content.'
                            )}
                        ], 'parent_ids': [n['id'] for n in mod_context]}
                    ], track_gradients=with_grad)

                    final_target_ids = [p + [self.eot_id] for p in sample['outputs']['final_tokens']]
                    _, logits = self.workflow.train_step(
                        [{'header': ('assistant', ''), 'prefill': '', 'parent_ids': [n['id'] for n in mod_context] + [final_prompt['id']]}],
                        final_target_ids
                    )
                    final_loss = F.cross_entropy(logits.squeeze(), reorder_targets(final_target_ids))
                    metrics['train/final_loss'] = final_loss.item()
                    return final_loss

            selected_loss = None
            if chunks["initial"] > 0:
                loss = process_initial(with_grad=(selected_chunk == "initial"))
                if selected_chunk == "initial":
                    selected_loss = loss

            if chunks["rounds_first_half"] > 0:
                loss = process_rounds(0, mid_point, with_grad=(selected_chunk == "rounds_first_half"))
                if selected_chunk == "rounds_first_half":
                    selected_loss = loss

            if chunks["rounds_second_half"] > 0:
                loss = process_rounds(mid_point, num_rounds, with_grad=(selected_chunk == "rounds_second_half"))
                if selected_chunk == "rounds_second_half":
                    selected_loss = loss

            if chunks["final"] > 0:
                loss = process_final(with_grad=(selected_chunk == "final"))
                if selected_chunk == "final":
                    selected_loss = loss

            total_loss = sum(v for k, v in metrics.items() if k.startswith('train/') and '_loss' in k)

            if debug:
                print(f"Total loss value: {total_loss:.4f}")
                print(f"Selected chunk: {selected_chunk} (loss: {selected_loss.item() if selected_loss is not None else 'None'})")

            metrics['train/selected_chunk'] = selected_chunk
            metrics['train/total_loss'] = total_loss

            return selected_loss, metrics

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f'Ran out of memory, skipping batch ({str(e)})')
                torch.cuda.empty_cache()
                return None
            else:
                raise e

    @torch.no_grad
    def evaluate(
        self,
        val_dataset: ListDataset,
        max_steps: Optional[int] = None,
        max_e2e: int = 20,
    ):
        self.workflow.model.eval()

        total_loss = 0
        all_metrics = defaultdict(float)
        for step, sample in enumerate(tqdm(val_dataset, desc="Validating")):
            if max_steps and step >= max_steps:
                break
            loss, metrics = self.step(sample)
            total_loss += metrics['train/total_loss']
            all_metrics['train/total_loss'] += metrics['train/total_loss']

        N = len(val_dataset)
        metrics = {
            'val/loss': total_loss / N,
            **{k.replace('train/', 'val/'): v / N for k, v in all_metrics.items()}
        }

        solutions = []
        problems = load_math_problems('/home/tbai4/llama3/data/MATH', split='val')
        problems = problems[:max_e2e]
        for problem in tqdm(problems, desc="Running e2e validation"):
            self.workflow.reset()
            outputs = mad_cached(
                workflow=self.workflow,
                problem=problem['problem'],
                max_rounds=3,
                temperature=0.7,
                top_p=1.0,
            )
            if isinstance(outputs['decision'], dict):
                solutions.append(outputs['decision']['Answer'])
            else:
                solutions.append(None)

        self.llama.model.reshape_cache(4)
        self.llama.model.set_adapter_state(enabled=False)
        try:
            correct = eval_solutions(
                self.llama,
                [s for s in solutions if s],
                [p for s, p in zip(solutions, problems) if s]
            )
            metrics['val/correct'] = sum(correct) / len(solutions)
            metrics['va/well_formed'] = len(correct) / len(solutions)
        finally:
            self.llama.model.set_adapter_state(enabled=True)
            self.llama.model.reshape_cache(1)

        self.workflow.model.train()
        return metrics
