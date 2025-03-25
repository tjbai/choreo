import random
from tqdm import tqdm
from typing import Dict, Optional
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from llama.workflows.trainers.base import LoraTrainer, ListDataset, reorder_targets
from llama.workflows.tot import load_math_problems
from llama.workflows.madpar import (
    starting_prompt,
    debate_prompt,
    summary_prompt,
    madpar_cached,
    eval_debate_solutions,
    parse_output
)

class MadparTrainer(LoraTrainer[ListDataset]):
    def step(self, sample: Dict, debug=False):
        self.workflow.reset()
        debate_tokens = sample['outputs']['debate_tokens']
        summary_tokens = sample['outputs']['summary_tokens']
        problem = sample['inputs']['problem']
        metrics = {}

        num_rounds = len(summary_tokens)
        initial_token_count = sum(len(tokens) for tokens in debate_tokens[0])

        round_token_counts = {}
        for i in range(num_rounds):
            summary_count = len(summary_tokens[i])
            debate_count = sum(len(tokens) for tokens in debate_tokens[i+1])
            round_token_counts[f"round_{i}"] = summary_count + debate_count

        chunks = {
            "initial": initial_token_count,
            **round_token_counts
        }

        weighted_chunks = list(chunks.items())
        chunk_names, chunk_weights = zip(*weighted_chunks)
        selected_chunk = random.choices(chunk_names, weights=chunk_weights, k=1)[0]

        if debug:
            print(f"Selected chunk: {selected_chunk} (weights: {chunks})")

        with torch.no_grad():
            [agent_node, debate_node, summary_prompt_node] = self.workflow.insert(
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

        def process_initial(with_grad=False):
            ctx = nullcontext() if with_grad else torch.no_grad()
            with ctx:
                initial_targets = [p + [self.eot_id] for p in debate_tokens[0]]
                initial_nodes, initial_logits = self.workflow.train_step([
                    {'header': ('assistant', None), 'prefill': f'From Agent {i+1}:\n', 'parent_ids': [agent_node['id']]}
                    for i in range(3)],
                    initial_targets
                )
                loss = F.cross_entropy(
                    initial_logits.squeeze(),
                    reorder_targets(initial_targets)
                )
                metrics['train/initial_loss'] = loss.item()
                return initial_nodes, loss

        def process_round(round_idx, last_round, contexts, with_grad=False):
            ctx = nullcontext() if with_grad else torch.no_grad()
            with ctx:
                summary_targets = [p + [self.eot_id] for p in summary_tokens[round_idx]]
                [summary_node], summary_logits = self.workflow.train_step([{
                        'header': ('assistant', None),
                        'prefill': 'Summary of agent responses:\n',
                        'parent_ids': [summary_prompt_node['id']] + [n['id'] for n in last_round]
                    }],
                    summary_targets
                )
                summary_loss = F.cross_entropy(summary_logits.squeeze(), reorder_targets(summary_targets))
                metrics[f'train/summary_{round_idx}_loss'] = summary_loss.item()

                update_targets = [p + [self.eot_id] for p in debate_tokens[round_idx+1]]
                update_nodes, update_logits = self.workflow.train_step([
                    {
                        "header": ("assistant", None),
                        "prefill": f"From Agent {i + 1}:\n",
                        "parent_ids": [debate_node["id"], summary_node["id"]] + [n["id"] for n in context],
                    }
                    for i, context in enumerate(contexts)
                ], update_targets)

                update_loss = F.cross_entropy(update_logits.squeeze(), reorder_targets(update_targets))
                metrics[f'train/update_{round_idx}_loss'] = update_loss.item()

                for update, context in zip(update_nodes, contexts):
                    context.append(update)

                return update_nodes, summary_loss + update_loss

        selected_loss = None

        initial_is_selected = selected_chunk == "initial"
        initial_nodes, initial_loss = process_initial(with_grad=initial_is_selected)
        if initial_is_selected:
            selected_loss = initial_loss

        last_round = initial_nodes
        contexts = [[node] for node in initial_nodes]

        for round_idx in range(num_rounds):
            round_is_selected = selected_chunk == f"round_{round_idx}"
            last_round, round_loss = process_round(
                round_idx,
                last_round,
                contexts,
                with_grad=round_is_selected
            )
            if round_is_selected:
                selected_loss = round_loss

        total_loss = sum(v for k, v in metrics.items() if k.startswith('train/') and '_loss' in k)
        metrics['train/total_loss'] = total_loss
        metrics['train/selected_chunk'] = selected_chunk

        if debug:
            print(f"Total loss value: {total_loss:.4f}")
            print(f"Selected chunk: {selected_chunk} (loss: {selected_loss.item() if selected_loss is not None else 'None'})")

        return selected_loss, metrics

    @torch.no_grad
    def evaluate(
        self,
        val_dataset: ListDataset,
        max_steps: Optional[int] = None,
        max_e2e: int = 20,
    ):
        self.workflow.model.eval()

        losses = []
        for step, sample in enumerate(tqdm(val_dataset)):
            if max_steps and step >= max_steps:
                break
            loss, metrics = self.step(sample)
            losses.append(metrics['train/total_loss'])

        outputs = []
        for step, sample in enumerate(tqdm(val_dataset)):
            if max_e2e and step >= max_e2e:
                break
            outputs.append(madpar_cached(
                self.workflow,
                problem=sample['inputs']['problem'],
            ))

        self.llama.model.reshape_cache(4)
        self.llama.model.set_adapter_state(enabled=False)
        problems = load_math_problems('/home/tbai4/llama3/data/MATH', split='train')
        p2s = {d['problem']: d['solution'] for d in problems}
        try:
            correct = eval_debate_solutions(
                self.llama,
                agent_solutions=[
                    [parse_output(a) for a in d['debate_tokens'][-1]]
                    for d in outputs
                ],
                problems=[{
                    'problem': sample['inputs']['problem'],
                    'solution': p2s[sample['inputs']['problem']]
                } for sample in val_dataset]
            )
        finally:
            self.llama.model.set_adapter_state(enabled=True)
            self.llama.model.reshape_cache(1)

        self.workflow.model.train()
        return {
            'val/total_loss': sum(losses) / len(losses),
            'val/correct': sum(correct) / len(correct)
        }
