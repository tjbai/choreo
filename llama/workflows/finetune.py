import os
import math
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable
from abc import abstractmethod
from collections import Counter, defaultdict
from operator import itemgetter as get

import wandb
import fire
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, random_split
import numpy as np
from tqdm import tqdm

from llama import Workflow, Llama
from llama.util import find_free_port
from llama.workflows.tot import (
    cot_prompt,
    finish_prompt,
    format_vote_system_prompt,
    format_problem,
    tot_cached,
    load_math_problems,
    eval_solutions
)
from llama.workflows.prisoners import (
    format_system_prompt as format_prisoners_system_prompt,
    plan_prompt,
    decide_prompt,
    cached_nll,
    prisoners_cached,
)

class TotDataset(Dataset):
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        with open(self.data_dir / 'metadata.json') as f:
            self.metadata = json.load(f)
        self.problem_paths = sorted(self.data_dir.glob('problem_*.pt'))

    def __len__(self):
        return len(self.problem_paths)

    def __getitem__(self, idx) -> Dict:
        return torch.load(self.problem_paths[idx], weights_only=True)

class PrisonersDataset(Dataset):
    def __init__(self, data_dir: str | Path, strategy: Optional[str] = None):
        self.data_dir = Path(data_dir)
        assert strategy in (None, 'always_cooperate', 'always_defect')
        self.paths = sorted(self.data_dir.glob(f'trace_{strategy}_*.pt'))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx) -> Dict:
        return torch.load(self.paths[idx], weights_only=True)

# this is a hack to match the header prefill -> content prefill setup we have
# a more principled way would be to just do one big prefill step and mask out the irrelevant header tokens
def reorder_targets(target_ids: List[List[int]]) -> torch.Tensor:
    return torch.tensor([t[0] for t in target_ids] + sum((t[1:] for t in target_ids), []), device='cuda')

class LoraTrainer:
    def __init__(self, workflow: Workflow, output_dir: str, learning_rate: float):
        self.workflow = workflow
        self.model = self.workflow.model
        self.tokenizer = self.workflow.tokenizer
        self.llama = Llama(self.model, self.tokenizer)
        self.optimizer = AdamW(self.model.get_trainable_parameters(), lr=learning_rate)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        num_total = sum(p.numel() for p in self.model.parameters())
        print(f"Training {num_trainable/1e6:.1f}M / {num_total/1e9:.1f}B parameters")

    def save_checkpoint(self, epoch: int, step: int):
        torch.save({
            "lora": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, self.output_dir / f"lora_epoch-{epoch}_step-{step}.pt")

    @abstractmethod
    def step(self, sample: Any) -> Any:
        pass

class TotTrainer(LoraTrainer):
    def __init__(self, workflow: Workflow, output_dir: str,  learning_rate: float, branching_factor: int, voters: int):
        super().__init__(workflow, output_dir, learning_rate)
        self.branching_factor = branching_factor
        self.voters = voters
        self.eot_id = self.workflow.tokenizer.eot_id

    def step(self, sample: Dict) -> Tuple[torch.Tensor, Dict]:
        self.workflow.reset()

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

class PrisonersTrainer(LoraTrainer):
    def __init__(self, workflow: Workflow, output_dir: str,  learning_rate: float):
        super().__init__(workflow, output_dir, learning_rate)
        self.eot_id = self.workflow.tokenizer.eot_id

    def step(self, sample: Dict) -> Tuple[torch.Tensor, Dict]:
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
        [alice_plan, bob_plan], plan_logits = self.workflow.ckpt_train_step([
            {'header': ('assistant', 'alice'), 'prefill': '', 'parent_ids': [alice_sys['id']]},
            {'header': ('assistant', 'bob'), 'prefill': '', 'parent_ids': [bob_sys['id']]},
        ], target_plan_ids)
        plan_targets = reorder_targets(target_plan_ids)
        metrics['train/plan_loss'] = F.cross_entropy(plan_logits.squeeze(0), plan_targets)

        alice_context = [alice_sys, alice_plan]
        bob_context = [bob_sys, bob_plan]
        for round, (alice_ids, bob_ids) in enumerate(zip(result['alice_message_ids'], result['bob_message_ids'])):
            if alice_first:
                alice_targets = [alice_ids + [self.eot_id]]
                [alice_msg], alice_logits = self.workflow.ckpt_train_step([{
                    'header': ('assistant', 'alice'),
                    'prefill': 'To Bob: ',
                    'parent_ids': [n['id'] for n in alice_context]
                }], alice_targets)
                alice_context.append(alice_msg)
                bob_context.append(alice_msg)
                metrics['train/alice_loss'] += F.cross_entropy(alice_logits.squeeze(), torch.tensor(alice_targets, device='cuda').squeeze())

                bob_targets = [bob_ids + [self.eot_id]]
                [bob_msg], bob_logits = self.workflow.ckpt_train_step([{
                    'header': ('assistant', 'bob'),
                    'prefill': 'To Alice: ',
                    'parent_ids': [n['id'] for n in bob_context]
                }], bob_targets)
                alice_context.append(bob_msg)
                bob_context.append(bob_msg)
                metrics['train/bob_loss'] += F.cross_entropy(bob_logits.squeeze(), torch.tensor(bob_targets, device='cuda').squeeze())
            else:
                bob_targets = [bob_ids + [self.eot_id]]
                [bob_msg], bob_logits = self.workflow.ckpt_train_step([{
                    'header': ('assistant', 'bob'),
                    'prefill': 'To Alice: ',
                    'parent_ids': [n['id'] for n in bob_context]
                }], bob_targets)
                alice_context.append(bob_msg)
                bob_context.append(bob_msg)
                metrics['train/bob_loss'] += F.cross_entropy(bob_logits.squeeze(), torch.tensor(bob_targets, device='cuda').squeeze())

                alice_targets = [alice_ids + [self.eot_id]]
                [alice_msg], alice_logits = self.workflow.ckpt_train_step([{
                    'header': ('assistant', 'alice'),
                    'prefill': 'To Bob: ',
                    'parent_ids': [n['id'] for n in alice_context]
                }], alice_targets)
                alice_context.append(alice_msg)
                bob_context.append(alice_msg)
                metrics['train/alice_loss'] += F.cross_entropy(alice_logits.squeeze(), torch.tensor(alice_targets, device='cuda').squeeze())

        [alice_ask, bob_ask] = self.workflow.insert([
            {'messages': [{'role': 'user', 'content': decide_prompt}], 'parent_ids': [n['id'] for n in alice_context]},
            {'messages': [{'role': 'user', 'content': decide_prompt}], 'parent_ids': [n['id'] for n in bob_context]},
        ], track_gradients=True)
        alice_context.append(alice_ask)
        bob_context.append(bob_ask)

        target_decision_ids = [p + [self.eot_id] for p in result['decision_ids']]
        _, decision_logits = self.workflow.ckpt_train_step([
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

@torch.no_grad()
def evaluate_tot(trainer: TotTrainer, val_dataset: TotDataset, max_steps=None, max_e2e=100):
    trainer.workflow.model.eval()

    total_loss = 0
    all_metrics = defaultdict(float)

    for step, sample in enumerate(tqdm(val_dataset, desc="Validating")):
        if max_steps and step >= max_steps:
            break
        loss, metrics = trainer.step(sample)
        total_loss += loss
        for k, v in metrics.items():
            all_metrics[k] += v.item()

    N = len(val_dataset)
    metrics = {
        'val/loss': total_loss / N,
        **{k.replace('train/', 'val/'): v / N for k, v in all_metrics.items()}
    }

    solutions = []
    problems = load_math_problems('/home/tbai4/llama3/data/MATH', split='val')
    problems = problems[:max_e2e]
    for problem in tqdm(problems, desc="Running e2e validation"):
        trainer.workflow.reset()
        solutions.append(tot_cached(
            workflow=trainer.workflow,
            problem=problem['problem'],
            branching_factor=trainer.branching_factor,
            voters=trainer.voters,
        ))

    trainer.llama.model.reshape_cache(4)
    trainer.llama.model.set_adapter_state(enabled=False)
    try:
        correct = eval_solutions(trainer.llama, solutions, problems)
        metrics['val/correct'] = sum(correct) / len(correct)
    finally:
        trainer.llama.model.set_adapter_state(enabled=True)
        trainer.llama.model.reshape_cache(1)

    trainer.workflow.model.train()
    return metrics

@torch.no_grad()
def evaluate_prisoners(
    trainer: PrisonersTrainer,
    val_dataset: PrisonersDataset,
    max_steps=None,
    max_e2e=50,
) -> Dict:
    trainer.workflow.model.eval()


    total_loss = 0
    for step, sample in enumerate(tqdm(val_dataset, desc='Validating')):
        if max_steps and step >= max_steps:
            break
        nll = cached_nll(
            workflow=trainer.workflow,
            outputs=sample['result'],
            payoff=sample['payoff'],
            alice_first=sample['alice_first'],
            alice_strategy=sample['strategy'],
        )
        total_loss += np.mean(nll['bob_nll'][0])

    metrics = {'val/bob_first_message_nll': total_loss / min(len(val_dataset), max_steps if max_steps else 1e9)}

    e2e = {'bob_decisions': [], 'alice_decisions': []}
    for seed in tqdm(range(max_e2e)):
        result = prisoners_cached(
            workflow=trainer.workflow,
            payoff=(5,3,1,0),
            alice_first=(seed < (max_e2e // 2)),
            alice_strategy=val_dataset[0]['strategy'],
            temperature=1.0,
            top_p=1.0,
            seed=seed+420,
        )
        e2e['bob_decisions'].append(trainer.workflow.tokenizer.decode(result['bob_context'][-1]['output_tokens']))
        e2e['alice_decisions'].append(trainer.workflow.tokenizer.decode(result['alice_context'][-1]['output_tokens']))

    metrics['val/bob_cooperate'] = sum(1 for d in e2e['bob_decisions'] if 'COOPERATE' in d) / max_e2e
    metrics['val/alice_cooperate'] = sum(1 for d in e2e['alice_decisions'] if 'COOPERATE' in d) / max_e2e

    trainer.workflow.model.train()
    return metrics

def get_lr_factor(step, warmup_steps=10, total_steps=100):
    # steps = number of updates != number of samples
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))

def set_model_env():
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(find_free_port())

def init_task(
    task: str,
    workflow: Workflow,
    data_path: str,
    output_dir: str,
    learning_rate: float,
    **task_params
) -> Tuple[LoraTrainer, Dataset, Callable]:
    if task == 'tot':
        trainer = TotTrainer(
            workflow,
            output_dir=output_dir,
            learning_rate=learning_rate,
            branching_factor=task_params['branching_factor'],
            voters=task_params['voters']
        )
        dataset = TotDataset(data_path)
        wandb.init(
            project="tot",
            config={
                "branching_factor": task_params['branching_factor'],
                "voters": task_params['voters'],
                "lora_rank": task_params['lora_rank'],
                "lora_alpha": task_params['lora_alpha'],
                "lora_dropout": task_params['lora_dropout'],
                "learning_rate": learning_rate,
            }
        )
        return trainer, dataset, evaluate_tot
    elif task == 'prisoners':
        trainer = PrisonersTrainer(
            workflow=workflow,
            output_dir=output_dir,
            learning_rate=learning_rate
        )
        dataset = PrisonersDataset(data_path, task_params['strategy'])
        wandb.init(
            project="prisoners",
            config={
                "strategy": task_params['strategy'],
                "lora_rank": task_params['lora_rank'],
                "lora_alpha": task_params['lora_alpha'],
                "lora_dropout": task_params['lora_dropout'],
                "learning_rate": learning_rate,
            }
        )
        return trainer, dataset, evaluate_prisoners
    raise NotImplementedError()

def training_schedule(
    epochs: Optional[int],
    steps: Optional[int],
    dataset_size: int,
    gradient_accumulation_steps: int
) -> Tuple[int, int, int]:
    if epochs is not None:
        total_steps = epochs * dataset_size // gradient_accumulation_steps
        warmup_steps = total_steps // 10
        return epochs, total_steps, warmup_steps

    assert steps is not None
    epochs = math.ceil(steps * gradient_accumulation_steps / dataset_size)
    warmup_steps = steps // 10
    return epochs, steps, warmup_steps

def finetune(
    data_path: str,
    ckpt_dir: str,
    tokenizer_path: str,
    task: str,
    output_dir: str = "checkpoints",
    max_seq_len: int = 4096,
    # training
    epochs: Optional[int] = 2,
    steps: Optional[int] = None,
    learning_rate: float = 2e-4,
    gradient_accumulation_steps: int = 4,
    # please set these wisely
    checkpoint_freq: int = 25,
    validation_freq: int = 25,
    # lora
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    # tot
    branching_factor: int = 8,
    voters: int = 4,
    # prisoners
    strategy: Optional[str] = None,
):
    set_model_env()
    workflow = Workflow.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=1,
        model_parallel_size=1,
        max_nodes=100,
        use_lora=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    trainer, dataset, eval_fn = init_task(
        task=task,
        workflow=workflow,
        data_path=data_path,
        output_dir=output_dir,
        learning_rate=learning_rate,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        branching_factor=branching_factor,
        voters=voters,
        strategy=strategy,
    )

    generator = torch.Generator(device="cuda").manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [0.9, 0.1], generator=generator)
    assert len(train_dataset) > 0 and len(val_dataset) > 0
    print(f"Train Dataset: {len(train_dataset)} samples")
    print(f"Val Dataset: {len(val_dataset)} samples")

    print("Sanity check:")
    eval_fn(trainer, val_dataset, max_steps=1, max_e2e=1)
    print("Passed!")

    epochs, steps, warmup_steps = training_schedule(
        epochs=epochs,
        steps=steps,
        dataset_size=len(train_dataset),
        gradient_accumulation_steps=gradient_accumulation_steps
    )

    print(f'Epochs: {epochs}, Steps: {steps}, Warmup: {warmup_steps}')

    global_step = 0
    for epoch in range(epochs):
        indices = torch.randperm(len(train_dataset)).tolist()
        for step, idx in enumerate(tqdm(indices, desc=f"Epoch {epoch}")):
            lr_factor = get_lr_factor(global_step, warmup_steps, steps)
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = learning_rate * lr_factor
            sample = dataset[idx]
            loss, metrics = trainer.step(sample)
            loss.backward()
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainer.workflow.model.parameters(), max_norm=1.0)
                trainer.optimizer.step()
                trainer.optimizer.zero_grad()
                global_step += 1
                if (global_step + 1) % validation_freq == 0:
                    val_metrics = eval_fn(trainer, val_dataset)
                    wandb.log(val_metrics)
                metrics.update({'lr': lr_factor})
                wandb.log(metrics)
                if (global_step + 1) % checkpoint_freq == 0:
                    trainer.save_checkpoint(epoch, step)
                if steps is not None and global_step == steps:
                    break

    trainer.save_checkpoint(epochs - 1, len(train_dataset) - 1)
    wandb.finish()

if __name__ == '__main__':
    fire.Fire(finetune)
