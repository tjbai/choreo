import os
import math
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from abc import abstractmethod
from collections import Counter, defaultdict

import wandb
import fire
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, random_split
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

class TotDataset(Dataset):
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        with open(self.data_dir / 'metadata.json') as f:
            self.metadata = json.load(f)
        self.problem_paths = sorted(self.data_dir.glob("problem_*.pt"))

    def __len__(self):
        return len(self.problem_paths)

    def __getitem__(self, idx) -> Dict:
        return torch.load(self.problem_paths[idx], weights_only=True)

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
    def __init__(self, workflow: Workflow, output_dir: str, branching_factor: int, voters: int, learning_rate: float):
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
        ], training=True)

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
        proposal_nodes, proposal_logprobs = self.workflow.train_step(proposal_tasks, target_proposal_ids)

        vote_tasks = [
            {'header': ('assistant', None),
             'prefill': 'BEST CHOICE: ',
             'parent_ids': [vote['id']] + [p['id'] for p in proposal_nodes]}
            for i in range(self.voters)
        ]
        _, vote_logprobs = self.workflow.train_step(vote_tasks, vote_target_ids)

        votes = [v for v in sample['result']['votes'] if v is not None]
        best = Counter(votes).most_common(1)[0][0]
        final_task = {
            'header': ('assistant', None),
            'prefill': None,
            'parent_ids': [finish['id'], proposal_nodes[best - 1]['id']]
        }
        _, final_logprobs = self.workflow.train_step([final_task], [final_target_ids])

        proposal_loss = F.cross_entropy(proposal_logprobs.squeeze(0), proposal_targets)
        vote_loss = F.cross_entropy(vote_logprobs.squeeze(0), vote_targets)
        final_loss = F.cross_entropy(final_logprobs.squeeze(0), final_targets)

        total_loss = proposal_loss + vote_loss + final_loss

        metrics = {
            'train/proposal_ppl': torch.exp(proposal_loss),
            'train/vote_ppl': torch.exp(vote_loss),
            'train/final_ppl': torch.exp(final_loss),
            'train/nll_loss': total_loss,
        }

        return total_loss, metrics

@torch.no_grad()
def evaluate(trainer: TotTrainer, val_dataset: TotDataset, max_steps=None, max_e2e=100):
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

def get_lr_factor(step, warmup_steps=10, total_steps=100):
    # steps = number of updates != number of samples
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))

def finetune(
    data_path: str,
    ckpt_dir: str,
    tokenizer_path: str,
    output_dir: str = "checkpoints",
    log_to_wandb: bool = True,
    epochs: int = 2,
    checkpoint_freq: int = 25,
    validation_freq: int = 25,
    branching_factor: int = 8,
    voters: int = 4,
    max_seq_len: int = 4096,
    max_batch_size: int = 1,
    model_parallel_size: int = 1,
    max_nodes: int = 20,
    learning_rate: float = 2e-4,
    gradient_accumulation_steps: int = 4,
    use_lora: bool = True,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    master_addr: str = "localhost",
    master_port: str = "29500"
):
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(find_free_port())

    workflow = Workflow.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        model_parallel_size=model_parallel_size,
        max_nodes=max_nodes,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    trainer = TotTrainer(
        workflow,
        output_dir=output_dir,
        branching_factor=branching_factor,
        voters=voters,
        learning_rate=learning_rate,
    )

    dataset = TotDataset(data_path)
    generator = torch.Generator(device="cuda").manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [0.9, 0.1], generator=generator)
    print(f"Train Dataset: {len(train_dataset)} samples")
    print(f"Val Dataset: {len(val_dataset)} samples")

    print("Sanity check:")
    evaluate(trainer, val_dataset, max_steps=1, max_e2e=1)
    print("Passed!")

    if log_to_wandb:
        wandb.init(
            project="tot",
            config={
                "epochs": epochs,
                "checkpoint_freq": checkpoint_freq,
                "branching_factor": branching_factor,
                "voters": voters,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
            }
        )

    total_steps = epochs * len(train_dataset) // gradient_accumulation_steps
    warmup_steps = total_steps // 10

    global_step = 0
    for epoch in range(epochs):
        indices = torch.randperm(len(train_dataset)).tolist()
        for step, idx in enumerate(tqdm(indices, desc=f"Epoch {epoch}")):
            lr_factor = get_lr_factor(global_step, warmup_steps, total_steps)
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
                val_metrics = evaluate(trainer, val_dataset)
                if log_to_wandb:
                    wandb.log(val_metrics)
            if log_to_wandb:
                metrics.update({'lr': lr_factor})
                wandb.log(metrics)
            if (global_step + 1) % checkpoint_freq == 0:
                trainer.save_checkpoint(epoch, step)

    trainer.save_checkpoint(epochs - 1, len(train_dataset) - 1)

    if log_to_wandb:
        wandb.finish()

if __name__ == '__main__':
    fire.Fire(finetune)

