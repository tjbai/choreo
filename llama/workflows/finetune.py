import os
import json
from pathlib import Path
from typing import Dict, Any, List
from abc import abstractmethod
from collections import Counter

import wandb
import fire
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset
from tqdm import tqdm

from llama import Workflow
from llama.workflows.tot import cot_prompt, finish_prompt, format_vote_system_prompt, format_problem

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
    def __init__(self, workflow: Workflow, learning_rate: float):
        self.workflow = workflow
        self.model = self.workflow.model
        self.tokenizer = self.workflow.tokenizer
        self.optimizer = AdamW(self.model.get_trainable_parameters(), lr=learning_rate)

        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        num_total = sum(p.numel() for p in self.model.parameters())
        print(f"Training {num_trainable/1e6:.1f}M / {num_total/1e9:.1f}B parameters")

    @abstractmethod
    def step(self, sample: Any) -> Any:
        pass

class TotTrainer(LoraTrainer):
    def __init__(self, workflow: Workflow, branching_factor: int, voters: int, learning_rate: float):
        super().__init__(workflow, learning_rate)
        self.branching_factor = branching_factor
        self.voters = voters
        self.eot_id = self.workflow.tokenizer.eot_id

    def step(self, sample: Dict):
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
        # proposal_targets = torch.tensor(sum(target_proposal_ids, []), device='cuda')
        # vote_targets = torch.tensor(sum(vote_target_ids, []), device='cuda')
        # final_targets = torch.tensor(final_target_ids, device='cuda')

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
            'loss/proposals': proposal_loss.item(),
            'loss/votes': vote_loss.item(),
            'loss/final': final_loss.item(),
            'loss/total': total_loss.item()
        }

        return total_loss, metrics

def save_checkpoint(trainer: LoraTrainer, epoch: int, output_dir: str):
    Path(output_dir).mkdir(exist_ok=True)
    path = Path(output_dir) / f"lora_epoch_{epoch}.pt"
    torch.save({"lora": trainer.model.state_dict(), "optimizer": trainer.optimizer.state_dict()}, path)
    print(f"Saved checkpoint to {path}")

def finetune(
    data_path: str,
    ckpt_dir: str,
    tokenizer_path: str,
    output_dir: str = "checkpoints",
    log_to_wandb: bool = True,
    epochs: int = 2,
    checkpoint_freq: int = 100,
    branching_factor: int = 8,
    voters: int = 4,
    max_seq_len: int = 4096,
    max_batch_size: int = 1,
    model_parallel_size: int = 1,
    max_nodes: int = 20,
    use_lora: bool = True,
    learning_rate: float = 1e-5,
    gradient_accumulation_steps: int = 1,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    master_addr: str = "localhost",
    master_port: str = "29500"
):
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

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
    )

    trainer = TotTrainer(
        workflow,
        branching_factor=branching_factor,
        voters=voters,
        learning_rate=learning_rate,
    )
    dataset = TotDataset(data_path)

    if log_to_wandb:
        wandb.init(
            project="tot",
            config={
                "epochs": epochs,
                "checkpoint_freq": checkpoint_freq,
                "branching_factor": branching_factor,
                "voters": voters,
            }
        )

    for epoch in range(epochs):
        indices = torch.randperm(len(dataset)).tolist()
        for step, idx in enumerate(tqdm(indices, desc=f"Epoch {epoch}")):
            sample = dataset[idx]
            loss, metrics = trainer.step(sample)
            loss.backward()
            if (step + 1) % gradient_accumulation_steps == 0:
                trainer.optimizer.step()
                trainer.optimizer.zero_grad()
            if log_to_wandb:
                metrics['epoch'] = epoch
                metrics['step'] = step
                wandb.log(metrics)
            if (step + 1) % checkpoint_freq == 0:
                save_checkpoint(trainer, epoch, output_dir)

        save_checkpoint(trainer, epoch, output_dir)

    if log_to_wandb:
        wandb.finish()

def main():
    fire.Fire(finetune)
