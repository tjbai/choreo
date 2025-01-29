import json
from pathlib import Path
from typing import Dict, Any
from abc import abstractmethod
from collections import Counter

import wandb
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset

from llama import Workflow
from llama.model import LoraTransformer
from llama.util import load_model_and_tokenizer
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
        return torch.load(self.problem_paths[idx])

class LoraTrainer:
    def __init__(self, workflow: Workflow):
        self.model = self.workflow.model
        self.tokenizer = self.workflow.tokenizer
        self.workflow = workflow
        self.optimizer = AdamW(self.model.get_trainable_parameters(), lr=2e-5)

        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        num_total = sum(p.numel() for p in self.model.parameters())
        print(f"Training {num_trainable/1e9:.1f}B / {num_total/1e9:.1f}B parameters")

    @abstractmethod
    def step(self, sample: Any) -> Any:
        pass

class TotTrainer(LoraTrainer):
    def __init__(self, workflow: Workflow, branching_factor: int, voters: int):
        super().__init__(workflow)
        self.branching_factor = branching_factor
        self.voters = voters

    def step(self, sample: Dict):
        self.workflow.reset()

        total_loss = 0.0
        metrics = {}

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
        ])

        # TODO TOMORROW -- handle the EOT. we're fading KD for now.

        proposal_tokens = sample['result']['proposal_tokens']
        proposal_tasks = [
            {'header': ('assistant', None),
             'prefill': f'Solution #{i+1}:\n\n',
             'parent_ids': [cot['id']]}
            for i in range(self.branching_factor)
        ]
        proposal_logprobs = self.workflow.train_step(proposal_tasks, proposal_tokens)

        vote_tokens = sample['result']['vote_tokens']
        vote_tasks = [
            {'header': ('assistant', None),
             'prefill': 'BEST CHOICE: ',
             'parent_ids': [vote['id']] + list(range(self.workflow.cur_id - self.branching_factor, self.workflow.cur_id))}
            for _ in range(self.voters)
        ]
        voter_logprobs = self.workflow.train_step(vote_tasks, vote_tokens)

        if sample['result']['final_logprobs'] is not None:
            final_tokens = sample['result']['final_tokens']
            votes = [v for v in sample['result']['votes'] if v is not None]
            assert len(votes) > 0
            best = Counter(votes).most_common(1)[0][0]
            final_tasks = [
                {'header': ('assistant', None),
                'prefill': None,
                'parent_ids': [finish['id']] + [self.workflow.cur_id - self.branching_factor - self.voters + best - 1]}
            ]
            student_final_logprobs = self.workflow.train_step(final_tasks, [final_tokens])

        return None # TODO

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
    epochs: int = 3,
    batch_size: int = 1, # bsz=1 for now
    log_to_wandb: bool = True
):
    # TODO -- load config and data

    model, tokenizer = load_model_and_tokenizer(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        model_class=LoraTransformer,
        # TODO -- figure out min size for these empirically
        max_seq_len=4096,
        max_batch_size=1,
    )
    workflow = Workflow(
        model=model,
        tokenizer=tokenizer,
        max_nodes=15,
    )
    trainer = LoraTrainer(workflow)

    # TODO
    if log_to_wandb:
        wandb.init(project="", config=None)
