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
        self.workflow = workflow
        self.model = self.workflow.model
        self.tokenizer = self.workflow.tokenizer
        self.optimizer = AdamW(self.model.get_trainable_parameters(), lr=2e-5)

        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        num_total = sum(p.numel() for p in self.model.parameters())
        print(f"Training {num_trainable/1e6:.1f}M / {num_total/1e9:.1f}B parameters")

    @abstractmethod
    def step(self, sample: Any) -> Any:
        pass

class TotTrainer(LoraTrainer):
    def __init__(self, workflow: Workflow, branching_factor: int, voters: int):
        super().__init__(workflow)
        self.branching_factor = branching_factor
        self.voters = voters
        self.eot_id = self.workflow.tokenizer.eot_id

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

        proposal_tasks = [
            {'header': ('assistant', None),
             'prefill': f'Solution #{i+1}:\n\n',
             'parent_ids': [cot['id']]}
            for i in range(self.branching_factor)
        ]
        target_proposal_ids = [p + [self.eot_id] for p in sample['result']['proposal_tokens']]
        proposal_nodes, proposal_logprobs = self.workflow.train_step(proposal_tasks, target_proposal_ids)

        vote_tasks = [
            {'header': ('assistant', None),
             'prefill': 'BEST CHOICE: ',
             'parent_ids': [vote['id']] + [p['id'] for p in proposal_nodes]}
            for i in range(self.voters)
        ]
        vote_target_ids = [p + [self.eot_id] for p in sample['result']['vote_tokens']]
        _, vote_logprobs = self.workflow.train_step(vote_tasks, vote_target_ids)

        votes = [v for v in sample['result']['votes'] if v is not None]
        best = Counter(votes).most_common(1)[0][0]

        final_task = {
            'header': ('assistant', None),
            'prefill': None,
            'parent_ids': [finish['id'], proposal_nodes[best - 1]['id']]
        }
        final_target_ids = sample['result']['final_tokens'] + [self.eot_id]
        _, final_logprobs = self.workflow.train_step([final_task], [final_target_ids])

        return proposal_logprobs, vote_logprobs, final_logprobs

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
    
    # TODO -- set parameters by config
    
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    workflow = Workflow.build(
        ckpt_dir='/scratch4/jeisner1/tjbai/llama_8b',
        tokenizer_path='/scratch4/jeisner1/tjbai/llama_8b/tokenizer.model',
        max_seq_len=4096,
        max_batch_size=1,
        model_parallel_size=1,
        max_nodes=20,
        use_lora=True,
        lora_rank=8,
        lora_alpha=16,
    )

    trainer = TotTrainer(workflow)

    # TODO
    if log_to_wandb:
        wandb.init(project="", config=None)
