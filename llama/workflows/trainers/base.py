import json
from typing import TypeVar, Generic, Any, Optional, Dict, List
from pathlib import Path
from abc import ABC, abstractmethod

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset

from llama import Workflow, Llama

# this is a hack to match the header prefill -> content prefill setup we have
# a more principled way would be to just do one big prefill step and mask out the irrelevant header tokens
def reorder_targets(target_ids: List[List[int]]) -> torch.Tensor:
    return torch.tensor([t[0] for t in target_ids] + sum((t[1:] for t in target_ids), []), device='cuda')

DataType = TypeVar('DataType', bound=Dataset)
class LoraTrainer(ABC, Generic[DataType]):
    def __init__(self, workflow: Workflow, output_dir: str, learning_rate: float):
        self.workflow = workflow
        self.model = self.workflow.model
        self.tokenizer = self.workflow.tokenizer
        self.llama = Llama(self.model, self.tokenizer)
        self.optimizer = AdamW(self.model.get_trainable_parameters(), lr=learning_rate)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.eot_id = self.workflow.tokenizer.eot_id
        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        num_total = sum(p.numel() for p in self.model.parameters())
        print(f"Training {num_trainable/1e6:.1f}M / {num_total/1e9:.1f}B parameters")

    def save_checkpoint(self, global_step: int):
        torch.save({
            "trainable_params": [p for p in self.model.get_trainable_parameters()],
            "optimizer": self.optimizer.state_dict()
        }, self.output_dir / f"lora_step-{global_step}.pt")

    @abstractmethod
    def step(self, sample: Any) -> Any:
        pass

    @abstractmethod
    def evaluate(
        self,
        val_dataset: DataType,
        max_steps: Optional[int] = int(1e9),
        max_e2e: int = 50
    ) -> Any:
        pass

class ListDataset(Dataset):
    def __init__(self, data_path: str | Path):
        with open(data_path) as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict:
        return self.data[idx]
