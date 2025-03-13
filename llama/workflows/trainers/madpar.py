from tqdm import tqdm
from typing import Dict, Optional
from collections import defaultdict

import torch

from llama.workflows.trainers.base import LoraTrainer, ListDataset, reorder_targets
from llama.workflows.tot import load_math_problems, eval_solutions

class MadparTrainer(LoraTrainer[ListDataset]):
    def step(self, sample: Dict, debug=False):
        pass

    @torch.no_grad
    def evaluate(
        self,
        val_dataset: ListDataset,
        max_steps: Optional[int] = None,
        max_e2e: int = 20,
    ):
        pass
