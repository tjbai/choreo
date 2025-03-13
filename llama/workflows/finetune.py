import os
import math
from typing import Tuple, Optional

import wandb
import fire
import torch
from torch.utils.data import Dataset, Subset, random_split
import numpy as np
from tqdm import tqdm

from llama import Workflow
from llama.util import find_free_port
from llama.workflows.prisoners import baseline_nll
from llama.workflows.trainers import (
    LoraTrainer,
    ListDataset,
    MadTrainer,
    BsmTrainer,
    PrisonersTrainer,
    PrisonersDataset,
    QaTrainer,
    TotTrainer,
    TotDataset
)

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
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    learning_rate: float,
    **task_params
) -> Tuple[LoraTrainer, Dataset]:
    if task == 'mad':
        trainer = MadTrainer(
            workflow,
            output_dir=output_dir,
            learning_rate=learning_rate
        )
        dataset = ListDataset(data_path)
        dataset = Subset(
            dataset=dataset,
            indices=[i for i, d in enumerate(dataset) if isinstance(d['outputs']['decision'], dict)]
        )
        print(f'Filtered to {len(dataset)}')
        wandb.init(
            project='mad',
            config={
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "learning_rate": learning_rate,
            }
        )
        return trainer, dataset
    if task == 'bsm':
        trainer = BsmTrainer(
            workflow,
            output_dir=output_dir,
            learning_rate=learning_rate
        )
        dataset = ListDataset(data_path)
        wandb.init(
            project='bsm',
            config={
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "learning_rate": learning_rate,
            }
        )
        return trainer, dataset
    elif task == 'tot':
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
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "learning_rate": learning_rate,
            }
        )
        return trainer, dataset
    elif task == 'prisoners':
        trainer = PrisonersTrainer(
            workflow=workflow,
            output_dir=output_dir,
            learning_rate=learning_rate
        )
        dataset = PrisonersDataset(data_path, task_params['strategy'])

        indices = []
        for i in tqdm(range(len(dataset)), desc='Filtering'):
            try:
                nll = baseline_nll(
                    workflow,
                    dataset[i]['result'],
                    payoff=dataset[i]['payoff'],
                    alice_first=dataset[i]['alice_first'],
                    alice_strategy=dataset[i]['strategy'],
                )
                if (np.mean(nll['bob_nll'][0] + nll['bob_nll'][1]) < 4 and
                    np.mean(nll['alice_nll'][0] + nll['alice_nll'][1]) < 4):
                    indices.append(i)
            except IndexError:
                continue
        dataset = Subset(dataset, indices)

        wandb.init(
            project="prisoners",
            config={
                "strategy": task_params['strategy'],
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "learning_rate": learning_rate,
            }
        )
        return trainer, dataset
    elif task == 'triviaqa':
        trainer = QaTrainer(
            workflow=workflow,
            output_dir=output_dir,
            learning_rate=learning_rate
        )
        dataset = ListDataset(data_path)
        wandb.init(
            project='triviaqa',
            config={
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "learning_rate": learning_rate,
            }
        )
        return trainer, dataset
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
    max_seq_len: int = 8192,
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
    # optionally, pass in a pre-configured workflow
    workflow: Optional[Workflow] = None,
):
    if workflow is None:
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

    # always reshape just in case
    workflow.model.reshape_cache(1)

    trainer, dataset = init_task(
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
    trainer.evaluate(val_dataset, max_steps=1, max_e2e=1)
    print("Passed!")

    epochs, steps, warmup_steps = training_schedule(
        epochs=epochs,
        steps=steps,
        dataset_size=len(train_dataset),
        gradient_accumulation_steps=gradient_accumulation_steps
    )

    print(f'Epochs: {epochs}, Steps: {steps}, Warmup: {warmup_steps}')
    validation_freq = min(validation_freq, steps // 2)
    print(f'Validation freq: {validation_freq}')

    global_step = 0
    for epoch in range(epochs):
        indices = torch.randperm(len(train_dataset)).tolist()
        for step, idx in enumerate(tqdm(indices, desc=f"Epoch {epoch}")):
            lr_factor = get_lr_factor(global_step, warmup_steps, steps)
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = learning_rate * lr_factor
            step_result = trainer.step(dataset[idx])
            if step_result is None:
                continue # will create jagged batches
            loss, metrics = step_result
            loss.backward()
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainer.workflow.model.parameters(), max_norm=1.0)
                trainer.optimizer.step()
                trainer.optimizer.zero_grad()
                global_step += 1
                if (global_step + 1) % int(validation_freq) == 0:
                    val_metrics = trainer.evaluate(val_dataset)
                    wandb.log(val_metrics)
                metrics.update({'lr': lr_factor})
                wandb.log(metrics)
                if (global_step + 1) % int(checkpoint_freq) == 0:
                    trainer.save_checkpoint(global_step)
                if steps is not None and global_step == steps:
                    break

    trainer.save_checkpoint(global_step)
    wandb.finish()

if __name__ == '__main__':
    fire.Fire(finetune)
