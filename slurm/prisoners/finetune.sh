#!/bin/bash

#SBATCH --job-name=finetune_prisoners
#SBATCH -A jeisner1_gpu
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/prisoners/finetune_%a.out
#SBATCH --array=0-2

strategies=("None" "always_defect" "always_cooperate")
strategy=${strategies[$SLURM_ARRAY_TASK_ID]}
uv run slurm/prisoners/finetune.py --strategy=${strategy}
