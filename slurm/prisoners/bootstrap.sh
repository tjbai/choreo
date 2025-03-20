#!/bin/bash

#SBATCH --job-name=prisoners/bootstrap
#SBATCH -A jeisner1_gpu
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=16:00:00
#SBATCH --output=slurm/prisoners/bootstrap_%a.out
#SBATCH --array=2

strategies=("None" "always_defect" "always_cooperate")
strategy=${strategies[$SLURM_ARRAY_TASK_ID]}
uv run slurm/prisoners/bootstrap.py --strategy=${strategy}
