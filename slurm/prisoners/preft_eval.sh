#!/bin/bash

#SBATCH --job-name=prisoners/prediction
#SBATCH -A jeisner1_gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --output=slurm/prisoners/prediction_%a.out
#SBATCH --array=0-2

strategies=("None" "always_defect" "always_cooperate")
strategy=${strategies[$SLURM_ARRAY_TASK_ID]}
uv run slurm/prisoners/preft_eval.py --strategy=${strategy}
