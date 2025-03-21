#!/bin/bash
#SBATCH --job-name=prisoners/ablations
#SBATCH -A jeisner1_gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/prisoners/ablations_%a.out
#SBATCH --array=0-5

base_idx=$((SLURM_ARRAY_TASK_ID / 2))
strategies=("None" "always_defect" "always_cooperate")
strategy=${strategies[$base_idx]}

leak_config=$((SLURM_ARRAY_TASK_ID % 2))
if [ $leak_config -eq 0 ]; then
    uv run slurm/prisoners/preft_eval.py --strategy=${strategy} --only_leak_sys=True --only_leak_plan=False
else
    uv run slurm/prisoners/preft_eval.py --strategy=${strategy} --only_leak_sys=False --only_leak_plan=True
fi
