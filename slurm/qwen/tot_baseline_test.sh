#!/bin/bash
#SBATCH --job-name=tot_baseline_test
#SBATCH -A jeisner1_gpu
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --array=0-4
#SBATCH --time=12:00:00
#SBATCH --output=tot_baseline_test_%A_%a.out

uv run python slurm/qwen/test.py --workflow_type=tot_baseline --shard_idx=${SLURM_ARRAY_TASK_ID}
