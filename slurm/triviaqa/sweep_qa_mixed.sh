#!/bin/bash
#SBATCH --job-name=qa-sweep
#SBATCH -A jeisner1_gpu
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --array=1-32
#SBATCH --time=12:00:00
#SBATCH --output=slurm/triviaqa/qa_sweep_mixed_%A_%a.out

SWEEP_ID=$(cat /home/tbai4/llama3/sweep/sweep_qa_mixed.txt)
uv run wandb agent --count 1 $SWEEP_ID

