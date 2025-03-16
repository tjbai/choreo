#!/bin/bash
#SBATCH --job-name=bsm/sweep
#SBATCH -A jeisner1_gpu
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --array=1-16
#SBATCH --time=12:00:00
#SBATCH --output=slurm/bsm/sweep_%A_%a.out

SWEEP_ID=$(cat /home/tbai4/llama3/sweep/sweep_bsm.txt)
uv run wandb agent --count 1 $SWEEP_ID

