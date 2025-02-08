#!/bin/bash
#SBATCH --job-name=tot-sweep
#SBATCH -A jeisner1_gpu
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --array=1-20
#SBATCH --time=12:00:00
#SBATCH --output=slurm/tot_sweep_%A_%a.out

SWEEP_ID=$(cat /home/tbai4/llama3/sweep/sweep_tot.txt)
uv run wandb agent --count 1 $SWEEP_ID

