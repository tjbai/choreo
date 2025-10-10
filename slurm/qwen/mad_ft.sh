#!/bin/bash
#SBATCH --job-name=mad_ft
#SBATCH -A jeisner1_gpu
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=mad_ft.out

uv run python slurm/qwen/ft.py --task mad --lora_rank 32
