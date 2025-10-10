#!/bin/bash
#SBATCH --job-name=madpar_ft
#SBATCH -A jeisner1_gpu
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=madpar_ft.out

uv run python slurm/qwen/ft.py --task madpar --lora_rank 8
