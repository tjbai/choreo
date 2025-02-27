#!/bin/bash
#SBATCH --job-name=finetune_n16
#SBATCH -A jeisner1_gpu
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/triviaqa/finetune_n8.out

uv run python -m llama.recipes.qa \
	--num_question=8 \
	--num_eval=50
