#!/bin/bash
#SBATCH --job-name=finetune_n4
#SBATCH -A jeisner1_gpu
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/triviaqa/finetune_n4.out

uv run python -m llama.recipes.qa \
	--num_questions=4 \
	--num_eval=50
