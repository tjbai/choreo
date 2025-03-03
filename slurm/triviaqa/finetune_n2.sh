#!/bin/bash
#SBATCH --job-name=finetune_n2
#SBATCH -A jeisner1_gpu
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/triviaqa/finetune_n2.out

uv run python -m llama.recipes.qa \
	--num_questions=2 \
	--num_eval=50 \
	--num_epochs=2 \
	--num_examples=500
