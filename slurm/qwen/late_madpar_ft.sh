#!/bin/bash
#SBATCH --job-name=late_madpar_ft
#SBATCH -A jeisner1_gpu
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --array=0-4
#SBATCH --time=12:00:00
#SBATCH --output=late_madpar_ft_%A_%a.out

uv run python slurm/qwen/test.py --workflow_type=madpar_cached --shard_idx=${SLURM_ARRAY_TASK_ID} --lora_ckpt_path /home/tbai4/llama3/checkpoints/lora_step-1199.pt
