#!/bin/bash

#SBATCH --job-name=tot-sweep
#SBATCH -A jeisner1_gpu
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --array=1-20
#SBATCH --time=12:00:00
#SBATCH --output=tot_sweep.out

if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    SWEEP_ID=$(uv run wandb sweep --project "tot" sweep.yaml --verbose | tail -1 | awk '{print $NF}')
    echo $SWEEP_ID > slurm/sweep_id.txt
else
    while [ ! -f sweep_id.txt ]; do sleep 1; done
    SWEEP_ID=$(cat slurm/sweep_id.txt)
fi

uv run wandb agent --count 1 $SWEEP_ID
