#!/bin/bash
# Full training run for one model.
#
# Usage (from this folder):
#   sbatch --export=ALL,MODEL=baseline train.slurm
#   sbatch --export=ALL,MODEL=improved train.slurm
#   sbatch --export=ALL,MODEL=improved,EPOCHS=80 train.slurm
#SBATCH --job-name=hipmri-train
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

MODEL=${MODEL:-improved}
EPOCHS=${EPOCHS:-50}

source $HOME/miniconda3/bin/activate
conda activate torch

echo "Job $SLURM_JOB_ID on $(hostname), started $(date) | model=$MODEL epochs=$EPOCHS"
nvidia-smi --query-gpu=name,memory.total --format=csv

python train.py --model "$MODEL" --epochs "$EPOCHS" --out "runs/$MODEL"

echo "Finished $(date)"
